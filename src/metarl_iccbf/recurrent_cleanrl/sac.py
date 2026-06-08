"""CleanRL-style recurrent SAC with SB3-compatible predict / save / load.

Off-policy recurrent SAC supporting Mamba2, GRU, and LSTM backends.
Key differences from PPO:

- Off-policy: uses replay buffer instead of rollout buffer
- Twin Q-networks instead of value function
- Squashed Gaussian policy with entropy regularization
- Target networks with Polyak (EMA) averaging
- Automatic entropy coefficient tuning
"""

from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from .sac_agent import Mamba2SACAgent, GRUSACAgent, LSTMSACAgent
from .replay_buffer import SequenceReplayBuffer, ChunkCollector


# ---------------------------------------------------------------------------
# Helpers (shared with ppo.py)
# ---------------------------------------------------------------------------

def _get_device(device: Union[str, torch.device]) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _wrap_env(env) -> VecEnv:
    if isinstance(env, VecEnv):
        return env
    return DummyVecEnv([lambda: env])


def _space_to_dict(space: gym.spaces.Space) -> dict[str, Any]:
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": list(space.shape),
            "dtype": str(space.dtype),
        }
    raise NotImplementedError(f"Cannot serialise {type(space)}")


def _dict_to_space(d: dict[str, Any]) -> gym.spaces.Space:
    if d["type"] == "Box":
        return gym.spaces.Box(
            low=np.array(d["low"], dtype=np.float32),
            high=np.array(d["high"], dtype=np.float32),
            shape=tuple(d["shape"]),
        )
    raise NotImplementedError(f"Cannot deserialise {d['type']}")


def _n_predict_envs(states, model_type: str) -> int:
    """Infer n_envs from recurrent states (architecture-agnostic)."""
    if model_type == "mamba2":
        # Mamba2: (conv_state, ssm_state) — first dim is n_envs
        return states[0].shape[0]
    elif model_type == "lstm":
        # LSTM: (h, c) where h is (1, n_envs, hidden_size)
        return states[0].shape[1]
    else:
        # GRU: (1, n_envs, hidden_size)
        return states.shape[1]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RecurrentSAC:
    """Recurrent SAC agent with CleanRL-style training loop.

    Supports Mamba2, GRU, and LSTM backends via ``model_type``.
    Provides SB3-compatible ``.predict()``, ``.save()``, and ``.load()``
    so existing evaluation scripts work unchanged.
    """

    def __init__(
        self,
        env: Optional[Union[gym.Env, VecEnv]],
        *,
        # -- Backend selection --
        model_type: Literal["mamba2", "gru", "lstm"] = "mamba2",
        # -- Mamba2 (ignored when model_type="gru") --
        mamba_d_model: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        # -- GRU (ignored when model_type="mamba2") --
        hidden_size: int = 64,
        # -- Network --
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        # -- SAC --
        learning_rate: Union[float, Callable[[float], float]] = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: Union[str, float] = "auto",
        ent_coef_lr: Optional[float] = None,
        ent_coef_min: float = 0.01,
        target_entropy: Union[str, float] = "auto",
        # -- Replay buffer --
        buffer_size: int = 100_000,
        chunk_len: int = 32,
        burn_in: int = 32,
        batch_size: int = 64,
        # -- Training --
        learning_starts: int = 1000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        max_grad_norm: float = 0.5,
        # -- Misc --
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "auto",
        verbose: int = 1,
        # -- Internal (used by load()) --
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
    ):
        self.device = _get_device(device)
        self.verbose = verbose
        self.model_type = model_type

        # ---- Environment ----
        if env is not None:
            self.env = _wrap_env(env)
            self.n_envs = self.env.num_envs
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        else:
            self.env = None  # type: ignore[assignment]
            self.n_envs = 1
            self.observation_space = None  # type: ignore[assignment]
            self.action_space = None  # type: ignore[assignment]

        obs_shape = self.observation_space.shape if self.observation_space is not None else ()
        act_shape = self.action_space.shape if self.action_space is not None else ()
        _obs_dim = obs_dim if obs_dim is not None else (int(np.prod(obs_shape)) if obs_shape else 1)
        _act_dim = act_dim if act_dim is not None else (int(np.prod(act_shape)) if act_shape else 1)
        obs_dim = _obs_dim
        act_dim = _act_dim

        # ---- Seed ----
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ---- Store config ----
        self._config = dict(
            model_type=model_type,
            mamba_d_model=mamba_d_model,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
            hidden_size=hidden_size,
            net_arch=net_arch,
            activation_fn=activation_fn.__name__
            if isinstance(activation_fn, type)
            else type(activation_fn).__name__,
            learning_rate=learning_rate
            if isinstance(learning_rate, (int, float))
            else None,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef if isinstance(ent_coef, (int, float)) else "auto",
            ent_coef_lr=ent_coef_lr,
            ent_coef_min=ent_coef_min,
            target_entropy=target_entropy if isinstance(target_entropy, (int, float)) else "auto",
            buffer_size=buffer_size,
            chunk_len=chunk_len,
            burn_in=burn_in,
            batch_size=batch_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            max_grad_norm=max_grad_norm,
            obs_dim=obs_dim,
            act_dim=act_dim,
            obs_space=_space_to_dict(self.observation_space) if self.observation_space is not None else None,
            act_space=_space_to_dict(self.action_space) if self.action_space is not None else None,
        )

        # ---- Hyperparams ----
        self._lr_schedule: Optional[Callable[[float], float]] = None
        if callable(learning_rate) and not isinstance(learning_rate, (int, float)):
            self._lr_schedule = learning_rate
            self._base_lr = learning_rate(1.0)
        else:
            self._base_lr = float(learning_rate)

        self.gamma = gamma
        self.tau = tau
        self.chunk_len = chunk_len
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.max_grad_norm = max_grad_norm

        # ---- Actor-Critic ----
        if model_type == "gru":
            self.agent = GRUSACAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=hidden_size,
                net_arch=net_arch,
                activation_fn=activation_fn,
            ).to(self.device)
        elif model_type == "lstm":
            self.agent = LSTMSACAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=hidden_size,
                net_arch=net_arch,
                activation_fn=activation_fn,
            ).to(self.device)
        else:
            self.agent = Mamba2SACAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                mamba_d_model=mamba_d_model,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_headdim=mamba_headdim,
                net_arch=net_arch,
                activation_fn=activation_fn,
            ).to(self.device)

        # ---- Target networks (backend-agnostic) ----
        src = self.agent.target_critic_source_modules()
        self.target_critic = nn.ModuleDict({
            k: deepcopy(v) for k, v in src.items()
        }).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # ---- Optimizers (backend-agnostic) ----
        self.actor_optimizer = torch.optim.Adam(
            list(self.agent.actor_parameters()),
            lr=self._base_lr,
            eps=1e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.agent.critic_parameters()),
            lr=self._base_lr,
            eps=1e-5,
        )

        # ---- Automatic entropy tuning ----
        self.ent_coef_min = ent_coef_min
        if ent_coef == "auto":
            if target_entropy == "auto":
                self.target_entropy = -float(act_dim)
            else:
                self.target_entropy = float(target_entropy)
            self.log_ent_coef = torch.zeros(1, requires_grad=True, device=self.device)
            _ent_lr = ent_coef_lr if ent_coef_lr is not None else self._base_lr
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef], lr=_ent_lr
            )
            self.ent_coef = self.log_ent_coef.exp().item()
        else:
            self.ent_coef = float(ent_coef)
            self.log_ent_coef = None
            self.ent_coef_optimizer = None
            self.target_entropy = None

        # ---- Replay buffer ----
        self.replay_buffer = SequenceReplayBuffer(
            capacity=buffer_size,
            chunk_len=chunk_len,
            burn_in=burn_in,
            obs_shape=obs_shape,
            act_shape=act_shape,
        )

        # ---- Chunk collector ----
        self._chunk_collector = ChunkCollector(
            n_envs=self.n_envs,
            chunk_len=chunk_len,
            burn_in=burn_in,
            obs_shape=obs_shape,
            act_shape=act_shape,
        )

        # ---- Running state ----
        self._last_obs: Optional[np.ndarray] = None
        self._last_episode_starts: Optional[np.ndarray] = None
        self._last_actor_states: Any = None

        # ---- Predict state (for inference / eval) ----
        self._predict_actor_states: Any = None

        # ---- Logging ----
        self.num_timesteps = 0
        self._n_updates = 0

    # ==================================================================
    # Training
    # ==================================================================

    def learn(
        self,
        total_timesteps: int,
        *,
        eval_env: Optional[Union[gym.Env, VecEnv]] = None,
        eval_freq: Optional[int] = None,
        n_eval_episodes: int = 10,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        wandb_run: Any = None,
        tb_log_dir: Optional[str] = None,
        tb_log_name: Optional[str] = None,
        progress_bar: bool = True,
    ) -> "RecurrentSAC":
        """Main training loop. Returns *self* for chaining."""
        # ---- TensorBoard ----
        tb_writer = None
        if tb_log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            tb_path = Path(tb_log_dir) / (tb_log_name or "RecurrentSAC")
            tb_writer = SummaryWriter(str(tb_path))

        # ---- Eval bookkeeping ----
        best_mean_reward = -float("inf")
        eval_results: dict[str, list] = {
            "timesteps": [], "results": [], "ep_lengths": [],
        }

        # ---- Progress bar ----
        pbar = None
        if progress_bar:
            from tqdm import tqdm
            pbar = tqdm(total=total_timesteps, desc="Training SAC", unit="step")

        # ---- Init env state ----
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            self._last_obs = reset_result[0]
        else:
            self._last_obs = reset_result
        self._last_episode_starts = np.ones(self.n_envs, dtype=np.float32)
        self._last_actor_states = self.agent.initial_actor_states(
            self.n_envs, self.device
        )

        if pbar is not None and self.num_timesteps > 0:
            pbar.update(self.num_timesteps)

        train_info: dict[str, float] = {}
        steps_since_train = 0

        while self.num_timesteps < total_timesteps:
            # ---- Action selection ----
            obs_t = torch.as_tensor(
                self._last_obs, dtype=torch.float32, device=self.device
            )
            ep_t = torch.as_tensor(
                self._last_episode_starts, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                if self.num_timesteps < self.learning_starts:
                    action_np = np.array([
                        self.action_space.sample() for _ in range(self.n_envs)
                    ])
                else:
                    action, _log_prob, self._last_actor_states = (
                        self.agent.step_actor(
                            obs_t, self._last_actor_states, ep_t,
                            deterministic=False,
                        )
                    )
                    action_np = action.cpu().numpy()

            # ---- Env step ----
            new_obs, rewards, dones, infos = self.env.step(action_np)
            if isinstance(new_obs, tuple):
                new_obs = new_obs[0]

            # ---- Store transitions in chunk collector ----
            for i in range(self.n_envs):
                chunk_ready = self._chunk_collector.add_step(
                    env_idx=i,
                    obs=self._last_obs[i],
                    action=action_np[i],
                    reward=rewards[i],
                    next_obs=new_obs[i],
                    episode_start=self._last_episode_starts[i],
                    done=float(dones[i]),
                )
                if chunk_ready:
                    chunk = self._chunk_collector.get_chunk(i)
                    self.replay_buffer.add(*chunk)

                # Reset recurrent states on done
                if dones[i]:
                    self.agent.zero_env_actor_states(
                        self._last_actor_states, i
                    )

            self._last_obs = new_obs
            self._last_episode_starts = dones.astype(np.float32)
            self.num_timesteps += self.n_envs
            steps_since_train += 1

            if pbar is not None:
                pbar.update(self.n_envs)

            # ---- LR schedule ----
            if self._lr_schedule is not None:
                progress_remaining = 1.0 - self.num_timesteps / total_timesteps
                new_lr = self._lr_schedule(progress_remaining)
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = new_lr
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = new_lr

            # ---- Gradient updates ----
            if (
                self.num_timesteps >= self.learning_starts
                and self.replay_buffer.size >= self.batch_size
                and steps_since_train >= self.train_freq
            ):
                steps_since_train = 0
                for _ in range(self.gradient_steps):
                    train_info = self._train_step()
                    self._n_updates += 1

                # ---- Logging ----
                if tb_writer is not None:
                    for k, v in train_info.items():
                        tb_writer.add_scalar(f"train/{k}", v, self.num_timesteps)

                if wandb_run is not None:
                    try:
                        import wandb
                        wandb.log(
                            {f"train/{k}": v for k, v in train_info.items()},
                            step=self.num_timesteps,
                        )
                    except ImportError:
                        pass

                if self.verbose >= 2:
                    c_loss = train_info.get("critic_loss", float("nan"))
                    a_loss = train_info.get("actor_loss", float("nan"))
                    ent = train_info.get("ent_coef", float("nan"))
                    print(
                        f"Step {self.num_timesteps}  "
                        f"critic_loss={c_loss:.4f}  actor_loss={a_loss:.4f}  "
                        f"ent_coef={ent:.4f}"
                    )

            # ---- Periodic evaluation ----
            if (
                eval_env is not None
                and eval_freq is not None
                and self.num_timesteps % eval_freq < self.n_envs
                and self.num_timesteps >= self.learning_starts
            ):
                mean_reward, mean_len = self._evaluate(eval_env, n_eval_episodes)
                eval_results["timesteps"].append(self.num_timesteps)
                eval_results["results"].append(mean_reward)
                eval_results["ep_lengths"].append(mean_len)

                if self.verbose >= 1:
                    print(
                        f"Eval @ {self.num_timesteps} steps: "
                        f"mean_reward={mean_reward:.2f}  mean_len={mean_len:.0f}"
                    )

                if tb_writer is not None:
                    tb_writer.add_scalar(
                        "eval/mean_reward", mean_reward, self.num_timesteps,
                    )

                if wandb_run is not None:
                    try:
                        import wandb
                        wandb.log(
                            {"eval/mean_reward": mean_reward},
                            step=self.num_timesteps,
                        )
                    except ImportError:
                        pass

                if mean_reward > best_mean_reward and best_model_save_path:
                    best_mean_reward = mean_reward
                    self.save(Path(best_model_save_path) / "best_model.zip")
                    if self.verbose >= 1:
                        print(f"  New best model saved (reward={mean_reward:.2f})")

        # ---- Cleanup ----
        if pbar is not None:
            pbar.close()
        if tb_writer is not None:
            tb_writer.close()

        if log_path and eval_results["timesteps"]:
            np.savez(
                Path(log_path) / "evaluations.npz",
                timesteps=np.array(eval_results["timesteps"]),
                results=np.array(eval_results["results"]),
                ep_lengths=np.array(eval_results["ep_lengths"]),
            )

        return self

    # ------------------------------------------------------------------
    # SAC update
    # ------------------------------------------------------------------

    def _train_step(self) -> dict[str, float]:
        """One SAC gradient update on a sampled batch."""
        self.agent.train()
        batch = self.replay_buffer.sample(self.batch_size, self.device)

        B = self.burn_in

        # Split sequences into burn-in context and training region
        ctx_obs = batch["obs"][:, :B, :]
        ctx_ep = batch["episode_starts"][:, :B]

        obs = batch["obs"][:, B:, :]
        actions = batch["actions"][:, B:, :]
        rewards = batch["rewards"][:, B:]
        next_obs = batch["next_obs"][:, B:, :]
        ep_starts = batch["episode_starts"][:, B:]
        dones = batch["dones"][:, B:]

        # Context for next_obs: shifted by 1 step
        ctx_next_obs = batch["obs"][:, 1:B + 1, :]
        ctx_next_ep = batch["episode_starts"][:, 1:B + 1]

        # Episode starts for next_obs
        next_ep_starts = dones.clone()

        # ---- Critic update ----
        with torch.no_grad():
            # Sample actions from current policy for next_obs
            next_actions, next_log_probs = self.agent.forward_actor(
                next_obs, next_ep_starts, ctx_next_obs, ctx_next_ep,
            )
            # Target Q-values
            next_q1, next_q2 = self.agent.forward_critic(
                next_obs, next_actions, next_ep_starts,
                ctx_next_obs, ctx_next_ep,
                target_modules=self.target_critic,
            )
            next_q = torch.min(next_q1, next_q2) - self.ent_coef * next_log_probs
            target_q = rewards + self.gamma * (1.0 - dones) * next_q
            target_q = target_q.clamp(-1e4, 1e4)

        # Online Q-values
        q1, q2 = self.agent.forward_critic(
            obs, actions, ep_starts, ctx_obs, ctx_ep,
        )

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Skip this entire update if critic loss is NaN/Inf to prevent
        # corrupting weights (which cascades into CUDA kernel crashes).
        if not torch.isfinite(critic_loss):
            if self.verbose >= 1:
                print(
                    f"[WARN] Step {self.num_timesteps}: non-finite critic_loss "
                    f"({critic_loss.item():.4g}), skipping update"
                )
            return {
                "critic_loss": float("nan"),
                "actor_loss": float("nan"),
                "ent_coef": self.ent_coef,
                "ent_coef_loss": 0.0,
                "q1_mean": float("nan"),
                "q2_mean": float("nan"),
            }

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent.critic_parameters()),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        # ---- Actor update ----
        # Freeze critic for actor update
        critic_params = list(self.agent.critic_parameters())
        for p in critic_params:
            p.requires_grad = False

        new_actions, log_probs = self.agent.forward_actor(
            obs, ep_starts, ctx_obs, ctx_ep,
        )
        q1_pi, q2_pi = self.agent.forward_critic(
            obs, new_actions, ep_starts, ctx_obs, ctx_ep,
        )
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.ent_coef * log_probs - q_pi).mean()

        # Skip actor update if loss is non-finite
        if torch.isfinite(actor_loss):
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.agent.actor_parameters()),
                self.max_grad_norm,
            )
            self.actor_optimizer.step()
        elif self.verbose >= 1:
            print(
                f"[WARN] Step {self.num_timesteps}: non-finite actor_loss "
                f"({actor_loss.item():.4g}), skipping actor update"
            )

        # Unfreeze critic
        for p in critic_params:
            p.requires_grad = True

        # ---- Entropy coefficient update ----
        ent_coef_loss = 0.0
        if self.log_ent_coef is not None:
            ent_coef_loss_val = (
                -self.log_ent_coef.exp() * (log_probs.detach() + self.target_entropy)
            ).mean()
            if torch.isfinite(ent_coef_loss_val):
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss_val.backward()
                self.ent_coef_optimizer.step()
            # Enforce entropy coefficient floor
            with torch.no_grad():
                log_min = np.log(self.ent_coef_min)
                self.log_ent_coef.data.clamp_(min=log_min)
            self.ent_coef = self.log_ent_coef.exp().item()
            ent_coef_loss = ent_coef_loss_val.item()

        # ---- Polyak update target networks ----
        self._polyak_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "ent_coef": self.ent_coef,
            "ent_coef_loss": ent_coef_loss,
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def _polyak_update(self) -> None:
        """Soft update target networks: theta_tgt = tau*theta + (1-tau)*theta_tgt."""
        with torch.no_grad():
            for p_online, p_target in zip(
                self.agent.critic_parameters(),
                self.target_critic.parameters(),
            ):
                p_target.data.mul_(1.0 - self.tau).add_(
                    p_online.data, alpha=self.tau
                )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_episodes: int,
    ) -> tuple[float, float]:
        """Run *n_episodes* deterministic rollouts, return mean reward & length."""
        if not isinstance(eval_env, VecEnv):
            eval_env_wrapped = DummyVecEnv([lambda: eval_env])
        else:
            eval_env_wrapped = eval_env

        n_eval_envs = eval_env_wrapped.num_envs

        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        running_rewards = np.zeros(n_eval_envs)
        running_lengths = np.zeros(n_eval_envs, dtype=int)

        obs = eval_env_wrapped.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        episode_starts = np.ones(n_eval_envs, dtype=np.float32)
        actor_states = self.agent.initial_actor_states(n_eval_envs, self.device)

        self.agent.eval()
        while len(ep_rewards) < n_episodes:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            ep_t = torch.as_tensor(
                episode_starts, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                action, _, actor_states = self.agent.step_actor(
                    obs_t, actor_states, ep_t, deterministic=True,
                )

            action_np = action.cpu().numpy()
            # SAC tanh output is in (-1, 1); clip for safety
            action_np = np.clip(action_np, -1.0, 1.0)

            obs, rewards, dones, infos = eval_env_wrapped.step(action_np)
            if isinstance(obs, tuple):
                obs = obs[0]

            running_rewards += rewards
            running_lengths += 1

            for i in range(n_eval_envs):
                if dones[i]:
                    ep_rewards.append(running_rewards[i])
                    ep_lengths.append(running_lengths[i])
                    running_rewards[i] = 0.0
                    running_lengths[i] = 0
                    self.agent.zero_env_actor_states(actor_states, i)

            episode_starts = dones.astype(np.float32)

        return float(np.mean(ep_rewards[:n_episodes])), float(
            np.mean(ep_lengths[:n_episodes])
        )

    # ==================================================================
    # Predict (SB3-compatible interface)
    # ==================================================================

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
        episode_start: Optional[Union[np.ndarray, bool]] = None,
        state: Optional[Any] = None,
    ) -> tuple[np.ndarray, None]:
        """SB3-compatible predict. Same interface as RecurrentPPO.predict()."""
        self.agent.eval()

        obs = np.asarray(observation, dtype=np.float32)
        was_single = obs.ndim == len(self.observation_space.shape)
        if was_single:
            obs = obs[np.newaxis, ...]
        n_envs = obs.shape[0]

        if (
            self._predict_actor_states is None
            or _n_predict_envs(self._predict_actor_states, self.model_type) != n_envs
        ):
            self._predict_actor_states = self.agent.initial_actor_states(
                n_envs, self.device
            )

        if episode_start is None:
            ep_starts = np.zeros(n_envs, dtype=np.float32)
        elif isinstance(episode_start, (bool, int, float)):
            ep_starts = np.full(n_envs, float(episode_start), dtype=np.float32)
        else:
            ep_starts = np.asarray(episode_start, dtype=np.float32).flatten()
            if ep_starts.shape[0] == 1 and n_envs > 1:
                ep_starts = np.full(n_envs, ep_starts[0], dtype=np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        ep_t = torch.as_tensor(ep_starts, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action, _, self._predict_actor_states = self.agent.step_actor(
                obs_t, self._predict_actor_states, ep_t,
                deterministic=deterministic,
            )

        action_np = action.cpu().numpy()
        # Tanh output is already in (-1, 1); clip for numerical safety
        action_np = np.clip(action_np, -1.0, 1.0)

        if was_single:
            action_np = action_np.squeeze(0)

        return action_np, None

    # ==================================================================
    # Save / Load
    # ==================================================================

    def save(self, path: Union[str, Path]) -> None:
        """Save model to a ``.zip`` file (config + weights)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_bytes = json.dumps(self._config, indent=2).encode()

        weights_buf = io.BytesIO()
        torch.save({
            "agent": self.agent.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_ent_coef": (
                self.log_ent_coef.detach().cpu()
                if self.log_ent_coef is not None
                else None
            ),
        }, weights_buf)
        weights_bytes = weights_buf.getvalue()

        training_state = json.dumps({
            "num_timesteps": self.num_timesteps,
            "n_updates": self._n_updates,
            "ent_coef": self.ent_coef,
        }).encode()

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("config.json", config_bytes)
            zf.writestr("policy.pt", weights_bytes)
            zf.writestr("training_state.json", training_state)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[Union[gym.Env, VecEnv]] = None,
        device: Union[str, torch.device] = "auto",
        **kwargs,
    ) -> "RecurrentSAC":
        """Load a model from a ``.zip`` file."""
        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            config = json.loads(zf.read("config.json"))
            weights_bytes = zf.read("policy.pt")

        # Resolve activation function
        act_fn_name = config.pop("activation_fn", "ReLU")
        act_fn_map = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "GELU": nn.GELU}
        activation_fn = act_fn_map.get(act_fn_name, nn.ReLU)

        # Resolve spaces
        obs_space_dict = config.pop("obs_space")
        act_space_dict = config.pop("act_space")
        obs_space = _dict_to_space(obs_space_dict)
        act_space = _dict_to_space(act_space_dict)

        # Pop lr if None
        lr = config.pop("learning_rate", None)
        if lr is None:
            lr = 3e-4

        instance = cls(
            env,
            activation_fn=activation_fn,
            learning_rate=lr,
            device=device,
            **config,
        )

        if instance.observation_space is None:
            instance.observation_space = obs_space
        if instance.action_space is None:
            instance.action_space = act_space

        # Load weights
        weights_buf = io.BytesIO(weights_bytes)
        checkpoint = torch.load(
            weights_buf, map_location=instance.device, weights_only=True,
        )
        instance.agent.load_state_dict(checkpoint["agent"])
        instance.target_critic.load_state_dict(checkpoint["target_critic"])

        if checkpoint.get("log_ent_coef") is not None and instance.log_ent_coef is not None:
            instance.log_ent_coef.data.copy_(checkpoint["log_ent_coef"].to(instance.device))
            instance.ent_coef = instance.log_ent_coef.exp().item()

        instance.agent.eval()

        # Restore training state
        with zipfile.ZipFile(path, "r") as zf:
            if "training_state.json" in zf.namelist():
                ts = json.loads(zf.read("training_state.json"))
                instance.num_timesteps = ts.get("num_timesteps", 0)
                instance._n_updates = ts.get("n_updates", 0)

        return instance


# Backward-compatible alias
Mamba2SAC = RecurrentSAC
