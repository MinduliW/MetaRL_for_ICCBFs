"""CleanRL-style recurrent PPO with SB3-compatible predict / save / load.

Supports multiple recurrent backends (Mamba2, GRU) via ``model_type``.

Standalone training loop — no dependency on SB3's ``RecurrentPPO``,
``RecurrentRolloutBuffer``, or ``RecurrentActorCriticPolicy``.

The only SB3 / gymnasium dependency is ``DummyVecEnv`` for wrapping
single environments and the standard ``VecEnv`` interface.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from .agent import GRUActorCritic, LSTMActorCritic, Mamba2ActorCritic
from .buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PopArtNormalizer:
    """Running PopArt normalizer for scalar reward targets (MOPPO paper eq. 18).

    Maintains exponential moving estimates of μ and σ over incoming targets.
    Use ``update()`` before GAE to fit statistics, then ``normalize()`` /
    ``unnormalize()`` to convert between raw and normalized scales.
    """

    def __init__(self, beta: float = 3e-4, epsilon: float = 1e-5):
        self.mu = 0.0
        self.sigma = 1.0
        self.beta = beta
        self.epsilon = epsilon
        self._nu = 1.0

    def update(self, targets: np.ndarray) -> None:
        batch_mu = float(targets.mean())
        batch_nu = float((targets ** 2).mean())
        self.mu = (1.0 - self.beta) * self.mu + self.beta * batch_mu
        self._nu = (1.0 - self.beta) * self._nu + self.beta * batch_nu
        self.sigma = max(float(np.sqrt(abs(self._nu - self.mu ** 2))), self.epsilon)

    def normalize(self, targets: np.ndarray) -> np.ndarray:
        return (targets - self.mu) / self.sigma

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.sigma + self.mu


def _get_device(device: Union[str, torch.device]) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _wrap_env(env) -> VecEnv:
    """Ensure *env* is a ``VecEnv``.  Wrap if necessary."""
    if isinstance(env, VecEnv):
        return env
    return DummyVecEnv([lambda: env])


def _space_to_dict(space: gym.spaces.Space) -> dict[str, Any]:
    """Serialise a Box space to a JSON-safe dict."""
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
    """Reconstruct a space from :func:`_space_to_dict` output."""
    if d["type"] == "Box":
        return gym.spaces.Box(
            low=np.array(d["low"], dtype=np.float32),
            high=np.array(d["high"], dtype=np.float32),
            shape=tuple(d["shape"]),
        )
    raise NotImplementedError(f"Cannot deserialise {d['type']}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RecurrentPPO:
    """Recurrent PPO agent with CleanRL-style training loop.

    Supports Mamba2 and GRU backends via ``model_type``.

    Provides SB3-compatible ``.predict()``, ``.save()``, and ``.load()``
    so existing evaluation scripts work unchanged.
    """

    def __init__(
        self,
        env: Optional[Union[gym.Env, VecEnv]],
        *,
        # -- Model selection --
        model_type: str = "mamba2",
        # -- GRU --
        hidden_size: int = 64,
        # -- Mamba2 --
        mamba_d_model: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        # -- Network --
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = -1.6094,
        # -- PPO --
        learning_rate: Union[float, Callable[[float], float]] = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.1,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.02,
        n_steps: int = 200,
        n_epochs: int = 10,
        batch_size: int = 512,
        normalize_advantage: bool = True,
        burn_in: int = 20,
        # -- MORL --
        morl: bool = False,
        morl_n_objectives: int = 2,
        morl_arch: str = "concat",
        morl_fixed_obj: int = 0,
        morl_w_fixed: float = 0.6,
        # -- MDMM entropy control --
        mdmm_entropy: bool = False,
        mdmm_H_target: float = 1.0,
        mdmm_H_decay: float = 0.0,
        mdmm_eta_tilde: Optional[float] = None,
        # -- PopArt value normalization --
        popart: bool = False,
        popart_beta: float = 3e-4,
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
            # Inference-only mode (no env, spaces injected by load())
            self.env = None  # type: ignore[assignment]
            self.n_envs = 1
            self.observation_space = None  # type: ignore[assignment]
            self.action_space = None  # type: ignore[assignment]

        obs_shape = self.observation_space.shape if self.observation_space is not None else ()
        act_shape = self.action_space.shape if self.action_space is not None else ()
        # Use explicit dims if provided (from load()), else infer from spaces
        _obs_dim = obs_dim if obs_dim is not None else (int(np.prod(obs_shape)) if obs_shape else 1)
        _act_dim = act_dim if act_dim is not None else (int(np.prod(act_shape)) if act_shape else 1)
        obs_dim = _obs_dim
        act_dim = _act_dim

        # ---- MORL setup ----
        self.morl = morl
        self.morl_n_objectives = morl_n_objectives
        self.morl_arch = morl_arch
        self.morl_fixed_obj = morl_fixed_obj
        self.morl_w_fixed = morl_w_fixed
        if morl:
            assert not obs_shape or len(obs_shape) == 1, "MORL requires a flat (1-D) observation space"
            _n = self.n_envs if env is not None else 1
            _varying = [i for i in range(morl_n_objectives) if i != morl_fixed_obj]
            self._last_w = np.empty((_n, morl_n_objectives), dtype=np.float32)
            self._last_w[:, morl_fixed_obj] = morl_w_fixed
            remainder = np.random.dirichlet(np.ones(len(_varying)), size=_n).astype(np.float32)
            self._last_w[:, _varying] = remainder * (1.0 - morl_w_fixed)
            if morl_arch == "concat":
                obs_shape = (obs_shape[0] + morl_n_objectives,)
                obs_dim += morl_n_objectives
        else:
            self._last_w = None

        self._rollout_episode_obj_returns: "np.ndarray | None" = None

        # ---- MDMM entropy control ----
        self.mdmm_entropy = mdmm_entropy
        self.mdmm_H_target = mdmm_H_target
        self.mdmm_H_decay = mdmm_H_decay
        if mdmm_entropy:
            _eta = mdmm_eta_tilde if mdmm_eta_tilde is not None else float(
                learning_rate if isinstance(learning_rate, (int, float)) else 1e-4
            ) / 10.0
            self.mdmm_eta_tilde = _eta
            self.mdmm_lambda = mdmm_H_target
            self._mdmm_update_count = 0

        # ---- PopArt ----
        self.popart = popart
        self.popart_norm = PopArtNormalizer(beta=popart_beta) if popart else None

        # ---- Seed ----
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ---- Store config (for save / load) ----
        self._config = dict(
            model_type=model_type,
            hidden_size=hidden_size,
            mamba_d_model=mamba_d_model,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
            net_arch=net_arch,
            activation_fn=activation_fn.__name__
            if isinstance(activation_fn, type)
            else type(activation_fn).__name__,
            ortho_init=ortho_init,
            log_std_init=log_std_init,
            learning_rate=learning_rate
            if isinstance(learning_rate, (int, float))
            else None,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            normalize_advantage=normalize_advantage,
            burn_in=burn_in,
            obs_dim=obs_dim,
            act_dim=act_dim,
            obs_space=_space_to_dict(self.observation_space) if self.observation_space is not None else None,
            act_space=_space_to_dict(self.action_space) if self.action_space is not None else None,
            # MORL
            morl=morl,
            morl_arch=morl_arch,
            morl_n_objectives=morl_n_objectives,
            morl_fixed_obj=morl_fixed_obj,
            morl_w_fixed=morl_w_fixed,
            # MDMM entropy control
            mdmm_entropy=mdmm_entropy,
            mdmm_H_target=mdmm_H_target,
            mdmm_H_decay=mdmm_H_decay,
            mdmm_eta_tilde=self.mdmm_eta_tilde if mdmm_entropy else None,
            # PopArt
            popart=popart,
            popart_beta=popart_beta,
        )

        # ---- Hyperparams ----
        self._lr_schedule: Optional[Callable[[float], float]] = None
        if callable(learning_rate) and not isinstance(learning_rate, (int, float)):
            self._lr_schedule = learning_rate
            self._base_lr = learning_rate(1.0)
        else:
            self._base_lr = float(learning_rate)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.normalize_advantage = normalize_advantage
        self.burn_in = burn_in

        # ---- Actor-Critic ----
        if model_type == "gru":
            self.agent = GRUActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=hidden_size,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
                log_std_init=log_std_init,
            ).to(self.device)
        elif model_type == "lstm":
            self.agent = LSTMActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=hidden_size,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
                log_std_init=log_std_init,
            ).to(self.device)
        else:
            self.agent = Mamba2ActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                mamba_d_model=mamba_d_model,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_headdim=mamba_headdim,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
                log_std_init=log_std_init,
                morl_arch=morl_arch,
                n_objectives=morl_n_objectives,
            ).to(self.device)

        # ---- Optimizer ----
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=self._base_lr, eps=1e-5,
        )

        # ---- Buffer ----
        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            n_envs=self.n_envs,
            obs_shape=obs_shape,
            act_shape=act_shape,
            gamma=gamma,
            gae_lambda=gae_lambda,
            burn_in=burn_in,
            n_objectives=morl_n_objectives if (morl and morl_arch == "multi_body") else 0,
        )

        # ---- Running state for rollout collection ----
        self._last_actor_states: Optional[Any] = None
        self._last_critic_states: Optional[Any] = None
        self._last_obs: Optional[Union[np.ndarray, dict[str, np.ndarray]]] = None
        self._last_episode_starts: Optional[np.ndarray] = None

        # ---- Burn-in context (previous rollout's tail) ----
        self._context_obs = np.zeros(
            (max(burn_in, 1), self.n_envs, *obs_shape), dtype=np.float32
        )
        self._context_episode_starts = np.ones(
            (max(burn_in, 1), self.n_envs), dtype=np.float32
        )

        # ---- Predict state (for inference / eval) ----
        self._predict_actor_states: Optional[Any] = None
        self._predict_critic_states: Optional[Any] = None

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
    ) -> "RecurrentPPO":
        """Main training loop.

        Returns *self* for chaining.
        """
        # ---- TensorBoard ----
        tb_writer = None
        if tb_log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            tb_path = Path(tb_log_dir) / (tb_log_name or "RecurrentPPO")
            tb_writer = SummaryWriter(str(tb_path))

        # ---- Eval bookkeeping ----
        best_mean_reward = -float("inf")
        eval_results: dict[str, list] = {
            "timesteps": [],
            "results": [],
            "ep_lengths": [],
        }

        # ---- Progress bar ----
        pbar = None
        if progress_bar:
            from tqdm import tqdm

            pbar = tqdm(total=total_timesteps, desc="Training", unit="step")

        # ---- Init env state ----
        reset_result = self.env.reset()
        # SB3 VecEnv.reset() may return (obs, info) or just obs
        if isinstance(reset_result, tuple):
            self._last_obs = reset_result[0]
        else:
            self._last_obs = reset_result
        self._last_episode_starts = np.ones(self.n_envs, dtype=np.float32)
        self._last_actor_states, self._last_critic_states = (
            self.agent.initial_states(self.n_envs, self.device)
        )

        steps_per_update = self.n_steps * self.n_envs
        num_updates = total_timesteps // steps_per_update
        start_update = self.num_timesteps // steps_per_update

        if pbar is not None and start_update > 0:
            pbar.update(start_update * steps_per_update)

        for update in range(start_update + 1, num_updates + 1):
            progress_remaining = 1.0 - (update - 1) / num_updates

            # ---- LR schedule ----
            if self._lr_schedule is not None:
                new_lr = self._lr_schedule(progress_remaining)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = new_lr

            # ---- Collect rollouts ----
            self._collect_rollouts()

            # ---- PPO update ----
            train_info = self._train_step()
            self._n_updates += 1

            self.num_timesteps += self.n_steps * self.n_envs

            if pbar is not None:
                pbar.update(self.n_steps * self.n_envs)

            # ---- Logging ----
            morl_metrics: dict[str, float] = {}
            if self.morl and self._rollout_obj_rewards is not None:
                for i, r in enumerate(self._rollout_obj_rewards):
                    morl_metrics[f"morl/reward_obj_{i}"] = float(r)
                for i, w in enumerate(self._rollout_mean_w):
                    morl_metrics[f"morl/w_mean_obj_{i}"] = float(w)
                for i, s in enumerate(self._rollout_w_std):
                    morl_metrics[f"morl/w_std_obj_{i}"] = float(s)
            if self.morl and self._rollout_episode_obj_returns is not None:
                for i, r in enumerate(self._rollout_episode_obj_returns):
                    morl_metrics[f"morl/episode_return_obj_{i}"] = float(r)

            if tb_writer is not None:
                for k, v in train_info.items():
                    tb_writer.add_scalar(f"train/{k}", v, self.num_timesteps)
                for k, v in morl_metrics.items():
                    tb_writer.add_scalar(k, v, self.num_timesteps)

            if wandb_run is not None:
                try:
                    import wandb

                    wandb.log(
                        {f"train/{k}": v for k, v in train_info.items()} | morl_metrics,
                        step=self.num_timesteps,
                    )
                except ImportError:
                    pass

            if self.verbose >= 1:
                pg_loss = train_info.get("policy_loss", float("nan"))
                vf_loss = train_info.get("value_loss", float("nan"))
                ent = train_info.get("entropy_loss", float("nan"))
                kl = train_info.get("approx_kl", float("nan"))
                msg = (
                    f"Update {update}/{num_updates}  "
                    f"policy_loss={pg_loss:.4f}  value_loss={vf_loss:.4f}  "
                    f"entropy={ent:.4f}  kl={kl:.4f}"
                )
                if self.mdmm_entropy:
                    msg += f"  mdmm_λ={train_info.get('mdmm_lambda', float('nan')):.4f}"
                if self.morl and self._rollout_obj_rewards is not None:
                    obj_str = "  ".join(
                        f"r{i}={v:.3f}" for i, v in enumerate(self._rollout_obj_rewards)
                    )
                    msg += f"  [{obj_str}]"

                if pbar is not None:
                    pbar.set_postfix({
                        "pg_loss": f"{pg_loss:.4f}",
                        "vf_loss": f"{vf_loss:.4f}",
                        "kl": f"{kl:.4f}",
                        "ent": f"{ent:.4f}",
                    })
                    pbar.write(msg)
                else:
                    print(msg)

            # ---- Periodic evaluation ----
            if (
                eval_env is not None
                and eval_freq is not None
                and update % max(1, eval_freq // (self.n_steps * self.n_envs)) == 0
            ):
                mean_reward, mean_len = self._evaluate(
                    eval_env, n_eval_episodes
                )
                eval_results["timesteps"].append(self.num_timesteps)
                eval_results["results"].append(mean_reward)
                eval_results["ep_lengths"].append(mean_len)

                if self.verbose >= 1:
                    eval_msg = (
                        f"Eval @ {self.num_timesteps} steps: "
                        f"mean_reward={mean_reward:.2f}  mean_len={mean_len:.0f}"
                    )
                    if pbar is not None:
                        pbar.write(eval_msg)
                    else:
                        print(eval_msg)

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
                        best_msg = f"  New best model saved (reward={mean_reward:.2f})"
                        if pbar is not None:
                            pbar.write(best_msg)
                        else:
                            print(best_msg)

        # ---- Cleanup ----
        if pbar is not None:
            pbar.close()
        if tb_writer is not None:
            tb_writer.close()

        # Save eval results
        if log_path and eval_results["timesteps"]:
            np.savez(
                Path(log_path) / "evaluations.npz",
                timesteps=np.array(eval_results["timesteps"]),
                results=np.array(eval_results["results"]),
                ep_lengths=np.array(eval_results["ep_lengths"]),
            )

        return self

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self) -> None:
        self.agent.eval()
        self.buffer.reset()

        # Set burn-in context from previous rollout
        if self.burn_in > 0:
            self.buffer.set_context(
                self._context_obs[: self.burn_in],
                self._context_episode_starts[: self.burn_in],
            )

        _obj_rewards_list: list[np.ndarray] = []  # (n_envs, n_objectives) per step
        if self.morl:
            _running_obj_rewards = np.zeros((self.n_envs, self.morl_n_objectives), dtype=np.float32)
            _completed_ep_obj_returns: list[np.ndarray] = []

        actor_states = self._last_actor_states
        critic_states = self._last_critic_states
        obs = self._last_obs
        episode_starts = self._last_episode_starts

        for step in range(self.n_steps):
            w_t = None
            if self.morl:
                if self.morl_arch == "concat":
                    obs_aug = np.concatenate([obs, self._last_w], axis=-1)
                else:
                    obs_aug = obs
                    w_t = torch.as_tensor(self._last_w, dtype=torch.float32, device=self.device)
            else:
                obs_aug = obs

            obs_t = torch.as_tensor(obs_aug, dtype=torch.float32, device=self.device)
            ep_t = torch.as_tensor(
                episode_starts, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                action, value, log_prob, actor_states, critic_states = (
                    self.agent.step(
                        obs_t,
                        actor_states,
                        critic_states,
                        ep_t,
                        w=w_t,
                    )
                )
            action_np = action.cpu().numpy()
            value_np = value.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()

            # Clip actions for the environment
            clipped_action = np.clip(
                action_np,
                self.action_space.low if isinstance(self.action_space, gym.spaces.Box) else -np.inf,
                self.action_space.high if isinstance(self.action_space, gym.spaces.Box) else np.inf,
            )

            self.buffer.add(
                obs_aug, action_np, np.zeros(self.n_envs), episode_starts,
                value_np, log_prob_np, w=self._last_w if self.morl_arch == "multi_body" else None,
            )

            new_obs, rewards, dones, infos = self.env.step(clipped_action)
            # SB3 VecEnv returns (obs, reward, done, info)
            if isinstance(new_obs, tuple):
                new_obs = new_obs[0]

            # MORL: scalarize vector rewards using current preference weights,
            # then resample w for any environment that just finished an episode
            if self.morl:
                # rewards shape: (n_envs, n_objectives) → scalarize to (n_envs,)
                _obj_rewards_list.append(rewards.copy())  # capture before scalarization
                _running_obj_rewards += rewards
                r_scalar = (rewards * self._last_w).sum(axis=-1)
                rewards = r_scalar

            # Store rewards (we couldn't store them before env.step)
            self.buffer.rewards[step] = rewards

            # Zero recurrent states for done environments; resample MORL weights
            for i in range(self.n_envs):
                if dones[i]:
                    self.agent.zero_env_states(actor_states, i)
                    self.agent.zero_env_states(critic_states, i)
                    if self.morl:
                        _completed_ep_obj_returns.append(_running_obj_rewards[i].copy())
                        _running_obj_rewards[i] = 0.0
                        _varying = [j for j in range(self.morl_n_objectives) if j != self.morl_fixed_obj]
                        self._last_w[i, self.morl_fixed_obj] = self.morl_w_fixed
                        remainder = np.random.dirichlet(np.ones(len(_varying))).astype(np.float32)
                        self._last_w[i, _varying] = remainder * (1.0 - self.morl_w_fixed)

            obs = new_obs
            episode_starts = dones.astype(np.float32)

        if self.morl:
            self._rollout_episode_obj_returns = (
                np.stack(_completed_ep_obj_returns).mean(axis=0)
                if _completed_ep_obj_returns else None
            )

        # Bootstrap value
        with torch.no_grad():
            w_t = None
            if self.morl:
                if self.morl_arch == "concat":
                    _obs_for_boot = np.concatenate([obs, self._last_w], axis=-1)
                else:
                    _obs_for_boot = obs
                    w_t = torch.as_tensor(self._last_w, dtype=torch.float32, device=self.device)
            else:
                _obs_for_boot = obs

            obs_t = torch.as_tensor(_obs_for_boot, dtype=torch.float32, device=self.device)
            ep_t = torch.as_tensor(
                episode_starts, dtype=torch.float32, device=self.device
            )
            last_values, critic_states = self.agent.get_value(
                obs_t, critic_states, ep_t, w=w_t
            )
            last_values = last_values.cpu().numpy()

        if self.popart and self.morl:
            self.popart_norm.update(self.buffer.rewards.flatten())
            self.buffer.rewards = self.popart_norm.normalize(self.buffer.rewards)
            # Do NOT unnormalize last_values here, as value loss target requires normalized rewards 
            # and trained values will natively be normalized.

        self.buffer.compute_returns_and_advantage(last_values, dones.astype(np.float32))

        # MORL rollout stats (logged in learn())
        if self.morl and _obj_rewards_list:
            _obj_arr = np.stack(_obj_rewards_list)  # (n_steps, n_envs, n_objectives)
            self._rollout_obj_rewards = _obj_arr.mean(axis=(0, 1))  # (n_objectives,)
            self._rollout_mean_w = self._last_w.mean(axis=0)        # (n_objectives,)
            self._rollout_w_std = self._last_w.std(axis=0)          # (n_objectives,)
        else:
            self._rollout_obj_rewards = None
            self._rollout_mean_w = None
            self._rollout_w_std = None

        # Save state for next rollout
        self._last_actor_states = actor_states
        self._last_critic_states = critic_states
        self._last_obs = obs
        self._last_episode_starts = episode_starts

        # Save context for next rollout's burn-in
        if self.burn_in > 0:
            K = self.burn_in
            if self.n_steps >= K:
                self._context_obs[:K] = self.buffer.obs[-K:]
                self._context_episode_starts[:K] = self.buffer.episode_starts[-K:]
            else:
                # Edge case: n_steps < burn_in, use what we have
                self._context_obs[:self.n_steps] = self.buffer.obs
                self._context_episode_starts[:self.n_steps] = (
                    self.buffer.episode_starts
                )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _train_step(self) -> dict[str, float]:
        self.agent.train()

        all_pg_losses: list[float] = []
        all_vf_losses: list[float] = []
        all_ent_losses: list[float] = []
        all_approx_kl: list[float] = []
        all_clip_fracs: list[float] = []
        early_stopped = False

        for epoch in range(self.n_epochs):
            if early_stopped:
                break

            for mb in self.buffer.get(self.batch_size, self.device):
                # Unpack
                mb_obs = mb["obs"]
                mb_actions = mb["actions"]
                mb_ep_starts = mb["episode_starts"]
                mb_old_values = mb["old_values"]
                mb_old_log_probs = mb["old_log_probs"]
                mb_advantages = mb["advantages"]
                mb_returns = mb["returns"]
                ctx_obs = mb.get("context_obs")
                ctx_ep = mb.get("context_episode_starts")
                mb_w = mb.get("weights")
                ctx_w = mb.get("context_weights")

                # Evaluate actions
                values, log_probs, entropy = self.agent.evaluate_actions(
                    mb_obs, mb_actions, mb_ep_starts, ctx_obs, ctx_ep, mb_w, ctx_w
                )

                # Flatten time dimension for loss computation
                values = values.reshape(-1)
                log_probs = log_probs.reshape(-1)
                entropy = entropy.reshape(-1)
                mb_advantages = mb_advantages.reshape(-1)
                mb_returns = mb_returns.reshape(-1)
                mb_old_log_probs = mb_old_log_probs.reshape(-1)
                mb_old_values = mb_old_values.reshape(-1)

                # Advantage normalization
                if self.normalize_advantage and mb_advantages.numel() > 1:
                    mb_advantages = (
                        (mb_advantages - mb_advantages.mean())
                        / (mb_advantages.std() + 1e-8)
                    )

                # Policy loss
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    values_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    vf_loss1 = (values - mb_returns).pow(2)
                    vf_loss2 = (values_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * (values - mb_returns).pow(2).mean()

                entropy_loss = -entropy.mean()

                if self.mdmm_entropy:
                    H_hat = -entropy_loss.detach()
                    H_target_t = max(0.0, self.mdmm_H_target - self.mdmm_H_decay * self._mdmm_update_count)
                    self.mdmm_lambda += self.mdmm_eta_tilde * (H_target_t - H_hat.item())
                    self.mdmm_lambda = max(0.0, self.mdmm_lambda)
                    ent_weight = self.mdmm_lambda
                else:
                    ent_weight = self.ent_coef

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + ent_weight * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm,
                )
                self.optimizer.step()

                # Logging
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - log_probs).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.clip_range)
                        .float()
                        .mean()
                        .item()
                    )

                all_pg_losses.append(policy_loss.item())
                all_vf_losses.append(value_loss.item())
                all_ent_losses.append(entropy_loss.item())
                all_approx_kl.append(approx_kl)
                all_clip_fracs.append(clip_frac)
                
            if self.mdmm_entropy:
                self._mdmm_update_count += 1

            # Early stopping on KL divergence
            if self.target_kl is not None and approx_kl > self.target_kl:
                early_stopped = True
                break

        ret_dict = {
            "policy_loss": float(np.mean(all_pg_losses)) if all_pg_losses else 0.0,
            "value_loss": float(np.mean(all_vf_losses)) if all_vf_losses else 0.0,
            "entropy_loss": float(np.mean(all_ent_losses)) if all_ent_losses else 0.0,
            "approx_kl": float(np.mean(all_approx_kl)) if all_approx_kl else 0.0,
            "clip_fraction": float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0,
            "n_epochs_actual": (
                len(all_pg_losses)
                / max(1, math.ceil(self.n_envs / self.batch_size))
            ),
        }
        if self.mdmm_entropy:
            ret_dict["mdmm_lambda"] = float(self.mdmm_lambda)
        return ret_dict

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_episodes: int,
    ) -> tuple[float, float]:
        """Run *n_episodes* deterministic rollouts, return mean reward & length."""
        # Create a temporary copy for evaluation
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
        actor_states, critic_states = self.agent.initial_states(
            n_eval_envs, self.device
        )

        # MORL eval: use balanced weight w=[0.5, 0.5] by default
        if self.morl:
            eval_w = np.full(
                (n_eval_envs, self.morl_n_objectives),
                1.0 / self.morl_n_objectives,
                dtype=np.float32,
            )
        else:
            eval_w = None

        self.agent.eval()
        while len(ep_rewards) < n_episodes:
            w_t = None
            if self.morl:
                if self.morl_arch == "concat":
                    _obs = np.concatenate([obs, eval_w], axis=-1)
                else:
                    _obs = obs
                    w_t = torch.as_tensor(eval_w, dtype=torch.float32, device=self.device)
            else:
                _obs = obs

            obs_t = torch.as_tensor(_obs, dtype=torch.float32, device=self.device)
            ep_t = torch.as_tensor(
                episode_starts, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                action, actor_states, critic_states = (
                    self.agent.deterministic_step(
                        obs_t, actor_states, critic_states, ep_t, w=w_t
                    )
                )

            action_np = action.cpu().numpy()
            action_np = np.clip(
                action_np,
                self.action_space.low if isinstance(self.action_space, gym.spaces.Box) else -np.inf,
                self.action_space.high if isinstance(self.action_space, gym.spaces.Box) else np.inf,
            )

            obs, rewards, dones, infos = eval_env_wrapped.step(action_np)
            if isinstance(obs, tuple):
                obs = obs[0]

            # MORL: scalarize eval rewards with balanced weight
            if self.morl:
                rewards = (rewards * eval_w).sum(axis=-1)

            running_rewards += rewards
            running_lengths += 1

            for i in range(n_eval_envs):
                if dones[i]:
                    ep_rewards.append(running_rewards[i])
                    ep_lengths.append(running_lengths[i])
                    running_rewards[i] = 0.0
                    running_lengths[i] = 0
                    self.agent.zero_env_states(actor_states, i)
                    self.agent.zero_env_states(critic_states, i)

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
        """SB3-compatible predict.

        Parameters
        ----------
        observation : array
            Current observation.
        deterministic : bool
            Use action mean (True) or sample (False).
        episode_start : array or bool or None
            ``True`` / ``1.0`` resets the internal recurrent state.

        Returns
        -------
        action : np.ndarray
        state : None   (kept for API compatibility)
        """
        self.agent.eval()

        # Ensure batch dimension
        obs = np.asarray(observation, dtype=np.float32)
        was_single = obs.ndim == len(self.observation_space.shape)
        if was_single:
            obs = obs[np.newaxis, ...]
        n_envs = obs.shape[0]

        # Init predict states on first call or if n_envs changed
        if self._predict_actor_states is None or self._n_predict_envs() != n_envs:
            self._predict_actor_states, self._predict_critic_states = (
                self.agent.initial_states(n_envs, self.device)
            )

        # Episode start handling
        if episode_start is None:
            ep_starts = np.zeros(n_envs, dtype=np.float32)
        elif isinstance(episode_start, (bool, int, float)):
            ep_starts = np.full(n_envs, float(episode_start), dtype=np.float32)
        else:
            ep_starts = np.asarray(episode_start, dtype=np.float32).flatten()
            if ep_starts.shape[0] == 1 and n_envs > 1:
                ep_starts = np.full(n_envs, ep_starts[0], dtype=np.float32)

        w_t = None
        if hasattr(self, "morl") and self.morl:
            # We assume predict uses uniform weight by default if w not supplied externally
            eval_w = np.full(
                (n_envs, self.morl_n_objectives),
                1.0 / self.morl_n_objectives,
                dtype=np.float32,
            )
            if self.morl_arch == "concat":
                obs = np.concatenate([obs, eval_w], axis=-1)
            else:
                w_t = torch.as_tensor(eval_w, dtype=torch.float32, device=self.device)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        ep_t = torch.as_tensor(ep_starts, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if deterministic:
                action, self._predict_actor_states, self._predict_critic_states = (
                    self.agent.deterministic_step(
                        obs_t,
                        self._predict_actor_states,
                        self._predict_critic_states,
                        ep_t,
                        w=w_t,
                    )
                )
            else:
                action, _, _, self._predict_actor_states, self._predict_critic_states = (
                    self.agent.step(
                        obs_t,
                        self._predict_actor_states,
                        self._predict_critic_states,
                        ep_t,
                        w=w_t,
                    )
                )

        action_np = action.cpu().numpy()
        action_np = np.clip(
            action_np,
            self.action_space.low if isinstance(self.action_space, gym.spaces.Box) else -np.inf,
            self.action_space.high if isinstance(self.action_space, gym.spaces.Box) else np.inf,
        )

        if was_single:
            action_np = action_np.squeeze(0)

        return action_np, None

    def _n_predict_envs(self) -> int:
        """Infer the number of envs from predict states (architecture-agnostic)."""
        s = self._predict_actor_states
        if s is None:
            return 0
        if self.model_type == "mamba2":
            # (conv_state, ssm_state): conv_state is (n_envs, conv_dim, d_conv)
            return s[0].shape[0]
        elif self.model_type == "lstm":
            # (h, c): h is (1, n_envs, hidden_size)
            return s[0].shape[1]
        else:
            # GRU: single tensor (1, n_envs, hidden_size)
            return s.shape[1]

    # ==================================================================
    # Save / Load
    # ==================================================================

    def save(self, path: Union[str, Path]) -> None:
        """Save model to a ``.zip`` file (config + weights)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_bytes = json.dumps(self._config, indent=2).encode()
        weights_buf = io.BytesIO()
        torch.save(self.agent.state_dict(), weights_buf)
        weights_bytes = weights_buf.getvalue()

        training_state = json.dumps({
            "num_timesteps": self.num_timesteps,
            "n_updates": self._n_updates,
            "mdmm_lambda": self.mdmm_lambda if self.mdmm_entropy else None,
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
    ) -> "RecurrentPPO":
        """Load a model from a ``.zip`` file.

        Parameters
        ----------
        path : str or Path
            Path to ``.zip`` file.
        env : Env or VecEnv or None
            If provided, overrides the saved observation/action spaces.
        device : str or torch.device
            Device to load onto.
        """
        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            config = json.loads(zf.read("config.json"))
            weights_bytes = zf.read("policy.pt")

        # Resolve activation function
        act_fn_name = config.pop("activation_fn", "Tanh")
        act_fn_map = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "GELU": nn.GELU}
        activation_fn = act_fn_map.get(act_fn_name, nn.Tanh)

        # Resolve spaces
        obs_space_dict = config.pop("obs_space")
        act_space_dict = config.pop("act_space")

        obs_space = _dict_to_space(obs_space_dict)
        act_space = _dict_to_space(act_space_dict)

        # Pop lr if None (was a callable during training)
        lr = config.pop("learning_rate", None)
        if lr is None:
            lr = 1e-4

        # Pass env if provided, otherwise None (inference-only mode)
        instance = cls(
            env,
            activation_fn=activation_fn,
            learning_rate=lr,
            device=device,
            **config,
        )

        # Ensure spaces are set from saved config when no env provided
        if instance.observation_space is None:
            instance.observation_space = obs_space
        if instance.action_space is None:
            instance.action_space = act_space

        # Load weights
        weights_buf = io.BytesIO(weights_bytes)
        state_dict = torch.load(
            weights_buf, map_location=instance.device, weights_only=True,
        )
        instance.agent.load_state_dict(state_dict)
        instance.agent.eval()

        # Restore training state (if saved)
        with zipfile.ZipFile(path, "r") as zf:
            if "training_state.json" in zf.namelist():
                ts = json.loads(zf.read("training_state.json"))
                instance.num_timesteps = ts.get("num_timesteps", 0)
                instance._n_updates = ts.get("n_updates", 0)
                if instance.mdmm_entropy and ts.get("mdmm_lambda") is not None:
                    instance.mdmm_lambda = ts["mdmm_lambda"]

        return instance


# Backward-compatible alias
Mamba2PPO = RecurrentPPO
