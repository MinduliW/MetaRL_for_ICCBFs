"""Self-contained training script for Mamba2 PPO on cruise control.

Usage::

    python -m metarl_iccbf.recurrent_cleanrl.train [OPTIONS]

    # Quick smoke test (1 env, 100 episodes)
    python -m metarl_iccbf.recurrent_cleanrl.train --num-env 1 --total-episodes 100

    # Full training run
    python -m metarl_iccbf.recurrent_cleanrl.train --num-env 28 --total-episodes 100000 \\
        --wandb-project my-project
"""

from __future__ import annotations

import gc
import argparse
import warnings
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional

import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .configs import cruise_control_config
from .ppo import Mamba2PPO

_CFG = cruise_control_config()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

class ConstantSchedule:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, progress_remaining: float) -> float:
        return self.value


class LinearSchedule:
    def __init__(self, initial: float, min_value: float):
        self.initial = initial
        self.min_value = min_value

    def __call__(self, progress_remaining: float) -> float:
        return max(self.initial * progress_remaining, self.min_value)


def _make_env(rank: int, dt: float, deterministic: bool, seed: int, env_cls):
    def _init():
        env = env_cls(dt=dt, deterministic=deterministic)
        env.action_space.seed(seed + rank)
        return Monitor(env)
    return _init


def _resolve_env_cls(env_type: str):
    if env_type == "iccbf":
        from metarl_iccbf.cruise_control.envs.rlcbf_env import RLCBFcontrol
        return RLCBFcontrol
    elif env_type == "rl_only":
        from metarl_iccbf.cruise_control.envs.rlonly_env import RLCBFcontrol
        return RLCBFcontrol
    else:
        raise ValueError(f"Unknown env_type: {env_type!r}. Expected 'iccbf' or 'rl_only'.")


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------

def train_mamba2(
    *,
    # ---- Environment ----
    env_type: Literal["iccbf", "rl_only"] = "iccbf",
    dt: float = 0.1,
    seed: int = 123,

    # ---- Network architecture ----
    layers: int = _CFG.layers,
    nodes: int = _CFG.nodes,
    activation_fn: type[nn.Module] = nn.Tanh,

    # ---- Mamba2 parameters ----
    mamba_d_model: int = _CFG.mamba_d_model,
    mamba_d_state: int = _CFG.mamba_d_state,
    mamba_d_conv: int = _CFG.mamba_d_conv,
    mamba_expand: int = _CFG.mamba_expand,
    mamba_headdim: int = _CFG.mamba_headdim,

    # ---- RL hyperparameters ----
    total_episodes: int = 100_000,
    approx_episode_len: int = _CFG.approx_episode_len,
    learning_rate: float = _CFG.learning_rate,
    lr_type: Literal["C", "D"] = "D",
    gamma: float = _CFG.gamma,
    gae_lambda: float = _CFG.gae_lambda,
    clip_range: float = _CFG.clip_range,
    ent_coef: float = _CFG.ent_coef,
    target_kl: Optional[float] = _CFG.target_kl,
    n_epochs: int = _CFG.n_epochs,
    batch_size: int = _CFG.batch_size,
    n_steps: int = _CFG.n_steps,

    # ---- Burn-in ----
    burn_in: int = _CFG.burn_in,

    # ---- Parallelism ----
    num_env: int = 1,

    # ---- Control ----
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "outputs/cruise_control",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
):
    """Train a Mamba2 PPO policy for cruise control.

    Returns
    -------
    model : Mamba2PPO
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    env_cls = _resolve_env_cls(env_type)

    # --------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------
    if lr_type == "D":
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    # --------------------------------------------------
    # Environments
    # --------------------------------------------------
    if num_env > 1:
        train_env = SubprocVecEnv([
            _make_env(i, dt, deterministic=False, seed=seed, env_cls=env_cls)
            for i in range(num_env)
        ])
    else:
        train_env = env_cls(dt=dt, deterministic=False)

    eval_env = env_cls(dt=dt, deterministic=True)

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"Mamba2TunedICCBF_CruiseControl_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)
    std_log = float(_CFG.log_std_init)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if trainLoad:
        model = Mamba2PPO.load(
            log_dir / "best_model.zip",
            env=train_env,
        )
    else:
        model = Mamba2PPO(
            train_env,
            mamba_d_model=mamba_d_model,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
            net_arch=dict(pi=[nodes] * layers, vf=[nodes] * layers),
            activation_fn=activation_fn,
            learning_rate=lr_schedule,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            target_kl=target_kl,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            normalize_advantage=True,
            burn_in=burn_in,
            seed=seed,
            ortho_init=True,
            log_std_init=std_log,
            verbose=1,
        )

    # --------------------------------------------------
    # W&B
    # --------------------------------------------------
    wandb_run = None
    if wandb_project is not None:
        import wandb

        wandb_kwargs = dict(
            project=wandb_project,
            name=training_name,
            config=dict(
                env_type=env_type,
                dt=dt,
                seed=seed,
                layers=layers,
                nodes=nodes,
                mamba_d_model=mamba_d_model,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_headdim=mamba_headdim,
                total_episodes=total_episodes,
                learning_rate=learning_rate,
                lr_type=lr_type,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                target_kl=target_kl,
                n_epochs=n_epochs,
                batch_size=batch_size,
                n_steps=n_steps,
                burn_in=burn_in,
                num_env=num_env,
            ),
            sync_tensorboard=True,
        )
        if wandb_resume_id is not None:
            wandb_kwargs["id"] = wandb_resume_id
            wandb_kwargs["resume"] = "must"
        wandb.init(**wandb_kwargs)
        wandb_run = wandb.run

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    model.learn(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=n_steps,
        n_eval_episodes=10,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        wandb_run=wandb_run,
        tb_log_dir=str(root_log_dir),
        tb_log_name=training_name,
        progress_bar=True,
    )

    model.save(log_dir / "final_model.zip")

    if wandb_project is not None:
        import wandb
        wandb.finish()

    return model, log_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Mamba2 PPO on cruise control (CleanRL-style)."
    )
    parser.add_argument(
        "--env-type",
        choices=["iccbf", "rl_only"],
        default="iccbf",
        help="Environment type (default: iccbf)",
    )
    parser.add_argument(
        "--num-env",
        type=int,
        default=28,
        help="Number of parallel environments (default: 28)",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=100_000,
        help="Total training episodes (default: 100000)",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=40,
        help="Context overlap steps for burn-in (default: 40, 0=disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (default: 123)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name. Omit to disable.",
    )
    parser.add_argument(
        "--training-name",
        type=str,
        default=None,
        help="Custom run name. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Resume from best_model.zip in the run directory.",
    )
    parser.add_argument(
        "--wandb-resume-id",
        type=str,
        default=None,
        help="W&B run ID to resume (e.g. 'ukceypri'). Requires --wandb-project.",
    )
    args = parser.parse_args()

    train_mamba2(
        env_type=args.env_type,
        num_env=args.num_env,
        total_episodes=args.total_episodes,
        burn_in=args.burn_in,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_resume_id=args.wandb_resume_id,
        training_name=args.training_name,
        trainLoad=args.load,
    )


if __name__ == "__main__":
    main()
