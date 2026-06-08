"""Self-contained training script for recurrent PPO/SAC on cruise control.

Supports Mamba2 and GRU backends via ``--model``, and PPO or SAC via ``--algo``.

Usage::

    python -m metarl_iccbf.recurrent_cleanrl.train [OPTIONS]

    # Quick smoke test (GRU PPO, 1 env, 100 episodes)
    python -m metarl_iccbf.recurrent_cleanrl.train --model gru --num-env 1 --total-episodes 100

    # Full Mamba2 PPO training run
    python -m metarl_iccbf.recurrent_cleanrl.train --model mamba2 --num-env 28 --total-episodes 100000 \\
        --wandb-project my-project

    # Mamba2 SAC ablation
    python -m metarl_iccbf.recurrent_cleanrl.train --algo sac --num-env 1 --total-episodes 50000
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

from .configs import (
    cruise_control_config,
    cruise_control_gru_config,
    cruise_control_lstm_ppo_config,
    cruise_control_sac_config,
    cruise_control_gru_sac_config,
    cruise_control_lstm_sac_config,
)
from .ppo import RecurrentPPO
from .sac import RecurrentSAC


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


def _make_env(rank: int, dt: float, deterministic: bool, seed: int, env_cls, **env_kwargs):
    def _init():
        env = env_cls(dt=dt, deterministic=deterministic, **env_kwargs)
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

def train_recurrent(
    *,
    # ---- Model ----
    model_type: Literal["mamba2", "gru", "lstm"] = "mamba2",

    # ---- Environment ----
    env_type: Literal["iccbf", "rl_only"] = "iccbf",
    dt: float = 0.1,
    seed: int = 123,

    # ---- Network architecture ----
    layers: Optional[int] = None,
    nodes: Optional[int] = None,
    activation_fn: type[nn.Module] = nn.Tanh,

    # ---- Mamba2 parameters (ignored when model_type="gru") ----
    mamba_d_model: Optional[int] = None,
    mamba_d_state: Optional[int] = None,
    mamba_d_conv: Optional[int] = None,
    mamba_expand: Optional[int] = None,
    mamba_headdim: Optional[int] = None,

    # ---- GRU parameters (ignored when model_type="mamba2") ----
    hidden_size: Optional[int] = None,

    # ---- RL hyperparameters ----
    total_episodes: int = 100_000,
    approx_episode_len: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lr_type: Literal["C", "D"] = "D",
    gamma: Optional[float] = None,
    gae_lambda: Optional[float] = None,
    clip_range: Optional[float] = None,
    ent_coef: Optional[float] = None,
    target_kl: Optional[float] = None,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_steps: Optional[int] = None,

    # ---- Burn-in ----
    burn_in: Optional[int] = None,

    # ---- Parallelism ----
    num_env: int = 28,

    # ---- Control ----
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "outputs/cruise_control",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
):
    """Train a recurrent PPO policy for cruise control.

    Returns
    -------
    model : RecurrentPPO
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # --------------------------------------------------
    # Select config defaults based on model type
    # --------------------------------------------------
    if model_type == "gru":
        _cfg = cruise_control_gru_config()
        _hidden_size = hidden_size if hidden_size is not None else _cfg.hidden_size
    elif model_type == "lstm":
        _cfg = cruise_control_lstm_ppo_config()
        _hidden_size = hidden_size if hidden_size is not None else _cfg.hidden_size
    else:
        _cfg = cruise_control_config()
        _hidden_size = hidden_size if hidden_size is not None else 64

    # Resolve defaults from config
    _layers = layers if layers is not None else _cfg.layers
    _nodes = nodes if nodes is not None else _cfg.nodes
    _learning_rate = learning_rate if learning_rate is not None else _cfg.learning_rate
    _gamma = gamma if gamma is not None else _cfg.gamma
    _gae_lambda = gae_lambda if gae_lambda is not None else _cfg.gae_lambda
    _clip_range = clip_range if clip_range is not None else _cfg.clip_range
    _ent_coef = ent_coef if ent_coef is not None else _cfg.ent_coef
    _target_kl = target_kl if target_kl is not None else _cfg.target_kl
    _n_epochs = n_epochs if n_epochs is not None else _cfg.n_epochs
    _batch_size = batch_size if batch_size is not None else _cfg.batch_size
    _n_steps = n_steps if n_steps is not None else _cfg.n_steps
    _burn_in = burn_in if burn_in is not None else _cfg.burn_in
    _approx_episode_len = approx_episode_len if approx_episode_len is not None else _cfg.approx_episode_len
    _log_std_init = float(_cfg.log_std_init)

    # Mamba-specific defaults (only used when model_type="mamba2")
    if model_type == "mamba2":
        _mamba_cfg = _cfg  # type: ignore[assignment]
        _mamba_d_model = mamba_d_model if mamba_d_model is not None else _mamba_cfg.mamba_d_model
        _mamba_d_state = mamba_d_state if mamba_d_state is not None else _mamba_cfg.mamba_d_state
        _mamba_d_conv = mamba_d_conv if mamba_d_conv is not None else _mamba_cfg.mamba_d_conv
        _mamba_expand = mamba_expand if mamba_expand is not None else _mamba_cfg.mamba_expand
        _mamba_headdim = mamba_headdim if mamba_headdim is not None else _mamba_cfg.mamba_headdim
    else:
        _mamba_d_model = mamba_d_model or 64
        _mamba_d_state = mamba_d_state or 16
        _mamba_d_conv = mamba_d_conv or 4
        _mamba_expand = mamba_expand or 2
        _mamba_headdim = mamba_headdim or 64

    env_cls = _resolve_env_cls(env_type)

    # --------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------
    if lr_type == "D":
        lr_schedule = LinearSchedule(_learning_rate, _learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(_learning_rate)

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
    model_label = {"mamba2": "Mamba2", "gru": "GRU", "lstm": "LSTM"}.get(model_type, model_type)
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"{model_label}TunedICCBF_CruiseControl_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * _approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if trainLoad:
        model = RecurrentPPO.load(
            log_dir / "best_model.zip",
            env=train_env,
        )
    else:
        model = RecurrentPPO(
            train_env,
            model_type=model_type,
            hidden_size=_hidden_size,
            mamba_d_model=_mamba_d_model,
            mamba_d_state=_mamba_d_state,
            mamba_d_conv=_mamba_d_conv,
            mamba_expand=_mamba_expand,
            mamba_headdim=_mamba_headdim,
            net_arch=dict(pi=[_nodes] * _layers, vf=[_nodes] * _layers),
            activation_fn=activation_fn,
            learning_rate=lr_schedule,
            gamma=_gamma,
            gae_lambda=_gae_lambda,
            clip_range=_clip_range,
            ent_coef=_ent_coef,
            target_kl=_target_kl,
            n_steps=_n_steps,
            n_epochs=_n_epochs,
            batch_size=_batch_size,
            normalize_advantage=True,
            burn_in=_burn_in,
            seed=seed,
            ortho_init=True,
            log_std_init=_log_std_init,
            verbose=1,
        )

    # --------------------------------------------------
    # W&B
    # --------------------------------------------------
    wandb_run = None
    if wandb_project is not None:
        import wandb

        wandb_config = dict(
            model_type=model_type,
            env_type=env_type,
            dt=dt,
            seed=seed,
            layers=_layers,
            nodes=_nodes,
            total_episodes=total_episodes,
            learning_rate=_learning_rate,
            lr_type=lr_type,
            gamma=_gamma,
            gae_lambda=_gae_lambda,
            clip_range=_clip_range,
            ent_coef=_ent_coef,
            target_kl=_target_kl,
            n_epochs=_n_epochs,
            batch_size=_batch_size,
            n_steps=_n_steps,
            burn_in=_burn_in,
            num_env=num_env,
        )
        if model_type == "mamba2":
            wandb_config.update(
                mamba_d_model=_mamba_d_model,
                mamba_d_state=_mamba_d_state,
                mamba_d_conv=_mamba_d_conv,
                mamba_expand=_mamba_expand,
                mamba_headdim=_mamba_headdim,
            )
        else:
            wandb_config["hidden_size"] = _hidden_size

        wandb_kwargs = dict(
            project=wandb_project,
            name=training_name,
            config=wandb_config,
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
        eval_freq=_n_steps,
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
# SAC training function
# ---------------------------------------------------------------------

def train_mamba2_sac(
    *,
    # ---- Model ----
    model_type: Literal["mamba2", "gru", "lstm"] = "mamba2",

    # ---- Environment ----
    env_type: Literal["iccbf", "rl_only"] = "iccbf",
    dt: float = 0.1,
    seed: int = 123,

    # ---- Network architecture ----
    layers: Optional[int] = None,
    nodes: Optional[int] = None,
    activation_fn: type[nn.Module] = nn.ReLU,

    # ---- Mamba2 parameters (ignored when model_type="gru") ----
    mamba_d_model: Optional[int] = None,
    mamba_d_state: Optional[int] = None,
    mamba_d_conv: Optional[int] = None,
    mamba_expand: Optional[int] = None,
    mamba_headdim: Optional[int] = None,

    # ---- GRU parameters (ignored when model_type="mamba2") ----
    hidden_size: Optional[int] = None,

    # ---- SAC hyperparameters ----
    total_episodes: int = 50_000,
    approx_episode_len: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lr_type: Literal["C", "D"] = "C",
    gamma: Optional[float] = None,
    tau: Optional[float] = None,
    ent_coef: str = "auto",
    target_entropy: str = "auto",
    ent_coef_min: Optional[float] = None,
    buffer_size: Optional[int] = None,
    chunk_len: Optional[int] = None,
    burn_in: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_starts: Optional[int] = None,
    train_freq: Optional[int] = None,
    gradient_steps: Optional[int] = None,

    # ---- Env reward shaping ----
    per_step_vel_penalty: bool = True,
    vel_penalty_coef: float = 0.1,
    vec_normalize: bool = False,

    # ---- Parallelism ----
    num_env: int = 4,

    # ---- Control ----
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "outputs/cruise_control",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
):
    """Train a recurrent SAC policy for cruise control.

    Returns
    -------
    model : RecurrentSAC
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    if model_type == "gru":
        _cfg = cruise_control_gru_sac_config()
        _hidden_size = hidden_size if hidden_size is not None else _cfg.hidden_size
    elif model_type == "lstm":
        _cfg = cruise_control_lstm_sac_config()
        _hidden_size = hidden_size if hidden_size is not None else _cfg.hidden_size
    else:
        _cfg = cruise_control_sac_config()
        _hidden_size = hidden_size if hidden_size is not None else 64

    # Resolve defaults from config
    _layers = layers if layers is not None else _cfg.layers
    _nodes = nodes if nodes is not None else _cfg.nodes
    _learning_rate = learning_rate if learning_rate is not None else _cfg.learning_rate
    _gamma = gamma if gamma is not None else _cfg.gamma
    _tau = tau if tau is not None else _cfg.tau
    _buffer_size = buffer_size if buffer_size is not None else _cfg.buffer_size
    _chunk_len = chunk_len if chunk_len is not None else _cfg.chunk_len
    _burn_in = burn_in if burn_in is not None else _cfg.burn_in
    _batch_size = batch_size if batch_size is not None else _cfg.batch_size
    _learning_starts = learning_starts if learning_starts is not None else _cfg.learning_starts
    _train_freq = train_freq if train_freq is not None else _cfg.train_freq
    _gradient_steps = gradient_steps if gradient_steps is not None else _cfg.gradient_steps
    _approx_episode_len = approx_episode_len if approx_episode_len is not None else _cfg.approx_episode_len
    _ent_coef_min = ent_coef_min if ent_coef_min is not None else _cfg.ent_coef_min

    # Mamba-specific defaults
    if model_type == "mamba2":
        _mamba_d_model = mamba_d_model if mamba_d_model is not None else _cfg.mamba_d_model
        _mamba_d_state = mamba_d_state if mamba_d_state is not None else _cfg.mamba_d_state
        _mamba_d_conv = mamba_d_conv if mamba_d_conv is not None else _cfg.mamba_d_conv
        _mamba_expand = mamba_expand if mamba_expand is not None else _cfg.mamba_expand
        _mamba_headdim = mamba_headdim if mamba_headdim is not None else _cfg.mamba_headdim
    else:
        _mamba_d_model = mamba_d_model or 64
        _mamba_d_state = mamba_d_state or 16
        _mamba_d_conv = mamba_d_conv or 4
        _mamba_expand = mamba_expand or 2
        _mamba_headdim = mamba_headdim or 64

    env_cls = _resolve_env_cls(env_type)

    # --------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------
    if lr_type == "D":
        lr_schedule = LinearSchedule(_learning_rate, _learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(_learning_rate)

    # --------------------------------------------------
    # Environments
    # --------------------------------------------------
    env_kwargs = dict(
        per_step_vel_penalty=per_step_vel_penalty,
        vel_penalty_coef=vel_penalty_coef,
    )

    if num_env > 1:
        train_env = SubprocVecEnv([
            _make_env(i, dt, deterministic=False, seed=seed, env_cls=env_cls, **env_kwargs)
            for i in range(num_env)
        ])
    else:
        train_env = env_cls(dt=dt, deterministic=False, **env_kwargs)

    if vec_normalize:
        from stable_baselines3.common.vec_env import VecNormalize
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = env_cls(dt=dt, deterministic=True)  # raw rewards for interpretability

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    model_label = {"mamba2": "Mamba2", "gru": "GRU", "lstm": "LSTM"}.get(model_type, model_type)
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"{model_label}SAC_CruiseControl_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * _approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if trainLoad:
        model = RecurrentSAC.load(
            log_dir / "best_model.zip",
            env=train_env,
        )
        model.gradient_steps = _gradient_steps
        model.train_freq = _train_freq
    else:
        model = RecurrentSAC(
            train_env,
            model_type=model_type,
            mamba_d_model=_mamba_d_model,
            mamba_d_state=_mamba_d_state,
            mamba_d_conv=_mamba_d_conv,
            mamba_expand=_mamba_expand,
            mamba_headdim=_mamba_headdim,
            hidden_size=_hidden_size,
            net_arch=dict(pi=[_nodes] * _layers, qf=[_nodes] * _layers),
            activation_fn=activation_fn,
            learning_rate=lr_schedule,
            gamma=_gamma,
            tau=_tau,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            ent_coef_min=_ent_coef_min,
            buffer_size=_buffer_size,
            chunk_len=_chunk_len,
            burn_in=_burn_in,
            batch_size=_batch_size,
            learning_starts=_learning_starts,
            train_freq=_train_freq,
            gradient_steps=_gradient_steps,
            seed=seed,
            verbose=1,
        )

    # --------------------------------------------------
    # W&B
    # --------------------------------------------------
    wandb_run = None
    if wandb_project is not None:
        import wandb

        wandb_config = dict(
            algo="sac",
            model_type=model_type,
            env_type=env_type,
            dt=dt,
            seed=seed,
            layers=_layers,
            nodes=_nodes,
            total_episodes=total_episodes,
            learning_rate=_learning_rate,
            lr_type=lr_type,
            gamma=_gamma,
            tau=_tau,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            ent_coef_min=_ent_coef_min,
            buffer_size=_buffer_size,
            chunk_len=_chunk_len,
            burn_in=_burn_in,
            batch_size=_batch_size,
            learning_starts=_learning_starts,
            train_freq=_train_freq,
            gradient_steps=_gradient_steps,
            num_env=num_env,
            per_step_vel_penalty=per_step_vel_penalty,
            vel_penalty_coef=vel_penalty_coef,
            vec_normalize=vec_normalize,
        )
        if model_type == "mamba2":
            wandb_config.update(
                mamba_d_model=_mamba_d_model,
                mamba_d_state=_mamba_d_state,
                mamba_d_conv=_mamba_d_conv,
                mamba_expand=_mamba_expand,
                mamba_headdim=_mamba_headdim,
            )
        else:
            wandb_config["hidden_size"] = _hidden_size

        wandb_kwargs = dict(
            project=wandb_project,
            name=training_name,
            config=wandb_config,
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
        eval_freq=_chunk_len * 10,
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
        description="Train recurrent PPO/SAC on cruise control (CleanRL-style)."
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm (default: ppo)",
    )
    parser.add_argument(
        "--model",
        choices=["mamba2", "gru", "lstm"],
        default="mamba2",
        help="Recurrent backbone (default: mamba2).",
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
        default=None,
        help="Context overlap steps for burn-in (default: from config, 0=disable)",
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

    # SAC-specific args
    parser.add_argument(
        "--tau", type=float, default=None,
        help="SAC Polyak averaging coefficient (default: from config)",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None,
        help="SAC replay buffer capacity (default: from config)",
    )
    parser.add_argument(
        "--chunk-len", type=int, default=None,
        help="SAC training sequence length (default: from config)",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=None,
        help="SAC random exploration steps before training (default: from config)",
    )
    parser.add_argument(
        "--train-freq", type=int, default=None,
        help="SAC env steps between gradient updates (default: from config)",
    )
    parser.add_argument(
        "--gradient-steps", type=int, default=None,
        help="SAC gradient steps per update (default: from config)",
    )
    parser.add_argument(
        "--target-entropy", type=str, default=None,
        help='SAC target entropy: "auto" or float e.g. "-2.0" (default: auto)',
    )
    parser.add_argument(
        "--ent-coef-min", type=float, default=None,
        help="SAC entropy coefficient floor (default: from config = 0.05)",
    )
    parser.add_argument(
        "--per-step-vel-penalty", action="store_true", default=None,
        help="Use per-step velocity tracking penalty (default: on for SAC)",
    )
    parser.add_argument(
        "--no-per-step-vel-penalty", dest="per_step_vel_penalty", action="store_false",
        help="Disable per-step velocity penalty (use episodic instead)",
    )
    parser.set_defaults(per_step_vel_penalty=True)
    parser.add_argument(
        "--vec-normalize", action="store_true", default=False,
        help="Wrap training env with VecNormalize reward normalization (default: off)",
    )

    args = parser.parse_args()

    if args.algo == "sac":
        train_mamba2_sac(
            model_type=args.model,
            env_type=args.env_type,
            num_env=args.num_env,
            total_episodes=args.total_episodes,
            burn_in=args.burn_in,
            seed=args.seed,
            tau=args.tau,
            buffer_size=args.buffer_size,
            chunk_len=args.chunk_len,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_entropy=args.target_entropy if args.target_entropy is not None else "auto",
            ent_coef_min=args.ent_coef_min,
            per_step_vel_penalty=args.per_step_vel_penalty,
            vec_normalize=args.vec_normalize,
            wandb_project=args.wandb_project,
            wandb_resume_id=args.wandb_resume_id,
            training_name=args.training_name,
            trainLoad=args.load,
        )
    else:
        train_recurrent(
            model_type=args.model,
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
