"""Mamba2 PPO training for docking ICCBF meta-policy (CleanRL-style).

Usage::

    python -m metarl_iccbf.docking.training.train [OPTIONS]

    # Quick smoke test (1 env, 100 episodes)
    python -m metarl_iccbf.docking.training.train --num-env 1 --total-episodes 100

    # Full training run
    python -m metarl_iccbf.docking.training.train --num-env 28 --total-episodes 200000 \\
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from metarl_iccbf.recurrent_cleanrl.configs import docking_config
from metarl_iccbf.recurrent_cleanrl.ppo import Mamba2PPO
from metarl_iccbf.recurrent_cleanrl.configs import docking_gru_config, docking_lstm_config
from metarl_iccbf.recurrent_cleanrl.configs import docking_sac_config, docking_gru_sac_config, docking_lstm_sac_config
from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO
from metarl_iccbf.recurrent_cleanrl.sac import RecurrentSAC

_CFG = docking_config()
_GRU_CFG = docking_gru_config()
_LSTM_CFG = docking_lstm_config()


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


def _make_env(rank: int, dt: float, deterministic: bool, seed: int,
              init_da: bool, da_order: int, da_vars: int,
          
    morl: bool = False,
    morl_arch: str = "concat",
    mdmm_entropy: bool = False,
    mdmm_H_target: float = 1.0,
    mdmm_H_decay: float = 0.0,
    popart: bool = False,
    popart_beta: float = 3e-4,

    morl: bool = False,
    morl_arch: str = "concat",
    mdmm_entropy: bool = False,
    mdmm_H_target: float = 1.0,
    mdmm_H_decay: float = 0.0,
    popart: bool = False,
    popart_beta: float = 3e-4,

    morl: bool = False,
    morl_arch: str = "concat",
    mdmm_entropy: bool = False,
    mdmm_H_target: float = 1.0,
    mdmm_H_decay: float = 0.0,
    popart: bool = False,
    popart_beta: float = 3e-4,
    adversarial: bool = False, adv_kwargs: Optional[dict] = None):
    """Factory for SubprocVecEnv. Must create a fresh env per subprocess."""
    def _init():
        # DA must be initialized per-process (process-global state)
        if init_da:
            try:
                from daceypy import DA
                DA.init(int(da_order), int(da_vars))
            except Exception:
                pass
        from metarl_iccbf.docking.RLCBF import RLCBFcontrol
        env = RLCBFcontrol(dt=dt, deterministic=deterministic,
                           tof=100.0 if adversarial else 50.0,
                           adversarial=adversarial, **(adv_kwargs or {}))
        env.action_space.seed(seed + rank)
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------

def train_docking_mamba2(
    *,
    # ---- Environment ----
    dt: float = 0.5,
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
    total_episodes: int = 200_000,
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
    num_env: int = 28,

    # ---- DA initialization ----
    init_da: bool = True,
    da_order: int = 4,
    da_vars: int = 2,

    # ---- Control ----
    trainLoad: bool = False,

    # ---- Adversarial ----
    adversarial: bool = False,
    adv_kwargs: Optional[dict] = None,

    # ---- Logging ----
    root_log_dir: str = "outputs/docking",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
):
    """Train a Mamba2 PPO policy for docking ICCBF.

    Returns
    -------
    model : Mamba2PPO
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

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
            _make_env(i, dt, deterministic=False, seed=seed,
                      init_da=init_da, da_order=da_order, da_vars=da_vars,
                      adversarial=adversarial, adv_kwargs=adv_kwargs)
            for i in range(num_env)
        ], start_method="spawn")
    else:
        if init_da:
            try:
                from daceypy import DA
                DA.init(int(da_order), int(da_vars))
            except Exception:
                pass
        from metarl_iccbf.docking.RLCBF import RLCBFcontrol
        train_env = RLCBFcontrol(dt=dt, deterministic=False,
                                 tof=100.0 if adversarial else 50.0,
                                 adversarial=adversarial, **(adv_kwargs or {}))

    # Eval env (main process)
    if init_da:
        try:
            from daceypy import DA
            DA.init(int(da_order), int(da_vars))
        except Exception:
            pass
    from metarl_iccbf.docking.RLCBF import RLCBFcontrol
    eval_env = RLCBFcontrol(dt=dt, deterministic=True,
                            tof=100.0 if adversarial else 50.0,
                            adversarial=adversarial, **(adv_kwargs or {}))

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"Mamba2TunedICCBF_Docking_{time_str}"
        # append adversarial tag if adversarial training is enabled
        if adversarial:
            training_name += "_Adversarial"

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
            morl=morl,
            morl_arch=morl_arch,
            mdmm_entropy=mdmm_entropy,
            mdmm_H_target=mdmm_H_target,
            mdmm_H_decay=mdmm_H_decay,
            popart=popart,
            popart_beta=popart_beta,
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
# GRU training function
# ---------------------------------------------------------------------

def train_docking_gru(
    *,
    # ---- Environment ----
    dt: float = 0.5,
    seed: int = 123,

    # ---- Network architecture ----
    layers: int = _GRU_CFG.layers,
    nodes: int = _GRU_CFG.nodes,
    activation_fn: type[nn.Module] = nn.Tanh,

    # ---- GRU parameters ----
    hidden_size: int = _GRU_CFG.hidden_size,

    # ---- RL hyperparameters ----
    total_episodes: int = 200_000,
    approx_episode_len: int = _GRU_CFG.approx_episode_len,
    learning_rate: float = _GRU_CFG.learning_rate,
    lr_type: Literal["C", "D"] = "D",
    gamma: float = _GRU_CFG.gamma,
    gae_lambda: float = _GRU_CFG.gae_lambda,
    clip_range: float = _GRU_CFG.clip_range,
    ent_coef: float = _GRU_CFG.ent_coef,
    target_kl: Optional[float] = _GRU_CFG.target_kl,
    n_epochs: int = _GRU_CFG.n_epochs,
    batch_size: int = _GRU_CFG.batch_size,
    n_steps: int = _GRU_CFG.n_steps,

    # ---- Burn-in ----
    burn_in: int = _GRU_CFG.burn_in,

    # ---- Parallelism ----
    num_env: int = 28,

    # ---- DA initialization ----
    init_da: bool = True,
    da_order: int = 4,
    da_vars: int = 2,

    # ---- Control ----
    trainLoad: bool = False,

    # ---- Adversarial ----
    adversarial: bool = False,
    adv_kwargs: Optional[dict] = None,

    # ---- Logging ----
    root_log_dir: str = "outputs/docking",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
):
    """Train a GRU PPO policy for docking ICCBF.

    Returns
    -------
    model : RecurrentPPO
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

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
            _make_env(i, dt, deterministic=False, seed=seed,
                      init_da=init_da, da_order=da_order, da_vars=da_vars,
                      adversarial=adversarial, adv_kwargs=adv_kwargs)
            for i in range(num_env)
        ], start_method="spawn")
    else:
        if init_da:
            try:
                from daceypy import DA
                DA.init(int(da_order), int(da_vars))
            except Exception:
                pass
        from metarl_iccbf.docking.RLCBF import RLCBFcontrol
        train_env = RLCBFcontrol(dt=dt, deterministic=False,
                                 tof=100.0 if adversarial else 50.0,
                                 adversarial=adversarial, **(adv_kwargs or {}))

    # Eval env (main process)
    if init_da:
        try:
            from daceypy import DA
            DA.init(int(da_order), int(da_vars))
        except Exception:
            pass
    from metarl_iccbf.docking.RLCBF import RLCBFcontrol
    eval_env = RLCBFcontrol(dt=dt, deterministic=True,
                            tof=100.0 if adversarial else 50.0,
                            adversarial=adversarial, **(adv_kwargs or {}))

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"GRUTunedICCBF_Docking_{time_str}"
        if adversarial:
            training_name += "_Adversarial"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)
    std_log = float(_GRU_CFG.log_std_init)

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
            model_type="gru",
            hidden_size=hidden_size,
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
            morl=morl,
            morl_arch=morl_arch,
            mdmm_entropy=mdmm_entropy,
            mdmm_H_target=mdmm_H_target,
            mdmm_H_decay=mdmm_H_decay,
            popart=popart,
            popart_beta=popart_beta,
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
                model_type="gru",
                dt=dt,
                seed=seed,
                layers=layers,
                nodes=nodes,
                hidden_size=hidden_size,
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
# LSTM training function
# ---------------------------------------------------------------------

def train_docking_lstm(
    *,
    # ---- Environment ----
    dt: float = 0.5,
    seed: int = 123,

    # ---- Network architecture ----
    layers: int = _LSTM_CFG.layers,
    nodes: int = _LSTM_CFG.nodes,
    activation_fn: type[nn.Module] = nn.Tanh,

    # ---- LSTM parameters ----
    hidden_size: int = _LSTM_CFG.hidden_size,

    # ---- RL hyperparameters ----
    total_episodes: int = 200_000,
    approx_episode_len: int = _LSTM_CFG.approx_episode_len,
    learning_rate: float = _LSTM_CFG.learning_rate,
    lr_type: Literal["C", "D"] = "C",
    gamma: float = _LSTM_CFG.gamma,
    gae_lambda: float = _LSTM_CFG.gae_lambda,
    clip_range: float = _LSTM_CFG.clip_range,
    ent_coef: float = _LSTM_CFG.ent_coef,
    target_kl: Optional[float] = _LSTM_CFG.target_kl,
    n_epochs: int = _LSTM_CFG.n_epochs,
    batch_size: int = _LSTM_CFG.batch_size,
    n_steps: int = _LSTM_CFG.n_steps,

    # ---- Burn-in ----
    burn_in: int = _LSTM_CFG.burn_in,

    # ---- Parallelism ----
    num_env: int = 28,

    # ---- DA initialization ----
    init_da: bool = True,
    da_order: int = 4,
    da_vars: int = 2,

    # ---- Control ----
    trainLoad: bool = False,

    # ---- Adversarial ----
    adversarial: bool = False,
    adv_kwargs: Optional[dict] = None,

    # ---- Logging ----
    root_log_dir: str = "outputs/docking",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
):
    """Train an LSTM PPO policy for docking ICCBF.

    Returns
    -------
    model : RecurrentPPO
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

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
            _make_env(i, dt, deterministic=False, seed=seed,
                      init_da=init_da, da_order=da_order, da_vars=da_vars,
                      adversarial=adversarial, adv_kwargs=adv_kwargs)
            for i in range(num_env)
        ], start_method="spawn")
    else:
        if init_da:
            try:
                from daceypy import DA
                DA.init(int(da_order), int(da_vars))
            except Exception:
                pass
        from metarl_iccbf.docking.RLCBF import RLCBFcontrol
        train_env = RLCBFcontrol(dt=dt, deterministic=False,
                                 tof=100.0 if adversarial else 50.0,
                                 adversarial=adversarial, **(adv_kwargs or {}))

    # Eval env (main process)
    if init_da:
        try:
            from daceypy import DA
            DA.init(int(da_order), int(da_vars))
        except Exception:
            pass
    from metarl_iccbf.docking.RLCBF import RLCBFcontrol
    eval_env = RLCBFcontrol(dt=dt, deterministic=True,
                            tof=100.0 if adversarial else 50.0,
                            adversarial=adversarial, **(adv_kwargs or {}))

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"LSTMTunedICCBF_Docking_{time_str}"
        if adversarial:
            training_name += "_Adversarial"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)
    std_log = float(_LSTM_CFG.log_std_init)

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
            model_type="lstm",
            hidden_size=hidden_size,
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
            morl=morl,
            morl_arch=morl_arch,
            mdmm_entropy=mdmm_entropy,
            mdmm_H_target=mdmm_H_target,
            mdmm_H_decay=mdmm_H_decay,
            popart=popart,
            popart_beta=popart_beta,
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
                model_type="lstm",
                dt=dt,
                seed=seed,
                layers=layers,
                nodes=nodes,
                hidden_size=hidden_size,
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
# SAC training function
# ---------------------------------------------------------------------

def train_docking_sac(
    *,
    # ---- Model ----
    model_type: Literal["mamba2", "gru", "lstm"] = "mamba2",

    # ---- Environment ----
    dt: float = 0.5,
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
    total_episodes: int = 200_000,
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

    # ---- Parallelism ----
    num_env: int = 28,

    # ---- DA initialization ----
    init_da: bool = True,
    da_order: int = 4,
    da_vars: int = 2,

    # ---- Control ----
    trainLoad: bool = False,

    # ---- Adversarial ----
    adversarial: bool = False,
    adv_kwargs: Optional[dict] = None,

    # ---- Logging ----
    root_log_dir: str = "outputs/docking",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
):
    """Train a recurrent SAC policy for docking ICCBF.

    Returns
    -------
    model : RecurrentSAC
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # --------------------------------------------------
    # Select config defaults based on model type
    # --------------------------------------------------
    if model_type == "gru":
        _cfg = docking_gru_sac_config()
        _hidden_size = hidden_size if hidden_size is not None else _cfg.hidden_size
    elif model_type == "lstm":
        _cfg = docking_lstm_sac_config()
        _hidden_size = hidden_size if hidden_size is not None else _cfg.hidden_size
    else:
        _cfg = docking_sac_config()
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
    _ent_coef_min = ent_coef_min if ent_coef_min is not None else getattr(_cfg, "ent_coef_min", 0.05)

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
            _make_env(i, dt, deterministic=False, seed=seed,
                      init_da=init_da, da_order=da_order, da_vars=da_vars,
                      adversarial=adversarial, adv_kwargs=adv_kwargs)
            for i in range(num_env)
        ], start_method="spawn")
    else:
        if init_da:
            try:
                from daceypy import DA
                DA.init(int(da_order), int(da_vars))
            except Exception:
                pass
        from metarl_iccbf.docking.RLCBF import RLCBFcontrol
        train_env = RLCBFcontrol(dt=dt, deterministic=False,
                                 tof=100.0 if adversarial else 50.0,
                                 adversarial=adversarial, **(adv_kwargs or {}))

    # Wrap training env with reward normalisation (eval env left raw for interpretability)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Eval env (main process)
    if init_da:
        try:
            from daceypy import DA
            DA.init(int(da_order), int(da_vars))
        except Exception:
            pass
    from metarl_iccbf.docking.RLCBF import RLCBFcontrol
    eval_env = RLCBFcontrol(dt=dt, deterministic=True,
                            tof=100.0 if adversarial else 50.0,
                            adversarial=adversarial, **(adv_kwargs or {}))

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    model_label = {"mamba2": "Mamba2", "gru": "GRU", "lstm": "LSTM"}.get(model_type, model_type)
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"{model_label}SAC_Docking_{time_str}"
        if adversarial:
            training_name += "_Adversarial"

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
        description="Train recurrent PPO/SAC for docking ICCBF (CleanRL-style)."
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
        help="Recurrent backbone (default: mamba2)",
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
        default=200_000,
        help="Total training episodes (default: 200000)",
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
        help="W&B run ID to resume (e.g. 'ksomsw9d'). Requires --wandb-project.",
    )
    parser.add_argument(
        "--no-da",
        action="store_true",
        help="Skip DA.init() (for envs that don't need it).",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Enable adversarial target rotation.",
    )


    # MORL Phase 2 flags
    parser.add_argument("--morl", action="store_true", help="Enable MORL")
    parser.add_argument("--morl-arch", choices=["concat", "multi_body"], default="concat")
    parser.add_argument("--mdmm-entropy", action="store_true")
    parser.add_argument("--mdmm-H-target", type=float, default=1.0)
    parser.add_argument("--mdmm-H-decay", type=float, default=0.0)
    parser.add_argument("--popart", action="store_true")
    parser.add_argument("--popart-beta", type=float, default=3e-4)

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
        help='SAC target entropy: "auto" or float e.g. "-6.0" (default: auto)',
    )
    parser.add_argument(
        "--ent-coef-min", type=float, default=None,
        help="SAC ent_coef floor (default: from config = 0.05)",
    )

    args = parser.parse_args()

    common_kwargs = dict(
        num_env=args.num_env,
        total_episodes=args.total_episodes,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_resume_id=args.wandb_resume_id,
        training_name=args.training_name,
        trainLoad=args.load,
        init_da=not args.no_da,
        adversarial=args.adversarial,

        morl=args.morl,
        morl_arch=args.morl_arch,
        mdmm_entropy=args.mdmm_entropy,
        mdmm_H_target=args.mdmm_H_target,
        mdmm_H_decay=args.mdmm_H_decay,
        popart=args.popart,
        popart_beta=args.popart_beta,

    )

    if args.algo == "sac":
        train_docking_sac(
            model_type=args.model,
            burn_in=args.burn_in,
            tau=args.tau,
            buffer_size=args.buffer_size,
            chunk_len=args.chunk_len,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_entropy=args.target_entropy or "auto",
            ent_coef_min=args.ent_coef_min,
            **common_kwargs,
        )
    elif args.model == "gru":
        train_docking_gru(
            burn_in=args.burn_in if args.burn_in is not None else _GRU_CFG.burn_in,
            **common_kwargs,
        )
    elif args.model == "lstm":
        train_docking_lstm(
            burn_in=args.burn_in if args.burn_in is not None else _LSTM_CFG.burn_in,
            **common_kwargs,
        )
    else:
        train_docking_mamba2(
            burn_in=args.burn_in if args.burn_in is not None else _CFG.burn_in,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
