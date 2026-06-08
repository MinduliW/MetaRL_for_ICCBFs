"""Unified training script for the inspection problem.

Supports three policy architectures (MLP, RNN, MAMBA) and two environment
modes (``iccbf`` for CBF-tuning, ``rl_only`` for thrust-only).

Usage::

    from metarl_iccbf.inspection.training.train import train_inspection

    model, log_dir = train_inspection(
        policy_type="MAMBA",
        env_type="iccbf",
        num_env=28,
        total_episodes=100_000,
    )
"""

from __future__ import annotations

import gc
import warnings
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional, Union
import numpy as np
import gymnasium as gym

import torch.nn as nn
from torch.optim import Adam
from torch.nn.modules import activation

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

try:
    from metarl_iccbf.recurrent_cleanrl.configs import inspection_config
    _CFG = inspection_config()
except Exception:
    import math
    import types
    _CFG = types.SimpleNamespace(
        mamba_d_model=64, mamba_d_state=32, mamba_d_conv=4, mamba_expand=2,
        mamba_headdim=64, layers=3, nodes=256, learning_rate=5e-5, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, target_kl=0.02,
        n_epochs=10, batch_size=64, n_steps=612, burn_in=100,
        approx_episode_len=400, log_std_init=math.log(0.2),
    )
    inspection_config = None  # type: ignore

from metarl_iccbf.recurrent_cleanrl.configs import inspection_gru_config, inspection_lstm_ppo_config
from metarl_iccbf.recurrent_cleanrl.configs import inspection_sac_config, inspection_gru_sac_config, inspection_lstm_sac_config

try:
    from wandb.integration.sb3 import WandbCallback
except Exception:
    WandbCallback = None  # type: ignore
_GRU_CFG = inspection_gru_config()
_LSTM_PPO_CFG = inspection_lstm_ppo_config()


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


def _make_env(
    rank: int,
    seed: int,
    dt: float,
    enable_param_randomisation: bool,
    enableNoise: bool,
    enableCBFtunning: bool,
    dvWeight: float,

    morl: bool = False,
    morl_arch: str = "concat",
    morl_objective_set: str = "fuel",
    fixed_ic: "np.ndarray | None" = None,
    mdmm_entropy: bool = False,
    mdmm_H_target: float = 1.0,
    mdmm_H_decay: float = 0.0,
    popart: bool = False,
    popart_beta: float = 3e-4,
    adversarial: bool = False,
    dv_budget_range: tuple = (0.5, 5.0),
):
    """Factory that creates one Monitor-wrapped InspectionEnv per subprocess."""
    def _init():
        from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv
        env = InspectionEnv(
            dt=dt,
            enable_param_randomisation=enable_param_randomisation,
            enableNoise=enableNoise,
            enableCBFtunning=enableCBFtunning,
            dvWeight=dvWeight,
            adversarial=adversarial,
            dv_budget_range=dv_budget_range,
            morl=morl,
            morl_objective_set=morl_objective_set,
            fixed_ic=fixed_ic,
        )
        env.action_space.seed(seed + rank)
        if morl:
            class MORLMonitor(gym.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.rewards = []
                    self.lengths = 0
                def step(self, action):
                    obs, rew, term, trunc, info = self.env.step(action)
                    self.rewards.append(rew)
                    self.lengths += 1
                    if term or trunc:
                        ep_rew = np.sum(self.rewards, axis=0) # vector
                        # Store vector return for custom logging if needed
                        info["morl_episode_return"] = ep_rew
                        # Provide a scalar for SB3's generic logger to avoid crashes (using sum of components as proxy)
                        info["episode"] = {"r": float(np.sum(ep_rew)), "l": self.lengths, "t": 0.0}
                        self.rewards = []
                        self.lengths = 0
                    return obs, rew, term, trunc, info
            return MORLMonitor(env)
        else:
            return Monitor(env)
    return _init


# ---------------------------------------------------------------------
# Unified training function
# ---------------------------------------------------------------------

def train_inspection(
    *,
    # ---- Policy / environment selection ----
    policy_type: Literal["MLP", "RNN", "GRU", "MAMBA", "LSTM"] = "RNN",
    algo: Literal["ppo", "sac"] = "ppo",
    env_type: Literal["iccbf", "rl_only"] = "iccbf",

    dt: float = 10.0,
    seed: int = 0,

    # ---- Inspection-specific environment parameters ----
    enable_param_randomisation: bool = True,
    enableNoise: bool = False,
    dvWeight: float = 10.0,

    morl: bool = False,
    morl_arch: str = "concat",
    morl_n_objectives: int = 2,
    morl_fixed_obj: int = 0,
    morl_w_fixed: float = 0.6,
    morl_objective_set: str = "fuel",
    fixed_ic: "np.ndarray | None" = None,
    mdmm_entropy: bool = False,
    mdmm_H_target: float = 1.0,
    mdmm_H_decay: float = 0.0,
    popart: bool = False,
    popart_beta: float = 3e-4,
    adversarial: bool = False,
    dv_budget_range: tuple = (0.5, 5.0),

    # ---- Network architecture ----
    layers: int = _CFG.layers,
    nodes: int = _CFG.nodes,
    activation_fn=activation.Tanh,

    # ---- LSTM parameters (ignored when policy_type != "RNN") ----
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 1,
    shared_lstm: bool = False,
    enable_critic_lstm: bool = True,

    # ---- Mamba2 parameters (only used when policy_type="MAMBA") ----
    mamba_d_model: int = _CFG.mamba_d_model,
    mamba_d_state: int = _CFG.mamba_d_state,
    mamba_d_conv: int = _CFG.mamba_d_conv,
    mamba_expand: int = _CFG.mamba_expand,
    mamba_headdim: int = _CFG.mamba_headdim,

    # ---- Mamba2 burn-in (only MAMBA) ----
    burn_in: int = _CFG.burn_in,

    # ---- RL hyperparameters ----
    total_episodes: int = 400_000,
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
    use_sde: bool = False,

    # ---- Parallelism ----
    num_env: int = 64,

    # ---- SAC-specific (ignored for PPO) ----
    target_entropy: Optional[str] = None,
    ent_coef_min: Optional[float] = None,
    learning_starts: Optional[int] = None,
    gradient_steps: Optional[int] = None,
    buffer_size: Optional[int] = None,
    chunk_len: Optional[int] = None,
    train_freq: Optional[int] = None,

    # ---- Control ----
    trainON: bool = True,
    trainLoad: Union[bool, str] = False,

    # ---- Logging ----
    root_log_dir: str = "outputs/inspection",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_resume_id: Optional[str] = None,
) -> tuple[Union[PPO, RecurrentPPO], Path]:
    """Train an inspection policy.

    Configurations:

    - **MLP + iccbf**: PPO with MLP that tunes ICCBF parameters (12D action).
    - **RNN + iccbf**: RecurrentPPO (LSTM) that tunes ICCBF parameters.
    - **GRU + iccbf**: GRU PPO (CleanRL) that tunes ICCBF parameters.
    - **MAMBA + iccbf**: Mamba2PPO (CleanRL) that tunes ICCBF parameters.
    - **MLP + rl_only**: PPO with 3D thrust-only output (no CBF tuning).

    Returns
    -------
    model : PPO | RecurrentPPO | Mamba2PPO
    log_dir : Path
    """
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    enableCBFtunning = (env_type == "iccbf")

    # --------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------
    if lr_type == "D" and algo != "sac":
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    # --------------------------------------------------
    # Environments
    # --------------------------------------------------
    env_kwargs = dict(
        dt=dt,
        enable_param_randomisation=enable_param_randomisation,
        enableNoise=enableNoise,
        enableCBFtunning=enableCBFtunning,
        dvWeight=dvWeight,
        adversarial=adversarial,
        dv_budget_range=dv_budget_range,
        morl=morl,
        morl_objective_set=morl_objective_set,
        fixed_ic=fixed_ic,
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating {num_env} environments...", flush=True)
    if num_env > 1:
        train_env = SubprocVecEnv([
            _make_env(i, seed, **env_kwargs)
            for i in range(num_env)
        ])
    else:
        from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv
        train_env = InspectionEnv(**env_kwargs)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Environments created. Initialising {policy_type} policy...", flush=True)

    if morl:
        eval_env = SubprocVecEnv([_make_env(0, seed, **env_kwargs)])
    else:
        from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv
        eval_env = InspectionEnv(**env_kwargs)

    # Wrap training env with reward normalisation for SAC only
    # (eval_env left raw so episode returns remain interpretable)
    if algo == "sac":
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # --------------------------------------------------
    # Policy kwargs (SB3 models)
    # --------------------------------------------------
    std_log = float(_CFG.log_std_init)

    base_policy_kwargs = dict(
        activation_fn=activation_fn,
        ortho_init=True,
        log_std_init=std_log,
        share_features_extractor=False,
        optimizer_class=Adam,
        net_arch=[dict(pi=[nodes] * layers, vf=[nodes] * layers)],
    )

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if algo == "sac":
            if policy_type == "MAMBA":
                method_tag = "Mamba2SAC"
            elif policy_type == "LSTM":
                method_tag = "LSTMSAC"
            else:
                method_tag = "GRUSAC"
        elif env_type == "rl_only":
            method_tag = "Baseline"
        elif policy_type == "MAMBA":
            method_tag = "Mamba2TunedICCBF"
        elif policy_type == "GRU":
            method_tag = "GRUTunedICCBF"
        elif policy_type == "RNN":
            method_tag = "RNNTunedICCBF"
        elif policy_type == "LSTM":
            method_tag = "LSTMTunedICCBF"
        else:
            method_tag = "MLPTunedICCBF"
        training_name = f"{method_tag}_Inspection_{time_str}"
        if adversarial:
            training_name += "_Adversarial"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if algo == "sac":
        if policy_type not in ("MAMBA", "GRU", "LSTM"):
            raise ValueError("SAC only supports MAMBA, GRU, and LSTM policy types")

        from metarl_iccbf.recurrent_cleanrl.sac import RecurrentSAC

        if policy_type == "MAMBA":
            _sac_cfg = inspection_sac_config()
            _sac_model_type = "mamba2"
        elif policy_type == "LSTM":
            _sac_cfg = inspection_lstm_sac_config()
            _sac_model_type = "lstm"
        else:  # GRU
            _sac_cfg = inspection_gru_sac_config()
            _sac_model_type = "gru"

        # Resolve SAC overrides (CLI args take precedence over config defaults)
        _ent_coef_min = ent_coef_min if ent_coef_min is not None else _sac_cfg.ent_coef_min
        _target_entropy = target_entropy if target_entropy is not None else _sac_cfg.target_entropy
        _learning_starts = learning_starts if learning_starts is not None else _sac_cfg.learning_starts
        _gradient_steps = gradient_steps if gradient_steps is not None else _sac_cfg.gradient_steps
        _buffer_size = buffer_size if buffer_size is not None else _sac_cfg.buffer_size
        _chunk_len = chunk_len if chunk_len is not None else _sac_cfg.chunk_len
        _train_freq = train_freq if train_freq is not None else _sac_cfg.train_freq
        _sac_chunk_len = _chunk_len

        sac_kwargs = dict(
            env=train_env,
            model_type=_sac_model_type,
            net_arch=dict(pi=[nodes] * layers, qf=[nodes] * layers),
            activation_fn=nn.ReLU,
            learning_rate=lr_schedule,
            gamma=_sac_cfg.gamma,
            tau=_sac_cfg.tau,
            ent_coef=ent_coef,
            target_entropy=_target_entropy,
            ent_coef_min=_ent_coef_min,
            buffer_size=_buffer_size,
            chunk_len=_chunk_len,
            burn_in=_sac_cfg.burn_in,
            batch_size=_sac_cfg.batch_size,
            learning_starts=_learning_starts,
            train_freq=_train_freq,
            gradient_steps=_gradient_steps,
            seed=seed,
            verbose=1,
        )

        if policy_type in ("GRU", "LSTM"):
            sac_kwargs["hidden_size"] = _sac_cfg.hidden_size
        else:
            sac_kwargs["mamba_d_model"] = mamba_d_model
            sac_kwargs["mamba_d_state"] = mamba_d_state
            sac_kwargs["mamba_d_conv"] = mamba_d_conv
            sac_kwargs["mamba_expand"] = mamba_expand
            sac_kwargs["mamba_headdim"] = mamba_headdim

        model = RecurrentSAC(**sac_kwargs)

    elif policy_type == "MAMBA":
        from metarl_iccbf.recurrent_cleanrl.ppo import Mamba2PPO

        model = Mamba2PPO(
            train_env,
            mamba_d_model=mamba_d_model,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
            net_arch=dict(pi=[nodes] * layers, vf=[nodes] * layers),
            activation_fn=nn.Tanh,
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
            morl_n_objectives=morl_n_objectives,
            morl_fixed_obj=morl_fixed_obj,
            morl_w_fixed=morl_w_fixed,
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

    elif policy_type == "GRU":
        from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO as GRUPPO

        _gru_std = float(_GRU_CFG.log_std_init)
        model = GRUPPO(
            train_env,
            model_type="gru",
            hidden_size=_GRU_CFG.hidden_size,
            net_arch=dict(pi=[nodes] * layers, vf=[nodes] * layers),
            activation_fn=nn.Tanh,
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
            morl_n_objectives=morl_n_objectives,
            morl_fixed_obj=morl_fixed_obj,
            morl_w_fixed=morl_w_fixed,
            mdmm_entropy=mdmm_entropy,
            mdmm_H_target=mdmm_H_target,
            mdmm_H_decay=mdmm_H_decay,
            popart=popart,
            popart_beta=popart_beta,
            normalize_advantage=True,
            burn_in=burn_in,
            seed=seed,
            ortho_init=True,
            log_std_init=_gru_std,
            verbose=1,
        )

    elif policy_type == "RNN":
        lstm_kwargs = dict(
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
        )
        policy_kwargs = {**base_policy_kwargs, **lstm_kwargs}

        model = RecurrentPPO(
            MlpLstmPolicy,
            train_env,
            verbose=1,
            learning_rate=lr_schedule,
            tensorboard_log=str(root_log_dir),
            normalize_advantage=True,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

    elif policy_type == "LSTM":
        from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO as CleanRLRecurrentPPO

        _lstm_lr = LinearSchedule(_LSTM_PPO_CFG.learning_rate, _LSTM_PPO_CFG.learning_rate / 100)
        model = CleanRLRecurrentPPO(
            train_env,
            model_type="lstm",
            hidden_size=_LSTM_PPO_CFG.hidden_size,
            net_arch=dict(pi=[_LSTM_PPO_CFG.nodes] * _LSTM_PPO_CFG.layers,
                          vf=[_LSTM_PPO_CFG.nodes] * _LSTM_PPO_CFG.layers),
            activation_fn=nn.Tanh,
            learning_rate=_lstm_lr,
            gamma=_LSTM_PPO_CFG.gamma,
            gae_lambda=_LSTM_PPO_CFG.gae_lambda,
            clip_range=_LSTM_PPO_CFG.clip_range,
            ent_coef=_LSTM_PPO_CFG.ent_coef,
            target_kl=_LSTM_PPO_CFG.target_kl,
            n_steps=_LSTM_PPO_CFG.n_steps,
            n_epochs=_LSTM_PPO_CFG.n_epochs,
            batch_size=_LSTM_PPO_CFG.batch_size,
            morl=morl,
            morl_arch=morl_arch,
            morl_n_objectives=morl_n_objectives,
            morl_fixed_obj=morl_fixed_obj,
            morl_w_fixed=morl_w_fixed,
            mdmm_entropy=mdmm_entropy,
            mdmm_H_target=mdmm_H_target,
            mdmm_H_decay=mdmm_H_decay,
            popart=popart,
            popart_beta=popart_beta,
            normalize_advantage=True,
            burn_in=_LSTM_PPO_CFG.burn_in,
            seed=seed,
            ortho_init=True,
            log_std_init=float(_LSTM_PPO_CFG.log_std_init),
            verbose=1,
        )

    else:  # MLP
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=lr_schedule,
            tensorboard_log=str(root_log_dir),
            normalize_advantage=True,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            target_kl=target_kl,
            use_sde=use_sde,
            policy_kwargs=base_policy_kwargs,
            seed=seed,
        )

    # --------------------------------------------------
    # Optionally load from checkpoint
    # --------------------------------------------------
    if trainLoad:
        # trainLoad can be True (use default path) or a string (custom path)
        model_path = Path(trainLoad) if isinstance(trainLoad, str) else log_dir / "best_model.zip"

        if algo == "sac":
            from metarl_iccbf.recurrent_cleanrl.sac import RecurrentSAC
            model = RecurrentSAC.load(model_path, env=train_env)
            model.gradient_steps = _sac_cfg.gradient_steps
            model.train_freq = _sac_cfg.train_freq
        elif policy_type == "MAMBA":
            from metarl_iccbf.recurrent_cleanrl.ppo import Mamba2PPO
            model = Mamba2PPO.load(
                model_path, env=train_env,
            )
        elif policy_type in ("GRU", "LSTM"):
            from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO as CleanRLRecurrentPPO
            model = CleanRLRecurrentPPO.load(
                model_path, env=train_env,
            )
        elif policy_type == "RNN":
            model = RecurrentPPO.load(
                model_path, env=train_env,
                tensorboard_log=str(root_log_dir),
            )
        else:
            model = PPO.load(
                model_path, env=train_env,
                tensorboard_log=str(root_log_dir),
            )

    # --------------------------------------------------
    # wandb
    # --------------------------------------------------
    if wandb_project is not None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initialising Weights & Biases (project={wandb_project})...", flush=True)
        import wandb

        wandb_config = dict(
            algo=algo,
            policy_type=policy_type,
            env_type=env_type,
            dt=dt,
            seed=seed,
            layers=layers,
            nodes=nodes,
            total_episodes=total_episodes,
            learning_rate=learning_rate,
            lr_type=lr_type,
            num_env=num_env,
            enable_param_randomisation=enable_param_randomisation,
            enableNoise=enableNoise,
            dvWeight=dvWeight,
        )
        wandb_config.update(
            morl=morl,
            morl_arch=morl_arch,
            mdmm_entropy=mdmm_entropy,
            popart=popart,
        )

        if algo == "sac":
            wandb_config.update(
                ent_coef=ent_coef,
                target_entropy=_target_entropy,
                ent_coef_min=_ent_coef_min,
                gamma=_sac_cfg.gamma,
                tau=_sac_cfg.tau,
                buffer_size=_buffer_size,
                chunk_len=_chunk_len,
                burn_in=_sac_cfg.burn_in,
                batch_size=_sac_cfg.batch_size,
                learning_starts=_learning_starts,
                train_freq=_train_freq,
                gradient_steps=_gradient_steps,
            )
        else:
            wandb_config.update(
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                target_kl=target_kl,
                n_epochs=n_epochs,
                batch_size=batch_size,
                n_steps=n_steps,
            )

        if wandb_resume_id is not None:
            wandb.init(
                project=wandb_project,
                name=training_name,
                config=wandb_config,
                sync_tensorboard=True,
                resume="allow",
                id=wandb_resume_id,
            )
        else:
            wandb.init(
                project=wandb_project,
                name=training_name,
                config=wandb_config,
                sync_tensorboard=True,
            )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if trainON:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {algo.upper()} training loop...", flush=True)
        if algo == "sac":
            wandb_run = None
            if wandb_project is not None:
                import wandb
                wandb_run = wandb.run

            model.learn(
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_freq=_sac_chunk_len * 10,
                n_eval_episodes=5,
                best_model_save_path=str(log_dir),
                log_path=str(log_dir),
                wandb_run=wandb_run,
                tb_log_dir=str(root_log_dir),
                tb_log_name=training_name,
                progress_bar=True,
            )
        elif policy_type in ("MAMBA", "GRU", "LSTM"):
            # Mamba2PPO / GRU / LSTM RecurrentPPO (CleanRL) have their own learn() with built-in eval
            wandb_run = None
            if wandb_project is not None:
                import wandb
                wandb_run = wandb.run

            model.learn(
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_freq=n_steps,
                n_eval_episodes=5,
                best_model_save_path=str(log_dir),
                log_path=str(log_dir),
                wandb_run=wandb_run,
                tb_log_dir=str(root_log_dir),
                tb_log_name=training_name,
                progress_bar=True,
            )
        else:
            # SB3 PPO / RecurrentPPO
            callbacks: list[BaseCallback] = [
                EvalCallback(
                    eval_env,
                    n_eval_episodes=5,
                    eval_freq=5 * n_steps,
                    best_model_save_path=str(log_dir),
                    log_path=str(log_dir),
                    deterministic=True,
                    verbose=1,
                ),
            ]

            if wandb_project is not None and WandbCallback is not None:
                callbacks.append(WandbCallback(
                    model_save_path=str(log_dir),
                    verbose=1,
                ))

            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=training_name,
                progress_bar=True,
            )

        model.save(log_dir / "final_model.zip")

    if wandb_project is not None:
        import wandb
        wandb.finish()

    return model, log_dir
