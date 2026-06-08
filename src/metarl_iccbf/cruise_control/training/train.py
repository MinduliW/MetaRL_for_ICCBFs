# src/metarl_iccbf/cruise_control/training/train.py

import gc
import warnings
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional

import numpy as np
from torch.optim import Adam
from torch.nn.modules import activation

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

try:
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    WandbCallback = None  # type: ignore


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
# Unified training function
# ---------------------------------------------------------------------

def train_cruise_control(
    *,
    # ---- Policy / environment selection ----
    policy_type: Literal["MLP", "RNN", "MAMBA"] = "RNN",
    env_type: Literal["iccbf", "rl_only"] = "iccbf",

    dt: float = 0.1,
    seed: int = 123,

    # ---- Network architecture ----
    layers: int = 3,
    nodes: int = 64,
    activation_fn=activation.Tanh,

    # ---- LSTM parameters (ignored when policy_type="MLP") ----
    lstm_hidden_size: int = 64,
    n_lstm_layers: int = 1,
    shared_lstm: bool = False,
    enable_critic_lstm: bool = True,

    # ---- Mamba2 parameters (only used when policy_type="MAMBA") ----
    mamba_d_model: int = 64,
    mamba_d_state: int = 16,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    mamba_headdim: int = 64,

    # ---- RL hyperparameters ----
    total_episodes: int = 100_000,
    approx_episode_len: int = 200,
    learning_rate: float = 1e-4,
    lr_type: Literal["C", "D"] = "C",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.1,
    ent_coef: float = 0.01,
    target_kl: Optional[float] = 0.02,
    n_epochs: int = 10,
    batch_size: int = 512,
    n_steps: int = 200,
    use_sde: bool = False,

    # ---- Parallelism ----
    num_env: int = 1,

    # ---- Control ----
    trainON: bool = True,
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "outputs/cruise_control",
    training_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
):
    """
    Train a cruise-control policy.

    Supports four configurations:

    - **MLP-Tuned ICCBF**: ``policy_type="MLP", env_type="iccbf"``
      PPO with a feedforward MLP that tunes the ICCBF parameters.
    - **RNN-Tuned ICCBF**: ``policy_type="RNN", env_type="iccbf"``
      Recurrent PPO (LSTM) that tunes the ICCBF parameters.
    - **Mamba2-Tuned ICCBF**: ``policy_type="MAMBA", env_type="iccbf"``
      Recurrent PPO (Mamba2 SSM) that tunes the ICCBF parameters.
    - **Baseline (learned nominal, no CBF)**: ``policy_type="MLP", env_type="rl_only"``
      PPO learning a nominal controller without safety constraints.

    Parameters
    ----------
    policy_type : "MLP", "RNN", or "MAMBA"
        "MLP" for MLP-Tuned ICCBF (PPO + MlpPolicy),
        "RNN" for RNN-Tuned ICCBF (RecurrentPPO + MlpLstmPolicy),
        "MAMBA" for Mamba2-Tuned ICCBF (Mamba2PPO + Mamba2ActorCriticPolicy).
    env_type : "iccbf" or "rl_only"
        "iccbf" for the ICCBF-augmented environment (RL-tuned CBF),
        "rl_only" for the unconstrained baseline (learned nominal control).

    Returns
    -------
    model : PPO | RecurrentPPO | Mamba2PPO
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
    # Policy kwargs
    # --------------------------------------------------
    std_log = float(np.log(0.2))

    base_policy_kwargs = dict(
        activation_fn=activation_fn,
        ortho_init=True,
        log_std_init=std_log,
        share_features_extractor=True,
        optimizer_class=Adam,
        net_arch=[dict(pi=[nodes] * layers, vf=[nodes] * layers)],
    )

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    if training_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if env_type == "rl_only":
            method_tag = "Baseline"
        elif policy_type == "MAMBA":
            method_tag = "Mamba2TunedICCBF"
        elif policy_type == "RNN":
            method_tag = "RNNTunedICCBF"
        else:
            method_tag = "MLPTunedICCBF"
        training_name = f"{method_tag}_CruiseControl_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if policy_type == "MAMBA":  # Mamba2-Tuned ICCBF
        from metarl_iccbf.mamba.mamba_policy import Mamba2ActorCriticPolicy
        from metarl_iccbf.mamba.mamba_ppo import Mamba2PPO

        mamba_kwargs = dict(
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            mamba_d_model=mamba_d_model,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
        )
        policy_kwargs = {**base_policy_kwargs, **mamba_kwargs}

        model = Mamba2PPO(
            Mamba2ActorCriticPolicy,  # type: ignore[arg-type]
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
    elif policy_type == "RNN":  # RNN-Tuned ICCBF
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
    else:  # MLP-Tuned ICCBF or Baseline
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
        if policy_type == "MAMBA":
            from metarl_iccbf.mamba.mamba_ppo import Mamba2PPO as _LoadCls
            load_cls = _LoadCls
        elif policy_type == "RNN":
            load_cls = RecurrentPPO
        else:
            load_cls = PPO
        model = load_cls.load(
            log_dir / "best_model.zip",
            env=train_env,
            tensorboard_log=str(root_log_dir),
        )

    # --------------------------------------------------
    # wandb
    # --------------------------------------------------
    if wandb_project is not None:
        import wandb

        wandb.init(
            project=wandb_project,
            name=training_name,
            config=dict(
                policy_type=policy_type,
                env_type=env_type,
                dt=dt,
                seed=seed,
                layers=layers,
                nodes=nodes,
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
                num_env=num_env,
            ),
            sync_tensorboard=True,
        )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if trainON:
        callbacks: list[BaseCallback] = [
            EvalCallback(
                eval_env,
                n_eval_episodes=50,
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
        wandb.finish()

    return model, log_dir
