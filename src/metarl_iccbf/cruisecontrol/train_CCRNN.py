# src/cruisecontrol/train_CCRNN.py

import os
import gc
import warnings
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.modules import activation

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from cruisecontrol.RLCBF import RLCBFcontrol


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


def make_env(rank: int, dt: float, deterministic: bool, seed: int):
    def _init():
        env = RLCBFcontrol(dt=dt, deterministic=deterministic)
        env.action_space.seed(seed + rank)
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------
# Training function (LSTM only)
# ---------------------------------------------------------------------

def train_cruise_control_lstm(
    *,
    dt: float = 0.1,
    seed: int = 123,

    # ---- Network architecture ----
    layers: int = 3,
    nodes: int = 64,
    activation_fn = activation.Tanh,

    # ---- LSTM parameters ----
    lstm_hidden_size: int = 64,
    n_lstm_layers: int = 1,
    shared_lstm: bool = False,
    enable_critic_lstm: bool = True,

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
    n_steps: int = 200,   # must be divisible by num_env

    # ---- Parallelism ----
    num_env: int = 1,

    # ---- Control ----
    trainON: bool = True,
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "TrainedModels",
):
    """
    Train a recurrent (LSTM) ICCBF meta-policy for cruise control.
    Returns (model, log_dir).
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
    train_env = SubprocVecEnv([
        make_env(i, dt, deterministic=False, seed=seed)
        for i in range(num_env)
    ])

    eval_env = RLCBFcontrol(dt=dt, deterministic=True)

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

    lstm_kwargs = dict(
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
        shared_lstm=shared_lstm,
        enable_critic_lstm=enable_critic_lstm,
    )

    policy_kwargs = {**base_policy_kwargs, **lstm_kwargs}

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_str = f"RNN_L{layers}_N{nodes}_H{lstm_hidden_size}"
    hyper_str = f"lr{learning_rate:g}_g{gamma}_gae{gae_lambda}_ent{ent_coef}"

    training_name = f"MetaRNN_CruiseControl_{arch_str}_{hyper_str}_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
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

    if trainLoad:
        model = RecurrentPPO.load(
            log_dir / "best_model.zip",
            env=train_env,
            tensorboard_log=str(root_log_dir),
        )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if trainON:
        callback = EvalCallback(
            eval_env,
            n_eval_episodes=10,
            eval_freq=n_steps,
            best_model_save_path=log_dir,
            log_path=log_dir,
            deterministic=True,
            verbose=1,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

        model.save(log_dir / "final_model.zip")

    return model, log_dir
