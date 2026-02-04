import os
import sys
import gc
import warnings
import shutil
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Tanh, ReLU, ELU

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from daceypy import DA
from cruisecontrol.RLCBF import RLCBFcontrol


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

ACTIVATION_MAP = {
    "tanh": Tanh,
    "relu": ReLU,
    "elu": ELU,
}


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


def make_env(dt, deterministic, rank, seed):
    def _init():
        env = RLCBFcontrol(dt=dt, deterministic=deterministic)
        DA.init(4, 2)
        env.action_space.seed(seed + rank)
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_cruise_control(
    *,
    dt: float = 0.1,
    total_episodes: int = 200_000,
    approx_episode_len: int = 200,
    seed: int = 123,

    # ---- Architecture ----
    layers: int = 3,
    nodes: int = 64,
    activation_fn: Literal["tanh", "relu", "elu"] = "tanh",
    policy_type: Literal["PPO", "RNN"] = "PPO",

    # ---- RL hyperparameters ----
    learning_rate: float = 1e-4,
    lr_type: Literal["C", "D"] = "C",
    gamma: float = 0.999,
    gae_lambda: float = 0.99,
    clip_range: float = 0.1,
    ent_coef: float = 0.01,
    n_epochs: int = 10,
    batch_size: int = 512,
    n_steps: int = 256,
    target_kl: Optional[float] = 0.02,

    # ---- Environment ----
    num_env: int = 1,
    trainON: bool = True,
    trainLoad: bool = False,

    # ---- Logging ----
    root_log_dir: str = "TrainedModels",
):
    """
    Train a cruise-control ICCBF meta-policy with configurable network architecture.
    """

    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # --------------------------------------------------
    # Activation + architecture
    # --------------------------------------------------
    act_cls = ACTIVATION_MAP[activation_fn.lower()]
    net_arch = [dict(pi=[nodes] * layers, vf=[nodes] * layers)]

    policy_kwargs = dict(
        activation_fn=act_cls,
        net_arch=net_arch,
        optimizer_class=Adam,
        ortho_init=True,
        log_std_init=np.log(0.2),
        share_features_extractor=False,
    )

    # --------------------------------------------------
    # Learning rate schedule
    # --------------------------------------------------
    if lr_type == "D":
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    # --------------------------------------------------
    # Environments
    # --------------------------------------------------
    train_env = SubprocVecEnv([
        make_env(dt, deterministic=False, rank=i, seed=seed)
        for i in range(num_env)
    ])

    eval_env = RLCBFcontrol(dt=dt, deterministic=True)

    # --------------------------------------------------
    # Naming + logging
    # --------------------------------------------------
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_str = f"{policy_type}_L{layers}_N{nodes}_{activation_fn}"
    hyper_str = f"lr{learning_rate:g}_g{gamma}_gae{gae_lambda}_ent{ent_coef}"
    training_name = f"MetaCNNCruiseControl_{arch_str}_{hyper_str}_{time_str}"

    log_dir = Path(root_log_dir) / training_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(total_episodes * approx_episode_len)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if policy_type == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr_schedule,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(root_log_dir),
            verbose=1,
            seed=seed,
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=lr_schedule,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(root_log_dir),
            verbose=1,
            seed=seed,
        )

    # --------------------------------------------------
    # Load pretrained
    # --------------------------------------------------
    if trainLoad:
        model = PPO.load(log_dir / "best_model.zip", env=train_env)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if trainON:
        callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=n_steps,
            n_eval_episodes=10,
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
