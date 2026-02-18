import sys

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
import os
import warnings
import shutil
import datetime
import gc
import pandas as pd
import copy
from stable_baselines3.common.policies import ActorCriticPolicy
from datetime import datetime
from stable_baselines3 import PPO
from scipy import integrate
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Type
from stable_baselines3.common.callbacks import BaseCallback
from scipy.io import savemat
import scipy as sp
import math
from math import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from typing import Callable

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
# ==== CHANGED ====
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import torch
from torch.optim import Adam
from torch.nn.modules import activation

import multiprocessing

from RLCBF import RLCBFcontrol


# ==== CHANGED: make_env now creates a NEW env instance per rank ====
def make_env(rank: int, dt: float, deterministic: bool = False, seed: int = 0):
    def _init_():
        env = RLCBFcontrol(dt=dt, deterministic=deterministic)
        env.action_space.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init_


class ConstantSchedule:
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def __call__(self, progress_remaining: float) -> float:
        return self.initial_value


class LinearSchedule:
    def __init__(self, initial_value: float, min_value: float):
        self.initial_value = initial_value
        self.min_value = min_value

    def __call__(self, progress_remaining: float) -> float:
        return max(progress_remaining * self.initial_value, self.min_value)


def is_debug_mode():
    return sys.gettrace() is not None


if __name__ == "__main__":
    # Clear command line
    os.system('cls' if os.name == 'nt' else 'clear')

    multiprocessing.freeze_support()

    # Set the warning filter to "ignore" for UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    # Collect garbage
    gc.collect()

    #### DEFINE SIMULATION ####
    trainON         = True      # TRAINING flag
    trainLoad       = True      # Load a trained model to use in Training
    modelLoad       = True       # Load a trained model to use in Validation

    totalEpisodes   = int(100*100*10)  # *150

    if is_debug_mode():
        num_env = 1
    else:
        num_env = 64

    if not trainON:
        num_env = 1

    dt = 0.1
    stoch_env = RLCBFcontrol(dt=dt, deterministic=False)
    deter_env = RLCBFcontrol(dt=dt, deterministic=True)

    # ==== CHANGED: vectorised env uses proper env factory ====
    if num_env > 1:
        train_env = SubprocVecEnv([make_env(rank=i, dt=dt, deterministic=False) for i in range(num_env)])
    else:
        train_env = stoch_env

    approxEpisode   = 200

    total_timesteps = int(totalEpisodes * approxEpisode)
    learning_rate   = 1e-4
    lr_type         = 'C'  # C = constant, W=warmup, D=decreasing
    gamma           = 0.99
    gae_lambda      = 0.95
    clip_range      = 0.1
    ent_coef        = 0.01  # .001 #0.01# 0.01->0.1
    target_kl       = 0.02
    n_epochs        = 10
    # ==== CHANGED: must be False for RecurrentPPO ====
    use_sde         = False
    
    

    # LR Schedules
    if lr_type == 'D':
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    custom_objects = {}

    # Stochasticity (for PPO Gaussian policy)
    std             = 0.2
    std_log         = float(np.log(std))

    layers          = 3
    nodes           = 64
    activation_fn   = activation.Tanh

    # ==== CHANGED: base MLP policy kwargs ====
    base_policy_kwargs = dict(
    activation_fn=activation_fn,
    ortho_init=True,
    log_std_init=std_log,
    share_features_extractor=True,   # <-- CHANGED to True
    optimizer_class=Adam,
    net_arch=[dict(pi=[nodes] * layers, vf=[nodes] * layers)],
    )

    # Extra bits for LSTM when using RecurrentPPO
    lstm_kwargs = dict(
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,        # <-- important to avoid earlier assertion
    )

        

    batch_size      = 512
    # ==== CHANGED: nicer for recurrent + 20 envs; keep 200 (multiple of 20) ====
    n_steps         = 200

    # ==== CHANGED: choose which algo to use ====
    # "PPO" for feedforward; "RNN" for RecurrentPPO (meta-RL)
    MLPtype = 'RNN'

    # ---- build training name ----
    act_name  = getattr(activation_fn, "__name__", activation_fn.__class__.__name__)
    time_str  = datetime.now().strftime("%Y%m%d_%H%M%S")

    arch_str   = f"{MLPtype}_L{layers}_N{nodes}_{act_name}"
    hyper_str  = (
        f"lr{learning_rate:g}"
        f"_g{gamma:.3f}"
        f"_gae{gae_lambda:.3f}"
        f"_ent{ent_coef:g}"
    )

    # training_name = f"NewRNNwNoiseCruiseControl_{arch_str}_{hyper_str}_{time_str}"

    training_name = "MarginMetaRNNwNoiseCruiseControl_CLFhigherweight"
    # If you want to overwrite with a specific name, you still can:
    # training_name = 'CruiseControl_PPO_L3_N64_Tanh_lr0.0001_g0.995_gae0.990_ent0.01_20251201_214227'

    log_dir       = f"TrainedModels/{training_name}"
    print("Logging to", log_dir)

    # Tensorboard
    tensorboard_log_dir_source = 'TrainedModels/'
    os.makedirs(tensorboard_log_dir_source, exist_ok=True)

    eval_log_dir = os.path.join(tensorboard_log_dir_source, training_name)
    os.makedirs(eval_log_dir, exist_ok=True)
    if trainON and os.path.exists(tensorboard_log_dir_source + training_name + '_1'):
        shutil.rmtree('TrainedModels/' + training_name + '_1')

    # Plot storage
    plot_log_dir = os.path.join('Results/', training_name)
    os.makedirs(plot_log_dir, exist_ok=True)

    # ==== CHANGED: build correct policy_kwargs & model ====
    if MLPtype == 'PPO':
        policy_kwargs = base_policy_kwargs
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=lr_schedule,
            tensorboard_log=tensorboard_log_dir_source,
            normalize_advantage=True,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            use_sde=use_sde,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            target_kl=target_kl,
        )
    else:
        # RecurrentPPO with MlpLstmPolicy
        policy_kwargs = {**base_policy_kwargs, **lstm_kwargs}
        model = RecurrentPPO(
            MlpLstmPolicy,
            train_env,
            verbose=1,
            learning_rate=lr_schedule,
            tensorboard_log=tensorboard_log_dir_source,
            normalize_advantage=True,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            target_kl=target_kl,
        )

    if trainLoad:
        if MLPtype == 'PPO':
            model = PPO.load(
                os.path.join(eval_log_dir, 'best_model.zip'),
                env=train_env,
                tensorboard_log=tensorboard_log_dir_source,
                custom_objects=custom_objects,
            )
        else:
            model = RecurrentPPO.load(
                os.path.join(eval_log_dir, 'best_model.zip'),
                env=train_env,
                tensorboard_log=tensorboard_log_dir_source,
                custom_objects=custom_objects,
            )

    if trainON:
        print('Training: ', total_timesteps, 'Batch: ', batch_size * max(num_env, 1), ' n_steps: ', n_steps * max(num_env, 1))

        # Call Back
        callbacks = EvalCallback(
            eval_env=deter_env,
            n_eval_episodes=10,
            eval_freq=n_steps,
            best_model_save_path=eval_log_dir,
            log_path=eval_log_dir,
            deterministic=True,
            verbose=1,
        )

        # Train Model
        print("--- STARTING LEARNING ---")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=training_name, progress_bar=True)
        print("--- DONE LEARNING ---")

        # Save and Load Model
        model.save(os.path.join(eval_log_dir, 'final_model.zip'))
        del model  # remove to demonstrate saving and loading

