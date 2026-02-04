import os
import sys
import warnings
import shutil
import datetime
import gc
import copy
import math
from math import *
from typing import Callable, Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import gymnasium as gym
from gymnasium import spaces

import torch
from torch.optim import Adam
from torch.nn.modules import activation

import multiprocessing

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from scipy.io import savemat
import scipy as sp
from scipy import integrate

from RLCBF import RLCBFcontrol


# ============================================================
# Threading / oversubscription control (helps when using many envs)
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass


# ============================================================
# Utilities
# ============================================================

def make_env(rank: int, dt: float, seed: int = 0):
    def _init():
        env = RLCBFcontrol(dt=dt, deterministic=False, nInitstates=1000, setconst=False)
        env = Monitor(env)

        # Seed env RNG (domain randomisation + measurement/actuation noise)
        env.reset(seed=seed + rank)

        # Seed action space too
        env.action_space.seed(seed + rank)
        return env
    return _init


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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Clear command line
    os.system('cls' if os.name == 'nt' else 'clear')

    # IMPORTANT for macOS: enforce spawn (safer with PyTorch + many subprocesses)
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set
        pass

    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    #### DEFINE SIMULATION ####
    trainON   = False      # TRAINING flag (set False if you only want eval)
    trainLoad = False     # Load a trained RNN model to continue training
    modelLoad = True      # Load a trained RNN model for validation

    totalEpisodes = int(2e5)

    if is_debug_mode():
        num_env = 1
    else:
        num_env = 64   # <<< CHANGED: from 8 to 64

    if not trainON:
        num_env = 1

    dt = 0.5

    # Deterministic env for evaluation / plotting
    deter_env = RLCBFcontrol(dt=dt, deterministic=True, nInitstates=10, setconst=False)
    deter_env.use_noise = False

    # Training env: parallelised if trainON
    base_seed = 0
    if num_env > 1:
        train_env = SubprocVecEnv(
            [make_env(rank=i, dt=dt, seed=base_seed) for i in range(num_env)],  # <<< FIXED: pass seed
            start_method="spawn",  # <<< IMPORTANT on macOS
        )
    else:
        train_env = RLCBFcontrol(dt=dt, deterministic=False, nInitstates=1000, setconst=False)
        train_env = Monitor(train_env)  # <<< FIXED: keep Monitor consistent

    approxEpisode   = 100
    total_timesteps = int(totalEpisodes * approxEpisode)

    learning_rate   = 5e-5
    lr_type         = 'C'  # C = constant, W = warmup, D = decreasing
    gamma           = 0.995
    gae_lambda      = 0.95
    clip_range      = 0.1
    ent_coef        = 0.01
    vf_coef         = 0.5

    n_epochs        = 10
    n_steps_factor  = 10
    use_sde         = True

    # LR schedule
    if lr_type == 'D':
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    custom_objects = {}

    # Stochasticity
    std = 0.2

    layers        = 4
    nodes         = 64
    activation_fn = activation.Tanh

    # Policy kwargs for RecurrentPPO with LSTM
    policy_kwargs = dict(
        activation_fn=activation_fn,
        ortho_init=True,
        log_std_init=float(np.log(std)),
        share_features_extractor=False,
        optimizer_class=Adam,
        net_arch=[dict(pi=[nodes] * layers, vf=[nodes] * layers)],
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )
    
    batch_size = 64

    # Your original intent: keep rollout size roughly constant when changing num_env.
    # With num_env=64: n_steps = 64*10*8/64 = 80, rollout = 80*64 = 5120.
    n_steps = int(batch_size * n_steps_factor * 8 / 64)

    training_name = (
        "NEWTest_New_MarginNoisyRotatingDockingCase_RNN"
        + "_l" + str(layers)
        + "_n" + str(nodes)
        + "_lr" + str(learning_rate) + lr_type
        + "_std" + str(std)
        + "_ne" + str(n_epochs)
        + "_ns" + str(n_steps / batch_size)
        + "_entropy" + str(ent_coef)
    )

    print('Training: ', training_name)

    # Tensorboard / model dirs
    tensorboard_log_dir_source = 'TrainedModels/'
    os.makedirs(tensorboard_log_dir_source, exist_ok=True)

    eval_log_dir = os.path.join(tensorboard_log_dir_source, training_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    # If you want a fresh run when training, delete previous dir
    if trainON and os.path.exists(tensorboard_log_dir_source + training_name + '_1'):
        shutil.rmtree('TrainedModels/' + training_name + '_1')

    # Plot storage
    plot_log_dir = os.path.join('Results/', training_name)
    os.makedirs(plot_log_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Define RNN Model (RecurrentPPO)
    # ------------------------------------------------------------
    model = RecurrentPPO(
        "MlpLstmPolicy",
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
        vf_coef=vf_coef,
        use_sde=use_sde,
        policy_kwargs=policy_kwargs,
    )

    if trainLoad:
        try:
            model = RecurrentPPO.load(
                os.path.join(eval_log_dir, 'best_model.zip'),
                env=train_env,
                tensorboard_log=tensorboard_log_dir_source,
                custom_objects=custom_objects,
            )
            print("Loaded existing RNN model for continued training.")
        except Exception as e:
            print("Unable to load model for training, starting from scratch:", e)

    print(train_env)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    if trainON:
        print('Training: ', total_timesteps,
              'Batch (SGD minibatch * env): ', batch_size * num_env,
              ' rollout steps collected per update: ', n_steps * num_env)

        # NOTE: eval_freq is in *callback steps*, not true timesteps.
        # With vec envs, 1 callback step advances num_env timesteps.
        # Keeping your original eval_freq=n_steps, this means evaluation every (n_steps * num_env) timesteps.
        callbacks = EvalCallback(
            eval_env=deter_env,
            n_eval_episodes=10,
            eval_freq=n_steps,
            best_model_save_path=eval_log_dir,
            log_path=eval_log_dir,
            deterministic=True,
            verbose=1,
        )

        print("--- STARTING RNN LEARNING ---")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=training_name,
            progress_bar=True,
        )
        print("--- DONE RNN LEARNING ---")

        model.save(os.path.join(eval_log_dir, 'final_model.zip'))
        del model

    # ------------------------------------------------------------
    # Load for evaluation
    # ------------------------------------------------------------
    if modelLoad:
        try:
            print('Loading Best RNN Model...')
            print(eval_log_dir)
            model = RecurrentPPO.load(
                os.path.join(eval_log_dir, 'best_model.zip'),
                env=deter_env,
                tensorboard_log=tensorboard_log_dir_source,
            )
        except Exception as e:
            print('Unable to load model, error:', e)
            raise

    # ------------------------------------------------------------
    # Deterministic Evaluation (10k samples) + .mat saving
    # ------------------------------------------------------------
    print('Validating RNN...')

    TOF = 50.0
    tvec_full = np.arange(0, TOF + dt, dt)

    lenvec   = len(tvec_full)
    nSamples = 10000

    uOpts       = np.zeros((nSamples, lenvec, 2))
    uOptmag     = np.zeros((nSamples, lenvec))
    uTotal      = np.zeros((nSamples, 1))
    actionStore = np.zeros((nSamples, lenvec, 4))

    tvec         = np.zeros((lenvec,))
    states       = np.zeros((nSamples, lenvec, 5))
    rewards      = np.zeros((nSamples, 1))
    hs           = np.zeros((nSamples, lenvec, 1))
    Vs           = np.zeros((nSamples, lenvec, 1))
    observations = np.zeros((nSamples, lenvec, deter_env.numObs))

    failcount = 0

    for i in range(nSamples):
        t = 0.0
        step = 0

        obs, _ = deter_env.reset(postProcess=False)
        rewardd = 0.0
        done = False

        # LSTM state and episode_start flag for RecurrentPPO
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            step += 1

            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )

            actionStore[i, step - 1, :] = action

            obs, reward_temp, done, _, _ = deter_env.step(action)

            rewardd += reward_temp
            observations[i, step - 1, :] = obs
            states[i, step - 1, :] = deter_env.x0 * 1e3

            uOpts[i, step - 1, :] = deter_env.control
            mag = np.linalg.norm(deter_env.control)
            uOptmag[i, step - 1] = mag
            hs[i, step - 1, :] = deter_env.dockingCase.originalh(deter_env.x0)

            V, _ = deter_env.dockingCase.calculate_V_and_dV(deter_env.x0)
            Vs[i, step - 1, :] = V

            t += dt
            tvec[step - 1] = t

            episode_starts = np.array([done], dtype=bool)

            if t >= TOF:
                done = True

        uTotal[i] = np.sum(uOptmag[i, :step]) * dt

        if np.any(hs[i, :step, 0] < 0.0):
            failcount += 1

        if (i + 1) % 100 == 0:
            print(f"Sample {i+1}/{nSamples} | reward = {rewardd:.3f}")

    print("Mean total u:", np.mean(uTotal[1:]))
    quartiles = np.percentile(uTotal[1:], [25, 50, 75])
    q1, q2, q3 = quartiles

    print("Q1 (25th percentile):", q1)
    print("Q2 (median):", q2)
    print("Q3 (75th percentile):", q3)
    print("Failures (h < 0):", failcount, "/", nSamples)

    savemat("MarginNoiseresultsStage1RNN.mat", {
        "uOpts": uOpts,
        "states": observations,
        "rewards": rewards,
        "hs": hs,
        "Vs": Vs,
        "actionStore": actionStore,
        "uTotal": uTotal,
        "tvec_full": tvec_full
    })

