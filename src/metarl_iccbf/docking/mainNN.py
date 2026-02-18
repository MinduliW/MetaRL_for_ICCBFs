import os
import sys
import warnings
import shutil
import gc
import math
from math import *
from typing import Callable, Type

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
from torch.optim import Adam
from torch.nn.modules import activation

import multiprocessing

import gymnasium as gym
from gymnasium import spaces

from scipy.io import savemat
from scipy import integrate
import scipy as sp

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from RLCBF import RLCBFcontrol


# ============================================================
# Threading / oversubscription control (important with many envs)
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
# VecEnv factory (IMPORTANT: must create a NEW env per subprocess)
# ============================================================
def make_env(rank: int, dt: float, seed: int = 0):
    """
    Create a *fresh* environment instance inside each subprocess.
    Do NOT pass a single pre-created env instance into SubprocVecEnv.
    """
    def _init():
        env = RLCBFcontrol(dt=dt, deterministic=False, nInitstates=1000, setconst=False)
        env = Monitor(env)

        # Seed env RNG (domain randomisation + noise)
        env.reset(seed=seed + rank)
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


if __name__ == "__main__":
    # Clear command line
    os.system('cls' if os.name == 'nt' else 'clear')

    # Multiprocessing safety (macOS/Windows)
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()
    
    

    #### DEFINE SIMULATION ####
    trainON   = True      # TRAINING flag
    trainLoad = True     # Load trained model for continued training
    modelLoad = True      # Load trained model for validation

    totalEpisodes = int(2e5)  # *150 originally

    if is_debug_mode():
        num_env = 1
    else:
        num_env = 64  # <<< parallel envs

    if not trainON:
        num_env = 1

    dt = 0.5

    # Create base envs (deterministic for eval/plots)
    stoch_env = RLCBFcontrol(dt=dt, deterministic=False, nInitstates=1000, setconst=False)
    deter_env = RLCBFcontrol(dt=dt, deterministic=True,  nInitstates=10,   setconst=False)

    # IMPORTANT: For SubprocVecEnv, do NOT pass `stoch_env` into workers.
    # Each worker must construct its own env via make_env(...).
    base_seed = 0
    if num_env > 1:
        train_env = SubprocVecEnv(
            [make_env(rank=i, dt=dt, seed=base_seed) for i in range(num_env)],
            start_method="spawn",
        )
    else:
        train_env = Monitor(stoch_env)

    approxEpisode   = 100
    total_timesteps = int(totalEpisodes * approxEpisode)

    learning_rate = 5e-5
    lr_type       = 'C'   # C=constant, W=warmup, D=decreasing
    gamma         = 0.995
    gae_lambda    = 0.95
    clip_range    = 0.2
    ent_coef      = 0.01
    vf_coef       = 0.5

    n_epochs       = 10
    n_steps_factor = 10
    use_sde        = True

    # LR Schedule
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

    # NOTE: SB3 expects float for log_std_init; your old code used a tensor.
    policy_kwargs = dict(
        activation_fn=activation_fn,
        ortho_init=True,
        log_std_init=float(np.log(std)),
        share_features_extractor=False,
        optimizer_class=Adam,
        net_arch=[dict(pi=[nodes] * layers, vf=[nodes] * layers)],
    )

    batch_size = 64

    # Your original intent: keep rollout size roughly constant when changing num_env.
    # With num_env=64: n_steps = 64*10*8/64 = 80, rollout = 80*64 = 5120.
    n_steps = int(batch_size * n_steps_factor * 8 / 64)

    MLPtype = 'PPO'

    if MLPtype == 'PPO':
        training_name = (
            'TEEEESTNoisyRotatingDockingCase'
            + '_' + '_l' + str(layers)
            + '_n' + str(nodes)
            + '_lr' + str(learning_rate) + lr_type
            + '_std' + str(std)
            + '_ne' + str(n_epochs)
            + '_ns' + str(n_steps / batch_size)
            + 'entropy' + str(ent_coef)
        )
    else:
        training_name = (
            'NoisyRotatingDockingCase_RNN'
            + '_' + '_l' + str(layers)
            + '_n' + str(nodes)
            + '_lr' + str(learning_rate) + lr_type
            + '_std' + str(std)
            + '_ne' + str(n_epochs)
            + '_ns' + str(n_steps / batch_size)
            + 'entropy' + str(ent_coef)
        )

    print('Training: ', training_name)

    # Tensorboard / model dirs
    tensorboard_log_dir_source = 'TrainedModels/'
    os.makedirs(tensorboard_log_dir_source, exist_ok=True)

    eval_log_dir = os.path.join(tensorboard_log_dir_source, training_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    if trainON and os.path.exists(tensorboard_log_dir_source + training_name + '_1'):
        shutil.rmtree('TrainedModels/' + training_name + '_1')

    plot_log_dir = os.path.join('Results/', training_name)
    os.makedirs(plot_log_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Define Model
    # ------------------------------------------------------------
    if MLPtype == 'PPO':
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
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
        )
    else:
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
            use_sde=use_sde,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
        )

    if trainLoad:
        try:
            if MLPtype == 'PPO':
                model = PPO.load(
                    os.path.join(eval_log_dir, 'best_model.zip'),
                    env=train_env,
                    tensorboard_log=tensorboard_log_dir_source,
                    custom_objects=custom_objects
                )
            else:
                model = RecurrentPPO.load(
                    os.path.join(eval_log_dir, 'best_model.zip'),
                    env=train_env,
                    tensorboard_log=tensorboard_log_dir_source,
                    custom_objects=custom_objects
                )
            print("Loaded existing model for continued training.")
        except Exception as e:
            print("Unable to load model for training, starting from scratch:", e)

    print(train_env)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    if trainON:
        print('Training: ', total_timesteps,
              'Batch(minibatch*env): ', batch_size * num_env,
              ' rollout n_steps*num_env: ', n_steps * num_env)

        callbacks = EvalCallback(
            eval_env=deter_env,
            n_eval_episodes=10,
            eval_freq=n_steps,   # callback steps (vec env => each step advances num_env timesteps)
            best_model_save_path=eval_log_dir,
            log_path=eval_log_dir,
            deterministic=True,
            verbose=1
        )

        print("--- STARTING LEARNING ---")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=training_name,
            progress_bar=True
        )
        print("--- DONE LEARNING ---")

        model.save(os.path.join(eval_log_dir, 'final_model.zip'))
        del model

    # ------------------------------------------------------------
    # Load for evaluation
    # ------------------------------------------------------------
    if modelLoad:
        try:
            print('Loading Best Model...')
            print(eval_log_dir)
            if MLPtype == 'PPO':
                model = PPO.load(
                    os.path.join(eval_log_dir, 'best_model.zip'),
                    env=deter_env,
                    tensorboard_log=tensorboard_log_dir_source
                )
            else:
                model = RecurrentPPO.load(
                    os.path.join(eval_log_dir, 'best_model.zip'),
                    env=deter_env,
                    tensorboard_log=tensorboard_log_dir_source
                )
        except Exception as e:
            print('Unable to load model, error:', e)
            raise

    # ------------------------------------------------------------
    # Deterministic evaluation + saving + plots (UNCHANGED BELOW)
    # ------------------------------------------------------------
    print('Validating...')

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

    for i in range(nSamples):
        t = 0.0
        step = 0

        obs, _ = deter_env.reset(postProcess=False)
        rewardd = 0.0
        done = False

        while done is False:
            step += 1
            action, _ = model.predict(obs, deterministic=True)

            actionStore[i, step - 1, :] = action
            obs, reward_temp, done, _, _ = deter_env.step(action)

            rewardd += reward_temp
            observations[i, step - 1, :] = obs
            states[i, step - 1, :] = deter_env.x0 * 1e3

            uOpts[i, step - 1, :] = deter_env.control
            mag = np.linalg.norm(deter_env.control)
            uOptmag[i, step - 1] = mag

            hs[i, step - 1, :] = deter_env.dockingCase.originalh(deter_env.x0)
            V, dV_dx_val = deter_env.dockingCase.calculate_V_and_dV(deter_env.x0)
            Vs[i, step - 1, :] = V

            t += dt
            tvec[step - 1] = t

        uTotal[i] = np.sum(uOptmag[i, :step]) * dt
        print(rewardd)

    print("Mean:", np.mean(uTotal[1:]))
    q1, q2, q3 = np.percentile(uTotal[1:], [25, 50, 75])
    print("Q1 (25th percentile):", q1)
    print("Q2 (median):", q2)
    print("Q3 (75th percentile):", q3)

    savemat("MarginDockingICCBF.mat", {
        "uOpts": uOpts,
        "states": observations,
        "rewards": rewards,
        "hs": hs,
        "Vs": Vs,
        "actionStore": actionStore,
        "uTotal": uTotal,
        "tvec_full": tvec_full
    })

