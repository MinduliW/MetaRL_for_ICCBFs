"""
Recurrent (LSTM) version of your JAIS Inspection training script.

Key changes vs your MLP file:
- Uses sb3-contrib RecurrentPPO + MlpLstmPolicy
- 1-layer LSTM
- Separate actor/critic LSTMs (shared_lstm=False, enable_critic_lstm=True)
- Separate actor/critic MLP heads (net_arch dict(pi=..., vf=...))
- Fixes SubprocVecEnv creation (must create a NEW env per worker)
- Evaluation rollouts keep LSTM hidden state via (state, episode_start)

Requirement:
    pip install sb3-contrib
"""

import os
import sys
import gc
import shutil
import warnings
import multiprocessing
import glob
import math
from math import *
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy as sp
from scipy.io import savemat

import gymnasium as gym
from gymnasium import spaces

import torch
from torch.optim import Adam
from torch.nn.modules import activation

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

# Your env
from inspectionEnvNoisy import InspectionEnv

warnings.filterwarnings("ignore", category=UserWarning, module=r"run_time_assurance\.rta\.asif")
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------
# utility: schedules
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# utility: env factory (parallel envs)  IMPORTANT: must create a NEW env per rank
# ----------------------------------------------------------------------
def make_env(rank: int, seed: int, dt: float, enable_param_randomisation: bool, enableNoise: bool, enableCBFtunning: bool,dvW: float):
    def _init_():
        env = InspectionEnv(
            dt=dt,
            enable_param_randomisation=enable_param_randomisation,
            enableNoise=enableNoise,
            enableCBFtunning=enableCBFtunning,
            dvWeight= dvW
        )
        env.action_space.seed(seed + rank)
        env = Monitor(env)
        return env

    return _init_


# ----------------------------------------------------------------------
# plotting helpers (your originals, lightly cleaned)
# ----------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401



# ----------------------------------------------------------------------
# main training + plotting function
# ----------------------------------------------------------------------
def run_jais_training():
    os.system('cls' if os.name == 'nt' else 'clear')

    multiprocessing.freeze_support()
    gc.collect()

    # ---------------- RL flags ----------------
    trainON = True
    trainLoad = True
    modelLoad = True

    # ---------------- env / timing ----------------
    totalEpisodes = int(1e5)  # not used directly
    if is_debug_mode():
        num_env = 1
    else:
        num_env = 64
    if not trainON:
        num_env = 1

    dt = 10

    # ---------------- PPO hyper-parameters ----------------
    approxEpisode = 400
    total_timesteps = int(totalEpisodes*approxEpisode)  # from paper

    learning_rate = 5e-5
    lr_type = 'D'  # C = constant, D = decreasing
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.01

    n_epochs = 10
    n_steps_factor = 50
    use_sde = False
    
      
    enable_param_randomisation = True
    enableNoise = False
    enableCBFtunning = True
    dvW = 10.0


    if lr_type == 'D':
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    custom_objects = {}

    # stochasticity (Gaussian policy std)
    std = 0.2
    log_std_init = float(np.log(std))

    # architecture params
    layers = 4
    nodes = 256  # you said your NN run uses 64 x 4; keep this
    activation_fn = activation.Tanh

    # Recurrent settings: 1-layer LSTM, separate actor/critic LSTMs + MLP heads
    lstm_hidden_size = 256
    n_lstm_layers = 1

    policy_kwargs = dict(
        activation_fn=activation_fn,
        ortho_init=True,
        log_std_init=log_std_init,
        share_features_extractor=False,
        optimizer_class=Adam,

        # recurrent bits
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
        shared_lstm=False,          # separate actor/critic LSTMs
        enable_critic_lstm=True,    # critic has its own LSTM

        # MLP heads after LSTM
        net_arch=[dict(pi=[nodes] * layers, vf=[nodes] * layers)],
    )

    batch_size = 64
    n_steps = int(batch_size * n_steps_factor * 8 / 64)

    training_name = (
        "NoisyInspectionRecurrentRNN"
        + "_l" + str(layers)
        + "_n" + str(nodes)
        + "_lstm" + str(n_lstm_layers)
        + "_lh" + str(lstm_hidden_size)
        + "_lr" + str(learning_rate) + lr_type
        + "_std" + str(std)
        + "_ne" + str(n_epochs)
        + "_ns" + str(n_steps / batch_size)
        + "_entropy" + str(ent_coef)
        + "_dvW" + str(dvW)
    )
    print("Training run name:", training_name)

    # ---------------- logging dirs ----------------
    tensorboard_log_dir_source = "TrainedModels/"
    os.makedirs(tensorboard_log_dir_source, exist_ok=True)

    eval_log_dir = os.path.join(tensorboard_log_dir_source, training_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_bank_path = os.path.join(eval_log_dir, "inspection_episode_spec_N1000_seed123_resetmeta_boundary.npz")

    # Deterministic evaluation environment with a fixed bank of 100 initial states
    deter_env_cb = InspectionEnv(
        dt=dt,
        enable_param_randomisation=enable_param_randomisation,
        enableNoise=enableNoise,
        enableCBFtunning=enableCBFtunning, dvWeight= dvW
    )
    
    # Callback prefers VecEnv
    # deter_env_cb_vec = DummyVecEnv([lambda: Monitor(deter_env_cb)])

    # Create Parallel Environments (VecEnv required)
    seed = 0
    if num_env > 1:
        train_env = SubprocVecEnv([make_env(rank=i, seed=seed, dt=dt, enable_param_randomisation=enable_param_randomisation, enableNoise=enableNoise, enableCBFtunning=enableCBFtunning, dvW=dvW) for i in range(num_env)])
    else:
        train_env = DummyVecEnv([make_env(rank=0, seed=seed, dt=dt, enable_param_randomisation=enable_param_randomisation, enableNoise=enableNoise, enableCBFtunning=enableCBFtunning, dvW=dvW)])

    # clear tb run dir if it exists
    if trainON and os.path.exists(tensorboard_log_dir_source + training_name + "_1"):
        shutil.rmtree(tensorboard_log_dir_source + training_name + "_1")

    plot_log_dir = os.path.join("Results/", training_name)
    os.makedirs(plot_log_dir, exist_ok=True)

    # ---------------- define model ----------------
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
            model = RecurrentPPO.load(
                os.path.join(eval_log_dir, "best_model.zip"),
                env=train_env,
                tensorboard_log=tensorboard_log_dir_source,
                custom_objects=custom_objects,
            )
        except Exception as e:
            print("Could not load previous model, starting fresh.", e)

    print("Train env:", train_env)

    # ---------------- training ----------------
    if trainON:
        print("Training:", total_timesteps,
              "Batch:", batch_size * num_env,
              "n_steps:", n_steps * num_env)

        callbacks = EvalCallback(
            eval_env=deter_env_cb,
            n_eval_episodes=5,
            eval_freq=n_steps,
            best_model_save_path=eval_log_dir,
            log_path=eval_log_dir,
            deterministic=True,
            verbose=1,
        )

        print("--- STARTING LEARNING ---")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=training_name,
            progress_bar=True,
        )
        print("--- DONE LEARNING ---")

        model.save(os.path.join(eval_log_dir, "final_model.zip"))
        del model
        gc.collect()

    # ---------------- load best model for plotting ----------------
    if modelLoad:
        print("Loading best model for plotting...")
        deter_env = InspectionEnv(
            dt=dt,  enable_param_randomisation=enable_param_randomisation,
            enableNoise = enableNoise,
            enableCBFtunning = enableCBFtunning, dvWeight=dvW
        )
        model = RecurrentPPO.load(
            os.path.join(eval_log_dir, "best_model.zip"),
            env=DummyVecEnv([lambda: Monitor(InspectionEnv(dt=dt,  enable_param_randomisation=enable_param_randomisation, enableNoise = enableNoise,
                                  enableCBFtunning = enableCBFtunning, dvWeight=dvW
                                    ))]),   
            tensorboard_log=tensorboard_log_dir_source,
        )
    else:
        deter_env = InspectionEnv(dt=dt,  enable_param_randomisation=enable_param_randomisation,
                                  enableNoise = enableNoise,
                                  enableCBFtunning = enableCBFtunning, dvWeight=dvW
                                    )

    # ------------------------------------------------------------------
    # Deterministic rollout + logging
    # ------------------------------------------------------------------
    print("Validating on deterministic JAIS env...")

    n_samples = 100

    TOF_samples = []
    DV_samples = []
    INSPECTED_PCT_FINAL = []
    REWARD_EP = []

    keep_traj = True
    traj_saved = False

    positions = None
    inspected_hist = None
    t_hist = None

    for i in range(n_samples):
        print(i)
        obs, _info = deter_env.reset()

        done = False
        truncated = False

        # recurrent state (IMPORTANT)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        t = 0.0
        dv_total = 0.0
        ep_reward = 0.0

        t_log = []
        inspected_pct_log = []
        dist_log = []
        speed_log = []
        u_safe_log = []
        u_rl_log = []
        theta_s_log = []
        theta_b_log_deg = []

        dv_inst_log = []
        dv_cum_log = []
        pos_log = []
        vel_log = []
        inspected_log = []

        if keep_traj and (not traj_saved):
            _positions = []
            _inspected_hist = []
            _t_hist = []

        while not (done or truncated):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )

            obs, reward, done, truncated, info = deter_env.step(action)
            ep_reward += float(reward)

            # mark episode start for next step
            episode_starts = np.array([done or truncated], dtype=bool)

            r = np.asarray(deter_env.state[:3], dtype=np.float64)
            v = np.asarray(deter_env.state[3:6], dtype=np.float64)

            theta_s = float(deter_env.state[6])
            theta_s_log.append(theta_s)

            sun_dir = deter_env._sun_direction(theta_s)
            rnorm = np.linalg.norm(r)
            boresight = -r / max(rnorm, 1e-12)

            cos_tb = np.clip(float(np.dot(boresight, sun_dir / max(np.linalg.norm(sun_dir), 1e-12))), -1.0, 1.0)
            theta_b_log_deg.append(np.degrees(np.arccos(cos_tb)))

            dist = float(np.linalg.norm(r))
            speed = float(np.linalg.norm(v))

            insp = np.asarray(deter_env.inspected, dtype=bool)
            insp_pct = 100.0 * float(np.mean(insp))

            u_safe = np.asarray(getattr(deter_env, "last_u_safe", np.zeros(3)), dtype=np.float64).reshape(3,)
            u_rl = np.asarray(getattr(deter_env, "last_u_rl", action), dtype=np.float64).reshape(-1)
            if u_rl.size >= 3:
                u_rl = u_rl[:3]
            else:
                u_rl = np.pad(u_rl, (0, 3 - u_rl.size))

            dv_inst = (np.linalg.norm(u_safe) / float(deter_env.m)) * float(deter_env.DT)
            dv_total += dv_inst

            t_log.append(t)
            inspected_pct_log.append(insp_pct)
            dist_log.append(dist)
            speed_log.append(speed)
            u_safe_log.append(u_safe.copy())
            u_rl_log.append(u_rl.copy())
            dv_inst_log.append(float(dv_inst))
            dv_cum_log.append(float(dv_total))
            pos_log.append(r.copy())
            vel_log.append(v.copy())
            inspected_log.append(insp.copy())

            if keep_traj and (not traj_saved):
                _positions.append(r.copy())
                _inspected_hist.append(insp.copy())
                _t_hist.append(t)

            t += float(deter_env.DT)

        insp_final = np.asarray(deter_env.inspected, dtype=bool)
        insp_pct_final = 100.0 * float(np.mean(insp_final))

        INSPECTED_PCT_FINAL.append(insp_pct_final)
        REWARD_EP.append(float(ep_reward))
        TOF_samples.append(t)
        DV_samples.append(dv_total)

        ep_dict = {
            "t_hist": np.asarray(t_log, dtype=np.float64),
            "inspected_pct_hist": np.asarray(inspected_pct_log, dtype=np.float64),
            "dist_hist": np.asarray(dist_log, dtype=np.float64),
            "speed_hist": np.asarray(speed_log, dtype=np.float64),
            "u_safe_hist": np.asarray(u_safe_log, dtype=np.float64),
            "u_rl_hist": np.asarray(u_rl_log, dtype=np.float64),
            "dv_inst_hist": np.asarray(dv_inst_log, dtype=np.float64),
            "dv_cum_hist": np.asarray(dv_cum_log, dtype=np.float64),
            "positions": np.asarray(pos_log, dtype=np.float64),
            "velocities": np.asarray(vel_log, dtype=np.float64),
            "inspected_hist": np.asarray(inspected_log, dtype=bool),
            "TOF": float(t),
            "DV_total": float(dv_total),
            "seed": int(i),
            "inspected_pct_final": float(insp_pct_final),
            "episode_reward": float(ep_reward),
            "eval_index": int(i),
            "theta_s_hist": np.asarray(theta_s_log, dtype=np.float64),
            "theta_b_deg_hist": np.asarray(theta_b_log_deg, dtype=np.float64),
        }
        savemat(os.path.join(plot_log_dir, f"eval_episode_{i:03d}.mat"), ep_dict)

        if keep_traj and (not traj_saved):
            positions = np.asarray(_positions)
            inspected_hist = np.asarray(_inspected_hist)
            t_hist = np.asarray(_t_hist)
            traj_saved = True

    TOF_samples = np.asarray(TOF_samples, dtype=np.float64)
    DV_samples = np.asarray(DV_samples, dtype=np.float64)
    INSPECTED_PCT_FINAL = np.asarray(INSPECTED_PCT_FINAL, dtype=np.float64)
    REWARD_EP = np.asarray(REWARD_EP, dtype=np.float64)

    savemat(
        os.path.join(plot_log_dir, "resultsInspection_summary.mat"),
        {
            "TOF_samples": TOF_samples,
            "DV_samples": DV_samples,
            "INSPECTED_PCT_FINAL": INSPECTED_PCT_FINAL,
            "REWARD_EP": REWARD_EP,
            "n_samples": np.array([n_samples], dtype=np.int32),
        },
    )

    print("TOF: mean ± std =", float(np.mean(TOF_samples)), float(np.std(TOF_samples)))
    print("Δv:  mean ± std =", float(np.mean(DV_samples)), float(np.std(DV_samples)))
    print("Final inspected (%): mean ± std =",
          float(np.mean(INSPECTED_PCT_FINAL)), float(np.std(INSPECTED_PCT_FINAL)))
    print("Episode reward: mean ± std =",
          float(np.mean(REWARD_EP)), float(np.std(REWARD_EP)))

    # Overlay constraint panels
    plot_all_constraints(plot_log_dir, deter_env, fft_stride=50)

    # Overlay plots (your originals)
    mat_files = sorted(glob.glob(os.path.join(plot_log_dir, "eval_episode_*.mat")))
    if len(mat_files) == 0:
        print("No eval_episode_*.mat files found; skipping overlay plots.")
    else:
        plt.figure()
        for f in mat_files:
            ep = sp.io.loadmat(f)
            t = ep["t_hist"].squeeze()
            y = ep["inspected_pct_hist"].squeeze()
            plt.plot(t, y, linewidth=1.0)
        plt.xlabel("t (s)")
        plt.ylabel("Inspected (%)")
        plt.title(f"Inspected % over time (n={len(mat_files)})")
        plt.savefig(os.path.join(plot_log_dir, "inspected_pct_over_time_all.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure()
        for f in mat_files:
            ep = sp.io.loadmat(f)
            t = ep["t_hist"].squeeze()
            y = ep["dist_hist"].squeeze()
            plt.plot(t, y, linewidth=1.0)
        plt.xlabel("t (s)")
        plt.ylabel(r"$||r||$ (m)")
        plt.title(f"Relative distance over time (n={len(mat_files)})")
        plt.savefig(os.path.join(plot_log_dir, "distance_over_time_all.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure()
        for f in mat_files:
            ep = sp.io.loadmat(f)
            t = ep["t_hist"].squeeze()
            y = ep["speed_hist"].squeeze()
            plt.plot(t, y, linewidth=1.0)
        plt.xlabel("t (s)")
        plt.ylabel(r"$||v||$ (m/s)")
        plt.title(f"Relative speed over time (n={len(mat_files)})")
        plt.savefig(os.path.join(plot_log_dir, "speed_over_time_all.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure()
        for f in mat_files:
            ep = sp.io.loadmat(f)
            t = ep["t_hist"].squeeze()
            y = ep["dv_cum_hist"].squeeze()
            plt.plot(t, y, linewidth=1.0)
        plt.xlabel("t (s)")
        plt.ylabel("Cumulative Δv (m/s)")
        plt.title(f"Cumulative Δv over time (n={len(mat_files)})")
        plt.savefig(os.path.join(plot_log_dir, "dv_over_time_all.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure()
        for f in mat_files:
            ep = sp.io.loadmat(f)
            t = ep["t_hist"].squeeze()
            u = ep["u_safe_hist"]
            plt.plot(t, u[:, 0], linewidth=1.0)
        plt.xlabel("t (s)")
        plt.ylabel("Fx (N)")
        plt.title(f"Fx over time (u_safe), n={len(mat_files)}")
        plt.savefig(os.path.join(plot_log_dir, "controls_Fx_over_time_all.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure()
        for f in mat_files:
            ep = sp.io.loadmat(f)
            t = ep["t_hist"].squeeze()
            u = ep["u_safe_hist"]
            plt.plot(t, u[:, 1], linewidth=1.0)
        plt.xlabel("t (s)")
        plt.ylabel("Fy (N)")
        plt.title(f"Fy over time (u_safe), n={len(mat_files)}")
        plt.savefig(os.path.join(plot_log_dir, "controls_Fy_over_time_all.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure()
        for f in mat_files:
            ep = sp.io.loadmat(f)
            t = ep["t_hist"].squeeze()
            u = ep["u_safe_hist"]
            plt.plot(t, u[:, 2], linewidth=1.0)
        plt.xlabel("t (s)")
        plt.ylabel("Fz (N)")
        plt.title(f"Fz over time (u_safe), n={len(mat_files)}")
        plt.savefig(os.path.join(plot_log_dir, "controls_Fz_over_time_all.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved overlay plots for {len(mat_files)} evaluation episodes to {plot_log_dir}")

    # Fig. 11 plot with cone
    if (positions is not None) and (inspected_hist is not None) and (t_hist is not None):
        fig11_path = os.path.join(plot_log_dir, "trajectory_fig11_cone.png")
        plot_fig11_like(deter_env, positions, inspected_hist, t_hist, save_path=fig11_path)
        print(f"Saved Fig. 11-style trajectory plot with FOV cone to {fig11_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_jais_training()
