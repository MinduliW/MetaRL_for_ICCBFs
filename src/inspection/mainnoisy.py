#NN_falseTr is running NN mainnoisy, but without any noise or param variation in training env atm, also nodes are 64 of 4 layers. 
import sys
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from Observation import ObservationModel

import os
import warnings
import shutil
import datetime
import gc
import pandas as pd
import copy

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from scipy import integrate
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Type
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from scipy.io import savemat

import scipy as sp
import math
from math import *
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import torch
from torch.optim import Adam
from torch.nn.modules import activation

import multiprocessing

from inspectionEnvNoisy import InspectionEnv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"run_time_assurance\.rta\.asif")


# ----------------------------------------------------------------------
# utility: env factory (parallel envs)
# ----------------------------------------------------------------------
def make_env(env_instance, rank, model=None, seed=0):
    def _init_():
        env = env_instance
        env.action_space.seed(seed + rank)
        env.model = model 
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


# ----------------------------------------------------------------------
# main training + plotting function
# ----------------------------------------------------------------------
def run_jais_training():
    # Clear command line
    os.system('cls' if os.name == 'nt' else 'clear')

    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # ---------------- RL flags ----------------
    trainON         = False       # actually do training
    trainLoad       = False      # resume from previous best_model
    modelLoad       = True       # load best_model for plotting at the end
    
    enable_param_randomisation = True
    enableNoise = False
    enableCBFtunning = False

    # ---------------- env / timing ----------------
    totalEpisodes   = int(1e6)   # just used to scale total_timesteps
    if is_debug_mode():
        num_env = 1
    else:
        num_env = 64
        
    if not trainON:
        num_env = 1
        
        
    dt = 10
    stoch_env = InspectionEnv(dt=dt,  enable_param_randomisation=enable_param_randomisation,enableNoise = enableNoise, enableCBFtunning = enableCBFtunning )
    

   
    # ---------------- PPO hyper-parameters ----------------
    approxEpisode   = 400    # average steps per episode (<= 1224 from paper)
    total_timesteps = int(5000000) #from paper.

    learning_rate   = 1e-4
    lr_type         = 'D'    # C = constant, D = decreasing
    gamma           = 0.99
    gae_lambda      = 0.95
    clip_range      = 0.2
    ent_coef        = 0.01
    # vf_coef         = 0.5

    n_epochs        = 10
    n_steps_factor  = 50
    use_sde         = False

    # learning-rate schedule
    if lr_type == 'D':
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    custom_objects = {}

    # stochasticity (Gaussian policy std)
    std             = 0.2
    log_std_init    = float(np.log(std))

    # network architecture: JAIS paper → 2 hidden layers × 256 tanh
    layers          = 2
    nodes           = 256
    activation_fn   = activation.Tanh

    policy_kwargs   = dict(
        activation_fn=activation_fn,
        ortho_init=True,
        log_std_init=log_std_init,
        share_features_extractor=False,
        optimizer_class=Adam,
        net_arch=[dict(pi=[nodes]*layers, vf=[nodes]*layers)],
    )

    batch_size      = 64
    n_steps         = int(batch_size * n_steps_factor*8/64)

#withRTA_
    training_name = (
        "uRLonlyNN"
        + "_l" + str(layers)
        + "_n" + str(nodes)
        + "_lr" + str(learning_rate) + lr_type
        + "_std" + str(std)
        + "_ne" + str(n_epochs)
        + "_ns" + str(n_steps / batch_size)
        + "_entropy" + str(ent_coef)
    )

    print("Training run name:", training_name)

  
        # ---------------- logging dirs ----------------
    tensorboard_log_dir_source = "TrainedModels/"
    os.makedirs(tensorboard_log_dir_source, exist_ok=True)

  
    eval_log_dir = os.path.join(tensorboard_log_dir_source, training_name)
   
    eval_bank_path = os.path.join(eval_log_dir, "eval_init_states_100.npz")


    # Deterministic evaluation environment with a fixed bank of 100 initial states
    deter_env = InspectionEnv(
        dt=dt,  enable_param_randomisation=enable_param_randomisation,
        enableNoise = enableNoise, 
        enableCBFtunning = enableCBFtunning )
    #     eval_mode=True,
    #     eval_bank_path=eval_bank_path,
    #     eval_bank_size=100,
    #     eval_bank_seed=12345,          # pick any fixed number you want
    #     regenerate_eval_bank=False,    # set True once if you want to overwrite
    # )


 
    
    #Create Parallel Environments
    if num_env > 1:
        train_env = SubprocVecEnv([make_env(stoch_env, rank=i) for i in range(num_env)])
    else:
        train_env = stoch_env


    os.makedirs(eval_log_dir, exist_ok=True)
    if trainON and os.path.exists(tensorboard_log_dir_source + training_name + "_1"):
        shutil.rmtree(tensorboard_log_dir_source + training_name + "_1")

    plot_log_dir = os.path.join("Results/", training_name)
    os.makedirs(plot_log_dir, exist_ok=True)

    # ---------------- define model ----------------
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
        # vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
    )

    if trainLoad:
        try:
            model = PPO.load(
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
            eval_env=deter_env,
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
        del model  # free memory

    # ---------------- load best model for plotting ----------------
    if modelLoad:
        print("Loading best model for plotting...")
        deter_env = InspectionEnv(dt, enable_param_randomisation=enable_param_randomisation, enableNoise=enableNoise, enableCBFtunning=enableCBFtunning )
        model = PPO.load(
            os.path.join(eval_log_dir, "best_model.zip"),
            env=deter_env,
            tensorboard_log=tensorboard_log_dir_source,
        )
    else:
        deter_env = InspectionEnv(dt, enable_param_randomisation=enable_param_randomisation, enableNoise=enableNoise, enableCBFtunning=enableCBFtunning )

    # ------------------------------------------------------------------
    # Deterministic rollout + Fig.11-style trajectory plots
    # ------------------------------------------------------------------
    print("Validating on deterministic JAIS env...")
  
    n_samples = 50  # evaluate across the full bank
    # deter_env.set_eval_index(0)  # start from the first stored state


    TOF_samples = []
    DV_samples  = []
    INSPECTED_PCT_FINAL = []   # final inspected % per episode
    # EP_LEN_S = []              # episode length in seconds (same as TOF, but kept explicit)
    REWARD_EP = []             # optional: total episode reward

    keep_traj = True
    traj_saved = False

    # For Fig. 11 plot (keep first episode)
    positions = None
    inspected_hist = None
    t_hist = None

    for i in range(n_samples):
        print(i)
        obs, _info = deter_env.reset()

        done = False
        truncated = False

        t = 0.0
        dv_total = 0.0
        ep_reward = 0.0

        # -------- per-episode logs --------
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

        # optionally store trajectory for Fig.11 (first episode)
        if keep_traj and (not traj_saved):
            _positions = []
            _inspected_hist = []
            _t_hist = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = deter_env.step(action)
            ep_reward += float(reward)

            # state
            r = np.asarray(deter_env.state[:3], dtype=np.float64)
            v = np.asarray(deter_env.state[3:6], dtype=np.float64)
            
            theta_s = float(deter_env.state[6])
            theta_s_log.append(theta_s)

            # theta_b: angle between boresight (deputy->chief) and sun direction
            sun_dir = deter_env.obs_model.sun_direction(theta_s)
            rnorm = np.linalg.norm(r)
            boresight = -r / max(rnorm, 1e-12)

            cos_tb = np.clip(float(np.dot(boresight, sun_dir / max(np.linalg.norm(sun_dir), 1e-12))), -1.0, 1.0)
            theta_b_log_deg.append(np.degrees(np.arccos(cos_tb)))


            dist = float(np.linalg.norm(r))
            
            # if dist < 50:
            #     print(dist)
            speed = float(np.linalg.norm(v))

            # inspected
            insp = np.asarray(deter_env.inspected, dtype=bool)
            insp_pct = 100.0 * float(np.mean(insp))

            # controls (prefer env logs if present)
            u_safe = np.asarray(getattr(deter_env, "last_u_safe", np.zeros(3)), dtype=np.float64).reshape(3,)
            u_rl   = np.asarray(getattr(deter_env, "last_u_rl", action), dtype=np.float64).reshape(-1)
            if u_rl.size >= 3:
                u_rl = u_rl[:3]
            else:
                u_rl = np.pad(u_rl, (0, 3 - u_rl.size))

            # Δv increment (assumes u_safe is FORCE [N])
            # If your last_u_safe is acceleration [m/s^2], remove "/ deter_env.m".
            dv_inst = (np.linalg.norm(u_safe) / float(deter_env.m)) * float(deter_env.DT)
            dv_total += dv_inst

            # -------- append logs --------
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

            # -------- Fig.11 first-episode storage --------
            if keep_traj and (not traj_saved):
                _positions.append(r.copy())
                _inspected_hist.append(insp.copy())
                _t_hist.append(t)

            t += float(deter_env.DT)

        insp_final = np.asarray(deter_env.inspected, dtype=bool)
        insp_pct_final = 100.0 * float(np.mean(insp_final))

        INSPECTED_PCT_FINAL.append(insp_pct_final)
        REWARD_EP.append(float(ep_reward))

        # episode summaries
        TOF_samples.append(t)
        DV_samples.append(dv_total)

        # save this episode
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
            "seed": int( i),
             "inspected_pct_final": float(insp_pct_final),
            "episode_reward": float(ep_reward),
            "eval_index": int(i),
                "theta_s_hist": np.asarray(theta_s_log, dtype=np.float64),
             "theta_b_deg_hist": np.asarray(theta_b_log_deg, dtype=np.float64),
        }
        savemat(os.path.join(plot_log_dir, f"eval_episode_{i:03d}.mat"), ep_dict)

        # store first trajectory for Fig.11 plotting
        if keep_traj and (not traj_saved):
            positions = np.asarray(_positions)
            inspected_hist = np.asarray(_inspected_hist)
            t_hist = np.asarray(_t_hist)
            traj_saved = True

    TOF_samples = np.asarray(TOF_samples, dtype=np.float64)
    DV_samples  = np.asarray(DV_samples, dtype=np.float64)
    
    INSPECTED_PCT_FINAL = np.asarray(INSPECTED_PCT_FINAL, dtype=np.float64)
    REWARD_EP = np.asarray(REWARD_EP, dtype=np.float64)

    # save summary stats
    savemat(
        os.path.join(plot_log_dir, "resultsInspection_summary.mat"),
        {
            "TOF_samples": TOF_samples,
            "DV_samples": DV_samples,
            "n_samples": np.array([n_samples], dtype=np.int32),
        },
    )
    


    # plot distributions
    # plt.figure()
    # plt.hist(TOF_samples, bins=20)
    # plt.xlabel("Time of flight (s)")
    # plt.ylabel("Count")
    # plt.title(f"TOF distribution (n={n_samples})")
    # tof_png = os.path.join(plot_log_dir, "tof_distribution.png")
    # plt.savefig(tof_png, dpi=300, bbox_inches="tight")
    # plt.show()

    # plt.figure()
    # plt.hist(DV_samples, bins=20)
    # plt.xlabel("Total Δv (m/s)")
    # plt.ylabel("Count")
    # plt.title(f"Total Δv distribution (n={n_samples})")
    # dv_png = os.path.join(plot_log_dir, "dv_distribution.png")
    # plt.savefig(dv_png, dpi=300, bbox_inches="tight")
    # plt.show()

    # print(f"Saved TOF histogram to {tof_png}")
    # print(f"Saved Δv histogram to {dv_png}")
    
    print("TOF: mean ± std =", float(np.mean(TOF_samples)), float(np.std(TOF_samples)))
    print("Δv:  mean ± std =", float(np.mean(DV_samples)), float(np.std(DV_samples)))
    print("Final inspected (%): mean ± std =",
      float(np.mean(INSPECTED_PCT_FINAL)), float(np.std(INSPECTED_PCT_FINAL)))

    print("Episode reward: mean ± std =",
        float(np.mean(REWARD_EP)), float(np.std(REWARD_EP)))
    
    import glob


    def plot_all_constraints(plot_log_dir: str, env: InspectionEnv, fft_stride: int = 50):
        """
        Replicates Fig.-style panels (a)-(f) overlaid for all eval_episode_*.mat files.
        fft_stride controls how often to draw FFT free-flight predictions in panel (e)
        (to avoid an unreadable plot).
        """
        mat_files = sorted(glob.glob(os.path.join(plot_log_dir, "eval_episode_*.mat")))
        if len(mat_files) == 0:
            print("No eval_episode_*.mat found; run evaluation first.")
            return

        r_coll = float(env.R_C + env.R_D)
        R_MAX  = float(env.R_MAX)
        V_MAX  = float(env.V_MAX)

        # paper-style dynamic speed bound
        # (your env uses v0 + nu1 * r)
        dist_grid = np.linspace(0.0, R_MAX, 300)
        # v_bound = env.v0 + env.nu1 * dist_grid

        # keepout threshold used in your CBF: theta_b - 0.5*alpha_fov >= 0
        theta_thresh_deg = float(np.degrees(0.5 * env.alpha_fov))

        # FFT settings (already in env)
        # fft_times = np.arange(0.0, float(env.T_fft) + 1e-9, float(env.dt_fft))

        def cwh_free_motion(p0, v0, t):
            n = float(env.n)
            snt = np.sin(n * t)
            cnt = np.cos(n * t)
            x0, y0, z0 = p0
            vx0, vy0, vz0 = v0
            x = (4.0 - 3.0 * cnt) * x0 + (snt / n) * vx0 + (2.0 / n) * (1.0 - cnt) * vy0
            y = 6.0 * (snt - n * t) * x0 + y0 + (2.0 / n) * (cnt - 1.0) * vx0 + ((4.0 * snt - 3.0 * n * t) / n) * vy0
            z = z0 * cnt + (vz0 / n) * snt
            return np.array([x, y, z], dtype=np.float64)

        fig, axs = plt.subplots(2, 3, figsize=(16, 7))
        ax_a, ax_b, ax_c = axs[0, 0], axs[0, 1], axs[0, 2]
        ax_d, ax_e, ax_f = axs[1, 0], axs[1, 1], axs[1, 2]

        # ---------- overlay all episodes ----------
        for f in mat_files:
            ep = sp.io.loadmat(f)

            t_hist = ep["t_hist"].squeeze()
            dist_hist = ep["dist_hist"].squeeze()
            speed_hist = ep["speed_hist"].squeeze()
            vel = ep["velocities"]  # (T,3)
            vx, vy, vz = vel[:, 0].squeeze(), vel[:, 1].squeeze(), vel[:, 2].squeeze()

            # a) Safe separation: ||p|| vs time (log-y like the paper)
            ax_a.plot (t_hist, dist_hist, linewidth=1.0)

            # b) Dynamic speed constraint: ||v|| vs ||p||
            ax_b.plot(dist_hist, speed_hist, linewidth=1.0)

            # c) Keepout: theta_b vs time (requires theta_b saved)
            if "theta_b_deg_hist" in ep:
                theta_b = ep["theta_b_deg_hist"].squeeze()
                ax_c.plot(t_hist, theta_b, linewidth=1.0)

            # d) Keep-in: ||p|| vs time
            ax_d.plot(t_hist, dist_hist, linewidth=1.0)

            # e) Passively safe manoeuvres: ||p|| vs time + FFT free-flight predictions
            # ax_e.semilogy(t_hist, dist_hist, linewidth=1.0)
            # pos = ep["positions"]   # (T,3)
            # for k in range(0, len(t_hist), max(int(fft_stride), 1)):
            #     p0 = pos[k, :]
            #     v0 = vel[k, :]
            #     norms = np.zeros_like(fft_times)
            #     for ii, tau in enumerate(fft_times):
            #         p_tau = cwh_free_motion(p0, v0, float(tau))
            #         norms[ii] = np.linalg.norm(p_tau)
            #     ax_e.plot(t_hist[k] + fft_times, norms, linewidth=0.5, alpha=0.15)

            # f) Axial velocity limits
            ax_f.plot(t_hist, vx, linewidth=0.8)
            ax_f.plot(t_hist, vy, linewidth=0.8)
            ax_f.plot(t_hist, vz, linewidth=0.8)

        # ---------- decorate axes ----------
        # (a)
        ax_a.axhline(r_coll, linestyle="--", linewidth=2.0)
        ax_a.set_xlabel("Time [s]")
        ax_a.set_ylabel(r"Relative Dist. ($||p||_2$) [m]")
        ax_a.set_title("a) Safe separation")

        # (b)
        # ax_b.plot(dist_grid, v_bound, "k--", linewidth=2.0)
        # ax_b.set_xlabel(r"Relative Dist. ($||p||_2$) [m]")
        # ax_b.set_ylabel(r"Relative Vel. ($||v||_2$) [m/s]")
        # ax_b.set_title("b) Dynamic speed constraint")

        # (c)
        ax_c.axhline(theta_thresh_deg, linestyle="--", linewidth=2.0)
        ax_c.set_xlabel("Time [s]")
        ax_c.set_ylabel(r"Angle to Sun ($\theta_b$) [deg]")
        ax_c.set_title("c) Keepout zone")

        # (d)
        ax_d.axhline(R_MAX, linestyle="--", linewidth=2.0)
        ax_d.set_xlabel("Time [s]")
        ax_d.set_ylabel(r"Relative Dist. ($||p||_2$) [m]")
        ax_d.set_title("d) Keep-in zone")

        # (e)
        ax_e.axhline(r_coll, linestyle="--", linewidth=2.0)
        ax_e.set_xlabel("Time [s]")
        ax_e.set_ylabel(r"Relative Dist. ($||p||_2$) [m]")
        ax_e.set_title("e) Passively safe manoeuvres")

        # (f)
        ax_f.axhline(V_MAX, linestyle="--", linewidth=2.0)
        ax_f.axhline(-V_MAX, linestyle="--", linewidth=2.0)
        ax_f.set_xlabel("Time [s]")
        ax_f.set_ylabel(r"$v$ [m/s]")
        ax_f.set_title("f) Axial velocity limits")

        plt.tight_layout()
        out_png = os.path.join(plot_log_dir, "constraints_all_episodes.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved: {out_png}")

    plot_all_constraints(plot_log_dir, deter_env, fft_stride=50)


    mat_files = sorted(glob.glob(os.path.join(plot_log_dir, "eval_episode_*.mat")))
    if len(mat_files) == 0:
        print("No eval_episode_*.mat files found; skipping overlay plots.")
    else:
        # 1) inspected % over time
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

        # 2) distance over time
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

        # 3) speed over time
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

        # 4) cumulative Δv over time
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

        # 5) controls over time (overlay all episodes)
        # Fx
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

        # Fy
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

        # Fz
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

    
 # -------------- Fig. 11-style plotting WITH VISIBILITY CONE --------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def draw_fov_cone(ax, apex, half_angle, length, color="r", alpha=0.15,
                      n_s=20, n_phi=40):
        """
        Draw a circular cone (FOV) with:

        - apex at `apex` (3D)
        - axis pointing from deputy to chief (i.e. toward origin)
        - half-angle `half_angle` (radians)
        - extending a distance `length` along the axis
        """
        apex = np.asarray(apex, dtype=float).ravel()
        if apex.size != 3:
            return

        # axis: from deputy to chief (origin)
        axis = -apex
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-6:
            return
        axis /= norm_axis

        # build orthonormal basis (u1, u2) ⟂ axis
        # choose a vector not parallel to axis
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(axis, up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        u1 = np.cross(axis, up)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(axis, u1)

        # mesh in (s, φ)
        s_vals = np.linspace(0.0, length, n_s)
        phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi)
        S, Phi = np.meshgrid(s_vals, phi_vals)

        # radius as function of s
        R = S * np.tan(half_angle)

        # cone surface
        X = (apex[0]
             + axis[0] * S
             + R * (np.cos(Phi) * u1[0] + np.sin(Phi) * u2[0]))
        Y = (apex[1]
             + axis[1] * S
             + R * (np.cos(Phi) * u1[1] + np.sin(Phi) * u2[1]))
        Z = (apex[2]
             + axis[2] * S
             + R * (np.cos(Phi) * u1[2] + np.sin(Phi) * u2[2]))

        ax.plot_surface(
            X, Y, Z,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=alpha,
            color=color,
        )

        # optional: boresight line to origin
        ax.plot(
            [apex[0], 0.0],
            [apex[1], 0.0],
            [apex[2], 0.0],
            linestyle="--",
            color=color,
            linewidth=1.0,
        )

    def plot_fig11_like(env, pos, inspected_hist, t_hist, save_path=None):
        """
        Fig. 11-style snapshots:

        - chief surface points as stars (blue = uninspected, green = inspected)
        - agent trajectory as a black line
        - start point = red circle, current point = black 'x'
        - sensor visibility cone drawn at the agent position
        - snapshots at t = 0, 1050, 2110, 3180 s (or closest available)
        """
        pts = env.obs_model.surface_points  # (N_POINTS, 3)

        # Desired snapshot times from the paper
        desired_times = np.array([0.0, 1050.0, 2110.0, 3180.0])
        snapshot_ids = []
        for t_target in desired_times:
            idx = int(np.argmin(np.abs(t_hist - t_target)))
            snapshot_ids.append(idx)

        titles = [f"t = {t_hist[k]:.0f} s" for k in snapshot_ids]

        fig = plt.figure(figsize=(9, 9))
        fig.suptitle("Inspection Trajectory (Fig. 11-style)", fontsize=14)

        for i, (k, title) in enumerate(zip(snapshot_ids, titles)):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")

            inspected = inspected_hist[k]

            # chief surface points
            ax.scatter(
                pts[~inspected, 0],
                pts[~inspected, 1],
                pts[~inspected, 2],
                marker="*",
                c="b",
                s=30,
                label="Uninspected" if i == 0 else None,
            )
            ax.scatter(
                pts[inspected, 0],
                pts[inspected, 1],
                pts[inspected, 2],
                marker="*",
                c="g",
                s=30,
                label="Inspected" if i == 0 else None,
            )

            # agent trajectory up to this time
            ax.plot(
                pos[:k + 1, 0],
                pos[:k + 1, 1],
                pos[:k + 1, 2],
                "k-",
                linewidth=1.5,
            )

            # start and current positions
            ax.scatter(
                pos[0, 0], pos[0, 1], pos[0, 2],
                c="r", marker="o", s=40,
                label="Start" if i == 0 else None,
            )
            ax.scatter(
                pos[k, 0], pos[k, 1], pos[k, 2],
                c="k", marker="x", s=40,
                label="Agent" if i == 0 else None,
            )

            # --- visibility cone at this snapshot ---
            apex = pos[k]
            # use distance to chief as cone length (so it intersects the sphere)
            length = np.linalg.norm(apex)
            if length > 1e-3:
                draw_fov_cone(
                    ax,
                    apex=apex,
                    half_angle=env.alpha_fov,   # this is γ_FOV/2 in the paper
                    length=length * 1.05,       # small overshoot so it passes the origin
                    color="r",
                    alpha=0.15,
                )

            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")
            ax.set_title(title)

            # make it roughly cube-shaped and nice to look at
            r_lim = max(env.R_C * 1.5, np.max(np.linalg.norm(pos, axis=1)) * 1.1)
            ax.set_xlim([-r_lim, r_lim])
            ax.set_ylim([-r_lim, r_lim])
            ax.set_zlim([-r_lim, r_lim])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=25, azim=45)

            if i == 0:
                ax.legend(loc="upper left", fontsize=8)

        plt.subplots_adjust(
            left=0.05, right=0.95,
            top=0.90, bottom=0.05,
            wspace=0.15, hspace=0.20,
        )

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        
               # Fig. 11 plot (your existing function)
        if keep_traj and traj_saved:
            fig11_path = os.path.join(plot_log_dir, "trajectory_fig11_cone.png")
            plot_fig11_like(deter_env, positions, inspected_hist, t_hist, save_path=fig11_path)
            print(f"Saved Fig. 11-style trajectory plot with FOV cone to {fig11_path}")


    # fig11_path = os.path.join(plot_log_dir, "trajectory_fig11_cone.png")
    # plot_fig11_like(deter_env, positions, inspected_hist, t_hist, save_path=fig11_path)
    # print(f"Saved Fig. 11-style trajectory plot with FOV cone to {fig11_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_jais_training()
