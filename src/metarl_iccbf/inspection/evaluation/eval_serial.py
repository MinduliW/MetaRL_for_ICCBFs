"""Serial evaluation for inspection policies.

Runs deterministic rollouts and collects inspection-specific metrics:
positions, velocities, inspected %, sun/boresight angles, fuel usage, etc.

Usage::

    from metarl_iccbf.inspection.evaluation.eval_serial import evaluate_serial

    out_path = evaluate_serial(
        "outputs/inspection/run_001/best_model.zip",
        policy_type="RNN",
        env_type="iccbf",
        n_episodes=10,
    )
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

# PPO and RecurrentPPO are imported lazily inside evaluate_serial()


def evaluate_serial(
    model_path: str,
    *,
    policy_type: Literal["MLP", "RNN", "GRU", "LSTM", "MAMBA"] = "MLP",
    algo: Literal["ppo", "sac"] = "ppo",
    env_type: Literal["iccbf", "rl_only"] = "iccbf",
    dt: float = 10.0,
    n_episodes: int = 10,
    enable_param_randomisation: bool = True,
    enableNoise: bool = False,
    dvWeight: float = 10.0,
    out_mat: Optional[str] = None,
    out_dir: str = "ResultsEval/inspection",
    plot: bool = True,
) -> str:
    """Run serial deterministic evaluation of an inspection policy.

    Parameters
    ----------
    model_path : str
        Path to the saved model (``.zip``).
    policy_type : "MLP", "RNN", or "MAMBA"
        Policy type used during training.
    env_type : "iccbf" or "rl_only"
        ``"iccbf"`` enables CBF tuning (12D action),
        ``"rl_only"`` for thrust-only (3D action).
    dt : float
        Environment timestep in seconds.
    n_episodes : int
        Number of evaluation episodes.
    enable_param_randomisation : bool
        Whether to randomise physical parameters each episode.
    enableNoise : bool
        Whether to enable actuation/measurement noise.
    dvWeight : float
        Fuel cost weight in reward.
    out_mat : str or None
        Output ``.mat`` filename. Auto-generated if None.
    out_dir : str
        Output directory for results.
    plot : bool
        Whether to generate summary plots.

    Returns
    -------
    str
        Path to the saved summary ``.mat`` file.
    """
    from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv

    enableCBFtunning = (env_type == "iccbf")

    env = InspectionEnv(
        dt=dt,
        enable_param_randomisation=enable_param_randomisation,
        enableNoise=enableNoise,
        enableCBFtunning=enableCBFtunning,
        dvWeight=dvWeight,
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    from stable_baselines3 import PPO
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        RecurrentPPO = None
    if algo == "sac":
        from metarl_iccbf.recurrent_cleanrl.sac import RecurrentSAC
        model = RecurrentSAC.load(model_path, device="cuda")
    elif policy_type == "MAMBA":
        from metarl_iccbf.recurrent_cleanrl.ppo import Mamba2PPO
        model = Mamba2PPO.load(model_path, device="cuda")
    elif policy_type in ("GRU", "LSTM"):
        from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO as CustomRNNPPO
        model = CustomRNNPPO.load(model_path, device="cuda")
    elif policy_type == "RNN":
        model = RecurrentPPO.load(model_path, device="cpu")
    else:
        model = PPO.load(model_path, device="cpu")

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    if out_mat is None:
        out_mat = f"inspection_eval_{policy_type}_{env_type}.mat"

    # --------------------------------------------------
    # Rollout collection
    # --------------------------------------------------
    TOF_samples = []
    DV_samples = []
    INSPECTED_PCT_FINAL = []
    REWARD_EP = []

    for ep_idx in tqdm(range(n_episodes), desc="Evaluating episodes", unit="ep"):
        obs, _info = env.reset()

        done = False
        truncated = False

        # Recurrent state management
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        episode_start = True

        t = 0.0
        dv_total = 0.0
        ep_reward = 0.0

        # Per-step logs
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
        comptime_log = []

        while not (done or truncated):
            # Policy-specific predict
            if algo == "sac" or policy_type in ("MAMBA", "GRU", "LSTM"):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    episode_start=episode_start,
                )
                episode_start = False
            elif policy_type == "RNN":
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                episode_starts = np.array([False], dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)

            t0 = time.perf_counter()
            obs, reward, done, truncated, info = env.step(action)
            comptime_log.append(time.perf_counter() - t0)

            ep_reward += float(reward)

            # Extract state
            r = np.asarray(env.state[:3], dtype=np.float64)
            v = np.asarray(env.state[3:6], dtype=np.float64)
            theta_s = float(env.state[6])

            # Boresight-to-sun angle
            sun_dir = env.obs_model.sun_direction(theta_s)
            rnorm = np.linalg.norm(r)
            boresight = -r / max(rnorm, 1e-12)
            cos_tb = np.clip(
                float(np.dot(boresight, sun_dir / max(np.linalg.norm(sun_dir), 1e-12))),
                -1.0, 1.0,
            )
            theta_b_deg = float(np.degrees(np.arccos(cos_tb)))

            dist = float(rnorm)
            speed = float(np.linalg.norm(v))

            insp = np.asarray(env.inspected, dtype=bool)
            insp_pct = 100.0 * float(np.mean(insp))

            u_safe = np.asarray(
                getattr(env, "last_u_safe", np.zeros(3)), dtype=np.float64
            ).reshape(3)
            u_rl = np.asarray(
                getattr(env, "last_u_rl", action), dtype=np.float64
            ).reshape(-1)
            if u_rl.size >= 3:
                u_rl = u_rl[:3]
            else:
                u_rl = np.pad(u_rl, (0, 3 - u_rl.size))

            dv_inst = (np.linalg.norm(u_safe) / float(env.m)) * float(env.DT)
            dv_total += dv_inst

            # Append logs
            t_log.append(t)
            inspected_pct_log.append(insp_pct)
            dist_log.append(dist)
            speed_log.append(speed)
            u_safe_log.append(u_safe.copy())
            u_rl_log.append(u_rl.copy())
            theta_s_log.append(theta_s)
            theta_b_log_deg.append(theta_b_deg)
            dv_inst_log.append(float(dv_inst))
            dv_cum_log.append(float(dv_total))
            pos_log.append(r.copy())
            vel_log.append(v.copy())
            inspected_log.append(insp.copy())

            t += float(env.DT)

        insp_final = np.asarray(env.inspected, dtype=bool)
        insp_pct_final = 100.0 * float(np.mean(insp_final))

        INSPECTED_PCT_FINAL.append(insp_pct_final)
        REWARD_EP.append(float(ep_reward))
        TOF_samples.append(t)
        DV_samples.append(dv_total)

        # Save per-episode .mat
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
            "theta_s_hist": np.asarray(theta_s_log, dtype=np.float64),
            "theta_b_deg_hist": np.asarray(theta_b_log_deg, dtype=np.float64),
            "comptimes": np.asarray(comptime_log, dtype=np.float64),
            "TOF": float(t),
            "DV_total": float(dv_total),
            "inspected_pct_final": float(insp_pct_final),
            "episode_reward": float(ep_reward),
            "eval_index": int(ep_idx),
        }
        savemat(os.path.join(out_dir, f"eval_episode_{ep_idx:03d}.mat"), ep_dict)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    TOF_samples = np.asarray(TOF_samples, dtype=np.float64)
    DV_samples = np.asarray(DV_samples, dtype=np.float64)
    INSPECTED_PCT_FINAL = np.asarray(INSPECTED_PCT_FINAL, dtype=np.float64)
    REWARD_EP = np.asarray(REWARD_EP, dtype=np.float64)

    summary_path = os.path.join(out_dir, out_mat)
    savemat(summary_path, {
        "TOF_samples": TOF_samples,
        "DV_samples": DV_samples,
        "INSPECTED_PCT_FINAL": INSPECTED_PCT_FINAL,
        "REWARD_EP": REWARD_EP,
        "n_episodes": np.array([n_episodes], dtype=np.int32),
        "model_path": np.array([model_path], dtype=object),
        "policy_type": np.array([policy_type], dtype=object),
        "env_type": np.array([env_type], dtype=object),
    })

    print(f"TOF:  mean +/- std = {float(np.mean(TOF_samples)):.1f} +/- {float(np.std(TOF_samples)):.1f}")
    print(f"dv:   mean +/- std = {float(np.mean(DV_samples)):.3f} +/- {float(np.std(DV_samples)):.3f}")
    print(f"Insp: mean +/- std = {float(np.mean(INSPECTED_PCT_FINAL)):.1f}% +/- {float(np.std(INSPECTED_PCT_FINAL)):.1f}%")
    print(f"Rew:  mean +/- std = {float(np.mean(REWARD_EP)):.2f} +/- {float(np.std(REWARD_EP)):.2f}")
    print(f"Saved: {summary_path}")

    return summary_path
