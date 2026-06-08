# src/metarl_iccbf/cruise_control/evaluation/eval_serial.py

import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm

# PPO and RecurrentPPO are imported lazily inside evaluate_serial()
# to avoid triggering heavy torch/SB3 imports in spawned worker processes.


def _resolve_env_cls(env_type: str):
    if env_type == "iccbf":
        from metarl_iccbf.cruise_control.envs.rlcbf_env import RLCBFcontrol
        return RLCBFcontrol
    elif env_type == "rl_only":
        from metarl_iccbf.cruise_control.envs.rlonly_env import RLCBFcontrol
        return RLCBFcontrol
    else:
        raise ValueError(f"Unknown env_type: {env_type!r}. Expected 'iccbf' or 'rl_only'.")


def evaluate_serial(
    model_path: str,
    *,
    policy_type: Literal["MLP", "RNN", "GRU", "MAMBA"] = "MLP",
    env_type: Literal["iccbf", "rl_only"] = "iccbf",
    algo: Literal["ppo", "sac"] = "ppo",
    dt: float = 0.1,
    TOF: float = 40.0,
    out_mat: Optional[str] = None,
    out_dir: str = "ResultsEval",
    plot: bool = True,
):
    """
    Run deterministic serial rollouts from fixed ICs, save a .mat file,
    and optionally plot a 6-panel summary figure.

    Parameters
    ----------
    model_path : str
        Path to a saved best_model.zip (or without .zip extension).
    policy_type : "MLP" or "RNN"
        Which SB3 class to use for loading.
    env_type : "rlcbf" or "rlonly"
        Which environment to evaluate in.
    dt, TOF : float
        Time step and total time-of-flight.
    out_mat : str or None
        Output .mat filename. Auto-generated if None.
    out_dir : str
        Directory to save outputs.
    plot : bool
        Whether to show matplotlib plots.

    Returns
    -------
    out_path : str
        Path to the saved .mat file.
    """
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    env_cls = _resolve_env_cls(env_type)
    deter_env = env_cls(dt=dt, deterministic=True)

    if algo == "sac":
        from metarl_iccbf.recurrent_cleanrl.sac import RecurrentSAC
        model = RecurrentSAC.load(model_path, device="cpu")
    elif policy_type == "MAMBA":
        from metarl_iccbf.recurrent_cleanrl.ppo import Mamba2PPO
        load_cls = Mamba2PPO
        model = load_cls.load(model_path, env=deter_env)
    elif policy_type == "GRU":
        from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO as CleanRLRecurrentPPO
        load_cls = CleanRLRecurrentPPO
        model = load_cls.load(model_path, env=deter_env)
    elif policy_type == "RNN":
        load_cls = RecurrentPPO
        model = load_cls.load(model_path, env=deter_env)
    else:
        load_cls = PPO
        model = load_cls.load(model_path, env=deter_env)

    lenvec = int(TOF / dt)
    tvec_full = np.arange(0, TOF, dt)

    points = deter_env.getValidPoints()
    nSamples = points.shape[0]
    print(f"Evaluating {nSamples} trajectories (serial)")

    # Data buffers
    uOpts = np.zeros((nSamples, lenvec, 1))
    observations = np.zeros((nSamples, lenvec, 2))
    uOptmag = np.zeros((nSamples, lenvec))
    uTotal = np.zeros((nSamples, 1))
    actionStore = np.zeros((nSamples, lenvec, 4))
    hs = np.zeros((nSamples, lenvec, 1))
    Vs = np.zeros((nSamples, lenvec, 1))
    comptimes = np.zeros((nSamples, lenvec))

    for i in tqdm(range(nSamples), desc="Evaluating trajectories", unit="traj"):
        t = 0.0
        step = 0
        episode_start = True

        obs, _ = deter_env.reset(postProcess=False)
        deter_env.x0 = points[i, :]
        obs = deter_env.scaleObservation(deter_env.x0)

        while t <= TOF and step < lenvec:
            step += 1

            action, _ = model.predict(obs, deterministic=True, episode_start=episode_start)
            episode_start = False  # Only first step of episode resets state
            t0 = time.perf_counter()
            obs, reward_temp, done, _, _ = deter_env.step(action)
            comptimes[i, step - 1] = time.perf_counter() - t0

            observations[i, step - 1, :] = deter_env.x0
            actionStore[i, step - 1, :] = action
            uOpts[i, step - 1, :] = deter_env.u
            uOptmag[i, step - 1] = np.linalg.norm(deter_env.u)
            hs[i, step - 1, :] = deter_env.x0[0] - 1.8 * deter_env.x0[1]
            Vs[i, step - 1, :] = (deter_env.x0[1] - deter_env.vmax) ** 2

            t += dt

        uTotal[i] = np.sum(uOptmag[i, :step]) * dt

    print(f"Completed {nSamples} trajectories")
    print("Mean uTotal:", np.mean(uTotal[1:]))
    q1, q2, q3 = np.percentile(uTotal[1:], [25, 50, 75])
    print(f"Q1={q1:.4f}  Q2={q2:.4f}  Q3={q3:.4f}")

    # Save .mat
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if out_mat is None:
        from datetime import datetime
        model_name = Path(model_path).resolve().parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_mat = f"eval_{model_name}_serial_{timestamp}.mat"
    out_path = str(Path(out_dir) / out_mat)

    savemat(out_path, {
        "uOpts": uOpts,
        "states": observations,
        "hs": hs,
        "Vs": Vs,
        "actionStore": actionStore,
        "uTotal": uTotal,
        "comptimes": comptimes,
        "tvec_full": tvec_full,
        "model_path": np.array([model_path], dtype=object),
    })
    print("Saved:", out_path)

    if plot:
        _plot_summary(observations, uOpts, Vs, hs, tvec_full, nSamples)

    return out_path


def _plot_summary(observations, uOpts, Vs, hs, tvec_full, nSamples):
    """6-panel summary figure matching the original mainRLonly/mainnoRL plots."""
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, nSamples))

    # x(t) vs v(t)
    for i in range(nSamples):
        axs[0, 0].plot(observations[i, :, 0], observations[i, :, 1], color=colors[i], alpha=0.8)
    v_line = np.linspace(0, 20, 300)
    axs[0, 0].plot(1.8 * v_line, v_line, 'r--', label='$x = 1.8 v$')
    axs[0, 0].set(title='$x(t)$ vs $v(t)$', xlabel='$x(t)$ (m)', ylabel='$v(t)$ (m/s)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # u(t) vs time
    for i in range(nSamples):
        axs[0, 1].plot(tvec_full, uOpts[i, :, 0], color=colors[i], alpha=0.8)
    axs[0, 1].set(title='$u(N)$ vs Time', xlabel='Time (s)', ylabel='$u(N)$')
    axs[0, 1].grid(True)

    # V(t) vs time
    for i in range(nSamples):
        axs[1, 0].plot(tvec_full, Vs[i, :, 0], color=colors[i], alpha=0.8)
    axs[1, 0].set(title='$V(t)$ vs Time', xlabel='Time (s)', ylabel='$V(t)$')
    axs[1, 0].grid(True)

    # h(t) vs time
    for i in range(nSamples):
        axs[1, 1].plot(tvec_full, hs[i, :, 0], color=colors[i], alpha=0.8)
    axs[1, 1].set(title='$h(t)$ original vs Time', xlabel='Time (s)', ylabel='$h(t)$')
    axs[1, 1].grid(True)

    # Velocity over time
    for i in range(nSamples):
        axs[2, 0].plot(tvec_full, observations[i, :, 1], color=colors[i], alpha=0.8)
    axs[2, 0].set(title='Velocity magnitude over time', xlabel='Time (s)', ylabel='velocity (m/s)')
    axs[2, 0].grid(True)

    # Position over time
    for i in range(nSamples):
        axs[2, 1].plot(tvec_full, observations[i, :, 0], color=colors[i], alpha=0.8)
    axs[2, 1].set(title='Position magnitude over time', xlabel='Time (s)', ylabel='Position (m)')
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.show()
