"""Serial evaluation of docking policies (MLP, RNN, MAMBA).

Runs deterministic rollouts from fixed initial conditions and saves results
to a .mat file.
"""

import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm

# PPO and RecurrentPPO are imported lazily inside evaluate_serial()


def evaluate_serial(
    model_path: str,
    *,
    policy_type: Literal["MLP", "RNN", "GRU", "MAMBA"] = "MLP",
    algo: Literal["ppo", "sac"] = "ppo",
    dt: float = 0.5,
    TOF: float = 50.0,
    out_mat: Optional[str] = None,
    out_dir: str = "ResultsEval",
    plot: bool = True,
    init_da: bool = True,
    da_order: int = 4,
    da_vars: int = 2,
):
    """Run deterministic serial rollouts from fixed ICs, save a .mat file.

    Parameters
    ----------
    model_path : str
        Path to a saved model (best_model.zip).
    policy_type : "MLP", "RNN", or "MAMBA"
        Which model class to use for loading.
    dt, TOF : float
        Time step and total time-of-flight.
    out_mat : str or None
        Output .mat filename. Auto-generated if None.
    out_dir : str
        Directory to save outputs.
    plot : bool
        Whether to show matplotlib plots.
    init_da : bool
        Whether to call DA.init() before creating the environment.

    Returns
    -------
    out_path : str
        Path to the saved .mat file.
    """
    # DA initialization (must happen before env construction)
    if init_da:
        try:
            from daceypy import DA
            DA.init(int(da_order), int(da_vars))
        except Exception:
            pass

    from metarl_iccbf.docking.RLCBF import RLCBFcontrol
    deter_env = RLCBFcontrol(dt=dt, deterministic=True)

    # Load model
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
        model = Mamba2PPO.load(model_path)
    elif policy_type == "GRU":
        from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO as GRUPPO
        model = GRUPPO.load(model_path, device="cuda")
    elif policy_type == "RNN":
        model = RecurrentPPO.load(model_path, device="cpu")
    else:
        model = PPO.load(model_path, device="cpu")

    lenvec = int(TOF / dt)
    tvec_full = np.arange(0, TOF, dt)

    points = deter_env.getValidPoints()
    nSamples = points.shape[0]
    print(f"Evaluating {nSamples} trajectories (serial)")

    state_dim = 5
    action_dim = 4

    # Data buffers
    states = np.zeros((nSamples, lenvec, state_dim))
    actions = np.zeros((nSamples, lenvec, action_dim))
    uOptmag = np.zeros((nSamples, lenvec))
    uTotal = np.zeros((nSamples, 1))
    hs = np.zeros((nSamples, lenvec, 1))
    comptimes = np.zeros((nSamples, lenvec))

    lstm_states = None

    for i in tqdm(range(nSamples), desc="Evaluating trajectories", unit="traj"):
        step = 0
        episode_start = True

        obs, _ = deter_env.reset(postProcess=False)
        deter_env.x0 = points[i, :]
        obs = deter_env.scaleObservation(deter_env.x0)

        # Reset recurrent state per episode
        if policy_type == "RNN" and algo != "sac":
            lstm_states = None

        for step in range(lenvec):
            if algo == "sac" or policy_type in ("MAMBA", "GRU"):
                action, _ = model.predict(
                    obs, deterministic=True, episode_start=episode_start
                )
            elif policy_type == "RNN":
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            episode_start = False

            t0 = time.perf_counter()
            obs, reward, done, truncated, info = deter_env.step(action)
            comptimes[i, step] = time.perf_counter() - t0

            states[i, step, :] = deter_env.x0
            actions[i, step, :] = action
            uOptmag[i, step] = np.linalg.norm(deter_env.control)
            hs[i, step, 0] = deter_env.dockingCase.originalh(deter_env.x0)

            if done or truncated:
                break

        uTotal[i] = np.sum(uOptmag[i, : step + 1]) * dt

    print(f"Completed {nSamples} trajectories")
    print("Mean uTotal:", np.mean(uTotal))
    q1, q2, q3 = np.percentile(uTotal, [25, 50, 75])
    print(f"Q1={q1:.6f}  Q2={q2:.6f}  Q3={q3:.6f}")

    # Save .mat
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if out_mat is None:
        out_mat = f"docking_eval_{policy_type.lower()}_serial.mat"
    out_path = str(Path(out_dir) / out_mat)

    savemat(out_path, {
        "states": states,
        "actions": actions,
        "uOptmag": uOptmag,
        "uTotal": uTotal,
        "hs": hs,
        "comptimes": comptimes,
        "tvec_full": tvec_full,
    })
    print("Saved:", out_path)

    if plot:
        _plot_summary(states, uOptmag, hs, tvec_full, nSamples, deter_env)

    return out_path


def _plot_summary(states, uOptmag, hs, tvec_full, nSamples, env):
    """6-panel summary figure for docking trajectories."""
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, nSamples))

    # px vs py (2D trajectory with docking cone)
    for i in range(nSamples):
        axs[0, 0].plot(
            states[i, :, 0] * 1e3,
            states[i, :, 1] * 1e3,
            color=colors[i],
            alpha=0.6,
        )
    axs[0, 0].set(
        title="Trajectory (px vs py)",
        xlabel="px (m)",
        ylabel="py (m)",
    )
    axs[0, 0].grid(True)

    # Control magnitude vs time
    for i in range(nSamples):
        axs[0, 1].plot(tvec_full, uOptmag[i, :], color=colors[i], alpha=0.6)
    axs[0, 1].set(
        title="Control magnitude vs Time",
        xlabel="Time (s)",
        ylabel="|u| (km/s²)",
    )
    axs[0, 1].grid(True)

    # CBF h(t) vs time
    for i in range(nSamples):
        axs[1, 0].plot(tvec_full, hs[i, :, 0], color=colors[i], alpha=0.6)
    axs[1, 0].axhline(0, color="r", linestyle="--", label="h=0")
    axs[1, 0].set(
        title="CBF h(t) vs Time",
        xlabel="Time (s)",
        ylabel="h(t)",
    )
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Position x vs time
    for i in range(nSamples):
        axs[1, 1].plot(
            tvec_full, states[i, :, 0] * 1e3, color=colors[i], alpha=0.6
        )
    axs[1, 1].set(
        title="Position px vs Time",
        xlabel="Time (s)",
        ylabel="px (m)",
    )
    axs[1, 1].grid(True)

    # Velocity vx vs time
    for i in range(nSamples):
        axs[2, 0].plot(
            tvec_full, states[i, :, 2] * 1e3, color=colors[i], alpha=0.6
        )
    axs[2, 0].set(
        title="Velocity vx vs Time",
        xlabel="Time (s)",
        ylabel="vx (m/s)",
    )
    axs[2, 0].grid(True)

    # Phase angle psi vs time
    for i in range(nSamples):
        axs[2, 1].plot(
            tvec_full,
            np.degrees(states[i, :, 4]),
            color=colors[i],
            alpha=0.6,
        )
    axs[2, 1].set(
        title="Phase angle psi vs Time",
        xlabel="Time (s)",
        ylabel="psi (deg)",
    )
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.show()
