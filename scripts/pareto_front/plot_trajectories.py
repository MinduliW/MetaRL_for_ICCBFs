"""Plot trajectories, inspection score, fuel, and safety margin for a single
starting condition across a sweep of preference weights (fuel vs safety tradeoff).

Weight structure (same as run_pareto.py):
  w_coverage = 0.6  (fixed)
  w_fuel     = t * 0.4
  w_safety   = (1 - t) * 0.4
  t sweeps [0, 1]

Usage:
  python plot_trajectories.py
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO
from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT      = "/home/aposadasn/MetaRL_for_ICCBFs/outputs/inspection/Mamba2TunedICCBF_Inspection_20260512_141238/final_model.zip"
BANK      = "/home/aposadasn/MetaRL_for_ICCBFs/outputs/data/inspection/inspection_episode_bank.npz"
EPISODE_IDX = 0     # set to int to pin a specific IC, or None to auto-find a fair one
N_WEIGHTS   = 20        # number of weight points to sweep
MAX_SCAN    = 0       # how many bank episodes to scan when auto-finding
W_COVERAGE  = 0.6
DEVICE      = "cuda"
OUT_FILE    = f"/home/aposadasn/MetaRL_for_ICCBFs/outputs/figures/pareto/trajectory_weight_sweep_episode_{EPISODE_IDX}.png"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
bank  = np.load(BANK, allow_pickle=True)
model = RecurrentPPO.load(CKPT, device=DEVICE)
model.agent.eval()

env = InspectionEnv(dt=10.0, morl=True, enableCBFtunning=True, enable_param_randomisation=False)

def apply_bank_episode(idx):
    """Load env params and return x0 for a given bank index."""
    meta = dict(
        m=float(bank["m_vec"][idx]),     R_D=float(bank["R_D_vec"][idx]),
        R_C=float(bank["R_C_vec"][idx]), U_MAX=float(bank["U_MAX_vec"][idx]),
        R_MAX=float(bank["R_MAX_vec"][idx]), r=float(bank["r_vec"][idx]),
    )
    for key, val in meta.items():
        if hasattr(env, key):
            setattr(env, key, val)
    if hasattr(env, "iccbf_da") and env.iccbf_da is not None:
        env.iccbf_da.m       = meta["m"]
        env.iccbf_da.rho_koz = float(meta["R_C"] + meta["R_D"])
        env.iccbf_da.rho_kiz = float(meta["R_MAX"])
    if hasattr(env, "_recompute_derived"): env._recompute_derived()
    if hasattr(env, "_rebuild_models"):    env._rebuild_models()
    return np.asarray(bank["x0s"][idx], dtype=np.float64)


def run_episode(w_np, x0):
    """Run one episode from x0 with weight vector w_np, return trajectory data."""
    # Reset to fixed IC
    env.state      = x0.copy()
    env.steps_done = 0
    env.inspected  = np.zeros(env.N_POINTS, dtype=bool)
    x_meas = env._noisy_measurement_state(env.state, env.np_random) \
        if hasattr(env, "_noisy_measurement_state") else env.state
    obs_dict = env.obs_model.get_observation(x_meas)
    env.inspected |= obs_dict["visible_mask"]
    for attr in ("last_cbfs", "last_u_rl", "last_u_safe"):
        if hasattr(env, attr):
            getattr(env, attr)[:] = 0.0
    obs = env._get_obs_vector() if hasattr(env, "_get_obs_vector") else np.asarray(x_meas, np.float32)

    w_t = torch.as_tensor(w_np[None], dtype=torch.float32, device=DEVICE)
    actor_states, critic_states = model.agent.initial_states(1, DEVICE)
    ep_starts = np.ones(1, dtype=np.float32)
    done = False

    xs, ys, zs           = [x0[0]], [x0[1]], [x0[2]]
    coverage_frac        = [float(env.inspected.sum()) / env.N_POINTS]
    cumulative_fuel      = [0.0]
    min_h_over_time      = []
    escaped_kiz          = False

    while not done:
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=DEVICE)
        ep_t  = torch.as_tensor(ep_starts, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            action, actor_states, critic_states = model.agent.deterministic_step(
                obs_t, actor_states, critic_states, ep_t, w=w_t,
            )
        action_np = np.clip(action.squeeze(0).cpu().numpy(),
                            env.action_space.low, env.action_space.high)
        obs, r_vec, terminated, truncated, info = env.step(action_np)
        done = bool(terminated or truncated)
        if np.linalg.norm(env.state[:3]) > env.R_MAX:
            escaped_kiz = True
        if done and (len(xs) < env.MAX_STEPS):
            cause = info.get("termination_reason", "terminated" if terminated else "truncated")
            print(f"    early exit at step {len(xs)}: {cause} | pos=({env.state[0]:.1f},{env.state[1]:.1f},{env.state[2]:.1f}) | r={np.linalg.norm(env.state[:3]):.1f}m", flush=True)
        ep_starts = np.ones(1, dtype=np.float32) if done else np.zeros(1, dtype=np.float32)

        xs.append(env.state[0])
        ys.append(env.state[1])
        zs.append(env.state[2])
        coverage_frac.append(float(env.inspected.sum()) / env.N_POINTS)
        cumulative_fuel.append(cumulative_fuel[-1] + float(-r_vec[1]))  # r_vec[1] = -fuel_cost_norm
        min_h_over_time.append(float(r_vec[2]))

    # Pad min_h to match length (first step has no CBF eval)
    min_h_over_time = [min_h_over_time[0]] + min_h_over_time

    final_coverage = float(env.inspected.sum()) / env.N_POINTS
    success = final_coverage >= env.morl_coverage_threshold

    return dict(
        xs=np.array(xs), ys=np.array(ys), zs=np.array(zs),
        coverage=np.array(coverage_frac),
        fuel=np.array(cumulative_fuel),
        min_h=np.array(min_h_over_time),
        final_coverage=final_coverage,
        success=success,
        n_steps=len(xs) - 1,
        inspected_mask=env.inspected.copy(),
        escaped_kiz=escaped_kiz,
    )


# ---------------------------------------------------------------------------
# Find a fair IC where all extreme weights survive full episode
# ---------------------------------------------------------------------------
t_vals  = np.linspace(0.0, 1.0, N_WEIGHTS)
weights = np.stack([
    np.full(N_WEIGHTS, W_COVERAGE),
    t_vals * (1.0 - W_COVERAGE),
    (1.0 - t_vals) * (1.0 - W_COVERAGE),
], axis=-1).astype(np.float32)  # (N_WEIGHTS, 3)

w_extreme = weights[[0, -1]]  # fuel-only and safety-only

if EPISODE_IDX is not None:
    chosen_idx = EPISODE_IDX
    x0 = apply_bank_episode(chosen_idx)
    print(f"Using pinned EPISODE_IDX={chosen_idx}")
else:
    print(f"Scanning up to {MAX_SCAN} bank episodes for a fair IC...")
    chosen_idx = None
    for scan_i in range(min(MAX_SCAN, len(bank["x0s"]))):
        x0 = apply_bank_episode(scan_i)
        all_survive = all(
            run_episode(w, x0)["n_steps"] >= env.MAX_STEPS - 1
            for w in w_extreme
        )
        status = "OK" if all_survive else "skip"
        print(f"  episode {scan_i}: {status} | x0=[{x0[0]:.1f},{x0[1]:.1f},{x0[2]:.1f}]", flush=True)
        if all_survive:
            chosen_idx = scan_i
            break
    if chosen_idx is None:
        print("WARNING: no fully-surviving IC found in scan; using episode 0")
        chosen_idx = 0
        x0 = apply_bank_episode(0)
    print(f"→ Selected episode {chosen_idx}\n")

# ---------------------------------------------------------------------------
# Full weight sweep on chosen IC
# ---------------------------------------------------------------------------
print(f"Running {N_WEIGHTS} weight sweep on episode {chosen_idx}...")
trajs = []
for wi, w in enumerate(weights):
    print(f"  [{wi+1}/{N_WEIGHTS}] w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]", flush=True)
    trajs.append(run_episode(w, x0))

print("\n=== Episode Summary ===")
for wi, (traj, t) in enumerate(zip(trajs, t_vals)):
    status = "✓ SUCCESS" if traj["success"] else "✗ TIMEOUT"
    escaped = " (ESCAPED KIZ)" if traj["escaped_kiz"] else ""
    print(f"  t={t:.2f}: {status} | coverage={traj['final_coverage']:.1%} | steps={traj['n_steps']}{escaped}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
ONLY_FULL_COVERAGE = False   # set False to show all trajectories

cmap   = cm.get_cmap("coolwarm")
colors = [cmap(t) for t in t_vals]

if ONLY_FULL_COVERAGE:
    mask       = [traj["final_coverage"] >= 1.0 for traj in trajs]
    trajs_plot  = [traj  for traj, m in zip(trajs,  mask) if m]
    colors_plot = [color for color, m in zip(colors, mask) if m]
    t_vals_plot = [t     for t,     m in zip(t_vals, mask) if m]
    print(f"\nPlotting {len(trajs_plot)}/{len(trajs)} trajectories with 100% coverage.")
else:
    trajs_plot, colors_plot, t_vals_plot = trajs, colors, list(t_vals)

# === Figure 1: X-Y trajectory ===
fig1, ax1 = plt.subplots(figsize=(10, 10))
fig1.suptitle(
    f"Episode {chosen_idx} — X-Y Trajectory\n"
    f"x₀ = [{x0[0]:.1f}, {x0[1]:.1f}, {x0[2]:.1f}] m",
    fontsize=12,
)

target_pts_xy = env.obs_model.surface_points[:, :2]  # (N_POINTS, 2)
n_plot = len(trajs_plot)
for wi, (traj, color, t) in enumerate(zip(trajs_plot, colors_plot, t_vals_plot)):
    ax1.plot(traj["xs"], traj["ys"], color=color, alpha=0.8, linewidth=1.5)
    marker = "o" if traj["success"] else "x"
    ax1.scatter([traj["xs"][-1]], [traj["ys"][-1]], color=color, s=50, marker=marker, zorder=4)

# Inspected targets (use first plotted trajectory)
ref_traj = trajs_plot[0] if trajs_plot else trajs[0]
inspected_pts = target_pts_xy[ref_traj["inspected_mask"]]
not_inspected_pts = target_pts_xy[~ref_traj["inspected_mask"]]
ax1.scatter(inspected_pts[:, 0], inspected_pts[:, 1], color="green", s=25, alpha=0.6, marker=".")
ax1.scatter(not_inspected_pts[:, 0], not_inspected_pts[:, 1], color="red", s=25, alpha=0.4, marker="x")

ax1.scatter([x0[0]], [x0[1]], color="black", s=120, zorder=5, marker="*")
# Draw keep-out zone and keep-in boundary
theta = np.linspace(0, 2 * np.pi, 200)
ax1.plot(env.R_C * np.cos(theta), env.R_C * np.sin(theta), "k--", linewidth=1.5, alpha=0.5)
ax1.plot(env.R_MAX * np.cos(theta), env.R_MAX * np.sin(theta), "k:", linewidth=1.5, alpha=0.5)
ax1.set_xlabel("x (m)", fontsize=20)
ax1.set_ylabel("y (m)", fontsize=20)
ax1.set_title("")
ax1.set_aspect("equal")

# Build legend with only the 3 desired entries
start_handle = mlines.Line2D([], [], color="black", marker="*", linestyle="None", markersize=10, label="Start")
endpoint_handle = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label="Endpoint")
inspected_handle = mlines.Line2D([], [], color="green", marker=".", linestyle="None", markersize=8, label="Inspected target")
ax1.legend(handles=[start_handle, endpoint_handle, inspected_handle], fontsize=9, loc="best")
ax1.grid(True, alpha=0.3)

# Colorbar for trajectory figure
t_range = t_vals_plot if t_vals_plot else list(t_vals)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(min(t_range), max(t_range)))
sm.set_array([])
cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.8, pad=0.02)
cbar1.set_label("t   (← Safety-focused | Fuel-focused →)", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_FILE.replace(".png", "_trajectory.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE.replace('.png', '_trajectory.png')}")
plt.close()

# === Figure 2: Time series (3 subplots) ===
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))

# Subplot 1: Inspection coverage
ax = axes2[0]
for wi, (traj, color) in enumerate(zip(trajs_plot, colors_plot)):
    steps = np.arange(len(traj["coverage"]))
    ax.plot(steps * env.DT, traj["coverage"], color=color, alpha=0.85, linewidth=1.5)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("Coverage (%)", fontsize=10)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Subplot 2: Cumulative fuel
ax = axes2[1]
for wi, (traj, color) in enumerate(zip(trajs_plot, colors_plot)):
    steps = np.arange(len(traj["fuel"]))
    ax.plot(steps * env.DT, traj["fuel"], color=color, alpha=0.85, linewidth=1.5)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("Fuel cost (norm)", fontsize=10)
ax.grid(True, alpha=0.3)

# Subplot 3: Safety margin
ax = axes2[2]
for wi, (traj, color) in enumerate(zip(trajs_plot, colors_plot)):
    steps = np.arange(len(traj["min_h"]))
    ax.plot(steps * env.DT, traj["min_h"], color=color, alpha=0.85, linewidth=1.5)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("min h(x) (Safety margin)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
