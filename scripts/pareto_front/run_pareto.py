"""Pareto front sweep for a multi_body MORL checkpoint — parallelized over weight points.

Each worker process initializes its own InspectionEnv (sympy compile) and loads
the model once, then evaluates all N_EPISODES for its assigned weight point(s).
"""
import time
import multiprocessing as mp
import numpy as np
import torch
import matplotlib.pyplot as plt

from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO
from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT      = "/home/aposadasn/MetaRL_for_ICCBFs/outputs/inspection/Mamba2TunedICCBF_Inspection_20260523_112748/final_model.zip"
N_WEIGHTS  = 20
N_EPISODES = 10
N_WORKERS  = 20    # multiple workers share GPU; increase if VRAM allows

# ---------------------------------------------------------------------------
# Helpers (inlined to avoid importing broken run_inspection_fixedspec_eval)
# ---------------------------------------------------------------------------

def apply_episode_params(env, meta):
    if "m"     in meta: env.m     = float(meta["m"])
    if "R_D"   in meta: env.R_D   = float(meta["R_D"])
    if "R_C"   in meta: env.R_C   = float(meta["R_C"])
    if "U_MAX" in meta: env.U_MAX = float(meta["U_MAX"])
    if "R_MAX" in meta: env.R_MAX = float(meta["R_MAX"])
    if "r" in meta:
        env.r = float(meta["r"])
        env.n = float(np.sqrt(env.mu / (env.r ** 3)))
    if hasattr(env, "iccbf_da") and env.iccbf_da is not None:
        try:
            env.iccbf_da.m       = float(env.m)
            env.iccbf_da.n       = float(env.n)
            env.iccbf_da.rho_koz = float(env.R_C + env.R_D)
            env.iccbf_da.rho_kiz = float(env.R_MAX)
            if hasattr(env.iccbf_da, "u_max"):
                env.iccbf_da.u_max = float(env.U_MAX)
            if hasattr(env.iccbf_da, "u_max_axis"):
                env.iccbf_da.u_max_axis = float(env.U_MAX)
        except Exception:
            pass
    if hasattr(env, "_recompute_derived"): env._recompute_derived()
    if hasattr(env, "_rebuild_models"):    env._rebuild_models()


# ---------------------------------------------------------------------------
# Worker globals (one copy per worker process)
# ---------------------------------------------------------------------------
_MODEL = None
_ENV   = None


def _init_worker(ckpt):
    global _MODEL, _ENV
    pid = mp.current_process().pid
    print(f"[worker {pid}] loading model + env...", flush=True)
    _MODEL = RecurrentPPO.load(ckpt, device="cuda")
    _MODEL.agent.eval()
    _ENV = InspectionEnv(
        dt=10.0, morl=True, enableCBFtunning=True,
        enable_param_randomisation=True,
        morl_objective_set="fuel",
    )
    print(f"[worker {pid}] ready.", flush=True)


def _eval_weight(args):
    wi, w, ep_seeds = args
    device = _MODEL.device
    w_t = torch.as_tensor(w[None], dtype=torch.float32, device=device)
    ep_r_vecs = []
    ep_diagnostics = []  # (pct_inspected, steps, dv_used, terminated_early)

    # Trace the very first episode of the first weight point (crash regime, high w_fuel).
    trace_this_weight = (wi == 0)
    trace_ep_idx = 0  # first episode

    for ep_idx, seed in enumerate(ep_seeds):
        obs, _ = _ENV.reset(seed=int(seed))
        do_trace = trace_this_weight and (ep_idx == trace_ep_idx)
        trace_rows = []  # (step, action_rl_norm, u_rl_norm, u_safe_norm, r_norm, h_sun, qp_solved)

        actor_states, critic_states = _MODEL.agent.initial_states(1, _MODEL.device)
        ep_starts = np.ones(1,  dtype=np.float32)
        ep_r_vec  = np.zeros(3, dtype=np.float32)
        done = False
        steps = 0
        last_info = {}
        last_terminated = False

        while not done:
            obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=_MODEL.device)
            ep_t  = torch.as_tensor(ep_starts, dtype=torch.float32, device=_MODEL.device)
            with torch.no_grad():
                action, actor_states, critic_states = _MODEL.agent.deterministic_step(
                    obs_t, actor_states, critic_states, ep_t, w=w_t,
                )
            action_np = np.clip(
                action.squeeze(0).cpu().numpy(),
                _ENV.action_space.low, _ENV.action_space.high,
            )
            obs, r_vec_step, terminated, truncated, info = _ENV.step(action_np)
            ep_r_vec += np.asarray(r_vec_step, dtype=np.float32)
            steps += 1
            last_info = info
            last_terminated = terminated
            done = bool(terminated or truncated)
            ep_starts = np.ones(1, dtype=np.float32) if done else np.zeros(1, dtype=np.float32)

            if do_trace:
                # raw policy thrust action (first 3 dims, before scaling by U_MAX); cbf tuning is the next 9
                a_rl_first3 = action_np[:3]
                cbf_tuning = action_np[3:] if action_np.shape[0] > 3 else np.zeros(0)
                u_rl_norm   = float(np.linalg.norm(info.get("u_rl",   np.zeros(3))))
                u_safe_norm = float(np.linalg.norm(info.get("u_safe", np.zeros(3))))
                trace_rows.append((
                    steps,
                    float(np.linalg.norm(a_rl_first3)),
                    u_rl_norm,
                    u_safe_norm,
                    float(info.get("r_norm", 0.0)),
                    float(info.get("h_sun", 0.0)),
                    bool(info.get("qp_solved", True)),
                    cbf_tuning.copy(),
                ))

        ep_r_vecs.append(ep_r_vec)
        pct_inspected = 100.0 * float(last_info.get("num_inspected", 0)) / float(_ENV.N_POINTS)
        dv_used = float(last_info.get("cum_dv", 0.0))
        impulse_used = float(last_info.get("cum_impulse", 0.0))
        ep_diagnostics.append((pct_inspected, steps, dv_used, impulse_used, last_terminated))

        if do_trace and trace_rows:
            print(f"\n  === trace: weight[{wi}] episode[{ep_idx}] (crash regime, w={w}) ===", flush=True)
            print("  step | |a_rl[:3]| |  |u_rl|  | |u_safe| |  r_norm |  h_sun  | qp_ok | cbf_tuning (9-dim)", flush=True)
            for row in trace_rows[:20]:
                step_i, a_rl, u_rl_n, u_safe_n, r_n, h_s, qp_ok, cbf = row
                cbf_str = "[" + " ".join(f"{x:+.2f}" for x in cbf) + "]"
                print(f"  {step_i:4d} | {a_rl:8.4f}  | {u_rl_n:7.4f} | {u_safe_n:7.4f} | {r_n:7.2f} | {h_s:+.4f} | {str(qp_ok):5s} | {cbf_str}", flush=True)
            print("", flush=True)

    r_arr = np.stack(ep_r_vecs)  # (n_episodes, 3)
    diag = np.array(ep_diagnostics, dtype=np.float64)  # (n_episodes, 5)
    pct, steps_arr, dv_arr, imp_arr, term_arr = diag[:, 0], diag[:, 1], diag[:, 2], diag[:, 3], diag[:, 4]
    print(
        f"  [{wi+1}/{N_WEIGHTS}] w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]  "
        f"coverage={r_arr[:,0].mean():.3f}±{r_arr[:,0].std():.3f}  "
        f"fuel_obj={r_arr[:,1].mean():.3f}±{r_arr[:,1].std():.3f}\n"
        f"    inspected: {pct.mean():.1f}%±{pct.std():.1f}% (range {pct.min():.0f}-{pct.max():.0f}%)\n"
        f"    steps:     {steps_arr.mean():.0f}±{steps_arr.std():.0f} (range {steps_arr.min():.0f}-{steps_arr.max():.0f})\n"
        f"    ΔV (m/s):  {dv_arr.mean():.2f}±{dv_arr.std():.2f}  |  impulse (N·s, eval_parallel scale): {imp_arr.mean():.2f}±{imp_arr.std():.2f}\n"
        f"    completed-early (terminated, not truncated): {int(term_arr.sum())}/{len(term_arr)}",
        flush=True,
    )
    return wi, r_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    W_SAFETY = 0.2
    t_vals  = np.linspace(0.0, 1.0, N_WEIGHTS, dtype=np.float32)
    # t=0: all weight on fuel, t=1: all weight on coverage
    weights = np.stack([
        t_vals * (1.0 - W_SAFETY),                      # w_coverage
        (1.0 - t_vals) * (1.0 - W_SAFETY),              # w_fuel
        np.full(N_WEIGHTS, W_SAFETY, dtype=np.float32),  # w_safety (fixed)
    ], axis=-1)  # (N_WEIGHTS, 3), sums to 1.0
    # Shared seeds: every weight point evaluates the same N_EPISODES ICs/params.
    # This cancels IC variance between weight points so the front reflects weight alone.
    ep_seeds = np.random.SeedSequence(42).generate_state(N_EPISODES)
    work_items = [(wi, w, ep_seeds) for wi, w in enumerate(weights)]

    n_workers = min(N_WORKERS, mp.cpu_count())
    print(f"Launching {n_workers} workers for {N_WEIGHTS} weight points x {N_EPISODES} episodes (shared seeds)...", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(CKPT,),
    ) as pool:
        results = pool.map(_eval_weight, work_items)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.", flush=True)

    # Reassemble in weight order
    all_r_vecs = [None] * N_WEIGHTS
    for wi, r_arr in results:
        all_r_vecs[wi] = r_arr

    r     = np.stack([a.mean(axis=0) for a in all_r_vecs])  # (N_WEIGHTS, 3)
    r_std = np.stack([a.std(axis=0)  for a in all_r_vecs])

    # ---- Plot 1: Pareto Front (coverage vs fuel) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(r[:, 0], r[:, 1], c=t_vals, cmap="coolwarm", s=60, zorder=3)
    ax.errorbar(r[:, 0], r[:, 1],
                xerr=r_std[:, 0], yerr=r_std[:, 1],
                fmt="none", ecolor="gray", alpha=0.4, zorder=2)
    plt.colorbar(sc, ax=ax, label="t  (w_coverage=t·0.8, w_fuel=(1-t)·0.8)")
    ax.set_xlabel("Coverage reward")
    ax.set_ylabel("Fuel objective (−fuel_norm, cumulative)")
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=150)
    print("Saved pareto_front.png")
    plt.close()

    # ---- Plot 2: Coverage Objective vs Weight ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_vals, r[:, 0], "g-o", label="Coverage obj", linewidth=2, markersize=6)
    ax.fill_between(t_vals, r[:, 0] - r_std[:, 0], r[:, 0] + r_std[:, 0],
                     alpha=0.2, color="green")
    ax.set_xlabel("t  (w_coverage = t · 0.8)")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("coverage_objective_vs_weight.png", dpi=150)
    print("Saved coverage_objective_vs_weight.png")
    plt.close()

    # ---- Plot 3: Fuel Objective vs Weight ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_vals, r[:, 1], "b-o", label="Fuel obj (−fuel_norm cumulative)", linewidth=2, markersize=6)
    ax.fill_between(t_vals, r[:, 1] - r_std[:, 1], r[:, 1] + r_std[:, 1],
                     alpha=0.2, color="blue")
    ax.set_xlabel("t  (w_fuel = (1-t) · 0.8)")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fuel_objective_vs_weight.png", dpi=150)
    print("Saved fuel_objective_vs_weight.png")
    plt.close()
