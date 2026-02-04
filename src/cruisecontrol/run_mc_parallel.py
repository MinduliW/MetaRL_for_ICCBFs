# run_mc_parallel.py
# Parallel Monte Carlo runner with an EPISODE-LEVEL progress bar (tqdm) over fixed ICs.
#
# Usage:
#   pip install tqdm
#   python run_mc_parallel.py
#
# Notes:
# - Each worker creates its own env + loads its own model (avoids pickling SB3 objects).
# - Uses spawn context (macOS-safe).
# - Progress bar updates per-episode via a multiprocessing Queue.
# - If MOSEK licence limits concurrency, reduce n_workers.

import os
import time
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import queue as pyqueue  # for queue.Empty in parent loop

# ----------- limit per-process threading to avoid oversubscription -----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# ---------------------------------------------------------------------------

from tqdm import tqdm
from scipy.io import savemat

# Your env (adjust path/import if needed)
from RLCBF import RLCBFcontrol

# SB3
from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None


# ---------- progress queue plumbing (set per worker via initializer) ----------
_PROGRESS_Q = None

def _init_worker(progress_q):
    global _PROGRESS_Q
    _PROGRESS_Q = progress_q
# -----------------------------------------------------------------------------


@dataclass
class MCConfig:
    ics_path: str                          # .npy containing (N,2) fixed ICs
    model_path: str                        # path to SB3 .zip model
    algo: str                              # "PPO" or "RNN" (RecurrentPPO)

    dt: float = 0.1
    TOF: float = 40.0

    n_workers: int = 4
    n_chunks: int = 80                     # number of chunks to split the 5000 ICs into

    base_seed: int = 12345                 # episode_seed = base_seed + global_index
    deterministic_policy: bool = True

    # progress reporting granularity: send +k to queue every k episodes
    progress_update_every: int = 5

    out_mat: str = "mc_parallel_results.mat"


def _load_model(model_path: str, algo: str):
    algo_u = algo.upper()
    if algo_u == "PPO":
        return PPO.load(model_path, device="cpu")
    if algo_u in ("RNN", "RECURRENT", "RECURRENTPPO"):
        if RecurrentPPO is None:
            raise RuntimeError("sb3_contrib.RecurrentPPO not available. Install sb3-contrib.")
        return RecurrentPPO.load(model_path, device="cpu")
    raise ValueError(f"Unknown algo={algo}. Use 'PPO' or 'RNN'.")


def _evaluate_chunk(args: Tuple[np.ndarray, np.ndarray, MCConfig]) -> Dict[str, Any]:
    """
    Worker evaluates a chunk of IC indices.
    Returns arrays aligned with idxs.
    """
    idxs, ics, cfg = args

    model = _load_model(cfg.model_path, cfg.algo)
    is_recurrent = (cfg.algo.upper() != "PPO")

    env = RLCBFcontrol(dt=cfg.dt, deterministic=False)
    lenvec = int(cfg.TOF / cfg.dt)

    K = idxs.shape[0]
    uTotal = np.zeros((K,), dtype=np.float64)
    min_h = np.full((K,), np.inf, dtype=np.float64)
    crashed = np.zeros((K,), dtype=np.int8)
    steps_taken = np.zeros((K,), dtype=np.int32)
    mean_step_time = np.zeros((K,), dtype=np.float64)

    # episode-level progress batching
    local_done = 0
    batch = cfg.progress_update_every

    for j, global_i in enumerate(idxs):
        d0, v0 = ics[global_i, :]
        ep_seed = int(cfg.base_seed + int(global_i))

        # Seed env RNG for reproducible sensor/actuation noise
        obs, _ = env.reset(seed=ep_seed, postProcess=False)

        # Force initial condition
        env.x0 = np.array([d0, v0], dtype=float)
        obs = env.scaleObservation(env.x0)

        # Recurrent state init
        if is_recurrent:
            lstm_states = None
            episode_start = np.array([True])

        t = 0.0
        step = 0
        u_accum = 0.0
        step_times: List[float] = []

        local_min_h = np.inf
        local_crash = 0

        while t <= cfg.TOF and step < lenvec:
            step += 1

            if is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=cfg.deterministic_policy,
                )
                episode_start = np.array([False])
            else:
                action, _ = model.predict(obs, deterministic=cfg.deterministic_policy)

            t0 = time.perf_counter()
            obs, reward, done, _, _ = env.step(action)
            step_times.append(time.perf_counter() - t0)

            u_mag = float(np.linalg.norm(env.u))
            u_accum += u_mag * cfg.dt

            h_now = float(env.x0[0] - 1.8 * env.x0[1])
            if h_now < local_min_h:
                local_min_h = h_now

            if env.x0[0] < 0.0:
                local_crash = 1
                break

            if done:
                break

            t += cfg.dt

        uTotal[j] = u_accum
        min_h[j] = local_min_h
        crashed[j] = local_crash
        steps_taken[j] = step
        mean_step_time[j] = float(np.mean(step_times)) if step_times else 0.0

        # report progress (batched)
        local_done += 1
        if _PROGRESS_Q is not None and (local_done % batch == 0):
            _PROGRESS_Q.put(batch)

    # flush remainder
    rem = local_done % batch
    if _PROGRESS_Q is not None and rem != 0:
        _PROGRESS_Q.put(rem)

    return dict(
        idxs=idxs,
        uTotal=uTotal,
        min_h=min_h,
        crashed=crashed,
        steps_taken=steps_taken,
        mean_step_time=mean_step_time,
    )


def run_parallel_mc(cfg: MCConfig):
    ics = np.load(cfg.ics_path)  # (N,2)
    if not (ics.ndim == 2 and ics.shape[1] == 2):
        raise ValueError(f"IC file must be shape (N,2). Got {ics.shape}")

    N = ics.shape[0]
    print(f"Loaded ICs: {N} from {cfg.ics_path}")
    print(f"Model: {cfg.model_path}")
    print(f"Algo: {cfg.algo}")
    print(f"TOF={cfg.TOF}, dt={cfg.dt}, workers={cfg.n_workers}, chunks={cfg.n_chunks}")

    all_idxs = np.arange(N, dtype=np.int64)
    chunks = np.array_split(all_idxs, max(cfg.n_chunks, cfg.n_workers))
    work_items = [(c, ics, cfg) for c in chunks if c.size > 0]

    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()

    # Submit async work
    with ctx.Pool(processes=cfg.n_workers, initializer=_init_worker, initargs=(progress_q,)) as pool:
        async_results = [pool.apply_async(_evaluate_chunk, (item,)) for item in work_items]

        # Episode-level progress bar
        pbar = tqdm(total=N, desc="MC episodes", unit="ep")

        # Drain queue until all work completes
        try:
            while True:
                # Update progress from workers
                try:
                    inc = progress_q.get(timeout=0.2)
                    pbar.update(int(inc))
                except pyqueue.Empty:
                    pass

                # Stop when all async tasks are ready
                if all(r.ready() for r in async_results):
                    break

            # Collect results (propagate worker exceptions here)
            results = [r.get() for r in async_results]

        finally:
            pbar.close()

    # Stitch back into global arrays (original order)
    uTotal = np.zeros((N,), dtype=np.float64)
    min_h = np.zeros((N,), dtype=np.float64)
    crashed = np.zeros((N,), dtype=np.int8)
    steps_taken = np.zeros((N,), dtype=np.int32)
    mean_step_time = np.zeros((N,), dtype=np.float64)

    for r in results:
        idxs = r["idxs"]
        uTotal[idxs] = r["uTotal"]
        min_h[idxs] = r["min_h"]
        crashed[idxs] = r["crashed"]
        steps_taken[idxs] = r["steps_taken"]
        mean_step_time[idxs] = r["mean_step_time"]

    print("Done.")
    print("uTotal mean:", float(np.mean(uTotal)))
    print("uTotal median:", float(np.median(uTotal)))
    print("uTotal q25/q75:", np.percentile(uTotal, [25, 75]).tolist())
    print("Crash rate:", float(np.mean(crashed)))
    print("Mean step time (s):", float(np.mean(mean_step_time)))

    savemat(cfg.out_mat, dict(
        uTotal=uTotal,
        min_h=min_h,
        crashed=crashed,
        steps_taken=steps_taken,
        mean_step_time=mean_step_time,
        dt=np.array([cfg.dt]),
        TOF=np.array([cfg.TOF]),
        ics=ics,
        model_path=np.array([cfg.model_path], dtype=object),
        algo=np.array([cfg.algo], dtype=object),
        base_seed=np.array([cfg.base_seed]),
    ))
    print("Saved:", cfg.out_mat)


if __name__ == "__main__":
    # Update these paths to match your setup.
    cfg = MCConfig(
        ics_path="fixed_initial_conditions_N5000_seed123.npy",
        model_path="TrainedModels/MarginMetaRNNwNoiseCruiseControl_CLFhigherweight/final_model.zip",
        algo="RNN",                 # "PPO" or "RNN"
        dt=0.1,
        TOF=40.0,
        n_workers=8,                # start small if MOSEK limits parallel solves
        n_chunks=80,                # more chunks => better load balancing
        base_seed=12345,
        deterministic_policy=True,
        progress_update_every=5,    # progress bar updates every 5 episodes per worker
        out_mat="mc_parallel_results.mat",
    )
    run_parallel_mc(cfg)
