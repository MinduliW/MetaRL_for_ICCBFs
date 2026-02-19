import os
import time
import glob
import numpy as np
import multiprocessing as mp
import queue as pyqueue
from dataclasses import dataclass
from typing import Tuple

from tqdm import tqdm
from scipy.io import savemat

# --- Avoid oversubscription when running multiple processes ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from stable_baselines3 import PPO
from metarl_iccbf.cruise_control.envs.rlcbf_env import RLCBFcontrol


# ------------------- Globals in each worker -------------------
_MODEL = None
_ENV = None
_CFG = None
_PROGRESS_Q = None

def _init_worker(progress_q, cfg):
    """Runs once per worker process: load model + create env once."""
    global _MODEL, _ENV, _CFG, _PROGRESS_Q
    _PROGRESS_Q = progress_q
    _CFG = cfg

    _MODEL = PPO.load(cfg.model_path, device="cpu")
    _ENV = RLCBFcontrol(dt=cfg.dt, deterministic=True)


@dataclass
class EvalConfig:
    ics_path: str
    model_path: str
    dt: float = 0.1
    TOF: float = 40.0

    n_workers: int = 8
    n_chunks: int = 32
    base_seed: int = 12345
    deterministic_policy: bool = True

    progress_update_every: int = 1

    out_dir: str = "ResultsEval"
    out_mat: str = "NoiseMetaICCBFMargin_NN_fixedICs_parallel.mat"


def _evaluate_chunk(args: Tuple[np.ndarray, np.ndarray, str]) -> str:
    """
    Evaluate a chunk of indices, save chunk arrays to disk, return the filename.
    We do NOT return big arrays over IPC.
    """
    idxs, ics, chunk_path = args
    cfg = _CFG
    env = _ENV
    model = _MODEL

    lenvec = int(cfg.TOF / cfg.dt)
    K = idxs.shape[0]

    uOpts = np.zeros((K, lenvec, 1), dtype=np.float64)
    states = np.zeros((K, lenvec, 2), dtype=np.float64)
    uOptmag = np.zeros((K, lenvec), dtype=np.float64)
    uTotal = np.zeros((K, 1), dtype=np.float64)
    actionStore = np.zeros((K, lenvec, 4), dtype=np.float64)
    hs = np.zeros((K, lenvec, 1), dtype=np.float64)
    Vs = np.zeros((K, lenvec, 1), dtype=np.float64)
    comptimes = np.zeros((K, lenvec), dtype=np.float64)
    steps_taken = np.zeros((K,), dtype=np.int32)

    local_done = 0
    batch = max(1, int(cfg.progress_update_every))

    for j, global_i in enumerate(idxs):
        d0, v0 = ics[int(global_i), :]
        ep_seed = int(cfg.base_seed + int(global_i))

        obs, _ = env.reset(seed=ep_seed, postProcess=False)

        env.x0 = np.array([d0, v0], dtype=float)
        obs = env.scaleObservation(env.x0)

        t = 0.0
        step = 0

        while t <= cfg.TOF and step < lenvec:
            step += 1

            action, _ = model.predict(obs, deterministic=cfg.deterministic_policy)

            t0 = time.perf_counter()
            obs, reward_temp, done, _, _ = env.step(action)
            comptimes[j, step - 1] = time.perf_counter() - t0

            states[j, step - 1, :] = env.x0
            actionStore[j, step - 1, :] = action

            # env.u might be scalar or vector; store robustly
            u_arr = np.atleast_1d(env.u).astype(float)
            uOpts[j, step - 1, 0] = float(u_arr[0])
            uOptmag[j, step - 1] = float(np.linalg.norm(u_arr))

            hs[j, step - 1, 0] = float(env.x0[0] - 1.8 * env.x0[1])
            Vs[j, step - 1, 0] = float((env.x0[1] - env.vmax) ** 2)

            t += cfg.dt
            if done:
                break

        steps_taken[j] = step
        uTotal[j, 0] = float(np.sum(uOptmag[j, :step]) * cfg.dt)

        local_done += 1
        if _PROGRESS_Q is not None and (local_done % batch == 0):
            _PROGRESS_Q.put(batch)

    rem = local_done % batch
    if _PROGRESS_Q is not None and rem != 0:
        _PROGRESS_Q.put(rem)

    np.savez_compressed(
        chunk_path,
        idxs=idxs,
        uOpts=uOpts,
        states=states,
        uOptmag=uOptmag,
        uTotal=uTotal,
        actionStore=actionStore,
        hs=hs,
        Vs=Vs,
        comptimes=comptimes,
        steps_taken=steps_taken,
    )
    return chunk_path


def evaluate_parallel(cfg: EvalConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load ICs robustly: allow .npy or .npz with 'ics'
    if cfg.ics_path.endswith(".npz"):
        D = np.load(cfg.ics_path, allow_pickle=True)
        ics = D["ics"] if "ics" in D.files else D[D.files[0]]
    else:
        ics = np.load(cfg.ics_path)

    if not (ics.ndim == 2 and ics.shape[1] == 2):
        raise ValueError(f"IC file must be shape (N,2). Got {ics.shape}")

    N = ics.shape[0]
    lenvec = int(cfg.TOF / cfg.dt)
    tvec_full = np.arange(0, cfg.TOF, cfg.dt)

    print(f"Loaded ICs: {N} from {cfg.ics_path}")
    print(f"Model: {cfg.model_path}")
    print(f"Workers: {cfg.n_workers}, chunks: {cfg.n_chunks}")

    all_idxs = np.arange(N, dtype=np.int64)
    chunks = np.array_split(all_idxs, max(cfg.n_chunks, cfg.n_workers))

    chunk_dir = os.path.join(cfg.out_dir, "_chunks_tmp")
    os.makedirs(chunk_dir, exist_ok=True)
    for f in glob.glob(os.path.join(chunk_dir, "chunk_*.npz")):
        try:
            os.remove(f)
        except OSError:
            pass

    work_items = []
    for k, c in enumerate(chunks):
        if c.size == 0:
            continue
        chunk_path = os.path.join(chunk_dir, f"chunk_{k:04d}.npz")
        work_items.append((c, ics, chunk_path))

    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()

    with ctx.Pool(
        processes=cfg.n_workers,
        initializer=_init_worker,
        initargs=(progress_q, cfg),
    ) as pool:

        async_results = [pool.apply_async(_evaluate_chunk, (item,)) for item in work_items]

        pbar = tqdm(total=N, desc="NN eval episodes", unit="ep")
        try:
            while True:
                try:
                    inc = progress_q.get(timeout=0.2)
                    pbar.update(int(inc))
                except pyqueue.Empty:
                    pass

                if all(r.ready() for r in async_results):
                    break

            chunk_files = [r.get() for r in async_results]
        finally:
            pbar.close()

    uOpts = np.zeros((N, lenvec, 1), dtype=np.float64)
    states = np.zeros((N, lenvec, 2), dtype=np.float64)
    uTotal = np.zeros((N, 1), dtype=np.float64)
    actionStore = np.zeros((N, lenvec, 4), dtype=np.float64)
    hs = np.zeros((N, lenvec, 1), dtype=np.float64)
    Vs = np.zeros((N, lenvec, 1), dtype=np.float64)
    comptimes = np.zeros((N, lenvec), dtype=np.float64)
    steps_taken = np.zeros((N,), dtype=np.int32)

    for cf in chunk_files:
        data = np.load(cf, allow_pickle=False)
        idxs = data["idxs"]

        uOpts[idxs, :, :] = data["uOpts"]
        states[idxs, :, :] = data["states"]
        uTotal[idxs, :] = data["uTotal"]
        actionStore[idxs, :, :] = data["actionStore"]
        hs[idxs, :, :] = data["hs"]
        Vs[idxs, :, :] = data["Vs"]
        comptimes[idxs, :] = data["comptimes"]
        steps_taken[idxs] = data["steps_taken"]

    print("Completed:", N, "episodes")
    print("uTotal mean:", float(np.mean(uTotal)))
    print("uTotal q25/q50/q75:", np.percentile(uTotal, [25, 50, 75]).tolist())

    out_path = os.path.join(cfg.out_dir, cfg.out_mat)
    savemat(out_path, {
        "uOpts": uOpts,
        "states": states,
        "hs": hs,
        "Vs": Vs,
        "actionStore": actionStore,
        "uTotal": uTotal,
        "comptimes": comptimes,
        "tvec_full": tvec_full,
        "steps_taken": steps_taken,
        "ics": ics,
        "model_path": np.array([cfg.model_path], dtype=object),
    })
    print("Saved:", out_path)
    return out_path
