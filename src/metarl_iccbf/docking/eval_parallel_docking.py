import os
import time
import glob
import numpy as np
import multiprocessing as mp
import queue as pyqueue
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

from tqdm import tqdm
from scipy.io import savemat

# --- Avoid oversubscription when running multiple processes ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

# IMPORTANT: adjust import to your repo layout
# If your docking env lives in docking/RLCBF.py, change to: from .RLCBF import RLCBFcontrol
from .RLCBF import RLCBFcontrol


# ------------------- Globals in each worker -------------------
_MODEL = None
_ENV = None
_CFG = None
_PROGRESS_Q = None
_SPEC = None  # dict of arrays loaded from cfg.ics_path


@dataclass
class EvalConfig:
    ics_path: str
    model_path: str

    # which RL algorithm/policy was trained?
    algo: Literal["PPO", "RNN"] = "PPO"

    dt: float = 0.5
    TOF: float = 50.0

    n_workers: int = 8
    n_chunks: int = 32
    base_seed: int = 12345
    deterministic_policy: bool = True

    progress_update_every: int = 1

    out_dir: str = "ResultsEval"
    out_mat: str = "DockingEval_parallel.mat"


def _load_episode_bank(path: str) -> dict:
    """
    Expected docking episode bank format (npz) with at least:
        x0s: (N,5)
        m_vec, rho_vec, om_vec, umax_vec, gamma_vec: (N,)
        r_vec: optional (N,)
    """
    if not path.endswith(".npz"):
        raise ValueError(f"Docking eval expects an .npz episode bank. Got: {path}")

    D = np.load(path, allow_pickle=True)
    files = set(D.files)

    def need(k: str):
        if k not in files:
            raise KeyError(f"Episode bank missing required key '{k}'. Keys={sorted(files)}")
        return np.asarray(D[k])

    spec = {
        "x0s": need("x0s").astype(np.float64),
        "m": need("m_vec").astype(np.float64).reshape(-1),
        "rho": need("rho_vec").astype(np.float64).reshape(-1),
        "om": need("om_vec").astype(np.float64).reshape(-1),
        "umax": need("umax_vec").astype(np.float64).reshape(-1),
        "gamma": need("gamma_vec").astype(np.float64).reshape(-1),
    }

    if "r_vec" in files:
        spec["r"] = np.asarray(D["r_vec"], dtype=np.float64).reshape(-1)
    else:
        spec["r"] = None

    # optional: per-episode seed vector (not required)
    spec["seed_vec"] = np.asarray(D["seed_vec"], dtype=np.int64).reshape(-1) if "seed_vec" in files else None
    spec["base_seed"] = int(np.asarray(D["base_seed"]).reshape(-1)[0]) if "base_seed" in files else None

    N = spec["x0s"].shape[0]
    if spec["x0s"].ndim != 2 or spec["x0s"].shape[1] != 5:
        raise ValueError(f"x0s must be shape (N,5). Got {spec['x0s'].shape}")
    for k in ("m", "rho", "om", "umax", "gamma"):
        if spec[k].shape[0] != N:
            raise ValueError(f"Key '{k}' must have length N={N}. Got {spec[k].shape}")

    if spec["r"] is not None and spec["r"].shape[0] != N:
        raise ValueError(f"Key 'r_vec' must have length N={N}. Got {spec['r'].shape}")

    return spec


def _init_worker(progress_q, cfg: EvalConfig):
    """Runs once per worker process: load model + create env + load episode bank once."""
    global _MODEL, _ENV, _CFG, _PROGRESS_Q, _SPEC
    _PROGRESS_Q = progress_q
    _CFG = cfg

    # load episode bank in each worker (avoids pickling huge arrays for each chunk)
    _SPEC = _load_episode_bank(cfg.ics_path)

    # load model
    if cfg.algo.upper() == "PPO":
        _MODEL = PPO.load(cfg.model_path, device="cpu")
    elif cfg.algo.upper() == "RNN":
        if RecurrentPPO is None:
            raise ImportError("sb3-contrib is required for RecurrentPPO. pip install sb3-contrib")
        _MODEL = RecurrentPPO.load(cfg.model_path, device="cpu")
    else:
        raise ValueError(f"Unknown algo={cfg.algo}. Use 'PPO' or 'RNN'.")

    # create env once
    _ENV = RLCBFcontrol(dt=cfg.dt, deterministic=True)


def _set_episode(env: RLCBFcontrol, spec: dict, i: int):
    """
    Force episode i into the env (state + parameters), then rebuild ICCBF/dynamics as needed.
    This mirrors what your reset() does when sampling parameters.
    """
    x0 = spec["x0s"][i].copy()

    env.x0 = x0
    env.m = float(spec["m"][i])
    env.rho = float(spec["rho"][i])
    env.om = float(spec["om"][i])
    env.umax = float(spec["umax"][i])
    env.gamma = float(spec["gamma"][i])

    if spec["r"] is not None:
        env.r = float(spec["r"][i])

    # keep derived quantities consistent
    try:
        env.n = float(np.sqrt(env.mu / env.r**3))
    except Exception:
        # env may use different naming; ignore if not present
        pass

    # rebuild modules exactly like your reset does
    try:
        from metarl_iccbf.docking.iccbfs import ICCBF
        from metarl_iccbf.docking.Dockingcase import DockingCase
        from metarl_iccbf.docking.dynamicsandControl import dynamicsAndControl
    except Exception:
        # if imports fail due to package layout, the env likely already has these
        ICCBF = None
        DockingCase = None
        dynamicsAndControl = None

    if ICCBF is not None:
        env.iccbf = ICCBF(mu=env.mu, r=env.r, gamma=env.gamma, rho=env.rho, m=env.m, om=env.om, umax=env.umax)
        env.getmargin_docking = env.iccbf.getmargin

    if DockingCase is not None:
        env.dockingCase = DockingCase(rho=env.rho, gamma=env.gamma)
        env.validPoints = env.dockingCase.generate_evenly_spread_cone_points_2d()

    if dynamicsAndControl is not None:
        env.dynamics = dynamicsAndControl(mu=env.mu, n=getattr(env, "n", None), r=env.r, m=env.m, om=env.om)

    # update observation scaling bounds that depend on omega if your env uses this
    if hasattr(env, "obshigh") and env.obshigh is not None and len(env.obshigh) >= 5:
        try:
            env.obshigh[4] = 1.2 * env.om * (env.TOF + 0.5)
        except Exception:
            pass


def _evaluate_chunk(args: Tuple[np.ndarray, str]) -> str:
    """
    Evaluate a chunk of indices, save chunk arrays to disk, return the filename.
    We do NOT return big arrays over IPC.
    """
    idxs, chunk_path = args
    cfg = _CFG
    env = _ENV
    model = _MODEL
    spec = _SPEC

    lenvec = int(cfg.TOF / cfg.dt)
    K = idxs.shape[0]

    # storage
    action_dim = int(np.prod(env.action_space.shape))
    state_dim = int(spec["x0s"].shape[1])

    actions = np.zeros((K, lenvec, action_dim), dtype=np.float64)
    states = np.zeros((K, lenvec, state_dim), dtype=np.float64)

    uOptmag = np.zeros((K, lenvec), dtype=np.float64)
    uTotal = np.zeros((K, 1), dtype=np.float64)
    comptimes = np.zeros((K, lenvec), dtype=np.float64)
    rewards = np.zeros((K, lenvec), dtype=np.float64)
    steps_taken = np.zeros((K,), dtype=np.int32)
    dones = np.zeros((K, lenvec), dtype=np.int8)

    # for recurrent policies
    recurrent = (cfg.algo.upper() == "RNN")
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    local_done = 0
    batch = max(1, int(cfg.progress_update_every))

    for j, global_i in enumerate(idxs):
        global_i = int(global_i)

        ep_seed = int(cfg.base_seed + global_i)
        obs, _ = env.reset(seed=ep_seed, postProcess=False)

        # force deterministic episode i
        _set_episode(env, spec, global_i)
        obs = env.scaleObservation(env.x0)

        # reset recurrent state per episode
        if recurrent:
            lstm_states = None
            episode_starts[:] = True

        step = 0
        for step in range(lenvec):
            if recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=cfg.deterministic_policy,
                )
            else:
                action, _ = model.predict(obs, deterministic=cfg.deterministic_policy)

            t0 = time.perf_counter()
            obs, reward_temp, terminated, truncated, _info = env.step(action)
            comptimes[j, step] = time.perf_counter() - t0

            # store
            states[j, step, :] = np.asarray(env.x0, dtype=np.float64).reshape(-1)
            actions[j, step, :] = np.asarray(action, dtype=np.float64).reshape(-1)
            rewards[j, step] = float(reward_temp)

            # magnitude of control (robust to scalar/vector)
            u_arr = np.atleast_1d(getattr(env, "u", np.zeros(action_dim))).astype(np.float64)
            uOptmag[j, step] = float(np.linalg.norm(u_arr))

            done = bool(terminated or truncated)
            dones[j, step] = 1 if done else 0

            if recurrent:
                episode_starts[:] = done

            if done:
                step += 1  # count this step
                break

        steps_taken[j] = int(step)
        uTotal[j, 0] = float(np.sum(uOptmag[j, :steps_taken[j]]) * cfg.dt)

        local_done += 1
        if _PROGRESS_Q is not None and (local_done % batch == 0):
            _PROGRESS_Q.put(batch)

    rem = local_done % batch
    if _PROGRESS_Q is not None and rem != 0:
        _PROGRESS_Q.put(rem)

    np.savez_compressed(
        chunk_path,
        idxs=idxs,
        states=states,
        actions=actions,
        rewards=rewards,
        uOptmag=uOptmag,
        uTotal=uTotal,
        comptimes=comptimes,
        steps_taken=steps_taken,
        dones=dones,
    )
    return chunk_path


def evaluate_parallel(cfg: EvalConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load episode bank once in main process as well (for N and metadata)
    spec_main = _load_episode_bank(cfg.ics_path)
    N = int(spec_main["x0s"].shape[0])

    lenvec = int(cfg.TOF / cfg.dt)
    tvec_full = np.arange(0, cfg.TOF, cfg.dt)

    print(f"Loaded episode bank: {N} from {cfg.ics_path}")
    print(f"Model: {cfg.model_path}")
    print(f"Algo: {cfg.algo}")
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
        work_items.append((c, chunk_path))

    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()

    with ctx.Pool(
        processes=cfg.n_workers,
        initializer=_init_worker,
        initargs=(progress_q, cfg),
    ) as pool:

        async_results = [pool.apply_async(_evaluate_chunk, (item,)) for item in work_items]

        pbar = tqdm(total=N, desc="Docking eval episodes", unit="ep")
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

    # allocate full outputs
    action_dim = 4
    state_dim = 5

    actions = np.zeros((N, lenvec, action_dim), dtype=np.float64)
    states = np.zeros((N, lenvec, state_dim), dtype=np.float64)
    rewards = np.zeros((N, lenvec), dtype=np.float64)
    uOptmag = np.zeros((N, lenvec), dtype=np.float64)
    uTotal = np.zeros((N, 1), dtype=np.float64)
    comptimes = np.zeros((N, lenvec), dtype=np.float64)
    steps_taken = np.zeros((N,), dtype=np.int32)
    dones = np.zeros((N, lenvec), dtype=np.int8)

    for cf in chunk_files:
        data = np.load(cf, allow_pickle=False)
        idxs = data["idxs"]
        states[idxs, :, :] = data["states"]
        actions[idxs, :, :] = data["actions"]
        rewards[idxs, :] = data["rewards"]
        uOptmag[idxs, :] = data["uOptmag"]
        uTotal[idxs, :] = data["uTotal"]
        comptimes[idxs, :] = data["comptimes"]
        steps_taken[idxs] = data["steps_taken"]
        dones[idxs, :] = data["dones"]

    print("Completed:", N, "episodes")
    print("uTotal mean:", float(np.mean(uTotal)))
    print("uTotal q25/q50/q75:", np.percentile(uTotal, [25, 50, 75]).tolist())

    out_path = os.path.join(cfg.out_dir, cfg.out_mat)
    savemat(out_path, {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "uOptmag": uOptmag,
        "uTotal": uTotal,
        "comptimes": comptimes,
        "tvec_full": tvec_full,
        "steps_taken": steps_taken,
        "dones": dones,
        "x0s": spec_main["x0s"],
        "m_vec": spec_main["m"],
        "rho_vec": spec_main["rho"],
        "om_vec": spec_main["om"],
        "umax_vec": spec_main["umax"],
        "gamma_vec": spec_main["gamma"],
        "r_vec": spec_main["r"] if spec_main["r"] is not None else np.array([], dtype=np.float64),
        "model_path": np.array([cfg.model_path], dtype=object),
        "ics_path": np.array([cfg.ics_path], dtype=object),
        "algo": np.array([cfg.algo], dtype=object),
    })
    print("Saved:", out_path)
    return out_path
