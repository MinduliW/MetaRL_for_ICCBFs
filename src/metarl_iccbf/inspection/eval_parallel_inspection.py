"""
inspection/eval_parallel_inspection.py

Parallel evaluation for Inspection policies (PPO MLP or RecurrentPPO LSTM),
modelled after your docking eval_parallel implementation.

Usage:
    from inspection.eval_parallel_inspection import EvalConfig, evaluate_parallel
"""

from __future__ import annotations

import os
import time
import glob
import numpy as np
import multiprocessing as mp
import queue as pyqueue
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Any, Dict

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
from inspection.inspectionEnvNoisy import InspectionEnv


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

    dt: float = 10.0
    TOF: float = 3.4 * 3600.0  # seconds

    n_workers: int = 8
    n_chunks: int = 16
    base_seed: int = 12345
    deterministic_policy: bool = True

    progress_update_every: int = 1

    out_dir: str = "../src/data/inspection"
    out_mat: str = "InspectionEval_parallel.mat"

    # env flags (match InspectionEnvNoisy)
    enable_param_randomisation: bool = False  # during eval, keep fixed episodes
    enableNoise: bool = False
    enableCBFtunning: bool = True
    dvWeight: float = 10.0


def _load_episode_bank(path: str) -> Dict[str, Any]:
    """
    Expected inspection episode bank format (npz) with at least:
        x0s: (N,7)  [x,y,z,vx,vy,vz,theta_s]
        m_vec, R_D_vec, R_C_vec, U_MAX_vec, R_MAX_vec, r_vec: (N,)

    Optional:
        n_vec: (N,)
        seed_vec: (N,)
        base_seed: scalar
    """
    if not path.endswith(".npz"):
        raise ValueError(f"Inspection eval expects an .npz episode bank. Got: {path}")

    D = np.load(path, allow_pickle=True)
    files = set(D.files)

    def need(k: str):
        if k not in files:
            raise KeyError(f"Episode bank missing required key '{k}'. Keys={sorted(files)}")
        return np.asarray(D[k])

    spec = {
        "x0s": need("x0s").astype(np.float64),
        "m": need("m_vec").astype(np.float64).reshape(-1),
        "R_D": need("R_D_vec").astype(np.float64).reshape(-1),
        "R_C": need("R_C_vec").astype(np.float64).reshape(-1),
        "U_MAX": need("U_MAX_vec").astype(np.float64).reshape(-1),
        "R_MAX": need("R_MAX_vec").astype(np.float64).reshape(-1),
        "r": need("r_vec").astype(np.float64).reshape(-1),
    }

    spec["n"] = np.asarray(D["n_vec"], dtype=np.float64).reshape(-1) if "n_vec" in files else None
    spec["seed_vec"] = np.asarray(D["seed_vec"], dtype=np.int64).reshape(-1) if "seed_vec" in files else None
    spec["base_seed"] = int(np.asarray(D["base_seed"]).reshape(-1)[0]) if "base_seed" in files else None

    N = spec["x0s"].shape[0]
    if spec["x0s"].ndim != 2 or spec["x0s"].shape[1] != 7:
        raise ValueError(f"x0s must be shape (N,7). Got {spec['x0s'].shape}")
    for k in ("m", "R_D", "R_C", "U_MAX", "R_MAX", "r"):
        if spec[k].shape[0] != N:
            raise ValueError(f"Key '{k}' must have length N={N}. Got {spec[k].shape}")
    if spec["n"] is not None and spec["n"].shape[0] != N:
        raise ValueError(f"Key 'n_vec' must have length N={N}. Got {spec['n'].shape}")
    if spec["seed_vec"] is not None and spec["seed_vec"].shape[0] != N:
        raise ValueError(f"Key 'seed_vec' must have length N={N}. Got {spec['seed_vec'].shape}")

    return spec


def _set_attr(obj: Any, names, value) -> bool:
    """Try setting attribute for any of the candidate names; return True if set."""
    for n in names:
        if hasattr(obj, n):
            try:
                setattr(obj, n, value)
                return True
            except Exception:
                pass
    return False


def _set_episode(env: InspectionEnv, spec: dict, i: int):
    """
    Force episode i into the env (state + parameters), then rebuild dynamics/CBFs as needed.
    This should mirror what your reset() does when sampling parameters.
    """
    x0s = spec["x0s"][i].copy().astype(np.float64).reshape(-1)

    r0v0 = x0s[:6]
    theta_s0 = float(x0s[6])

    # state
    _set_attr(env, ["x0", "x", "state", "x_state"], r0v0)

    # sun angle / direction parameter
    _set_attr(env, ["theta_s", "thetaS", "theta_s0", "thetaS0"], theta_s0)

    # parameters (try common names)
    _set_attr(env, ["m", "m_c", "mass", "mc"], float(spec["m"][i]))
    _set_attr(env, ["R_D", "R_Dp", "R_dep", "Rdeputy", "RD"], float(spec["R_D"][i]))
    _set_attr(env, ["R_C", "R_ch", "Rchief", "RC"], float(spec["R_C"][i]))
    _set_attr(env, ["U_MAX", "u_max", "umax", "uMax"], float(spec["U_MAX"][i]))
    _set_attr(env, ["R_MAX", "r_max", "Rmax"], float(spec["R_MAX"][i]))
    _set_attr(env, ["r", "r_orbit", "rOrbit"], float(spec["r"][i]))
    if spec.get("n") is not None:
        _set_attr(env, ["n", "mean_motion"], float(spec["n"][i]))

    # Rebuild any derived modules if your env exposes a method
    for fn in ["_rebuild", "rebuild", "build", "reset_modules", "resetModules", "init_modules", "initModules"]:
        if hasattr(env, fn):
            try:
                getattr(env, fn)()
                break
            except Exception:
                pass


def _init_worker(progress_q, cfg: EvalConfig):
    """Runs once per worker process: load model + create env + load episode bank once."""
    global _MODEL, _ENV, _CFG, _PROGRESS_Q, _SPEC
    _PROGRESS_Q = progress_q
    _CFG = cfg

    _SPEC = _load_episode_bank(cfg.ics_path)

    if cfg.algo.upper() == "PPO":
        _MODEL = PPO.load(cfg.model_path, device="cpu")
    elif cfg.algo.upper() == "RNN":
        if RecurrentPPO is None:
            raise ImportError("sb3-contrib is required for RecurrentPPO. pip install sb3-contrib")
        _MODEL = RecurrentPPO.load(cfg.model_path, device="cpu")
    else:
        raise ValueError(f"Unknown algo={cfg.algo}. Use 'PPO' or 'RNN'.")

    _ENV = InspectionEnv(
        dt=cfg.dt,
        enable_param_randomisation=cfg.enable_param_randomisation,
        enableNoise=cfg.enableNoise,
        enableCBFtunning=cfg.enableCBFtunning,
        dvWeight=cfg.dvWeight,
    )


def _evaluate_chunk(args: Tuple[np.ndarray, str]) -> str:
    idxs, chunk_path = args
    cfg = _CFG
    env: InspectionEnv = _ENV
    model = _MODEL
    spec = _SPEC

    lenvec = int(np.floor(cfg.TOF / cfg.dt))
    K = idxs.shape[0]

    action_dim = int(np.prod(env.action_space.shape))
    state_dim = 6  # store [r;v] only; theta_s saved separately

    actions = np.zeros((K, lenvec, action_dim), dtype=np.float64)
    states = np.zeros((K, lenvec, state_dim), dtype=np.float64)
    theta_s_hist = np.zeros((K, lenvec), dtype=np.float64)

    uOptmag = np.zeros((K, lenvec), dtype=np.float64)
    uTotal = np.zeros((K, 1), dtype=np.float64)
    comptimes = np.zeros((K, lenvec), dtype=np.float64)
    rewards = np.zeros((K, lenvec), dtype=np.float64)
    steps_taken = np.zeros((K,), dtype=np.int32)
    dones = np.zeros((K, lenvec), dtype=np.int8)

    recurrent = (cfg.algo.upper() == "RNN")
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    local_done = 0
    batch = max(1, int(cfg.progress_update_every))

    for j, global_i in enumerate(idxs):
        global_i = int(global_i)

        if spec.get("seed_vec") is not None:
            ep_seed = int(spec["seed_vec"][global_i])
        else:
            ep_seed = int(cfg.base_seed + global_i)

        obs, _ = env.reset(seed=ep_seed)

        _set_episode(env, spec, global_i)

        if hasattr(env, "scaleObservation"):
            try:
                obs = env.scaleObservation(getattr(env, "x0"))
            except Exception:
                pass

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

            x_store = np.asarray(getattr(env, "x0", np.zeros(state_dim)), dtype=np.float64).reshape(-1)
            if x_store.shape[0] >= state_dim:
                states[j, step, :] = x_store[:state_dim]
            else:
                states[j, step, :x_store.shape[0]] = x_store

            actions[j, step, :] = np.asarray(action, dtype=np.float64).reshape(-1)
            rewards[j, step] = float(reward_temp)

            theta_s_val = None
            for nm in ["theta_s", "thetaS", "theta_s0", "thetaS0"]:
                if hasattr(env, nm):
                    try:
                        theta_s_val = float(getattr(env, nm))
                        break
                    except Exception:
                        pass
            if theta_s_val is None:
                theta_s_val = float(spec["x0s"][global_i, 6])
            theta_s_hist[j, step] = theta_s_val

            u_arr = np.atleast_1d(getattr(env, "u", np.zeros(action_dim))).astype(np.float64)
            uOptmag[j, step] = float(np.linalg.norm(u_arr))

            done = bool(terminated or truncated)
            dones[j, step] = 1 if done else 0

            if recurrent:
                episode_starts[:] = done

            if done:
                step += 1
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
        theta_s_hist=theta_s_hist,
        actions=actions,
        rewards=rewards,
        uOptmag=uOptmag,
        uTotal=uTotal,
        comptimes=comptimes,
        steps_taken=steps_taken,
        dones=dones,
    )
    return chunk_path


def evaluate_parallel(cfg: EvalConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)

    spec_main = _load_episode_bank(cfg.ics_path)
    N = int(spec_main["x0s"].shape[0])

    lenvec = int(np.floor(cfg.TOF / cfg.dt))
    tvec_full = np.arange(0.0, lenvec * cfg.dt, cfg.dt, dtype=np.float64)

    print(f"Loaded episode bank: N={N} from {cfg.ics_path}")
    print(f"Model: {cfg.model_path}")
    print(f"Algo: {cfg.algo}")
    print(f"dt={cfg.dt}, TOF={cfg.TOF} -> lenvec={lenvec}")
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

        pbar = tqdm(total=N, desc="Inspection eval episodes", unit="ep")
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

    # Create a tmp env to get action dim
    env_tmp = InspectionEnv(
        dt=cfg.dt,
        enable_param_randomisation=False,
        enableNoise=False,
        enableCBFtunning=cfg.enableCBFtunning,
        dvWeight=cfg.dvWeight,
    )
    action_dim = int(np.prod(env_tmp.action_space.shape))
    state_dim = 6

    actions = np.zeros((N, lenvec, action_dim), dtype=np.float64)
    states = np.zeros((N, lenvec, state_dim), dtype=np.float64)
    theta_s_hist = np.zeros((N, lenvec), dtype=np.float64)

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
        theta_s_hist[idxs, :] = data["theta_s_hist"]
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
        "theta_s_hist": theta_s_hist,
        "actions": actions,
        "rewards": rewards,
        "uOptmag": uOptmag,
        "uTotal": uTotal,
        "comptimes": comptimes,
        "tvec_full": tvec_full,
        "steps_taken": steps_taken,
        "dones": dones,
        # episode bank parameters
        "x0s": spec_main["x0s"],
        "m_vec": spec_main["m"],
        "R_D_vec": spec_main["R_D"],
        "R_C_vec": spec_main["R_C"],
        "U_MAX_vec": spec_main["U_MAX"],
        "R_MAX_vec": spec_main["R_MAX"],
        "r_vec": spec_main["r"],
        "n_vec": spec_main["n"] if spec_main["n"] is not None else np.array([], dtype=np.float64),
        # provenance
        "model_path": np.array([cfg.model_path], dtype=object),
        "ics_path": np.array([cfg.ics_path], dtype=object),
        "algo": np.array([cfg.algo], dtype=object),
        "dt": np.array([cfg.dt], dtype=np.float64),
        "TOF": np.array([cfg.TOF], dtype=np.float64),
    })
    print("Saved:", out_path)
    return out_path
