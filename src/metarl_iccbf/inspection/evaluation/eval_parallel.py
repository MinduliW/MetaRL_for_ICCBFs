"""Parallel evaluation for inspection policies (MLP, RNN, MAMBA).

Uses multiprocessing with chunked episode batches. Each worker loads the
model and environment once, then evaluates a sequence of episodes from a
pre-generated episode bank (``.npz``).

Usage::

    from metarl_iccbf.inspection.evaluation.eval_parallel import (
        EvalConfig, evaluate_parallel,
    )

    cfg = EvalConfig(
        ics_path="episode_bank.npz",
        model_path="best_model.zip",
        policy_type="MAMBA",
        device="cuda",
    )
    out_path = evaluate_parallel(cfg)
"""

from __future__ import annotations

import os
import time
import glob
import numpy as np
import multiprocessing as mp
import queue as pyqueue
from dataclasses import dataclass
from typing import Literal, Optional, Any, Dict

from tqdm import tqdm
from scipy.io import savemat

# Avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# stable_baselines3 imports deferred to _init_worker()


# ------------------- Worker globals -------------------
_MODEL = None
_ENV = None
_CFG = None
_PROGRESS_Q = None
_SPEC = None


@dataclass
class EvalConfig:
    """Configuration for parallel inspection evaluation."""
    ics_path: str
    model_path: str

    policy_type: Literal["MLP", "RNN", "GRU", "LSTM", "MAMBA"] = "MLP"
    algo: str = "ppo"

    dt: float = 10.0
    TOF: float = 3.4 * 3600.0  # 12240 seconds

    n_workers: int = 8
    n_chunks: int = 16
    base_seed: int = 12345
    deterministic_policy: bool = True
    device: str = "cpu"

    progress_update_every: int = 1

    out_dir: str = "ResultsEval/inspection"
    out_mat: str = "InspectionEval_parallel.mat"

    # Episode subsetting
    num_episodes: Optional[int] = None  # None = use all episodes in bank
    sample_seed: int = 42

    # Env flags
    enable_param_randomisation: bool = False
    enableNoise: bool = False
    enableCBFtunning: bool = True
    dvWeight: float = 10.0

    adversarial: bool = False

    merge_only: bool = False  # skip workers, just merge existing chunks


def _load_episode_bank(path: str) -> Dict[str, Any]:
    """Load a ``.npz`` episode bank.

    Expected keys: ``x0s`` (N,7), ``m_vec``, ``R_D_vec``, ``R_C_vec``,
    ``U_MAX_vec``, ``R_MAX_vec``, ``r_vec`` (all (N,)).
    Optional: ``n_vec``, ``seed_vec``.
    """
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

    spec["n"] = (
        np.asarray(D["n_vec"], dtype=np.float64).reshape(-1)
        if "n_vec" in files else None
    )
    spec["seed_vec"] = (
        np.asarray(D["seed_vec"], dtype=np.int64).reshape(-1)
        if "seed_vec" in files else None
    )

    N = spec["x0s"].shape[0]
    if spec["x0s"].ndim != 2 or spec["x0s"].shape[1] != 7:
        raise ValueError(f"x0s must be shape (N,7). Got {spec['x0s'].shape}")
    for k in ("m", "R_D", "R_C", "U_MAX", "R_MAX", "r"):
        if spec[k].shape[0] != N:
            raise ValueError(f"Key '{k}' must have length N={N}. Got {spec[k].shape}")

    return spec


def _set_attr(obj: Any, names, value) -> bool:
    """Try setting attribute for any of the candidate names."""
    for n in names:
        if hasattr(obj, n):
            try:
                setattr(obj, n, value)
                return True
            except Exception:
                pass
    return False


def _set_episode(env, spec: dict, i: int):
    """Force episode *i* into the env (state + parameters)."""
    x0s = spec["x0s"][i].copy().astype(np.float64).reshape(-1)

    # Set the full 7-element state [x,y,z,vx,vy,vz,theta_s] so that
    # env.state[6] remains accessible for the sun-direction lookup.
    _set_attr(env, ["state"], x0s[:7] if x0s.shape[0] >= 7 else x0s)
    _set_attr(env, ["x0", "x", "x_state"], x0s[:6])
    _set_attr(env, ["theta_s", "thetaS", "theta_s0", "thetaS0"], float(x0s[6]))

    _set_attr(env, ["m", "m_c", "mass", "mc"], float(spec["m"][i]))
    _set_attr(env, ["R_D", "R_Dp", "R_dep", "Rdeputy", "RD"], float(spec["R_D"][i]))
    _set_attr(env, ["R_C", "R_ch", "Rchief", "RC"], float(spec["R_C"][i]))
    _set_attr(env, ["U_MAX", "u_max", "umax", "uMax"], float(spec["U_MAX"][i]))
    _set_attr(env, ["R_MAX", "r_max", "Rmax"], float(spec["R_MAX"][i]))
    _set_attr(env, ["r", "r_orbit", "rOrbit"], float(spec["r"][i]))
    if spec.get("n") is not None:
        _set_attr(env, ["n", "mean_motion"], float(spec["n"][i]))

    # Rebuild derived modules if env exposes a method
    for fn in ("_rebuild_models", "_rebuild", "rebuild", "reset_modules"):
        if hasattr(env, fn):
            try:
                getattr(env, fn)()
                break
            except Exception:
                pass


def _init_worker(progress_q, cfg: EvalConfig):
    """Runs once per worker process: load model + env + episode bank."""
    global _MODEL, _ENV, _CFG, _PROGRESS_Q, _SPEC
    _PROGRESS_Q = progress_q
    _CFG = cfg

    _SPEC = _load_episode_bank(cfg.ics_path)

    from metarl_iccbf.inspection.inspectionEnvNoisy import InspectionEnv

    from stable_baselines3 import PPO
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        RecurrentPPO = None

    if cfg.algo == "sac":
        from metarl_iccbf.recurrent_cleanrl.sac import RecurrentSAC
        _MODEL = RecurrentSAC.load(cfg.model_path, device=cfg.device)
    elif cfg.policy_type == "MAMBA":
        from metarl_iccbf.recurrent_cleanrl.ppo import Mamba2PPO
        _MODEL = Mamba2PPO.load(cfg.model_path, device=cfg.device)
    elif cfg.policy_type in ("GRU", "LSTM"):
        from metarl_iccbf.recurrent_cleanrl.ppo import RecurrentPPO as CustomRNNPPO
        _MODEL = CustomRNNPPO.load(cfg.model_path, device=cfg.device)
    elif cfg.policy_type == "RNN":
        if RecurrentPPO is None:
            raise ImportError("sb3-contrib is required for RecurrentPPO.")
        _MODEL = RecurrentPPO.load(cfg.model_path, device=cfg.device)
    else:
        _MODEL = PPO.load(cfg.model_path, device=cfg.device)

    _ENV = InspectionEnv(
        dt=cfg.dt,
        enable_param_randomisation=cfg.enable_param_randomisation,
        enableNoise=cfg.enableNoise,
        enableCBFtunning=cfg.enableCBFtunning,
        dvWeight=cfg.dvWeight,
        adversarial=cfg.adversarial,
    )


def _evaluate_chunk(args):
    """Evaluate a chunk of episodes in a single worker."""
    idxs, chunk_path = args
    cfg = _CFG
    env = _ENV
    model = _MODEL
    spec = _SPEC

    lenvec = int(np.floor(cfg.TOF / cfg.dt))
    K = idxs.shape[0]

    action_dim = int(np.prod(env.action_space.shape))
    state_dim = 6  # store [r;v], theta_s saved separately

    actions = np.zeros((K, lenvec, action_dim), dtype=np.float64)
    states = np.zeros((K, lenvec, state_dim), dtype=np.float64)
    theta_s_hist = np.zeros((K, lenvec), dtype=np.float64)
    u_safe = np.zeros((K, lenvec, 3), dtype=np.float64)
    uOptmag = np.zeros((K, lenvec), dtype=np.float64)
    uTotal = np.zeros((K, 1), dtype=np.float64)
    comptimes = np.zeros((K, lenvec), dtype=np.float64)
    rewards = np.zeros((K, lenvec), dtype=np.float64)
    steps_taken = np.zeros((K,), dtype=np.int32)
    dones = np.zeros((K, lenvec), dtype=np.int8)
    num_inspected = np.zeros((K, lenvec), dtype=np.float64)
    h_sun_hist = np.zeros((K, lenvec), dtype=np.float64)

    sac = (cfg.algo == "sac")
    recurrent = (cfg.policy_type == "RNN") and not sac
    mamba_or_gru = (cfg.policy_type in ("MAMBA", "GRU"))
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    batch = max(1, int(cfg.progress_update_every))
    local_done = 0

    for j, global_i in enumerate(idxs):
        global_i = int(global_i)

        ep_seed = (
            int(spec["seed_vec"][global_i])
            if spec.get("seed_vec") is not None
            else int(cfg.base_seed + global_i)
        )

        obs, _ = env.reset(seed=ep_seed)
        _set_episode(env, spec, global_i)

        # Recompute inspected mask and observation for the new state/params
        env.inspected = np.zeros(env.inspected.shape, dtype=bool)
        obs_dict = env.obs_model.get_observation(env.state)
        env.inspected |= obs_dict["visible_mask"]
        obs = env._get_obs_vector()

        if recurrent:
            lstm_states = None
            episode_starts[:] = True
        episode_start = True

        step = 0
        for step in range(lenvec):
            if sac or mamba_or_gru:
                action, _ = model.predict(
                    obs,
                    deterministic=cfg.deterministic_policy,
                    episode_start=episode_start,
                )
            elif recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=cfg.deterministic_policy,
                )
            else:
                action, _ = model.predict(
                    obs, deterministic=cfg.deterministic_policy,
                )

            t0 = time.perf_counter()
            obs, reward_temp, terminated, truncated, _info = env.step(action)
            comptimes[j, step] = time.perf_counter() - t0

            x_store = np.asarray(
                getattr(env, "state", np.zeros(state_dim)), dtype=np.float64
            ).reshape(-1)
            if x_store.shape[0] >= state_dim:
                states[j, step, :] = x_store[:state_dim]
            else:
                states[j, step, : x_store.shape[0]] = x_store

            theta_s_val = float(x_store[6]) if x_store.shape[0] > 6 else 0.0
            theta_s_hist[j, step] = theta_s_val

            actions[j, step, :] = np.asarray(action, dtype=np.float64).reshape(-1)
            rewards[j, step] = float(reward_temp)

            u_arr = np.atleast_1d(
                getattr(env, "last_u_safe", np.zeros(3))
            ).astype(np.float64).reshape(3)
            u_safe[j, step, :] = u_arr
            if step == 0 and j == 0:
                print(f"✓ u_safe collected: {u_arr}, shape will be {u_safe.shape}")
            uOptmag[j, step] = float(np.linalg.norm(u_arr))

            num_inspected[j, step] = float(_info.get("num_inspected", 0))
            h_sun_hist[j, step] = float(_info.get("h_sun", np.nan))

            done = bool(terminated or truncated)
            dones[j, step] = 1 if done else 0

            if recurrent:
                episode_starts[:] = done
            episode_start = False

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

    print(f"Saving chunk with u_safe shape {u_safe.shape}")
    np.savez_compressed(
        chunk_path,
        idxs=idxs,
        states=states,
        theta_s_hist=theta_s_hist,
        actions=actions,
        rewards=rewards,
        u_safe=u_safe,
        uOptmag=uOptmag,
        uTotal=uTotal,
        comptimes=comptimes,
        steps_taken=steps_taken,
        dones=dones,
        num_inspected=num_inspected,
        h_sun_hist=h_sun_hist,
    )
    return chunk_path


def evaluate_parallel(cfg: EvalConfig) -> str:
    """Run parallel evaluation of an inspection policy.

    Returns the path to the output ``.mat`` file.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    spec_main = _load_episode_bank(cfg.ics_path)
    N = int(spec_main["x0s"].shape[0])

    lenvec = int(np.floor(cfg.TOF / cfg.dt))
    tvec_full = np.arange(0.0, lenvec * cfg.dt, cfg.dt, dtype=np.float64)

    print(f"Loaded episode bank: N={N} from {cfg.ics_path}")
    print(f"Model: {cfg.model_path}")
    print(f"Policy: {cfg.policy_type}, device: {cfg.device}")
    print(f"dt={cfg.dt}, TOF={cfg.TOF} -> lenvec={lenvec}")
    print(f"Workers: {cfg.n_workers}, chunks: {cfg.n_chunks}")

    if cfg.num_episodes is not None and cfg.num_episodes < N:
        rng = np.random.default_rng(cfg.sample_seed)
        all_idxs = rng.choice(N, size=cfg.num_episodes, replace=False)
        all_idxs.sort()
        N = cfg.num_episodes
        print(f"Subsampled {N} episodes (seed={cfg.sample_seed})")
    else:
        all_idxs = np.arange(N, dtype=np.int64)
    chunks = np.array_split(all_idxs, max(cfg.n_chunks, cfg.n_workers))

    chunk_dir = os.path.join(cfg.out_dir, "_chunks_tmp")
    os.makedirs(chunk_dir, exist_ok=True)

    if cfg.merge_only:
        chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npz")))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
        print(f"--merge-only: reusing {len(chunk_files)} existing chunks")
    else:
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
            async_results = [
                pool.apply_async(_evaluate_chunk, (item,))
                for item in work_items
            ]

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

    # Merge chunks – infer action_dim from the first chunk to stay consistent
    # with whatever the worker actually saved (avoids env mismatch bugs).
    _first = np.load(chunk_files[0], allow_pickle=False)
    action_dim = _first["actions"].shape[-1]
    state_dim = 6

    actions_all = np.zeros((N, lenvec, action_dim), dtype=np.float64)
    states_all = np.zeros((N, lenvec, state_dim), dtype=np.float64)
    theta_s_all = np.zeros((N, lenvec), dtype=np.float64)
    rewards_all = np.zeros((N, lenvec), dtype=np.float64)
    u_safe_all = np.zeros((N, lenvec, 3), dtype=np.float64)
    uOptmag_all = np.zeros((N, lenvec), dtype=np.float64)
    uTotal_all = np.zeros((N, 1), dtype=np.float64)
    comptimes_all = np.zeros((N, lenvec), dtype=np.float64)
    steps_all = np.zeros((N,), dtype=np.int32)
    dones_all = np.zeros((N, lenvec), dtype=np.int8)
    num_inspected_all = np.zeros((N, lenvec), dtype=np.float64)
    h_sun_all = np.zeros((N, lenvec), dtype=np.float64)

    for cf in chunk_files:
        data = np.load(cf, allow_pickle=False)
        idxs = data["idxs"]
        states_all[idxs] = data["states"]
        theta_s_all[idxs] = data["theta_s_hist"]
        actions_all[idxs] = data["actions"]
        rewards_all[idxs] = data["rewards"]
        if "u_safe" in data:
            u_safe_all[idxs] = data["u_safe"]
        uOptmag_all[idxs] = data["uOptmag"]
        uTotal_all[idxs] = data["uTotal"]
        comptimes_all[idxs] = data["comptimes"]
        steps_all[idxs] = data["steps_taken"]
        dones_all[idxs] = data["dones"]
        num_inspected_all[idxs] = data["num_inspected"]
        h_sun_all[idxs] = data["h_sun_hist"]

    print(f"Completed: {N} episodes")
    print(f"u_safe_all shape: {u_safe_all.shape}")
    print(f"uTotal mean: {float(np.mean(uTotal_all)):.3f}")
    print(f"uTotal q25/q50/q75: {np.percentile(uTotal_all, [25, 50, 75]).tolist()}")

    out_path = os.path.join(cfg.out_dir, cfg.out_mat)
    savemat(out_path, {
        "states": states_all,
        "theta_s_hist": theta_s_all,
        "actions": actions_all,
        "rewards": rewards_all,
        "u_safe": u_safe_all,
        "uOptmag": uOptmag_all,
        "uTotal": uTotal_all,
        "comptimes": comptimes_all,
        "tvec_full": tvec_full,
        "steps_taken": steps_all,
        "dones": dones_all,
        "num_inspected": num_inspected_all,
        "h_sun": h_sun_all,
        # Episode bank parameters
        "x0s": spec_main["x0s"],
        "m_vec": spec_main["m"],
        "R_D_vec": spec_main["R_D"],
        "R_C_vec": spec_main["R_C"],
        "U_MAX_vec": spec_main["U_MAX"],
        "R_MAX_vec": spec_main["R_MAX"],
        "r_vec": spec_main["r"],
        "n_vec": (
            spec_main["n"]
            if spec_main["n"] is not None
            else np.array([], dtype=np.float64)
        ),
        # Provenance
        "model_path": np.array([cfg.model_path], dtype=object),
        "ics_path": np.array([cfg.ics_path], dtype=object),
        "policy_type": np.array([cfg.policy_type], dtype=object),
        "dt": np.array([cfg.dt], dtype=np.float64),
        "TOF": np.array([cfg.TOF], dtype=np.float64),
    })
    print(f"Saved: {out_path}")
    return out_path
