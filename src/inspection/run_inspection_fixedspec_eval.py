#!/usr/bin/env python3
"""
run_inspection_fixedspec_eval_simple.py

Purpose
-------
Evaluate a trained SB3 policy (PPO or RecurrentPPO) on a FIXED episode bank (.npz)
for your InspectionEnv, and save a MATLAB .mat file with trajectories + diagnostics.

Episode bank (.npz) must contain:
  - init_states:      (N, 7)  [x,y,z,vx,vy,vz,theta_s]
  - meta_params:      (N, P)
  - meta_param_names: (P,)
Optional:
  - seed_vec:         (N,)

What this script does (high-level)
----------------------------------
For each episode i:
  1) Reset env (only for housekeeping / RNG).
  2) Apply meta parameters from the spec (so reset randomisation does not matter).
  3) Force the env state to init_states[i] and initialise inspection bookkeeping.
  4) Roll out the trained policy for up to max_steps.
  5) Save everything to .mat.

Key design choice
-----------------
The env is constructed via `make_env_from_cfg(cfg)` in BOTH:
  - worker processes
  - main process (for dimension inference + allocation)

This prevents action/observation dimension mismatches when enableCBFtunning toggles
between 3D and 12D action spaces.
"""

import os
import time
import numpy as np
import multiprocessing as mp
import queue as pyqueue
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from tqdm import tqdm
from scipy.io import savemat

from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

# CHANGE THIS if your env lives elsewhere
from inspectionEnvNoisy import InspectionEnv


# Reduce BLAS oversubscription when using multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class EvalConfig:
    spec_path: str
    model_path: str
    algo: str  # "PPO" or "RNN"

    dt: float = 10.0
    max_steps: int = 1224

    n_workers: int = 8
    n_chunks: int = 32

    deterministic_policy: bool = True

    use_actuation_noise: bool = True
    use_state_meas_noise: bool = True

    out_mat: str = "Inspection_FIXEDSPEC.mat"

    # how often each worker reports progress (episodes)
    progress_update_every: int = 1

    # --- Env construction flags (set once in main) ---
    enable_param_randomisation: bool = False
    enableNoise: bool = False
    enableCBFtunning: bool = False
    dvWeight: float = 10.0


# ---------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------
def load_spec(spec_path: str) -> Dict[str, Any]:
    """
    Load the fixed episode spec (.npz) and normalise types.
    """
    raw = dict(np.load(spec_path, allow_pickle=True))

    required = ["init_states", "meta_params", "meta_param_names"]
    for k in required:
        if k not in raw:
            raise RuntimeError(f"Spec missing '{k}'. Found keys: {list(raw.keys())}")

    spec: Dict[str, Any] = {}
    spec["init_states"] = np.asarray(raw["init_states"], dtype=np.float64)
    spec["meta_params"] = np.asarray(raw["meta_params"], dtype=np.float64)

    # normalise meta names to strings
    names = []
    for n in list(raw["meta_param_names"]):
        if isinstance(n, (bytes, bytearray)):
            names.append(n.decode("utf-8"))
        else:
            names.append(str(n))
    spec["meta_param_names"] = np.array(names, dtype=object)

    if "seed_vec" in raw:
        spec["seed_vec"] = np.asarray(raw["seed_vec"], dtype=np.int64)

    return spec


def meta_row_to_dict(meta_row: np.ndarray, meta_names: np.ndarray) -> Dict[str, float]:
    """
    Convert one row of meta_params into a dict: name -> value.
    """
    out: Dict[str, float] = {}
    for j, name in enumerate(meta_names):
        out[str(name)] = float(meta_row[j])
    return out


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_model(model_path: str, algo: str):
    """
    Returns:
      model, is_recurrent
    """
    a = algo.upper().strip()

    if a == "PPO":
        return PPO.load(model_path, device="cpu"), False

    if a in ("RNN", "RECURRENT", "RECURRENTPPO"):
        if RecurrentPPO is None:
            raise RuntimeError("RecurrentPPO not available. Install sb3-contrib.")
        return RecurrentPPO.load(model_path, device="cpu"), True

    raise ValueError(f"Unknown algo '{algo}'. Use 'PPO' or 'RNN'.")


# ---------------------------------------------------------------------
# Env construction (single source of truth)
# ---------------------------------------------------------------------
def make_env_from_cfg(cfg: EvalConfig) -> InspectionEnv:
    """
    Create an InspectionEnv with flags controlled from the main config.
    Use this everywhere (workers + main) to avoid action/obs dimension mismatches.
    """
    env = InspectionEnv(
        dt=cfg.dt,
        enable_param_randomisation=cfg.enable_param_randomisation,
        enableNoise=cfg.enableNoise,
        enableCBFtunning=cfg.enableCBFtunning,
        dvWeight = cfg.dvWeight
        
    )

    # Ensure eval noise flags match cfg (if your env exposes them)
    if hasattr(env, "enable_actuation_noise"):
        env.enable_actuation_noise = bool(cfg.use_actuation_noise)
    if hasattr(env, "enable_state_meas_noise"):
        env.enable_state_meas_noise = bool(cfg.use_state_meas_noise)

    return env


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------
def apply_episode_params(env: InspectionEnv, meta: Dict[str, float]) -> None:
    """
    Apply the episode meta-parameters to the env and rebuild anything needed.
    This is how we avoid relying on reset() randomisation.
    """
    if "m" in meta:
        env.m = float(meta["m"])
    if "R_D" in meta:
        env.R_D = float(meta["R_D"])
    if "R_C" in meta:
        env.R_C = float(meta["R_C"])
    if "U_MAX" in meta:
        env.U_MAX = float(meta["U_MAX"])
    if "R_MAX" in meta:
        env.R_MAX = float(meta["R_MAX"])

    # n is derived from r in your reset logic
    if "r" in meta:
        env.r = float(meta["r"])
        env.n = float(np.sqrt(env.mu / (env.r ** 3)))

    # Update ICCBF DA params if the object exists
    if hasattr(env, "iccbf_da") and env.iccbf_da is not None:
        try:
            env.iccbf_da.m = float(env.m)
            env.iccbf_da.n = float(env.n)
            env.iccbf_da.rho_koz = float(env.R_C + env.R_D)
            env.iccbf_da.rho_kiz = float(env.R_MAX)
            env.iccbf_da.alpha_fov = float(env.alpha_fov)
            # some versions use u_max_axis / u_max; keep best-effort
            if hasattr(env.iccbf_da, "u_max"):
                env.iccbf_da.u_max = float(env.U_MAX)
            if hasattr(env.iccbf_da, "u_max_axis"):
                env.iccbf_da.u_max_axis = float(env.U_MAX)
        except Exception:
            pass

    # Rebuild any internal derived quantities / models
    if hasattr(env, "_recompute_derived"):
        env._recompute_derived()
    if hasattr(env, "_rebuild_models"):
        env._rebuild_models()


def set_fixed_initial_condition(env: InspectionEnv, x0: np.ndarray):
    """
    Force the env initial state to x0 and initialise inspection bookkeeping
    similarly to reset().

    Returns:
      obs (what the policy sees at time 0)
    """
    env.state = np.asarray(x0, dtype=np.float64).copy()
    env.steps_done = 0

    # reset inspected flags
    env.inspected = np.zeros(env.N_POINTS, dtype=bool)

    # At t=0, mark any visible points as inspected (same as your reset behaviour)
    if hasattr(env, "_noisy_measurement_state"):
        x_meas = env._noisy_measurement_state(env.state, env.np_random)
    else:
        x_meas = env.state

    obs_dict = env.obs_model.get_observation(x_meas)
    env.inspected |= obs_dict["visible_mask"]

    # clear any "last step" buffers if present
    if hasattr(env, "last_cbfs"):
        env.last_cbfs[:] = 0.0
    if hasattr(env, "last_u_rl"):
        env.last_u_rl[:] = 0.0
    if hasattr(env, "last_u_safe"):
        env.last_u_safe[:] = 0.0

    # build observation vector
    if hasattr(env, "_get_obs_vector"):
        obs = env._get_obs_vector()
    else:
        obs = np.asarray(x_meas, dtype=np.float32)

    return obs


# ---------------------------------------------------------------------
# Multiprocessing progress bar support
# ---------------------------------------------------------------------
_PROGRESS_Q = None


def init_worker(progress_q):
    global _PROGRESS_Q
    _PROGRESS_Q = progress_q


# ---------------------------------------------------------------------
# Worker function: evaluate a chunk of episodes
# ---------------------------------------------------------------------
def evaluate_chunk(args: Tuple[np.ndarray, EvalConfig]) -> Dict[str, Any]:
    """
    Runs evaluation for a subset of episode indices.
    Each worker creates its own env + loads its own model.
    """
    idxs, cfg = args

    spec = load_spec(cfg.spec_path)
    X0 = spec["init_states"]
    meta_params = spec["meta_params"]
    meta_names = spec["meta_param_names"]
    seed_vec = spec.get("seed_vec", None)

    model, is_recurrent = load_model(cfg.model_path, cfg.algo)

    # Construct env consistently with cfg
    env = make_env_from_cfg(cfg)

    # Dimensions
    T = int(cfg.max_steps)
    K = int(idxs.shape[0])

    act_dim = int(np.prod(env.action_space.shape))
    obs_dim = int(env.numObs) if hasattr(env, "numObs") else int(np.prod(env.observation_space.shape))
    n_points = int(env.N_POINTS)

    # Allocate outputs for this chunk
    states = np.zeros((K, T, 7), dtype=np.float64)
    observations = np.zeros((K, T, obs_dim), dtype=np.float64)
    actions = np.zeros((K, T, act_dim), dtype=np.float64)
    rewards = np.zeros((K, T, 1), dtype=np.float64)

    u_rl_hist = np.zeros((K, T, 3), dtype=np.float64)
    u_safe_hist = np.zeros((K, T, 3), dtype=np.float64)

    num_inspected = np.zeros((K, T, 1), dtype=np.int32)
    newly_inspected = np.zeros((K, T, 1), dtype=np.int32)

    r_norm_hist = np.zeros((K, T, 1), dtype=np.float64)
    h_sun_hist = np.zeros((K, T, 1), dtype=np.float64)
    qp_solved_hist = np.zeros((K, T, 1), dtype=np.int8)

    terminated = np.zeros((K, 1), dtype=np.int8)
    truncated = np.zeros((K, 1), dtype=np.int8)
    steps_taken = np.zeros((K,), dtype=np.int32)
    success_all_inspected = np.zeros((K, 1), dtype=np.int8)

    comptimes = np.zeros((K, T), dtype=np.float64)

    # Progress reporting
    batch = max(1, int(cfg.progress_update_every))
    local_done = 0

    for j, global_i in enumerate(idxs):
        # Decide per-episode seed (for deterministic noise sequences if you want that)
        if seed_vec is not None:
            ep_seed = int(seed_vec[global_i])
        else:
            ep_seed = int(12345 + int(global_i))

        # Reset for housekeeping / env RNG
        env.reset(seed=ep_seed)

        # Apply meta params from spec
        meta = meta_row_to_dict(meta_params[global_i], meta_names)
        apply_episode_params(env, meta)

        # Force initial state
        obs = set_fixed_initial_condition(env, X0[global_i])

        # Recurrent state init
        if is_recurrent:
            lstm_states = None
            episode_starts = np.array([True], dtype=bool)

        done = False
        step = 0

        while (not done) and step < T:
            # Save current obs/state
            observations[j, step, :] = np.asarray(obs, dtype=np.float64).reshape(-1)
            states[j, step, :] = np.asarray(env.state, dtype=np.float64).reshape(7,)

            # Query policy action
            t0 = time.perf_counter()
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=cfg.deterministic_policy,
                )
                episode_starts = np.array([False], dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=cfg.deterministic_policy)
            comptimes[j, step] = time.perf_counter() - t0

            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.size != act_dim:
                raise RuntimeError(f"Policy action dim {action.size} != env act_dim {act_dim}")
            actions[j, step, :] = action

            # Step environment
            obs, r, term, trunc, info = env.step(action)

            rewards[j, step, 0] = float(r)

            # Controls
            if "u_rl" in info:
                u_rl_hist[j, step, :] = np.asarray(info["u_rl"], dtype=np.float64).reshape(3,)
            elif hasattr(env, "last_u_rl"):
                u_rl_hist[j, step, :] = np.asarray(env.last_u_rl, dtype=np.float64).reshape(3,)

            if "u_safe" in info:
                u_safe_hist[j, step, :] = np.asarray(info["u_safe"], dtype=np.float64).reshape(3,)
            elif hasattr(env, "last_u_safe"):
                u_safe_hist[j, step, :] = np.asarray(env.last_u_safe, dtype=np.float64).reshape(3,)

            # Inspection progress
            if "num_inspected" in info:
                num_inspected[j, step, 0] = int(info["num_inspected"])
            else:
                num_inspected[j, step, 0] = int(np.sum(getattr(env, "inspected",
                                                             np.zeros(n_points, dtype=bool))))

            if "newly_inspected" in info:
                newly_inspected[j, step, 0] = int(info["newly_inspected"])

            # Diagnostics (if present)
            if "r_norm" in info:
                r_norm_hist[j, step, 0] = float(info["r_norm"])
            if "h_sun" in info:
                h_sun_hist[j, step, 0] = float(info["h_sun"])
            if "qp_solved" in info:
                qp_solved_hist[j, step, 0] = 1 if bool(info["qp_solved"]) else 0

            step += 1
            done = bool(term) or bool(trunc)

        steps_taken[j] = step
        terminated[j, 0] = 1 if bool(term) else 0
        truncated[j, 0] = 1 if bool(trunc) else 0

        # Success: inspected all points at end
        try:
            success_all_inspected[j, 0] = 1 if bool(env.inspected.all()) else 0
        except Exception:
            success_all_inspected[j, 0] = 0

        # Progress update
        local_done += 1
        if _PROGRESS_Q is not None and (local_done % batch == 0):
            _PROGRESS_Q.put(batch)

    # Flush any remaining progress
    rem = local_done % batch
    if _PROGRESS_Q is not None and rem != 0:
        _PROGRESS_Q.put(rem)

    return {
        "idxs": idxs,
        "states": states,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "u_rl": u_rl_hist,
        "u_safe": u_safe_hist,
        "num_inspected": num_inspected,
        "newly_inspected": newly_inspected,
        "r_norm": r_norm_hist,
        "h_sun": h_sun_hist,
        "qp_solved": qp_solved_hist,
        "terminated": terminated,
        "truncated": truncated,
        "steps_taken": steps_taken,
        "success_all_inspected": success_all_inspected,
        "comptimes": comptimes,
    }


# ---------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------
def run_eval(cfg: EvalConfig) -> None:
    spec = load_spec(cfg.spec_path)
    N = int(spec["init_states"].shape[0])
    meta_names = spec["meta_param_names"]

    print("Loaded spec:", cfg.spec_path)
    print("Episodes:", N)
    print("Meta names:", list(meta_names))
    print("Model:", cfg.model_path)
    print("Algo:", cfg.algo)
    print("dt:", cfg.dt, "max_steps:", cfg.max_steps)
    print("workers:", cfg.n_workers, "chunks:", cfg.n_chunks)
    print("noise actuation:", cfg.use_actuation_noise, "meas:", cfg.use_state_meas_noise)
    print("Env flags:",
          "param_rand=", cfg.enable_param_randomisation,
          "Noise=", cfg.enableNoise,
          "CBFtune=", cfg.enableCBFtunning)
    print("Saving ->", cfg.out_mat)

    # Split episode indices into chunks for multiprocessing
    all_idxs = np.arange(N, dtype=np.int64)
    chunks = np.array_split(all_idxs, max(cfg.n_chunks, cfg.n_workers))
    work_items = [(c, cfg) for c in chunks if c.size > 0]

    # Start multiprocessing pool with a progress queue
    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()

    with ctx.Pool(processes=cfg.n_workers, initializer=init_worker, initargs=(progress_q,)) as pool:
        async_results = [pool.apply_async(evaluate_chunk, (item,)) for item in work_items]

        pbar = tqdm(total=N, desc="Inspection MC episodes", unit="ep")
        try:
            while True:
                try:
                    inc = progress_q.get(timeout=0.2)
                    pbar.update(int(inc))
                except pyqueue.Empty:
                    pass

                if all(r.ready() for r in async_results):
                    break

            results = [r.get() for r in async_results]
        finally:
            pbar.close()

    # Build global arrays (N episodes)
    T = int(cfg.max_steps)

    # Create a temp env in main with the SAME flags to get dims
    tmp_env = make_env_from_cfg(cfg)
    act_dim = int(np.prod(tmp_env.action_space.shape))
    obs_dim = int(tmp_env.numObs) if hasattr(tmp_env, "numObs") else int(np.prod(tmp_env.observation_space.shape))

    # Defensive check against regressions
    if len(results) > 0:
        a_dim_worker = int(results[0]["actions"].shape[2])
        o_dim_worker = int(results[0]["observations"].shape[2])
        if a_dim_worker != act_dim:
            raise RuntimeError(f"Act dim mismatch: main={act_dim} vs worker={a_dim_worker}")
        if o_dim_worker != obs_dim:
            raise RuntimeError(f"Obs dim mismatch: main={obs_dim} vs worker={o_dim_worker}")

    states = np.zeros((N, T, 7), dtype=np.float64)
    observations = np.zeros((N, T, obs_dim), dtype=np.float64)
    actions = np.zeros((N, T, act_dim), dtype=np.float64)
    rewards = np.zeros((N, T, 1), dtype=np.float64)

    u_rl = np.zeros((N, T, 3), dtype=np.float64)
    u_safe = np.zeros((N, T, 3), dtype=np.float64)

    num_inspected = np.zeros((N, T, 1), dtype=np.int32)
    newly_inspected = np.zeros((N, T, 1), dtype=np.int32)

    r_norm = np.zeros((N, T, 1), dtype=np.float64)
    h_sun = np.zeros((N, T, 1), dtype=np.float64)
    qp_solved = np.zeros((N, T, 1), dtype=np.int8)

    terminated = np.zeros((N, 1), dtype=np.int8)
    truncated = np.zeros((N, 1), dtype=np.int8)
    steps_taken = np.zeros((N,), dtype=np.int32)
    success_all_inspected = np.zeros((N, 1), dtype=np.int8)

    comptimes = np.zeros((N, T), dtype=np.float64)

    # Copy each chunk result into the correct episode indices
    for r in results:
        idxs = r["idxs"]
        states[idxs, :, :] = r["states"]
        observations[idxs, :, :] = r["observations"]
        actions[idxs, :, :] = r["actions"]
        rewards[idxs, :, :] = r["rewards"]
        u_rl[idxs, :, :] = r["u_rl"]
        u_safe[idxs, :, :] = r["u_safe"]
        num_inspected[idxs, :, :] = r["num_inspected"]
        newly_inspected[idxs, :, :] = r["newly_inspected"]
        r_norm[idxs, :, :] = r["r_norm"]
        h_sun[idxs, :, :] = r["h_sun"]
        qp_solved[idxs, :, :] = r["qp_solved"]
        terminated[idxs, :] = r["terminated"]
        truncated[idxs, :] = r["truncated"]
        steps_taken[idxs] = r["steps_taken"]
        success_all_inspected[idxs, :] = r["success_all_inspected"]
        comptimes[idxs, :] = r["comptimes"]

    # Simple summary stats
    total_reward = rewards.sum(axis=1).reshape(-1)

    last_idx = np.maximum(steps_taken - 1, 0).reshape(-1, 1)
    inspected_final = np.take_along_axis(num_inspected[:, :, 0], last_idx, axis=1).reshape(-1)

    print("Done.")
    print("Total reward mean/median:", float(np.mean(total_reward)), float(np.median(total_reward)))
    print("Final inspected mean/median:", float(np.mean(inspected_final)), float(np.median(inspected_final)))
    print("Success (all inspected):", int(success_all_inspected.sum()), "/", N)

    # Save to .mat
    savemat(cfg.out_mat, {
        # Rollout data
        "states": states,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "u_rl": u_rl,
        "u_safe": u_safe,
        "num_inspected": num_inspected,
        "newly_inspected": newly_inspected,
        "r_norm": r_norm,
        "h_sun": h_sun,
        "qp_solved": qp_solved,
        "comptimes": comptimes,

        # Episode bookkeeping
        "terminated": terminated,
        "truncated": truncated,
        "steps_taken": steps_taken,
        "success_all_inspected": success_all_inspected,

        # Copy spec into the .mat for provenance
        "init_states": spec["init_states"],
        "meta_params": spec["meta_params"],
        "meta_param_names": spec["meta_param_names"],
        "seed_vec": spec.get("seed_vec", np.arange(N, dtype=np.int64)),

        # Metadata
        "model_path": np.array([cfg.model_path], dtype=object),
        "algo": np.array([cfg.algo], dtype=object),
        "dt": np.array([cfg.dt]),
        "max_steps": np.array([cfg.max_steps], dtype=np.int32),
        "use_actuation_noise": np.array([int(cfg.use_actuation_noise)], dtype=np.int8),
        "use_state_meas_noise": np.array([int(cfg.use_state_meas_noise)], dtype=np.int8),

        # Env flags for provenance
        "enable_param_randomisation": np.array([int(cfg.enable_param_randomisation)], dtype=np.int8),
        "enableNoise": np.array([int(cfg.enableNoise)], dtype=np.int8),
        "enableCBFtunning": np.array([int(cfg.enableCBFtunning)], dtype=np.int8),
    })

    print("Saved:", cfg.out_mat)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = EvalConfig(
        spec_path="N500.npz",

        # model_path="TrainedModels/uRLCBFNN_l4_n256_lr0.0001D_std0.2_ne10_ns6.25_entropy0.01_dvW10.0/best_model",
        # algo="PPO",
        
        model_path="TrainedModels/NoisyInspectionRecurrentRNN_l4_n256_lstm1_lh256_lr5e-05D_std0.2_ne10_ns6.25_entropy0.01_dvW10.0/best_model",
        algo="RNN",
        
        # model_path="TrainedModels/uRLonlyNN_l2_n256_lr0.0001D_std0.2_ne10_ns6.25_entropy0.01/best_model",
        # algo="PPO",
             
             
        dt=10.0,
        max_steps=1224,
        n_workers=8,
        n_chunks=32,
        deterministic_policy=True,
        use_actuation_noise=False,
        use_state_meas_noise=False,
        

        # --- Set env flags ONCE here ---
        enable_param_randomisation=True,  # recommended for fixed-spec eval
        enableNoise=True,
        enableCBFtunning=True,             # True => 12D action space
        dvWeight = 10.0,
        
        # concatanate dvWeight and algo and model_path to out_mat
        out_mat = f"Inspection_RNNRL_FIXEDSPECdvW10.0.mat",
      
    )

    run_eval(cfg)



# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
# if __name__ == "__main__":
#     cfg = EvalConfig(
#         spec_path="N100.npz",
        
#         # model_path="TrainedModels/uRLonlyNN_l2_n256_lr0.0001D_std0.2_ne10_ns6.25_entropy0.01/best_model",
#         # algo="PPO",
        
        
        
#         model_path="TrainedModels/uRLCBFNN_l2_n256_lr0.0001D_std0.2_ne10_ns6.25_entropy0.01/best_model",
#         algo="PPO",
        
#         # model_path="TrainedModels/NoisyInspectionRecurrentRNN_l2_n256_lstm1_lh256_lr0.0002D_std0.2_ne10_ns6.25_entropy0.01/best_model",
#         # algo="RNN",
        
#         dt=10.0,
#         max_steps=1224,
#         n_workers=8,
#         n_chunks=32,
#         deterministic_policy=True,
#         use_actuation_noise=False,
#         use_state_meas_noise=False,
#         out_mat="Inspection_NNRL_FIXEDSPEC.mat",
#     )

#     run_eval(cfg)
