import os
import time
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import queue as pyqueue

# Limit oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from tqdm import tqdm
from scipy.io import savemat

# Your environment + dependencies (must import cleanly)
from RLCBF import RLCBFcontrol
from Dockingcase import DockingCase
from dynamicsandControl import dynamicsAndControl
from iccbfs import ICCBF

from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

_PROGRESS_Q = None

def _init_worker(progress_q):
    global _PROGRESS_Q
    _PROGRESS_Q = progress_q

@dataclass
class EvalConfig:
    spec_path: str
    model_path: str
    algo: str                 # "PPO" or "RNN"

    dt: float = 0.5
    TOF: float = 50.0

    n_workers: int = 8
    n_chunks: int = 32
    deterministic_policy: bool = True

    # Evaluation noise handling
    use_noise: bool = True  

    progress_update_every: int = 1

    out_mat: str = "Docking_FIXEDSPEC.mat"

    # If your env constructor needs these
    env_nInitstates: int = 1000
    env_setconst: bool = True


def _load_model(model_path: str, algo: str):
    a = algo.upper()
    if a == "PPO":
        return PPO.load(model_path, device="cpu")
    if a in ("RNN", "RECURRENT", "RECURRENTPPO"):
        if RecurrentPPO is None:
            raise RuntimeError("sb3_contrib.RecurrentPPO not available. Install sb3-contrib.")
        return RecurrentPPO.load(model_path, device="cpu")
    raise ValueError(f"Unknown algo={algo}. Use 'PPO' or 'RNN'.")


def _apply_episode_params(env: RLCBFcontrol, m: float, rho: float, om: float, umax: float):
    """
    Override episode parameters WITHOUT touching reset().
    Rebuild the same internal objects reset() would rebuild.
    """
    env.m = float(m)
    env.rho = float(rho)
    env.om = float(om)
    env.umax = float(umax)

    # Rebuild ICCBF / margin function
    env.iccbf = ICCBF(mu=env.mu, r=env.r, gamma=env.gamma, rho=env.rho, m=env.m, om=env.om, umax=env.umax)
    env.getmargin_docking = env.iccbf.getmargin

    # Rebuild dynamics and docking case
    env.dockingCase = DockingCase(rho=env.rho, gamma=env.gamma)
    env.dynamics = dynamicsAndControl(mu=env.mu, n=env.n, r=env.r, m=env.m, om=env.om)

    # If you rely on this
    env.validPoints = env.dockingCase.generate_evenly_spread_cone_points_2d()

    # Update bounds that depend on om (only index 4 used for 5D observation)
    try:
        env.obshigh[4] = env.om * (env.TOF + 0.5)
    except Exception:
        pass


def _evaluate_chunk(args: Tuple[np.ndarray, Dict[str, np.ndarray], EvalConfig]) -> Dict[str, Any]:
    idxs, spec, cfg = args

    model = _load_model(cfg.model_path, cfg.algo)
    is_recurrent = (cfg.algo.upper() != "PPO")

    env = RLCBFcontrol(dt=cfg.dt, deterministic=False, nInitstates=cfg.env_nInitstates, setconst=cfg.env_setconst)
    env.TOF = float(cfg.TOF)
    env.use_noise = bool(cfg.use_noise)

    # time grid matching your original docking script
    tvec_full = np.arange(0.0, cfg.TOF + cfg.dt, cfg.dt)
    T = len(tvec_full)

    K = idxs.shape[0]
    
    cert_valid      = np.zeros((K, T, 1), dtype=np.float64)
    cert_zeta_lb    = np.full((K, T, 1), np.nan, dtype=np.float64)
    cert_base_lb    = np.full((K, T, 1), np.nan, dtype=np.float64)
    cert_base_ub    = np.full((K, T, 1), np.nan, dtype=np.float64)
    cert_support_lb = np.full((K, T, 1), np.nan, dtype=np.float64)
    cert_normLg_lb  = np.full((K, T, 1), np.nan, dtype=np.float64)
    cert_nu         = np.full((K, T, 1), np.nan, dtype=np.float64)
    cert_branch_ok  = np.full((K, T, 1), np.nan, dtype=np.float64)
    cert_branch_m   = np.full((K, T, 1), np.nan, dtype=np.float64)

    # intervals packed as 4 numbers: [Lg1_lb, Lg1_ub, Lg2_lb, Lg2_ub]
    cert_Lg_interval = np.full((K, T, 4), np.nan, dtype=np.float64)

    # branch intervals packed as 4 numbers: [Lgb1_1_lb, Lgb1_1_ub, Lgb1_2_lb, Lgb1_2_ub]
    cert_Lgb1_interval = np.full((K, T, 4), np.nan, dtype=np.float64)


    # Allocate outputs for this chunk
    uOpts = np.zeros((K, T, 2), dtype=np.float64)
    actionStore = np.zeros((K, T, 4), dtype=np.float64)
    rewards_step = np.zeros((K, T, 1), dtype=np.float64)

    # physical state history: store x,y,vx,vy in meters and m/s, phi in rad
    states_phys = np.zeros((K, T, 5), dtype=np.float64)
    obs_hist = np.zeros((K, T, 5), dtype=np.float64)

    hs = np.zeros((K, T, 1), dtype=np.float64)
    Vs = np.zeros((K, T, 1), dtype=np.float64)
    comptimes = np.zeros((K, T), dtype=np.float64)

    uTotal = np.zeros((K, 1), dtype=np.float64)
    steps_taken = np.zeros((K,), dtype=np.int32)

    # ICCBF diagnostics you asked for
    LfhICCBF = np.zeros((K, T, 1), dtype=np.float64)
    LghICCBF = np.zeros((K, T, 2), dtype=np.float64)
    Lgh_norm = np.zeros((K, T, 1), dtype=np.float64)
    nuMargin_hist = np.zeros((K, T, 1), dtype=np.float64)

    # progress batching
    local_done = 0
    batch = max(1, int(cfg.progress_update_every))

    for j, global_i in enumerate(idxs):
        # 1) reset (for housekeeping), then override parameters from spec
        obs, _ = env.reset(seed=int(spec["seed_vec"][global_i]), postProcess=False)

        _apply_episode_params(
            env,
            m=float(spec["m_vec"][global_i]),
            rho=float(spec["rho_vec"][global_i]),
            om=float(spec["om_vec"][global_i]),
            umax=float(spec["umax_vec"][global_i]),
        )

        # 2) set initial state from spec
        env.x0 = spec["x0s"][global_i].astype(np.float64).copy()

        # initial observation (policy input)
        obs = env._get_observation(env.x0) if hasattr(env, "_get_observation") else env.scaleObservation(env.x0)

        if is_recurrent:
            lstm_states = None
            episode_start = np.array([True])

        # roll out one episode
        u_accum = 0.0
        step = 0
        done = False

        while (not done) and step < T:
            # store obs at this step
            obs_hist[j, step, :] = obs

            # policy
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

            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape[0] != 4:
                raise RuntimeError(f"Policy action dim is {action.shape[0]} but env expects 4.")

            actionStore[j, step, :] = action

            # --- ICCBF diagnostics *for this step* (pre-step x0 and mapped coefs) ---
            # Mirror your env.step mapping exactly:
            acoef1 = 2.0 * (action[0] + 1.0) / 2.0   # = action[0] + 1
            acoef2 = 2.0 * (action[1] + 1.0) / 2.0
            hslack = 0.01 + 0.5 * (action[2] + 1.0) * (1.0 - 0.01)
            # Lslack not needed for getmargin

            nuMargin, vals = env.getmargin_docking(env.x0, acoef1, acoef2, hslack, env.tstep)
            Lgh1, Lgh2, Lfh, hI = vals

            LfhICCBF[j, step, 0] = float(Lfh)
            LghICCBF[j, step, 0] = float(Lgh1)
            LghICCBF[j, step, 1] = float(Lgh2)
            Lgh_norm[j, step, 0] = float(np.hypot(Lgh1, Lgh2))
            nuMargin_hist[j, step, 0] = float(nuMargin)

            # step env
            t0 = time.perf_counter()
            obs, r, done, _, info = env.step(action)
            comptimes[j, step] = time.perf_counter() - t0

            rewards_step[j, step, 0] = float(r)
            
            # --- pull packed cert from env and store ---
            c = getattr(env, "_cert_last", None)
            if c is not None:
                cert_valid[j, step, 0]      = float(c.get("valid", 0))
                cert_zeta_lb[j, step, 0]    = float(c.get("zeta_lb", np.nan))
               
             


            # physical state history in meters/m/s (phi left in rad)
            x_phys = env.x0.astype(np.float64).copy()
            x_phys[:4] *= 1e3
            states_phys[j, step, :] = x_phys

            # store control (2D) and accumulate fuel proxy
            u = np.asarray(env.control, dtype=np.float64).reshape(2,)
            uOpts[j, step, :] = u
            u_accum += float(np.linalg.norm(u)) * cfg.dt

            # original CBF h and CLF V
            hs[j, step, 0] = float(env.dockingCase.originalh(env.x0))
            V_raw, _ = env.dockingCase.calculate_V_and_dV(env.x0)
            Vs[j, step, 0] = float(V_raw)

            step += 1

        steps_taken[j] = step
        uTotal[j, 0] = u_accum

        # progress update
        local_done += 1
        if _PROGRESS_Q is not None and (local_done % batch == 0):
            _PROGRESS_Q.put(batch)

    rem = local_done % batch
    if _PROGRESS_Q is not None and rem != 0:
        _PROGRESS_Q.put(rem)

    return dict(
        idxs=idxs,
        cert_valid=cert_valid,
        cert_zeta_lb=cert_zeta_lb,
        uOpts=uOpts,
        actionStore=actionStore,
        rewards_step=rewards_step,
        states_phys=states_phys,
        obs_hist=obs_hist,
        hs=hs,
        Vs=Vs,
        comptimes=comptimes,
        uTotal=uTotal,
        steps_taken=steps_taken,
        LfhICCBF=LfhICCBF,
        LghICCBF=LghICCBF,
        Lgh_norm=Lgh_norm,
        nuMargin=nuMargin_hist,
        tvec_full=tvec_full,
    )


def run_eval(cfg: EvalConfig):
    spec = dict(np.load(cfg.spec_path, allow_pickle=False))
    x0s = spec["x0s"]
    N = x0s.shape[0]

    print(f"Loaded spec: {cfg.spec_path} with N={N}")
    print(f"Model: {cfg.model_path}")
    print(f"Algo: {cfg.algo}")
    print(f"TOF={cfg.TOF}, dt={cfg.dt}, workers={cfg.n_workers}, chunks={cfg.n_chunks}, noise={cfg.use_noise}")

    all_idxs = np.arange(N, dtype=np.int64)
    chunks = np.array_split(all_idxs, max(cfg.n_chunks, cfg.n_workers))
    work_items = [(c, spec, cfg) for c in chunks if c.size > 0]

    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()

    with ctx.Pool(processes=cfg.n_workers, initializer=_init_worker, initargs=(progress_q,)) as pool:
        async_results = [pool.apply_async(_evaluate_chunk, (item,)) for item in work_items]

        pbar = tqdm(total=N, desc="Docking MC episodes", unit="ep")

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

    # Build global arrays
    tvec_full = results[0]["tvec_full"]
    T = len(tvec_full)

    cert_valid      = np.zeros((N, T, 1))
    cert_zeta_lb    = np.full((N, T, 1), np.nan)
    uOpts        = np.zeros((N, T, 2))
    actionStore  = np.zeros((N, T, 4))
    rewards_step = np.zeros((N, T, 1))
    states_phys  = np.zeros((N, T, 5))
    obs_hist     = np.zeros((N, T, 5))
    hs           = np.zeros((N, T, 1))
    Vs           = np.zeros((N, T, 1))
    comptimes    = np.zeros((N, T))
    uTotal       = np.zeros((N, 1))
    steps_taken  = np.zeros((N,), dtype=np.int32)

    LfhICCBF     = np.zeros((N, T, 1))
    LghICCBF     = np.zeros((N, T, 2))
    Lgh_norm     = np.zeros((N, T, 1))
    nuMargin     = np.zeros((N, T, 1))

    for r in results:
        idxs = r["idxs"]
        cert_valid[idxs, :, :]      = r["cert_valid"]
        cert_zeta_lb[idxs, :, :]    = r["cert_zeta_lb"]
        uOpts[idxs, :, :]        = r["uOpts"]
        actionStore[idxs, :, :]  = r["actionStore"]
        rewards_step[idxs, :, :] = r["rewards_step"]
        states_phys[idxs, :, :]  = r["states_phys"]
        obs_hist[idxs, :, :]     = r["obs_hist"]
        hs[idxs, :, :]           = r["hs"]
        Vs[idxs, :, :]           = r["Vs"]
        comptimes[idxs, :]       = r["comptimes"]
        uTotal[idxs, :]          = r["uTotal"]
        steps_taken[idxs]        = r["steps_taken"]

        LfhICCBF[idxs, :, :]     = r["LfhICCBF"]
        LghICCBF[idxs, :, :]     = r["LghICCBF"]
        Lgh_norm[idxs, :, :]     = r["Lgh_norm"]
        nuMargin[idxs, :, :]     = r["nuMargin"]

    # Print quick stats
    print("Done.")
    print("uTotal mean:", float(np.mean(uTotal)))
    print("uTotal median:", float(np.median(uTotal)))
    print("uTotal q25/q75:", np.percentile(uTotal, [25, 75]).tolist())

    # Save FULL mat (the set you demanded, plus extras)
    savemat(cfg.out_mat, {
        # Core
        "uOpts": uOpts,
        "cert_valid": cert_valid,
        "cert_zeta_lb": cert_zeta_lb,
        "states": states_phys,         # physical states (m, m/s, rad)
        "observations": obs_hist,      # policy inputs (scaled)
        "rewards": rewards_step,       # per-step reward
        "hs": hs,
        "Vs": Vs,
        "actionStore": actionStore,
        "uTotal": uTotal,
        "comptimes": comptimes,
        "tvec_full": tvec_full,

        # Episode bookkeeping
        "steps_taken": steps_taken,

        # Fixed episode spec (critical for NN vs RNN comparison)
        "x0s": spec["x0s"],
        "m_vec": spec["m_vec"],
        "rho_vec": spec["rho_vec"],
        "om_vec": spec["om_vec"],
        "umax_vec": spec["umax_vec"],
        "seed_vec": spec["seed_vec"],

        # ICCBF diagnostics you asked for
        "LfhICCBF": LfhICCBF,
        "LghICCBF": LghICCBF,          # (Lgh1, Lgh2)
        "Lgh_norm": Lgh_norm,
        "nuMargin": nuMargin,

        # Metadata
        "model_path": np.array([cfg.model_path], dtype=object),
        "algo": np.array([cfg.algo], dtype=object),
        "dt": np.array([cfg.dt]),
        "TOF": np.array([cfg.TOF]),
        "use_noise": np.array([int(cfg.use_noise)]),
    })

    print("Saved:", cfg.out_mat)


if __name__ == "__main__":
    cfg = EvalConfig(
        spec_path="docking_episode_spec_N5000_seed123.npz",

        # CHANGE THESE:
        
        # model_path="TrainedModels/Test_New_MarginNoisyRotatingDockingCase_RNN_l4_n64_lr5e-05C_std0.2_ne10_ns1.25_entropy0.01/best_model.zip",
        # algo="RNN",  # or "RNN"

        model_path="TrainedModels/TEEEESTNoisyRotatingDockingCase__l4_n64_lr5e-05C_std0.2_ne10_ns1.25entropy0.01/final_model.zip",
        algo="PPO",  # or "RNN"
        
        # model_path="TrainedModels/NEWTest_New_MarginNoisyRotatingDockingCase_RNN_l4_n64_lr5e-05C_std0.2_ne10_ns1.25_entropy0.01/best_model.zip",
          

        dt=0.5,
        TOF=50.0,
        n_workers=8,
        n_chunks=32,
        deterministic_policy=True,
        use_noise=True,

        out_mat="Docking_RNN.mat",

        env_nInitstates=1000,
        env_setconst=False,  # IMPORTANT: True would ignore policy actions
    )
    run_eval(cfg)
