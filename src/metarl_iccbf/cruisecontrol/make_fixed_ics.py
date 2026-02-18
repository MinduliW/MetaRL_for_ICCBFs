import numpy as np
import pandas as pd
from pathlib import Path

from cruisecontrol.RLCBF import RLCBFcontrol  # adjust import path if needed


def make_cruise_control_episode_bank(
    dt: float = 0.1,
    N: int = 5000,
    seed: int = 123,
    out_path: str = "../src/data/cruise_control/cruise_control_episode_bank.npz",
    # Paper bounds: p = [m, v0, vmax, umax]
    p_min = (1320.0, 12.501, 21.6, 0.20),
    p_max = (1980.0, 15.279, 26.4, 0.30),
    save_csv: bool = True,
):
    """
    Creates a fixed Monte-Carlo episode bank for cruise control:
      - init_states: (N,2) [d0, v0_follow]
      - meta_params: (N,4) [m, v0_lead, vmax, umax]
      - meta_param_names: (4,)
      - seed_vec: (N,)
    Saves a single compressed NPZ to out_path.
    Optionally also saves CSVs next to it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ----- 1) Initial states from your env pool -----
    env = RLCBFcontrol(dt=dt, deterministic=False)

    points = env.getValidPoints()  # (M,2) expected
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"getValidPoints() must return shape (M,2). Got {points.shape}")

    M = points.shape[0]
    print(f"Candidate initial states available: {M}")

    if M >= N:
        idx = rng.choice(M, size=N, replace=False)
    else:
        print(f"Warning: only {M} points available; sampling with replacement to reach {N}.")
        idx = rng.choice(M, size=N, replace=True)

    init_states = points[idx].astype(float)  # (N,2)

    # ----- 2) Meta-parameters sampled uniformly in paper bounds -----
    p_min = np.array(p_min, dtype=float)
    p_max = np.array(p_max, dtype=float)
    if p_min.shape != (4,) or p_max.shape != (4,):
        raise ValueError("p_min and p_max must be length-4: [m, v0, vmax, umax]")

    meta_params = rng.uniform(low=p_min, high=p_max, size=(N, 4)).astype(float)
    meta_param_names = np.array(["m", "v0", "vmax", "umax"], dtype=object)

    # Per-episode seed vector (useful for reproducibility/debug)
    seed_vec = rng.integers(low=0, high=2**31-1, size=(N,), dtype=np.int64)

    # ----- 3) Save NPZ (single canonical artefact) -----
    np.savez_compressed(
        out_path,
        init_states=init_states,
        meta_params=meta_params,
        meta_param_names=meta_param_names,
        seed_vec=seed_vec,
        dt=float(dt),
        N=int(N),
        bank_seed=int(seed),
        p_min=p_min,
        p_max=p_max,
    )

    print("Saved episode bank:", out_path)

    # ----- 4) Optional CSVs for human inspection -----
    if save_csv:
        base = out_path.with_suffix("")  # remove .npz
        df_x0 = pd.DataFrame(init_states, columns=["d0", "v0_follow"])
        df_p  = pd.DataFrame(meta_params, columns=meta_param_names.tolist())
        df_x0.to_csv(str(base) + "_init_states.csv", index=False)
        df_p.to_csv(str(base) + "_meta_params.csv", index=False)
        print("Saved CSVs:")
        print(" -", str(base) + "_init_states.csv")
        print(" -", str(base) + "_meta_params.csv")

    return init_states, meta_params


if __name__ == "__main__":
    make_cruise_control_episode_bank()
