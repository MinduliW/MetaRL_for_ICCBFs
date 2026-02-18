import numpy as np


# -------------------------
# Meta-RL parameter ranges (UNIFORM)
# Tune these bounds as you like.
# Units match your env: m (kg), n (rad/s), radii (m), angles (rad), etc.
# -------------------------
META_RANGES = {
    "m": (10.0, 15.0),                          # kg
    "n": (0.00085, 0.00120),                    # rad/s
    "R_D": (4.0, 6.0),                          # m
    "R_C": (8.0, 12.0),                         # m
    "alpha_fov": (np.deg2rad(45.0), np.deg2rad(75.0)),  # rad (half-angle)
    "R_MAX": (600.0, 1000.0),                   # m
    "V_MAX": (3.0, 7.0),                        # m/s
    "U_MAX": (0.6, 1.4),                        # N
    "v0": (0.10, 0.35),                         # m/s (if you want to randomise it)
    "nu1_factor": (5.0, 10.0),                  # dimensionless
    "w_dv": (0.001, 0.1),                       # reward weight, uniform
    "T_fft": (80.0, 140.0),                     # s
    "dt_fft": (5.0, 20.0),                      # s
    # Integers: sample uniformly over integers in [low, high]
    "N_POINTS": (80, 120),                      # count (int)
}


def sun_direction(theta_s: float) -> np.ndarray:
    """
    Replace this with your ObservationModel.sun_direction(theta_s) convention if different.
    """
    return np.array([np.cos(theta_s), np.sin(theta_s), 0.0], dtype=np.float64)


def sample_initial_state(
    rng: np.random.Generator,
    init_range_min: float = 50.0,
    init_range_max: float = 100.0,
    theta_b_min_deg: float = 40.0,
) -> np.ndarray:
    """
    Mirrors your InspectionEnv._sample_initial_state():
    - range ∈ [50, 100] m
    - az ∈ [0, 2π], el ∈ [-π/2, π/2]
    - v0 = 0
    - theta_s ∈ [0, 2π]
    - if boresight-to-sun angle < 40°, flip r0
    """
    r0_mag = rng.uniform(init_range_min, init_range_max)
    az = rng.uniform(0.0, 2.0 * np.pi)
    el = rng.uniform(-0.5 * np.pi, 0.5 * np.pi)

    r0 = np.array([
        r0_mag * np.cos(el) * np.cos(az),
        r0_mag * np.cos(el) * np.sin(az),
        r0_mag * np.sin(el),
    ], dtype=np.float64)

    v0 = np.zeros(3, dtype=np.float64)
    theta_s0 = rng.uniform(0.0, 2.0 * np.pi)

    s_dir = sun_direction(theta_s0)
    r_norm = np.linalg.norm(r0)
    boresight = -r0 / max(r_norm, 1e-9)  # deputy -> chief

    cos_theta_b = np.clip(float(boresight.dot(s_dir)), -1.0, 1.0)
    theta_b = np.arccos(cos_theta_b)

    if theta_b < np.deg2rad(theta_b_min_deg):
        r0 = -r0

    return np.concatenate([r0, v0, [theta_s0]]).astype(np.float64)


def sample_meta_params_uniform(
    rng: np.random.Generator,
    meta_ranges: dict,
) -> tuple[np.ndarray, list[str]]:
    """
    Samples meta parameters independently, uniformly.
    Returns:
      - vec: (P,) float64 (N_POINTS stored as float but representing int)
      - names: list[str] length P (same order as vec)
    """
    names = []
    vals = []

    for k, (lo, hi) in meta_ranges.items():
        names.append(k)
        if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):
            # discrete uniform for integer parameters
            v = rng.integers(lo, hi + 1)  # inclusive
            vals.append(float(v))
        else:
            vals.append(float(rng.uniform(lo, hi)))

    return np.asarray(vals, dtype=np.float64), names


def make_inspection_episode_spec(
    out_path: str = "inspection_episode_spec_N5000_seed123.npz",
    N: int = 5000,
    seed: int = 123,
    init_range_min: float = 50.0,
    init_range_max: float = 100.0,
    theta_b_min_deg: float = 40.0,
    meta_ranges: dict | None = None,
):
    """
    Creates a fixed MC dataset of episodes:
      - init_states: (N,7) [x,y,z,vx,vy,vz,theta_s]
      - meta_params: (N,P) sampled uniformly per META_RANGES
      - meta_param_names: (P,)
      - seed_vec: (N,) per-episode seeds
    """
    if meta_ranges is None:
        meta_ranges = META_RANGES

    rng = np.random.default_rng(seed)

    init_states = np.zeros((N, 7), dtype=np.float64)

    # Determine meta param names/order once
    meta0, meta_names = sample_meta_params_uniform(rng, meta_ranges)
    P = meta0.size
    meta_params = np.zeros((N, P), dtype=np.float64)
    meta_params[0, :] = meta0

    # Fill episode 0 state
    init_states[0, :] = sample_initial_state(
        rng,
        init_range_min=init_range_min,
        init_range_max=init_range_max,
        theta_b_min_deg=theta_b_min_deg,
    )

    for i in range(1, N):
        init_states[i, :] = sample_initial_state(
            rng,
            init_range_min=init_range_min,
            init_range_max=init_range_max,
            theta_b_min_deg=theta_b_min_deg,
        )
        meta_params[i, :], _ = sample_meta_params_uniform(rng, meta_ranges)

    # Optional per-episode seeds (useful if you later want deterministic noise per episode)
    seed_vec = rng.integers(low=1, high=2**31 - 1, size=N, dtype=np.int64)

    # Save range bounds too (for provenance)
    # Store as arrays so np.savez_compressed is happy
    meta_lows = np.array([meta_ranges[k][0] for k in meta_names], dtype=np.float64)
    meta_highs = np.array([meta_ranges[k][1] for k in meta_names], dtype=np.float64)

    np.savez_compressed(
        out_path,
        init_states=init_states,
        meta_params=meta_params,
        meta_param_names=np.array(meta_names, dtype=object),
        meta_lows=meta_lows,
        meta_highs=meta_highs,
        seed_vec=seed_vec,
        base_seed=np.array([seed], dtype=np.int64),
        init_range_min=np.array([init_range_min], dtype=np.float64),
        init_range_max=np.array([init_range_max], dtype=np.float64),
        theta_b_min_deg=np.array([theta_b_min_deg], dtype=np.float64),
    )

    print("Saved:", out_path)
    print("init_states:", init_states.shape)
    print("meta_params:", meta_params.shape)
    print("meta_param_names:", meta_names)
    print("Example init state:", init_states[0])
    print("Example meta params:", dict(zip(meta_names, meta_params[0])))


if __name__ == "__main__":
    make_inspection_episode_spec(
        out_path="inspection_episode_spec_N5000_seed123_with_meta.npz",
        N=5000,
        seed=123,
    )

if __name__ == "__main__":
    make_inspection_episode_spec(
        out_path="inspection_episode_spec_N5000_seed123_with_meta.npz",
        N=5000,
        seed=123,
    )
