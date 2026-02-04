#!/usr/bin/env python3
"""
make_inspection_episode_spec_paperIC.py

Purpose
-------
Create a fixed Monte-Carlo episode bank (.npz) for your InspectionEnv where:

  (1) Episode meta-parameters are sampled first (the same ones you randomise in reset)
  (2) The initial condition is generated using the *paper* method:
        - range r in [50, 100] m
        - random azimuth/elevation
        - zero relative velocity
        - theta_s in [0, 2pi]
        - if boresight-to-sun angle theta_b < 40 deg, flip r -> -r

What gets saved
---------------
- init_states:      (N, 7)  [x,y,z,vx,vy,vz,theta_s]
- meta_params:      (N, 6)  [m, R_D, R_C, U_MAX, R_MAX, r_orbit]
- meta_param_names: (6,)
- n_derived:        (N,)    n computed from r_orbit using your formula
- episode_vec14:    (N,14)  [init_states(7), meta_params(6), n_derived(1)]  (optional)

Notes
-----
- This sampler intentionally does NOT do fancy boundary-focused conditioning.
- KOZ/KIZ will be trivially satisfied at t=0 for your typical ranges because r0 is 50–100 m.
"""

import numpy as np

# Import your sun-direction function used in the env (same as env._sun_direction)
from inspection.Observation import ObservationModel


# -----------------------------
# User-editable settings
# -----------------------------
DEFAULTS = {
    # How many episodes
    "N": 500,
    "seed": 123,
    "out_path": "N500.npz",

    # Nominal parameters (match env)
    "base_m": 12.0,
    "base_R_D": 5.0,
    "base_R_C": 10.0,
    "base_U_MAX": 1.0,
    "base_R_MAX": 800.0,
    "base_r_orbit_m": 6771.0e3,  # m

    # Randomisation fractions (match env reset)
    "frac_m": 0.10,
    "frac_R_D": 0.10,
    "frac_R_C": 0.10,
    "frac_U_MAX": 0.10,
    "frac_R_MAX": 0.10,
    "frac_r": 0.10,

    # Orbit dynamics constant used by your n-from-r line
    "mu_m3_s2": 3.986004418e14,

    # Paper initial condition distribution
    "init_range_min_m": 50.0,
    "init_range_max_m": 100.0,
    "theta_b_min_deg": 40.0,

    # Save episode_vec14 convenience array
    "include_episode_vec14": True,
}

META_PARAM_NAMES = ["m", "R_D", "R_C", "U_MAX", "R_MAX", "r"]


# -----------------------------
# Small helper functions
# -----------------------------
def pm_range(base: float, frac: float) -> tuple[float, float]:
    """Return (min,max) for uniform sampling around a base value with ±frac."""
    return base * (1.0 - frac), base * (1.0 + frac)


def derive_n_from_r(mu_m3_s2: float, r_m: float) -> float:
  

    return float(np.sqrt(mu_m3_s2 / (r_m ** 3)))


def sample_meta_params(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """
    Sample ONLY the parameters your reset randomises:
      [m, R_D, R_C, U_MAX, R_MAX, r_orbit]
    """
    m_min, m_max = pm_range(cfg["base_m"], cfg["frac_m"])
    RD_min, RD_max = pm_range(cfg["base_R_D"], cfg["frac_R_D"])
    RC_min, RC_max = pm_range(cfg["base_R_C"], cfg["frac_R_C"])
    U_min, U_max = pm_range(cfg["base_U_MAX"], cfg["frac_U_MAX"])
    RMAX_min, RMAX_max = pm_range(cfg["base_R_MAX"], cfg["frac_R_MAX"])
    rmin, rmax = pm_range(cfg["base_r_orbit_m"], cfg["frac_r"])

    return np.array([
        rng.uniform(m_min, m_max),
        rng.uniform(RD_min, RD_max),
        rng.uniform(RC_min, RC_max),
        rng.uniform(U_min, U_max),
        rng.uniform(RMAX_min, RMAX_max),
        rng.uniform(rmin, rmax),
    ], dtype=np.float64)


def sample_initial_state_paper(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """
    Paper initial condition sampler:
      - r0 magnitude uniform in [50,100] m
      - random direction via azimuth/elevation
      - v0 = 0
      - theta_s uniform in [0,2pi]
      - if theta_b < 40 deg, flip r0 -> -r0

    Returns x0 = [x,y,z,vx,vy,vz,theta_s]
    """
    # 1) Random position with r in [min,max]
    r_mag = float(rng.uniform(cfg["init_range_min_m"], cfg["init_range_max_m"]))
    az = float(rng.uniform(0.0, 2.0 * np.pi))
    el = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))

    r0 = np.array([
        r_mag * np.cos(el) * np.cos(az),
        r_mag * np.cos(el) * np.sin(az),
        r_mag * np.sin(el),
    ], dtype=np.float64)

    # 2) Zero velocity
    v0 = np.zeros(3, dtype=np.float64)

    # 3) Random sun angle parameter
    theta_s0 = float(rng.uniform(0.0, 2.0 * np.pi))

    # 4) Compute boresight-to-sun angle theta_b and apply “flip” rule
    sun_dir = np.asarray(ObservationModel.sun_direction(theta_s0), dtype=np.float64)
    sun_dir /= max(1e-12, float(np.linalg.norm(sun_dir)))

    r_norm = max(1e-12, float(np.linalg.norm(r0)))
    boresight = -r0 / r_norm  # deputy -> chief

    cos_theta_b = float(np.clip(np.dot(boresight, sun_dir), -1.0, 1.0))
    theta_b = float(np.arccos(cos_theta_b))

    if theta_b < np.deg2rad(float(cfg["theta_b_min_deg"])):
        r0 = -r0

    return np.concatenate([r0, v0, [theta_s0]]).astype(np.float64)


# -----------------------------
# Main generator
# -----------------------------
def make_episode_spec(cfg: dict) -> None:
    N = int(cfg["N"])
    seed = int(cfg["seed"])
    out_path = str(cfg["out_path"])

    rng = np.random.default_rng(seed)

    init_states = np.zeros((N, 7), dtype=np.float64)
    meta_params = np.zeros((N, 6), dtype=np.float64)
    n_derived = np.zeros((N,), dtype=np.float64)

    # Optional sanity counters (KOZ/KIZ should basically never fail here)
    bad_koz = 0
    bad_kiz = 0

    for i in range(N):
        # 1) Sample meta first
        mp = sample_meta_params(rng, cfg)
        meta_params[i, :] = mp

        m_i, R_D_i, R_C_i, U_MAX_i, R_MAX_i, r_orbit_i = mp.tolist()
        n_derived[i] = derive_n_from_r(cfg["mu_m3_s2"], r_orbit_i)

        # 2) Sample paper initial state
        x0 = sample_initial_state_paper(rng, cfg)
        init_states[i, :] = x0

        # 3) Sanity check (optional)
        r_coll = float(R_D_i + R_C_i)
        r0_norm = float(np.linalg.norm(x0[:3]))
        if r0_norm <= r_coll:
            bad_koz += 1
        if r0_norm >= float(R_MAX_i):
            bad_kiz += 1

    payload = {
        "init_states": init_states,
        "meta_params": meta_params,
        "meta_param_names": np.array(META_PARAM_NAMES, dtype=object),
        "n_derived": n_derived,
        "base_seed": np.array([seed], dtype=np.int64),
        "bad_koz_count": np.array([bad_koz], dtype=np.int64),
        "bad_kiz_count": np.array([bad_kiz], dtype=np.int64),
    }

    if bool(cfg["include_episode_vec14"]):
        payload["episode_vec14"] = np.concatenate(
            [init_states, meta_params, n_derived[:, None]], axis=1
        ).astype(np.float64)

    np.savez_compressed(out_path, **payload)

    print("Saved:", out_path)
    print("init_states shape:", init_states.shape)
    print("meta_params shape:", meta_params.shape)
    print("n_derived shape:", n_derived.shape)
    print("bad_koz_count:", int(bad_koz))
    print("bad_kiz_count:", int(bad_kiz))
    if "episode_vec14" in payload:
        print("episode_vec14 shape:", payload["episode_vec14"].shape)


if __name__ == "__main__":
    make_episode_spec(DEFAULTS)
