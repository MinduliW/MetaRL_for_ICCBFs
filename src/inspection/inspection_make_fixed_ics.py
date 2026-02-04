"""
inspection/make_fixed_ics.py

Fixed episode-bank generator for the Inspection environment.

This mirrors the *style* of your docking episode-bank generator (API + .npz payload),
but uses the inspection paper-style initial condition distribution:

  - ||r0|| uniform in [50, 100] m with random azimuth/elevation
  - v0 = 0
  - theta_s uniform in [0, 2pi]
  - if boresight-to-sun angle theta_b < theta_b_min_deg, flip r0 -> -r0

It also optionally samples per-episode physical/meta parameters (m, R_D, R_C, U_MAX, R_MAX, r_orbit)
using multiplicative factor ranges.
"""

from __future__ import annotations

import numpy as np

# Uses the same sun-direction utility as your env / paper sampler.
# Your Inspection codebase already imports this as: `from Observation import ObservationModel`
from inspection.Observation import ObservationModel


def _sun_direction(theta_s: float) -> np.ndarray:
    """Unit vector toward the Sun, consistent with the env."""
    s = np.asarray(ObservationModel.sun_direction(theta_s), dtype=np.float64).reshape(3,)
    n = float(np.linalg.norm(s))
    return s / max(1e-12, n)


def _sample_initial_state_paper(
    rng: np.random.Generator,
    r_min_m: float,
    r_max_m: float,
    theta_b_min_deg: float,
) -> np.ndarray:
    """
    Returns x0 = [x,y,z,vx,vy,vz,theta_s] following the paper distribution,
    including the "flip away from Sun if theta_b < theta_b_min_deg" rule.
    """
    r_mag = float(rng.uniform(r_min_m, r_max_m))
    az = float(rng.uniform(0.0, 2.0 * np.pi))
    el = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))

    r0 = np.array(
        [
            r_mag * np.cos(el) * np.cos(az),
            r_mag * np.cos(el) * np.sin(az),
            r_mag * np.sin(el),
        ],
        dtype=np.float64,
    )

    v0 = np.zeros(3, dtype=np.float64)
    theta_s0 = float(rng.uniform(0.0, 2.0 * np.pi))

    sun_dir = _sun_direction(theta_s0)
    r_norm = float(np.linalg.norm(r0))
    boresight = -r0 / max(1e-12, r_norm)  # deputy -> chief

    cos_theta_b = float(np.clip(np.dot(boresight, sun_dir), -1.0, 1.0))
    theta_b = float(np.arccos(cos_theta_b))

    if theta_b < np.deg2rad(float(theta_b_min_deg)):
        r0 = -r0

    return np.concatenate([r0, v0, [theta_s0]]).astype(np.float64)


def make_inspection_episode_bank(
    out_path: str = "../src/data/inspection/inspection_episode_bank.npz",
    N: int = 500,
    seed: int = 123,
    # -------------------------
    # Nominal meta-parameters
    # -------------------------
    base_m: float = 12.0,
    base_R_D: float = 5.0,
    base_R_C: float = 10.0,
    base_U_MAX: float = 1.0,
    base_R_MAX: float = 800.0,
    base_r_orbit_m: float = 6771.0e3,
    mu_m3_s2: float = 3.986004418e14,
    # -------------------------
    # Per-episode variation factors (multiplicative)
    # -------------------------
    m_factor_range=(0.9, 1.1),
    R_D_factor_range=(0.9, 1.1),
    R_C_factor_range=(0.9, 1.1),
    U_MAX_factor_range=(0.9, 1.1),
    R_MAX_factor_range=(0.9, 1.1),
    r_factor_range=(0.9, 1.1),
    # -------------------------
    # Paper initial-condition distribution
    # -------------------------
    init_range_min_m: float = 50.0,
    init_range_max_m: float = 100.0,
    theta_b_min_deg: float = 40.0,
    # -------------------------
    # Optional basic t=0 feasibility checks
    # -------------------------
    enforce_koz_kiz: bool = False,
    koz_margin_m: float = 0.0,
    kiz_margin_m: float = 0.0,
    max_tries_per_episode: int = 200,
):
    """
    Creates a fixed MC dataset of inspection *episodes* and saves a .npz containing:
      - x0s:     (N,7) initial state [x,y,z,vx,vy,vz,theta_s]
      - m_vec, R_D_vec, R_C_vec, U_MAX_vec, R_MAX_vec, r_vec: (N,)
      - n_vec:   (N,) mean motion derived from r_vec: n = sqrt(mu / r^3)
      - seed_vec:(N,) per-episode seeds (useful for deterministic noise later)

    If enforce_koz_kiz=True, the sampler rejects x0s that violate simple geometric checks at t=0:
      - KOZ: ||r0|| > (R_D + R_C + koz_margin_m)
      - KIZ: ||r0|| < (R_MAX - kiz_margin_m)
    """
    rng = np.random.default_rng(seed)

    # Per-episode factors
    mass_factor = rng.uniform(*m_factor_range, size=N)
    RD_factor   = rng.uniform(*R_D_factor_range, size=N)
    RC_factor   = rng.uniform(*R_C_factor_range, size=N)
    U_factor    = rng.uniform(*U_MAX_factor_range, size=N)
    RMAX_factor = rng.uniform(*R_MAX_factor_range, size=N)
    r_factor    = rng.uniform(*r_factor_range, size=N)

    # Per-episode parameters
    m_vec     = (base_m      * mass_factor).astype(np.float64)
    R_D_vec   = (base_R_D    * RD_factor).astype(np.float64)
    R_C_vec   = (base_R_C    * RC_factor).astype(np.float64)
    U_MAX_vec = (base_U_MAX  * U_factor).astype(np.float64)
    R_MAX_vec = (base_R_MAX  * RMAX_factor).astype(np.float64)
    r_vec     = (base_r_orbit_m * r_factor).astype(np.float64)

    # Derived mean motion (per-episode if r varies)
    n_vec = np.sqrt(mu_m3_s2 / (r_vec ** 3)).astype(np.float64)

    # Optional per-episode seeds (handy for deterministic noise later)
    seed_vec = rng.integers(low=1, high=2**31 - 1, size=N, dtype=np.int64)

    # Allocate x0s
    x0s = np.zeros((N, 7), dtype=np.float64)
    tries_used = np.zeros((N,), dtype=np.int64)

    for i in range(N):
        if not enforce_koz_kiz:
            x0s[i, :] = _sample_initial_state_paper(
                rng,
                r_min_m=init_range_min_m,
                r_max_m=init_range_max_m,
                theta_b_min_deg=theta_b_min_deg,
            )
            tries_used[i] = 1
            continue

        # Rejection sample based on KOZ/KIZ geometry at t=0
        accepted = False
        for t in range(int(max_tries_per_episode)):
            x_try = _sample_initial_state_paper(
                rng,
                r_min_m=init_range_min_m,
                r_max_m=init_range_max_m,
                theta_b_min_deg=theta_b_min_deg,
            )
            r0 = x_try[:3]
            r0_norm = float(np.linalg.norm(r0))

            koz = float(R_D_vec[i] + R_C_vec[i] + koz_margin_m)
            kiz = float(R_MAX_vec[i] - kiz_margin_m)

            if (r0_norm > koz) and (r0_norm < kiz):
                x0s[i, :] = x_try
                tries_used[i] = t + 1
                accepted = True
                break

        if not accepted:
            raise RuntimeError(
                f"Could not find feasible x0 for episode {i} after {max_tries_per_episode} tries. "
                f"Consider disabling enforce_koz_kiz or relaxing margins."
            )

    from pathlib import Path

    # Ensure output directory exists
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)


    np.savez_compressed(
        str(out_path_p),
        x0s=x0s,
        m_vec=m_vec,
        R_D_vec=R_D_vec,
        R_C_vec=R_C_vec,
        U_MAX_vec=U_MAX_vec,
        R_MAX_vec=R_MAX_vec,
        r_vec=r_vec,
        n_vec=n_vec,
        base_seed=np.array([seed], dtype=np.int64),
        seed_vec=seed_vec,
        tries_used=tries_used,
        # Save nominals/ranges for provenance
        base_m=np.array([base_m], dtype=np.float64),
        base_R_D=np.array([base_R_D], dtype=np.float64),
        base_R_C=np.array([base_R_C], dtype=np.float64),
        base_U_MAX=np.array([base_U_MAX], dtype=np.float64),
        base_R_MAX=np.array([base_R_MAX], dtype=np.float64),
        base_r_orbit_m=np.array([base_r_orbit_m], dtype=np.float64),
        mu_m3_s2=np.array([mu_m3_s2], dtype=np.float64),
        init_range=np.array([init_range_min_m, init_range_max_m], dtype=np.float64),
        theta_b_min_deg=np.array([theta_b_min_deg], dtype=np.float64),
        enforce_koz_kiz=np.array([int(enforce_koz_kiz)], dtype=np.int64),
        koz_margin_m=np.array([koz_margin_m], dtype=np.float64),
        kiz_margin_m=np.array([kiz_margin_m], dtype=np.float64),
        ranges=np.array([
            *m_factor_range,
            *R_D_factor_range,
            *R_C_factor_range,
            *U_MAX_factor_range,
            *R_MAX_factor_range,
            *r_factor_range,
        ], dtype=np.float64),
    )

    print("Saved:", out_path)
    print("x0s:", x0s.shape)
    print("Params:", m_vec.shape, R_D_vec.shape, R_C_vec.shape, U_MAX_vec.shape, R_MAX_vec.shape, r_vec.shape)
    print("n_vec:", n_vec.shape)
    if enforce_koz_kiz:
        print("t=0 KOZ/KIZ checks enforced. tries_used stats (min/median/max):",
              int(np.min(tries_used)), int(np.median(tries_used)), int(np.max(tries_used)))


if __name__ == "__main__":
    make_inspection_episode_bank()
