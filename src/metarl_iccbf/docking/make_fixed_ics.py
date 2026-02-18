import numpy as np

def make_docking_episode_bank(
    out_path: str = "../src/data/docking/docking_episode_bank.npz",
    N: int = 5000,
    seed: int = 123,
    # Nominals (match your env)
    base_m: float = 1000.0,
    base_rho: float = 2.4 / 1e3,
    base_om: float = 0.6 * np.pi / 180.0,
    base_umax: float = 0.25,
    base_gamma: float = 10.0 * np.pi / 180.0,
    base_r: float = 6771.0,
    # Variation factors
    m_factor_range=(0.9, 1.1),
    umax_factor_range=(0.9, 1.1),
    rho_factor_range=(0.9, 1.1),
    om_factor_range=(0.9, 1.1),
    gamma_factor_range=(0.9, 1.1),
    r_factor_range=(0.9, 1.1),
    # Initial geometry (x fixed, y sampled; vx,vy,phi fixed unless you randomise)
    x_init: float = 100.0 / 1e3,
    vx0: float = 0.0,
    vy0: float = 0.0,
    phi0: float = 0.0,
    # -------------------------
    # Inner-safe-set enforcement
    # -------------------------
    enforce_inner_safe: bool = True,
    h_margin: float = 0.0,
    inner_band_frac: float = 0.00,
    max_tries_per_episode: int = 200,
    relax_band_on_fail: bool = False,
    relax_factor: float = 0.0,
    relax_rounds: int = 6,
    randomise_phi: bool = False,
    phi_range=(0.0, 2.0 * np.pi),
):
    """
    Creates a fixed MC dataset of *episodes*:
      - x0s: (N,5) initial state [x, y, vx, vy, phi]
      - m_vec, rho_vec, om_vec, umax_vec, gamma_vec, r_vec: (N,)
      - seed_vec: (N,)

    Inner-safe-set meaning here:
      - x0 must satisfy DockingCase.originalh(x0) >= h_margin
      - additionally, y is sampled from a shrunk band: [y_low, y_high] shrunk by inner_band_frac
        to avoid points near the boundary.

    If enforce_inner_safe=True, this routine will ONLY write points that pass.
    """

    rng = np.random.default_rng(seed)

    # Per-episode factors
    mass_factor   = rng.uniform(*m_factor_range,   size=N)
    thrust_factor = rng.uniform(*umax_factor_range, size=N)
    rho_factor    = rng.uniform(*rho_factor_range, size=N)
    om_factor     = rng.uniform(*om_factor_range,  size=N)
    gamma_factor  = rng.uniform(*gamma_factor_range, size=N)
    r_factor      = rng.uniform(*r_factor_range,   size=N)

    # Per-episode parameters
    m_vec     = base_m     * mass_factor
    umax_vec  = base_umax  * thrust_factor
    rho_vec   = base_rho   * rho_factor
    om_vec    = base_om    * om_factor
    gamma_vec = base_gamma * gamma_factor
    r_vec     = base_r     * r_factor

    # Optional per-episode seeds (useful for deterministic noise later)
    seed_vec = rng.integers(low=1, high=2**31 - 1, size=N, dtype=np.int64)

    # Allocate outputs
    x0s = np.zeros((N, 5), dtype=np.float64)
    h0_vec = np.full((N,), np.nan, dtype=np.float64)  # provenance: inner-safe margin at x0 (if enforced)

    if enforce_inner_safe:
        try:
            from docking.Dockingcase import DockingCase
        except Exception as e:
            raise ImportError(
                "enforce_inner_safe=True requires `Dockingcase.DockingCase` to be importable "
                "so `originalh(x)` can be evaluated."
            ) from e

    for i in range(N):
        rho_i = float(rho_vec[i])
        gam_i = float(gamma_vec[i])

        # y-range in your reset depends on rho and gamma
        # (same structure as your earlier reset)
        tg = np.tan(gam_i)
        y_low  = -((90.0/1e3 - rho_i)        * tg)
        y_high =  ((90.0/1e3 - rho_i - 1e-3) * tg)

        if y_high < y_low:
            y_low, y_high = y_high, y_low

        # shrink away from boundary for "inner" sampling
        width = (y_high - y_low)
        band = float(inner_band_frac)
        yL = y_low  + band * width
        yU = y_high - band * width

        # handle degenerate case
        if yU <= yL:
            mid = 0.5 * (y_low + y_high)
            yL, yU = mid, mid

        # Choose phi
        if randomise_phi:
            phi_i = float(rng.uniform(*phi_range))
        else:
            phi_i = float(phi0)

        # If not enforcing, just sample once from inner band
        if not enforce_inner_safe:
            y_i = float(rng.uniform(yL, yU))
            x0s[i, :] = np.array([x_init, y_i, vx0, vy0, phi_i], dtype=np.float64)
            continue

        # Enforce using DockingCase.originalh(x) >= h_margin
        dc = DockingCase(rho=rho_i, gamma=gam_i)

        accepted = False
        band_local = band
        for rr in range(max(1, int(relax_rounds))):
            # recompute inner bounds if we relax
            yLr = y_low + band_local * width
            yUr = y_high - band_local * width
            if yUr <= yLr:
                mid = 0.5 * (y_low + y_high)
                yLr, yUr = mid, mid

            for _ in range(int(max_tries_per_episode)):
                y_i = float(rng.uniform(yLr, yUr))
                x_try = np.array([x_init, y_i, vx0, vy0, phi_i], dtype=np.float64)

                h0 = float(dc.originalh(x_try))
                if h0 >= float(h_margin):
                    x0s[i, :] = x_try
                    h0_vec[i] = h0
                    accepted = True
                    break

            if accepted:
                break

            # Relax band (allow closer to boundary) if still failing
            if relax_band_on_fail:
                band_local *= float(relax_factor)
            else:
                break

        if not accepted:
            raise RuntimeError(
                f"Could not find inner-safe x0 for episode {i} "
                f"after {max_tries_per_episode} tries × {relax_rounds} relax rounds. "
                f"Consider decreasing h_margin or inner_band_frac."
            )

    np.savez(
        out_path,
        x0s=x0s,
        m_vec=m_vec.astype(np.float64),
        rho_vec=rho_vec.astype(np.float64),
        om_vec=om_vec.astype(np.float64),
        umax_vec=umax_vec.astype(np.float64),
        gamma_vec=gamma_vec.astype(np.float64),
        r_vec=r_vec.astype(np.float64),
        base_seed=np.array([seed], dtype=np.int64),
        seed_vec=seed_vec,

        # provenance / diagnostics
        enforce_inner_safe=np.array([int(enforce_inner_safe)], dtype=np.int64),
        h_margin=np.array([float(h_margin)], dtype=np.float64),
        inner_band_frac=np.array([float(inner_band_frac)], dtype=np.float64),
        h0_vec=h0_vec.astype(np.float64),

        # save nominals (handy)
        base_m=np.array([base_m], dtype=np.float64),
        base_rho=np.array([base_rho], dtype=np.float64),
        base_om=np.array([base_om], dtype=np.float64),
        base_umax=np.array([base_umax], dtype=np.float64),
        base_gamma=np.array([base_gamma], dtype=np.float64),
        base_r=np.array([base_r], dtype=np.float64),
        ranges=np.array([
            *m_factor_range, *umax_factor_range, *rho_factor_range, *om_factor_range,
            *gamma_factor_range, *r_factor_range
        ], dtype=np.float64),
    )

    print("Saved:", out_path)
    print("x0s:", x0s.shape)
    if enforce_inner_safe:
        print("Inner-safe enforced.")
        print("h0_vec min/median:", float(np.nanmin(h0_vec)), float(np.nanmedian(h0_vec)))


# if __name__ == "__main__":
#     make_docking_episode_spec(
#         out_path="T2docking_episode_spec_N5000_seed123.npz",
#         N=5000,
#         seed=123,
#         enforce_inner_safe=True,
#         h_margin=1e-3,         # tighten/loosen this as needed
#         inner_band_frac=0.20,  # 20% away from boundary by construction
#         max_tries_per_episode=200,
#         relax_band_on_fail=True,
#     )
