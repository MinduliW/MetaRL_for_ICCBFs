import math
import numpy as np

# DACEyPy: pip install daceypy
from daceypy import DA  # DACEyPy DA type (Differential Algebra)



class CruiseControlICCBF_DA:

    def __init__(self, m, g0, ulim, v0, f0, f1, f2, da_order=4):
        self.m = float(m)
        self.g0 = float(g0)
        self.ulim = float(ulim)
        self.v0 = float(v0)
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.f2 = float(f2)
        self.da_order = int(da_order)

    def getICCBFvars_dace(
        self,
        x0: np.ndarray,
        a1: float = 4.0,
        bcoef1: float = 1.0,
        a2: float = 7.0,
        bcoef2: float = 0.5,
        return_polys: bool = False,
    ):
        """
        Returns (LghICCBF, LfhICCBF, hICCBF) evaluated at x0 (floats).
        Optionally returns the DA polynomials too.

        IMPORTANT: branching is frozen using scalar evaluations at x0.
        """
       
        d0 = float(x0[0])
        v0 = float(x0[1])

        # DA variables (global expansion around 0); evaluate at (d0, v0) later.
        d = DA(1)
        v = DA(2)

        # Drag/rolling resistance model F(v) = f0 + f1 v + f2 v^2
        F = self.f0 + self.f1 * v + self.f2 * (v * v)

        # Drift dynamics (u = 0) for Lie derivatives:
        d_dot = (self.v0 - v)
        v_dot0 = -(F / self.m)

        # Barrier
        h = d - 1.8 * v

        # Lf h and Lg h (with g = [0, g0])
        # dh/dt = d_dot - 1.8 v_dot = (v0 - v) - 1.8 (-(F/m) + g0 u)
        # => Lf h = (v0 - v) + 1.8 F/m,  Lg h = -1.8 g0
        Lfh = d_dot + 1.8 * (F / self.m)
        Lgh = -1.8 * self.g0

        # k1(h) = a1 * h^bcoef1  (you forced bcoef1=1 earlier; keep general here)
        k1 = a1 * h**bcoef1

        # First layer ICCBF (your implementation used + Lgh * ulim)
        b1 = Lfh + Lgh * self.ulim + k1

        # Compute Lg b1 = ∂b1/∂v * g0
        b1_v = b1.deriv(2)
        Lgb1 = self.g0 * b1_v

        # Freeze worst-case u1inf based on sign of Lg b1 at x0
        Lgb1_x0 = float(Lgb1.eval([d0, v0]))
        u1inf = (-self.ulim) if (Lgb1_x0 > 0.0) else (self.ulim)

        # Lf b1 = ∇b1 · f (drift f = [d_dot, v_dot0])
        b1_d = b1.deriv(1)
        Lfb1 = b1_d * d_dot + b1_v * v_dot0

        # Freeze sign for the (potentially non-analytic) b1 power at x0
        b1_x0 = float(b1.eval([d0, v0]))
        if b1_x0 >= 0.0:
            k2 = a2 * b1**bcoef2 
        else:
            # Use (-b1)^bcoef2 on the region where b1 remains negative
            k2 = 0.0

        # Second layer
        b2 = Lfb1 + Lgb1 * u1inf + k2

        # ICCBF is hICCBF = b2; Lie derivatives for control-affine constraint
        b2_d = b2.deriv(1)
        b2_v = b2.deriv(2)

        LfhICCBF = b2_d * d_dot + b2_v * v_dot0
        LghICCBF = self.g0 * b2_v
        hICCBF = b2

        # Return numeric values at x0 (matches your existing API expectation)
        Lfh_val = float(LfhICCBF.eval([d0, v0]))
        Lgh_val = float(LghICCBF.eval([d0, v0]))
        h_val = float(hICCBF.eval([d0, v0]))

        if return_polys:
            return (Lgh_val, Lfh_val, h_val), {
                "hICCBF": hICCBF,
                "LfhICCBF": LfhICCBF,
                "LghICCBF": LghICCBF,
                "b1": b1,
                "b2": b2,
                "u1inf": u1inf,
                "branch_info": {"Lgb1_x0": Lgb1_x0, "b1_x0": b1_x0},
            }

        return Lgh_val, Lfh_val, h_val


def lipschitz_bound_from_da(poly, center, half_width, grid=7) -> float:
    """
    Practical (sampling-based) Lipschitz bound on poly over a box:
      d in [d_c - hw_d, d_c + hw_d], v in [v_c - hw_v, v_c + hw_v]
    using L = sup ||∇poly||_2 estimated on a grid.

    This is not a formal proof bound unless you combine it with ADS/range bounding,
    but it is the easiest way to get a usable margin quickly while still using DA
    for exact derivatives.
    """
    d_c, v_c = float(center[0]), float(center[1])
    hw_d, hw_v = float(half_width[0]), float(half_width[1])

    dp = poly.deriv(1)
    vp = poly.deriv(2)

    ds = np.linspace(d_c - hw_d, d_c + hw_d, grid)
    vs = np.linspace(v_c - hw_v, v_c + hw_v, grid)

    Lmax = 0.0
    for di in ds:
        for vi in vs:
            gd = float(dp.eval([di, vi]))
            gv = float(vp.eval([di, vi]))
            Lmax = max(Lmax, math.sqrt(gd * gd + gv * gv))
    return Lmax

def phi0g_lemma2(center, half_width, pm, *, T, u_max, order=4,
                 safe_is_h_leq_0=True, alpha_gain=1.0, grid=7):
    """
    Returns:
      phi0g_val at x=center (float),
      plus (l_Lfh, l_Lgh, l_alpha, Delta, l1, l2)
    using DACEyPy polynomials and your lipschitz sampler.
    """
    polys = build_da_acc_polys(center, pm, order=order,
                               safe_is_h_leq_0=safe_is_h_leq_0,
                               alpha_gain=alpha_gain)

    h = polys["h"]
    Lfh = polys["Lfh"]
    Lgh = polys["Lgh"]
    alpha_minus_h = polys["alpha_minus_h"]

    # Lipschitz constants (estimated) on the box
    l_Lfh = lipschitz_bound_from_da(Lfh, center, half_width, grid=grid)
    l_Lgh = lipschitz_bound_from_da(Lgh, center, half_width, grid=grid)
    l_alpha = lipschitz_bound_from_da(alpha_minus_h, center, half_width, grid=grid)

    # Δ bound
    Delta = delta_bound_on_box(center, half_width, polys["f_vec"], polys["g_vec"], u_max, grid=grid)

    l1 = l_Lfh + l_Lgh * u_max + l_alpha
    l2 = l_Lfh + l_Lgh * u_max

    # α(-h(x)) evaluated at x=center
    alpha_minus_h_x = float(alpha_minus_h.eval([float(center[0]), float(center[1])]))

    # Lemma-2 expression
    if abs(l2) < 1e-12:
        # limit as l2 -> 0: (l1*Δ/l2)(e^{l2T}-1) -> l1*Δ*T
        corr = l1 * Delta * T
    else:
        corr = (l1 * Delta / l2) * (math.exp(l2 * T) - 1.0)

    phi0g_val = alpha_minus_h_x - corr

    return phi0g_val, {
        "l_Lfh": l_Lfh,
        "l_Lgh": l_Lgh,
        "l_alpha": l_alpha,
        "Delta": Delta,
        "l1": l1,
        "l2": l2,
        "alpha_minus_h_x": alpha_minus_h_x,
    }



def simplest_margin_nuT(poly_h, center, half_width, dt, rate_bound, grid=7) -> float:
    """
    Conservative inter-sample tightening:
      nu_T ≈ (sup ||∇h||) * (dt * sup ||x_dot||)
    where sup ||∇h|| is estimated from DA and sup ||x_dot|| is provided/estimated.

    This matches the common “Lipschitz * motion bound” structure used in sampled-data
    tightening ideas (including feasibility-margin style constructions), albeit conservatively.
    """
    Lh = lipschitz_bound_from_da(poly_h, center, half_width, grid=grid)
    return float(Lh * dt * rate_bound)
