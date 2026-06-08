import symengine as se
from sympy.utilities.lambdify import lambdify
import numpy as np
from daceypy import DA


class ICCBF:
    """
    ICCBF helper with DA-based local certificate + "slight adjustment until valid".

    Notes:
      - DA.init(...) is done *inside* certificate / expansion routines to avoid
        cross-contamination of DA state.
      - Support-term lower bound uses per-component interval bounds (robust when
        (Lg1^2+Lg2^2).bound() collapses).
      - Sampled-data tightening nu is included consistently in the base term.

    """

    def __init__(self, mu, r, gamma, rho, m, om, umax):
        self.mu = float(mu)
        self.r = float(r)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.m = float(m)
        self.om = float(om)
        self.umax = float(umax)

    # -----------------------------
    # Interval helpers
    # -----------------------------
    @staticmethod
    def _interval_endpoints(I):
        for a, b in [("lb", "ub"), ("lower", "upper"), ("l", "u"), ("inf", "sup")]:
            if hasattr(I, a) and hasattr(I, b):
                return float(getattr(I, a)), float(getattr(I, b))
        return float(I[0]), float(I[1])

    @staticmethod
    def interval_maxabs(I):
        for a, b in [("lb", "ub"), ("lower", "upper"), ("l", "u"), ("inf", "sup")]:
            if hasattr(I, a) and hasattr(I, b):
                return max(abs(float(getattr(I, a))), abs(float(getattr(I, b))))
        return max(abs(float(I[0])), abs(float(I[1])))

    @staticmethod
    def minabs_interval(lb, ub, eps=0.0):
        # certified inf_{v in [lb,ub]} |v|
        if (lb <= eps) and (ub >= -eps):
            return 0.0
        return min(abs(lb), abs(ub))

    # -----------------------------
    # DA bounding over a physical box
    # -----------------------------
    def _bound_over_box(self, poly, half_width):
        hw = np.asarray(half_width, dtype=float).flatten()
        if hw.size != 5:
            raise ValueError(f"half_width must be length 5, got {hw.size}")

        p = poly
        for j in range(1, 6):
            p = p.scaleVariable(j, float(hw[j - 1]))

        I = p.bound()
        return self._interval_endpoints(I)

    # -----------------------------
    # Lipschitz bound (your code, unchanged)
    # -----------------------------
    def lipschitz_bound_bounder_docking(self, poly, half_width):
        hw = np.asarray(half_width, dtype=float).flatten()
        sup_abs = []
        for i in range(1, 6):
            dpi = poly.deriv(i)
            for j in range(1, 6):
                dpi = dpi.scaleVariable(j, float(hw[j - 1]))
            sup_abs_i = self.interval_maxabs(dpi.bound())
            sup_abs.append(float(sup_abs_i))
        return float(np.linalg.norm(sup_abs, ord=2))

    # -----------------------------
    # Delta bound (your code, unchanged)
    # -----------------------------
    def delta_box_docking_bounder_L2(self, center, half_width, da_order: int = 4):
        c = np.asarray(center, dtype=float).flatten()
        hw = np.asarray(half_width, dtype=float).flatten()
        if c.size != 5 or hw.size != 5:
            raise ValueError(f"center and half_width must be length 5, got {c.size}, {hw.size}")

        x10, x20, x30, x40, x50 = map(float, c)
        r1, r2, r3, r4, r5 = map(float, hw)

        DA.init(int(da_order), 5)

        dx1, dx2, dx3, dx4, dx5 = DA(1), DA(2), DA(3), DA(4), DA(5)
        x1 = x10 + dx1
        x2 = x20 + dx2
        x3 = x30 + dx3
        x4 = x40 + dx4
        x5 = x50 + dx5

        mu = float(self.mu)
        r = float(self.r)
        n = float(np.sqrt(mu / (r**3)))
        om = float(self.om)
        m = float(self.m)
        umax = float(self.umax)

        rc = ((x1 + r) * (x1 + r) + x2 * x2).sqrt()
        rc3 = rc * rc * rc

        f1 = x3
        f2 = x4
        f3 = (n * n) * x1 + 2.0 * n * x4 + mu / (r * r) - mu * (r + x1) / rc3
        f4 = (n * n) * x2 - 2.0 * n * x3 - mu * x2 / rc3

        M5 = abs(om)

        def scale_all(p):
            return (p.scaleVariable(1, r1)
                    .scaleVariable(2, r2)
                    .scaleVariable(3, r3)
                    .scaleVariable(4, r4)
                    .scaleVariable(5, r5))

        M1 = float(self.interval_maxabs(scale_all(f1).bound()))
        M2 = float(self.interval_maxabs(scale_all(f2).bound()))
        M3 = float(self.interval_maxabs(scale_all(f3).bound()))
        M4 = float(self.interval_maxabs(scale_all(f4).bound()))

        Mf34 = float(np.sqrt(M3 * M3 + M4 * M4))
        Mf34_tot = Mf34 + abs(umax) / abs(m)

        return float(np.sqrt(M1 * M1 + M2 * M2 + Mf34_tot * Mf34_tot + M5 * M5))

    def delta_box_from_polys(self, f_polys, half_width):
        """Compute the delta bound reusing pre-built dynamics polynomials."""
        hw = np.asarray(half_width, dtype=float).flatten()
        f1, f2, f3, f4, f5 = f_polys

        def scale_all(p):
            return (p.scaleVariable(1, float(hw[0]))
                    .scaleVariable(2, float(hw[1]))
                    .scaleVariable(3, float(hw[2]))
                    .scaleVariable(4, float(hw[3]))
                    .scaleVariable(5, float(hw[4])))

        M1 = float(self.interval_maxabs(scale_all(f1).bound()))
        M2 = float(self.interval_maxabs(scale_all(f2).bound()))
        M3 = float(self.interval_maxabs(scale_all(f3).bound()))
        M4 = float(self.interval_maxabs(scale_all(f4).bound()))
        M5 = abs(float(self.om))

        Mf34 = float(np.sqrt(M3 * M3 + M4 * M4))
        Mf34_tot = Mf34 + abs(float(self.umax)) / abs(float(self.m))

        return float(np.sqrt(M1 * M1 + M2 * M2 + Mf34_tot * Mf34_tot + M5 * M5))

    # -----------------------------
    # nu (your simplified Lemma-2)
    # -----------------------------
    @staticmethod
    def nu_lemma2(T, l1, l2, Delta, eps=1e-12):
        return float(l1) * float(T) * float(Delta)

    # -----------------------------
    # margin routine (your code, unchanged)
    # -----------------------------
    def getmargin(self, x0, a1, a2, hslack, tstep):
        x0 = np.asarray(x0, dtype=float).flatten()

        half_width = np.array([0.002, 0.002, 0.000005, 0.000005, 0.0002], dtype=float)

        vals, polys = self.getICCBFvars_dace_docking(
            x0=x0, a1=a1, a2=a2, return_polys=True, da_order=4
        )

        Lf_poly = polys["LfhICCBF"]
        Lg1_poly = polys["Lgh1ICCBF"]
        Lg2_poly = polys["Lgh2ICCBF"]
        h_poly = polys["hICCBF"]

        l_Lfh = self.lipschitz_bound_bounder_docking(Lf_poly, half_width)
        l_Lg1 = self.lipschitz_bound_bounder_docking(Lg1_poly, half_width)
        l_Lg2 = self.lipschitz_bound_bounder_docking(Lg2_poly, half_width)
        l_Lg_vec = float(np.sqrt(l_Lg1 * l_Lg1 + l_Lg2 * l_Lg2))

        l_h = self.lipschitz_bound_bounder_docking(h_poly, half_width)
        l_alpha = abs(float(hslack)) * l_h

        Delta = self.delta_box_from_polys(polys["f_polys"], half_width)

        u_max = abs(float(self.umax))
        l2 = l_Lfh + l_Lg_vec * u_max
        l1 = l2 + l_alpha

        nu = self.nu_lemma2(float(tstep), float(l1), float(l2), float(Delta))
        return float(nu), vals

    # -----------------------------
    # DA construction (your code, mostly unchanged)
    # -----------------------------
    def getICCBFvars_dace_docking(self, x0, a1=0.25, a2=0.85, return_polys=False, da_order=4):
        x0 = np.asarray(x0, dtype=float).flatten()
        if x0.size != 5:
            raise ValueError("x0 must be length 5")

        DA.init(int(da_order), 5)

        x10, x20, x30, x40, x50 = map(float, x0[:5])
        x1 = DA(1) + x10
        x2 = DA(2) + x20
        x3 = DA(3) + x30
        x4 = DA(4) + x40
        x5 = DA(5) + x50

        mu = float(self.mu)
        r = float(self.r)
        n = float(np.sqrt(mu / (r**3)))
        gamma = float(self.gamma)

        rho = float(self.rho)
        m = float(self.m)
        om = float(self.om)
        umax = float(self.umax)

        c5 = x5.cos()
        s5 = x5.sin()

        rc = ((x1 + r) * (x1 + r) + x2 * x2).sqrt()
        rc3 = rc * rc * rc

        term1 = (n * n) * x1 + 2.0 * n * x4 + mu / (r * r) - mu * (r + x1) / rc3
        term2 = (n * n) * x2 - 2.0 * n * x3 - mu * x2 / rc3

        f1, f2, f3, f4, f5 = x3, x4, term1, term2, om

        invm = 1.0 / m
        g1 = [DA(0.0), DA(0.0), DA(invm), DA(0.0), DA(0.0)]
        g2 = [DA(0.0), DA(0.0), DA(0.0), DA(invm), DA(0.0)]

        rcp1 = x1 - rho * c5
        rcp2 = x2 - rho * s5

        dot = rcp1 * c5 + rcp2 * s5
        norm_rcp = (rcp1 * rcp1 + rcp2 * rcp2).sqrt()

        # Your h definition (no offset term)
        h = dot / norm_rcp - np.cos(gamma)

        dh = [h.deriv(1), h.deriv(2), h.deriv(3), h.deriv(4), h.deriv(5)]

        Lfb0 = dh[0]*f1 + dh[1]*f2 + dh[2]*f3 + dh[3]*f4 + dh[4]*f5

        b1 = Lfb0 + float(a1) * h
        db1 = [b1.deriv(1), b1.deriv(2), b1.deriv(3), b1.deriv(4), b1.deriv(5)]

        Lgb1_1 = db1[0]*g1[0] + db1[1]*g1[1] + db1[2]*g1[2] + db1[3]*g1[3] + db1[4]*g1[4]
        Lgb1_2 = db1[0]*g2[0] + db1[1]*g2[1] + db1[2]*g2[2] + db1[3]*g2[3] + db1[4]*g2[4]

        Lgb1_1_x0 = float(Lgb1_1.eval([0.0, 0.0, 0.0, 0.0, 0.0]))
        Lgb1_2_x0 = float(Lgb1_2.eval([0.0, 0.0, 0.0, 0.0, 0.0]))
        u1inf = (-umax) if (Lgb1_1_x0 > 0.0) else (umax)
        u2inf = (-umax) if (Lgb1_2_x0 > 0.0) else (umax)

        Lfb1 = db1[0]*f1 + db1[1]*f2 + db1[2]*f3 + db1[3]*f4 + db1[4]*f5

        b2 = Lfb1 + Lgb1_1 * u1inf + Lgb1_2 * u2inf + float(a2) * b1

        db2 = [b2.deriv(1), b2.deriv(2), b2.deriv(3), b2.deriv(4), b2.deriv(5)]
        Lfb2 = db2[0]*f1 + db2[1]*f2 + db2[2]*f3 + db2[3]*f4 + db2[4]*f5
        Lgb2_1 = db2[0]*g1[0] + db2[1]*g1[1] + db2[2]*g1[2] + db2[3]*g1[3] + db2[4]*g1[4]
        Lgb2_2 = db2[0]*g2[0] + db2[1]*g2[1] + db2[2]*g2[2] + db2[3]*g2[3] + db2[4]*g2[4]

        h_val = float(b2.eval([0.0, 0.0, 0.0, 0.0, 0.0]))
        Lfh_val = float(Lfb2.eval([0.0, 0.0, 0.0, 0.0, 0.0]))
        Lgh1_val = float(Lgb2_1.eval([0.0, 0.0, 0.0, 0.0, 0.0]))
        Lgh2_val = float(Lgb2_2.eval([0.0, 0.0, 0.0, 0.0, 0.0]))

        if return_polys:
            return (Lgh1_val, Lgh2_val, Lfh_val, h_val), {
                "hICCBF": b2,
                "LfhICCBF": Lfb2,
                "Lgh1ICCBF": Lgb2_1,
                "Lgh2ICCBF": Lgb2_2,
                "b1": b1,
                "b2": b2,
                "Lgb1_1": Lgb1_1,
                "Lgb1_2": Lgb1_2,
                "u1inf": u1inf,
                "u2inf": u2inf,
                "branch_info": {"Lgb1_1_x0": Lgb1_1_x0, "Lgb1_2_x0": Lgb1_2_x0},
                "f_polys": (f1, f2, f3, f4, f5),
            }

        return Lgh1_val, Lgh2_val, Lfh_val, h_val

    # -----------------------------
    # Certificate (FIXED) + optional branch checking
    # -----------------------------
    def certify_local_validity_best(
        self,
        x0,
        a1,
        a2,
        hslack,
        tstep,
        half_width,
        *,
        da_order=4,
        eps_zero=1e-16,
        margin_req=0.0,
        branch_tau=1e-6,
        check_branch=False,
        verbose=False,
    ):
        x0 = np.asarray(x0, dtype=float).flatten()
        half_width = np.asarray(half_width, dtype=float).flatten()
        if x0.size != 5 or half_width.size != 5:
            raise ValueError("x0 and half_width must be length 5")

        # Build nu
        nu, _ = self.getmargin(x0=x0, a1=a1, a2=a2, hslack=hslack, tstep=tstep)
        nu = float(nu)

        # Build polynomials
        (_vals_num, polys) = self.getICCBFvars_dace_docking(
            x0=x0, a1=a1, a2=a2, return_polys=True, da_order=da_order
        )

        b2_poly = polys["hICCBF"]
        Lf_poly = polys["LfhICCBF"]
        Lg1_poly = polys["Lgh1ICCBF"]
        Lg2_poly = polys["Lgh2ICCBF"]

        # Base: inf_x [Lf b2 + hslack*b2 - nu]
        base_poly = Lf_poly + float(hslack) * b2_poly 
        base_lb, base_ub = self._bound_over_box(base_poly, half_width)

        # Support: inf_x [umax * ||Lg||] with robust interval minabs per component
        Lg1_lb, Lg1_ub = self._bound_over_box(Lg1_poly, half_width)
        Lg2_lb, Lg2_ub = self._bound_over_box(Lg2_poly, half_width)

        minabs_Lg1 = float(self.minabs_interval(Lg1_lb, Lg1_ub, eps=eps_zero))
        minabs_Lg2 = float(self.minabs_interval(Lg2_lb, Lg2_ub, eps=eps_zero))

        normLg_lb = float(np.sqrt(minabs_Lg1**2 + minabs_Lg2**2))
        support_lb = abs(float(self.umax)) * normLg_lb

        zeta_lb = float(base_lb + support_lb)
        valid = bool(zeta_lb >= float(margin_req))

        # Optional branch check
        branch_ok = True
        branch_margin_lb = None
        Lgb1_1_int = None
        Lgb1_2_int = None

        if check_branch:
            Lgb1_1_poly = polys["Lgb1_1"]
            Lgb1_2_poly = polys["Lgb1_2"]
            b1_1_lb, b1_1_ub = self._bound_over_box(Lgb1_1_poly, half_width)
            b1_2_lb, b1_2_ub = self._bound_over_box(Lgb1_2_poly, half_width)

            Lgb1_1_int = (float(b1_1_lb), float(b1_1_ub))
            Lgb1_2_int = (float(b1_2_lb), float(b1_2_ub))

            crosses1 = (b1_1_lb <= eps_zero) and (b1_1_ub >= -eps_zero)
            crosses2 = (b1_2_lb <= eps_zero) and (b1_2_ub >= -eps_zero)

            minabs1 = self.minabs_interval(b1_1_lb, b1_1_ub, eps=eps_zero)
            minabs2 = self.minabs_interval(b1_2_lb, b1_2_ub, eps=eps_zero)
            branch_margin_lb = float(min(minabs1, minabs2))

            branch_ok = (not crosses1) and (not crosses2) and (branch_margin_lb >= float(branch_tau))

            # if you want validity to require branch stability too:
            # valid = valid and branch_ok

        if verbose:
            print(f"[cert] base_lb={base_lb:.3e}, support_lb={support_lb:.3e}, zeta_lb={zeta_lb:.3e}, nu={nu:.3e}")

        return {
            "valid": valid,
            "zeta_lb": zeta_lb,
            "margin_req": float(margin_req),
            "base_lb": float(base_lb),
            "base_ub": float(base_ub),
            "support_lb": float(support_lb),
            "normLg_lb": float(normLg_lb),
            "nu": float(nu),
            "Lg_interval": ((float(Lg1_lb), float(Lg1_ub)), (float(Lg2_lb), float(Lg2_ub))),
            "branch_ok": bool(branch_ok),
            "branch_margin_lb": None if branch_margin_lb is None else float(branch_margin_lb),
            "Lgb1_1_interval": Lgb1_1_int,
            "Lgb1_2_interval": Lgb1_2_int,
        }

    # -----------------------------
    # Adjustment until valid (ADDED)
    # -----------------------------
    def adjust_until_valid(
        self,
        x0,
        a1,
        a2,
        hslack,
        tstep,
        half_width,
        *,
        margin_req=0.0,
        da_order=4,
        eps_zero=1e-16,
        max_iters=15,
        # small adjustment knobs
        hslack_mult=1.10,
        a2_mult=1.03,
        a1_mult=1.01,
        half_width_mult=0.90,
        # caps / floors
        hslack_max=100.0,
        a1_max=50.0,
        a2_max=50.0,
        half_width_min=None,
        try_shrink_box=True,
        try_increase_gains=True,
        check_branch=False,
        branch_tau=1e-6,
        verbose=False,
    ):
        """
        Returns:
          (a1_new, a2_new, hslack_new, half_width_new, cert, history)
        """
        x0 = np.asarray(x0, dtype=float).flatten()
        hw = np.asarray(half_width, dtype=float).flatten()
        if hw.size != 5:
            raise ValueError("half_width must be length 5")

        if half_width_min is not None:
            half_width_min = np.asarray(half_width_min, dtype=float).flatten()
            if half_width_min.size != 5:
                raise ValueError("half_width_min must be length 5")

        a1_cur, a2_cur, hs_cur = float(a1), float(a2), float(hslack)
        hw_cur = hw.copy()

        history = []

        for k in range(max_iters):
            cert = self.certify_local_validity_best(
                x0=x0,
                a1=a1_cur,
                a2=a2_cur,
                hslack=hs_cur,
                tstep=float(tstep),
                half_width=hw_cur,
                da_order=int(da_order),
                eps_zero=float(eps_zero),
                margin_req=float(margin_req),
                check_branch=bool(check_branch),
                branch_tau=float(branch_tau),
                verbose=verbose,
            )

            history.append({
                "iter": k,
                "a1": a1_cur,
                "a2": a2_cur,
                "hslack": hs_cur,
                "half_width": hw_cur.copy(),
                "valid": bool(cert["valid"]),
                "zeta_lb": float(cert["zeta_lb"]),
                "base_lb": float(cert["base_lb"]),
                "support_lb": float(cert["support_lb"]),
                "normLg_lb": float(cert["normLg_lb"]),
            })

            if cert["valid"]:
                return a1_cur, a2_cur, hs_cur, hw_cur, cert, history

            # ---- slight adjustment ----
            if try_increase_gains:
                hs_cur = min(hs_cur * hslack_mult, hslack_max)
                a2_cur = min(a2_cur * a2_mult, a2_max)
                a1_cur = min(a1_cur * a1_mult, a1_max)

            if try_shrink_box:
                hw_next = hw_cur * half_width_mult
                if half_width_min is not None:
                    hw_next = np.maximum(hw_next, half_width_min)
                # stop shrinking if we hit the floor
                if half_width_min is not None and np.allclose(hw_next, hw_cur):
                    pass
                else:
                    hw_cur = hw_next

        # return last attempt
        cert = self.certify_local_validity_best(
            x0=x0,
            a1=a1_cur,
            a2=a2_cur,
            hslack=hs_cur,
            tstep=float(tstep),
            half_width=hw_cur,
            da_order=int(da_order),
            eps_zero=float(eps_zero),
            margin_req=float(margin_req),
            check_branch=bool(check_branch),
            branch_tau=float(branch_tau),
            verbose=verbose,
        )
        return a1_cur, a2_cur, hs_cur, hw_cur, cert, history



    def getICCBFs(self):
        # ----------------- Symbols -----------------
        x1, x2, x3, x4, x5 = se.symbols("x1 x2 x3 x4 x5")
        u1inf, u2inf = se.symbols("u1inf u2inf")
        acoef1, acoef2 = se.symbols("acoef1 acoef2")

        # Parameters that change per episode
        rho_s, m_s, om_s, gamma_s, r_s = se.symbols("rho m om gamma r")

        # Fixed constants (substitute numerically)
        mu = self.mu
        r = r_s
        n =  se.sqrt(mu / r_s**3)
        gamma = gamma_s

        # ----------------- Dynamics f, g -----------------
        rc = se.sqrt((x1 + r) ** 2 + x2**2)

        term1 = (
            n**2 * x1
            + 2 * n * x4
            + mu / r**2
            - mu * (r + x1) / rc**3
        )
        term2 = n**2 * x2 - 2 * n * x3 - mu * x2 / rc**3

        # x5_dot = om_s (parameter)
        f = se.Matrix([x3, x4, term1, term2, om_s])

        # g depends on mass m_s
        g = (1 / m_s) * se.Matrix(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
                [0, 0],
            ]
        )

        # ----------------- CLF V -----------------
        V = (x3 + (x1 - rho_s * se.cos(x5)) / 10) ** 2 + \
            (x4 + (x2 - rho_s * se.sin(x5)) / 10) ** 2
        dV_dx = se.Matrix([se.diff(V, var) for var in [x1, x2, x3, x4, x5]])

        # ----------------- CBF h -----------------
        rcp = se.Matrix(
            [
                x1 - rho_s * se.cos(x5),
                x2 - rho_s * se.sin(x5),
            ]
        )
        ehat = se.Matrix([se.cos(x5), se.sin(x5)])

        h = (rcp.dot(ehat)) / se.sqrt(rcp.dot(rcp)) - se.cos(gamma)
        dh_dx = se.Matrix([se.diff(h, var) for var in [x1, x2, x3, x4, x5]])

        Lgb0_1 = dh_dx.dot(g.col(0))
        Lgb0_2 = dh_dx.dot(g.col(1))
        Lfb0 = dh_dx.dot(f)

        # ----------------- ICCBF layer 1: b1 -----------------
        b1 = Lfb0 + acoef1 * h + Lgb0_1*u1inf + Lgb0_2*u2inf
        db1_dx = se.Matrix([se.diff(b1, var) for var in [x1, x2, x3, x4, x5]])

        Lgb1_1 = db1_dx.dot(g.col(0))
        Lgb1_2 = db1_dx.dot(g.col(1))
        Lfb1 = db1_dx.dot(f)

        # ----------------- ICCBF layer 2: b2 -----------------
        b2 = Lfb1 + Lgb1_1 * u1inf + Lgb1_2 * u2inf + acoef2 * b1
        db2_dx = se.Matrix([se.diff(b2, var) for var in [x1, x2, x3, x4, x5]])

        Lfb2 = db2_dx.dot(f)
        Lgb2_1 = db2_dx.dot(g.col(0))
        Lgb2_2 = db2_dx.dot(g.col(1))

        # ----------------- Lambdify -----------------
        # Order: [x1..x5, a1, a2, rho, m, om]
        state_param_vars = [
            x1, x2, x3, x4, x5,
            acoef1, acoef2,
            rho_s, m_s, om_s, gamma_s, r_s,
        ]
        all_vars = state_param_vars + [u1inf, u2inf]

        b1_func = lambdify(state_param_vars, b1, modules="numpy")
        Lgb1_1_func = lambdify(state_param_vars, Lgb1_1, modules="numpy")
        Lgb1_2_func = lambdify(state_param_vars, Lgb1_2, modules="numpy")

        b2_func = lambdify(all_vars, b2, modules="numpy")
        Lfb2_func = lambdify(all_vars, Lfb2, modules="numpy")
        Lgb2_1_func = lambdify(all_vars, Lgb2_1, modules="numpy")
        Lgb2_2_func = lambdify(all_vars, Lgb2_2, modules="numpy")

        return (
            b2_func,
            Lfb2_func,
            b1_func,
            Lgb1_1_func,
            Lgb1_2_func,
            Lgb2_1_func,
            Lgb2_2_func,
        )
        
    

def _eval_true_iccbf_at_points(iccbf, pts, a1, a2):
    """
    Evaluate the *true* (lambdified nonlinear) ICCBF quantities at many points.

    Uses the same branching logic as your env:
      - compute Lgb1_1, Lgb1_2
      - choose u1inf, u2inf based on sign
      - evaluate b2, Lfb2, Lgb2_1, Lgb2_2

    Returns dict of arrays length N.
    """
    # Grab lambdified functions (fresh; OK for diagnostics)
    (b2_func, Lfb2_func, b1_func, Lgb1_1_func, Lgb1_2_func, Lgb2_1_func, Lgb2_2_func) = iccbf.getICCBFs()

    N = pts.shape[0]
    out = {
        "b2": np.zeros(N),
        "Lfb2": np.zeros(N),
        "Lgb2_1": np.zeros(N),
        "Lgb2_2": np.zeros(N),
        "u1inf": np.zeros(N),
        "u2inf": np.zeros(N),
        "Lgb1_1": np.zeros(N),
        "Lgb1_2": np.zeros(N),
    }

    # parameters from iccbf object (episode-level)
    rho   = float(iccbf.rho)
    m     = float(iccbf.m)
    om    = float(iccbf.om)
    gamma = float(iccbf.gamma)
    r     = float(iccbf.r)
    umax  = float(iccbf.umax)

    for i in range(N):
        x1, x2, x3, x4, x5 = map(float, pts[i, :])

        stateAndcoefs = [x1, x2, x3, x4, x5, float(a1), float(a2), rho, m, om, gamma, r]

        Lgb11 = float(Lgb1_1_func(*stateAndcoefs))
        Lgb12 = float(Lgb1_2_func(*stateAndcoefs))
        out["Lgb1_1"][i] = Lgb11
        out["Lgb1_2"][i] = Lgb12

        u1inf = -umax if (Lgb11 > 0.0) else umax
        u2inf = -umax if (Lgb12 > 0.0) else umax
        out["u1inf"][i] = u1inf
        out["u2inf"][i] = u2inf

        stateCControl = stateAndcoefs + [u1inf, u2inf]

        out["b2"][i]     = float(b2_func(*stateCControl))
        out["Lfb2"][i]   = float(Lfb2_func(*stateCControl))
        out["Lgb2_1"][i] = float(Lgb2_1_func(*stateCControl))
        out["Lgb2_2"][i] = float(Lgb2_2_func(*stateCControl))

    return out


def plot_da_bounds_vs_true_samples(
    iccbf,
    x0,
    a1,
    a2,
    hslack,
    tstep,
    half_width,
    *,
    da_order=4,
    Nsamp=5000,
    seed=0,
    eps_zero=1e-16,
    check_branch=True,
    savepath=None,
    show=True,
):
    """
    Compares DA interval bounds (from DA polynomials) vs true nonlinear values
    (from lambdified symengine/sympy expressions) over the SAME physical box.

    Produces:
    - For each quantity q in {b2, Lfb2, Lgb2_1, Lgb2_2}:
        * plot samples with DA [lb, ub]
        * plot slack-to-bounds: (q - lb) and (ub - q)
        * report max violation if any (should be <= 0)
    - Also checks your composed certificate pieces:
        base_poly = Lf + hslack*b2
        support lower bound via minabs intervals (as in certify_local_validity_best)
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float).reshape(5,)
    hw = np.asarray(half_width, dtype=float).reshape(5,)

    # ---- DA polynomials around x0
    (_vals, polys) = iccbf.getICCBFvars_dace_docking(
        x0=x0, a1=a1, a2=a2, return_polys=True, da_order=da_order
    )
    b2_poly  = polys["hICCBF"]
    Lf_poly  = polys["LfhICCBF"]
    Lg1_poly = polys["Lgh1ICCBF"]
    Lg2_poly = polys["Lgh2ICCBF"]

    # DA bounds for each polynomial
    b2_lb,  b2_ub  = iccbf._bound_over_box(b2_poly,  hw)
    Lf_lb,  Lf_ub  = iccbf._bound_over_box(Lf_poly,  hw)
    Lg1_lb, Lg1_ub = iccbf._bound_over_box(Lg1_poly, hw)
    Lg2_lb, Lg2_ub = iccbf._bound_over_box(Lg2_poly, hw)

    # ---- Sample points in the physical box
    # Uniform in box: x = x0 + U[-hw, hw]
    U = rng.uniform(low=-1.0, high=1.0, size=(Nsamp, 5))
    pts = x0[None, :] + U * hw[None, :]

    # ---- True values (lambdified nonlinear)
    true = _eval_true_iccbf_at_points(iccbf, pts, a1=a1, a2=a2)

    # ---- Helpers for plotting / violations
    def _violations(vals, lb, ub):
        # positive means violation
        v_low  = lb - vals
        v_high = vals - ub
        return np.maximum(v_low, 0.0), np.maximum(v_high, 0.0)

    # ---- Figure: 4 quantities x 2 panels each
    quantities = [
        ("b2",     true["b2"],     b2_lb,  b2_ub),
        ("Lfb2",   true["Lfb2"],   Lf_lb,  Lf_ub),
        ("Lgb2_1", true["Lgb2_1"], Lg1_lb, Lg1_ub),
        ("Lgb2_2", true["Lgb2_2"], Lg2_lb, Lg2_ub),
    ]

    fig, axs = plt.subplots(len(quantities), 2, figsize=(12, 3.2 * len(quantities)))
    if len(quantities) == 1:
        axs = np.array([axs])

    report = {}

    for i, (name, vals, lb, ub) in enumerate(quantities):
        vals = np.asarray(vals, dtype=float)

        # Panel A: samples with DA bounds
        ax = axs[i, 0]
        idx = np.arange(vals.size)
        ax.plot(idx, vals, linestyle="none", marker=".", markersize=2, alpha=0.6, label="true samples")
        ax.axhline(lb, linestyle="--", linewidth=1.5, label="DA lb")
        ax.axhline(ub, linestyle="--", linewidth=1.5, label="DA ub")
        ax.set_title(f"{name}: true samples vs DA interval")
        ax.set_xlabel("sample index")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")

        # Panel B: margin-to-bounds and violations
        ax = axs[i, 1]
        slack_low  = vals - lb
        slack_high = ub - vals
        vlow, vhigh = _violations(vals, lb, ub)

        ax.plot(idx, slack_low,  linestyle="none", marker=".", markersize=2, alpha=0.6, label="(val - lb)")
        ax.plot(idx, slack_high, linestyle="none", marker=".", markersize=2, alpha=0.6, label="(ub - val)")
        ax.axhline(0.0, linestyle=":", linewidth=1.5)
        ax.set_title(f"{name}: slack to bounds (negatives imply violation)")
        ax.set_xlabel("sample index")
        ax.set_ylabel("slack")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")

        max_v = float(max(np.max(vlow), np.max(vhigh)))
        report[name] = {
            "lb": float(lb), "ub": float(ub),
            "min_val": float(np.min(vals)), "max_val": float(np.max(vals)),
            "max_violation": max_v,
            "violations_count": int(np.sum((vlow > 0) | (vhigh > 0))),
        }

    plt.tight_layout()

    # ---- Certificate-piece consistency check (optional but recommended)
    # This is checking your *polynomial* bound vs *sampled true* evaluation
    base_poly = Lf_poly + float(hslack) * b2_poly
    base_lb, base_ub = iccbf._bound_over_box(base_poly, hw)

    # true base samples
    base_true = true["Lfb2"] + float(hslack) * true["b2"]
    vlow_b, vhigh_b = _violations(base_true, base_lb, base_ub)

    # support lower bound via minabs intervals (same as certify_local_validity_best)
    minabs_Lg1 = float(iccbf.minabs_interval(Lg1_lb, Lg1_ub, eps=eps_zero))
    minabs_Lg2 = float(iccbf.minabs_interval(Lg2_lb, Lg2_ub, eps=eps_zero))
    normLg_lb = float(np.sqrt(minabs_Lg1**2 + minabs_Lg2**2))
    support_lb = abs(float(iccbf.umax)) * normLg_lb

    # True support samples use ||[Lg1,Lg2]|| (with the same Lg2_1, Lg2_2)
    support_true = abs(float(iccbf.umax)) * np.sqrt(true["Lgb2_1"]**2 + true["Lgb2_2"]**2)
    support_violation = float(np.max(np.maximum(support_lb - support_true, 0.0)))

    report["base_poly"] = {
        "lb": float(base_lb), "ub": float(base_ub),
        "min_val": float(np.min(base_true)), "max_val": float(np.max(base_true)),
        "max_violation": float(max(np.max(vlow_b), np.max(vhigh_b))),
        "violations_count": int(np.sum((vlow_b > 0) | (vhigh_b > 0))),
    }
    report["support_lb_check"] = {
        "support_lb": float(support_lb),
        "min_support_true": float(np.min(support_true)),
        "max_support_true": float(np.max(support_true)),
        "max_violation_(lb_minus_true)": support_violation,
    }

    if check_branch:
        # whether your box crosses the switching surfaces for uinf selection
        Lgb1_1_poly = polys["Lgb1_1"]
        Lgb1_2_poly = polys["Lgb1_2"]
        b1_1_lb, b1_1_ub = iccbf._bound_over_box(Lgb1_1_poly, hw)
        b1_2_lb, b1_2_ub = iccbf._bound_over_box(Lgb1_2_poly, hw)
        crosses1 = (b1_1_lb <= eps_zero) and (b1_1_ub >= -eps_zero)
        crosses2 = (b1_2_lb <= eps_zero) and (b1_2_ub >= -eps_zero)
        report["branch_surfaces"] = {
            "Lgb1_1_interval": (float(b1_1_lb), float(b1_1_ub)),
            "Lgb1_2_interval": (float(b1_2_lb), float(b1_2_ub)),
            "crosses_Lgb1_1_zero": bool(crosses1),
            "crosses_Lgb1_2_zero": bool(crosses2),
            "minabs_Lgb1_1": float(iccbf.minabs_interval(b1_1_lb, b1_1_ub, eps=eps_zero)),
            "minabs_Lgb1_2": float(iccbf.minabs_interval(b1_2_lb, b1_2_ub, eps=eps_zero)),
        }

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Print a compact summary (so you can paste into paper / logs)
    print("\n=== DA bounds vs true-sample check ===")
    for k, v in report.items():
        if isinstance(v, dict) and "max_violation" in v:
            print(f"{k:>10s}: DA[{v['lb']:+.3e},{v['ub']:+.3e}]  "
                f"true[{v['min_val']:+.3e},{v['max_val']:+.3e}]  "
                f"max_violation={v['max_violation']:.3e}  "
                f"count={v['violations_count']}")
    if "support_lb_check" in report:
        s = report["support_lb_check"]
        print(f"{'support':>10s}: support_lb={s['support_lb']:+.3e}  "
            f"true_min={s['min_support_true']:+.3e}  "
            f"max(lb-true)={s['max_violation_(lb_minus_true)']:.3e}")
    if "branch_surfaces" in report:
        b = report["branch_surfaces"]
        print(f"{'branch':>10s}: Lgb1_1 in [{b['Lgb1_1_interval'][0]:+.3e},{b['Lgb1_1_interval'][1]:+.3e}], "
            f"crosses0={b['crosses_Lgb1_1_zero']};  "
            f"Lgb1_2 in [{b['Lgb1_2_interval'][0]:+.3e},{b['Lgb1_2_interval'][1]:+.3e}], "
            f"crosses0={b['crosses_Lgb1_2_zero']}")

    return report


