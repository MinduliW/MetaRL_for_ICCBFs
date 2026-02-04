# iccbf_inspection_dace.py
import numpy as np
from daceypy import DA


class ICCBFInspectionDA:
    """
    DACEyPy-based ICCBF + sampled-data margin nu for 3D inspection.

    State (6): x = [x, y, z, vx, vy, vz]  (SI units: m, m/s)
    Dynamics (CW):
        xdot = vx
        ydot = vy
        zdot = vz
        vxdot = 3 n^2 x + 2 n vy + (1/m) ux
        vydot = -2 n vx        + (1/m) uy
        vzdot = -n^2 z         + (1/m) uz

    Control constraint: per-axis box |u_i| <= u_max_axis.

    ICCBF (relative degree 2, N=2):
        b1 = Lf h + k1 h
        b2 = Lf b1 + (Lg b1) u_inf + k2 b1
    Branching "u_inf" frozen at x0:
        u_inf,i = -u_max if (Lg_i b1)(x0) > 0 else +u_max

    Margin form (same style as your docking code):
        nu = l1 * T * Delta
        l2 = Lip(Lf b2) + u_max * sum_i Lip(Lg_i b2)     (box-friendly)
        l1 = l2 + Lip(alpha(b2)), alpha(s)=hslack*s => Lip(alpha(b2)) <= |hslack| Lip(b2)

    Notes:
    - We use box-friendly combination for Lg: sum_i Lip(Lg_i) (conservative).
    - Delta bounds sup ||f(x)+g(x)u||_2 over a local box using closed-form CW structure.
    """

    def __init__(self, mu, n, m, rho_koz, rho_kiz, alpha_fov, u_max_axis):
        self.mu = float(mu)  # unused by CW here, kept for consistency
        self.n = float(n)
        self.m = float(m)

        self.rho_koz = float(rho_koz)   # collision / keep-out radius
        self.rho_kiz = float(rho_kiz)   # keep-in radius
        self.alpha_fov = float(alpha_fov)
        self.u_max = float(u_max_axis)

    # ------------------------
    # Interval helper
    # ------------------------
    
    @staticmethod
    def _interval_endpoints(I):
        for a, b in [("lb", "ub"), ("lower", "upper"), ("l", "u"), ("inf", "sup")]:
            if hasattr(I, a) and hasattr(I, b):
                return float(getattr(I, a)), float(getattr(I, b))
        return float(I[0]), float(I[1])

    @staticmethod
    def _interval_maxabs(I):
        for a, b in [("lb", "ub"), ("lower", "upper"), ("l", "u"), ("inf", "sup")]:
            if hasattr(I, a) and hasattr(I, b):
                return max(abs(float(getattr(I, a))), abs(float(getattr(I, b))))
        return max(abs(float(I[0])), abs(float(I[1])))

    @staticmethod
    def minabs_interval(lb, ub, eps=0.0):
        """
        certified inf_{v in [lb,ub]} |v|
        """
        if (lb <= eps) and (ub >= -eps):
            return 0.0
        return min(abs(lb), abs(ub))

    def _bound_over_box(self, poly, half_width6):
        """
        Bound DA poly over the physical box defined by half_width6 about the expansion point.
        """
        hw = np.asarray(half_width6, dtype=float).reshape(6,)
        p = poly
        for j in range(1, 7):
            p = p.scaleVariable(j, float(hw[j - 1]))
        I = p.bound()
        return self._interval_endpoints(I)
    
 
    # ------------------------
    # Lipschitz via DA bounds
    # ------------------------
    def lipschitz_bound(self, poly, half_width6):
        """
        Upper bound on sup ||∇poly||_2 over box centred at DA expansion point.
        half_width6: physical half-widths for [x,y,z,vx,vy,vz].
        """
        hw = np.asarray(half_width6, dtype=float).reshape(6,)
        sup_abs = []
        for i in range(1, 7):  # DA vars 1..6
            dpi = poly.deriv(i)
            # scale each variable to map physical box -> [-1,1]^6 for bound()
            for j in range(1, 7):
                dpi = dpi.scaleVariable(j, float(hw[j - 1]))
            sup_abs_i = self._interval_maxabs(dpi.bound())
            sup_abs.append(float(sup_abs_i))
        return float(np.linalg.norm(sup_abs, ord=2))

    # ------------------------
    # Delta bound for CW + box input
    # ------------------------
    def delta_box_bound_L2(self, center6, half_width6):
        """
        Conservative bound:
            sup_{x in box, |u_i|<=u_max} || f(x) + g u ||_2

        Uses CW structure + triangle inequality componentwise.
        """
        c = np.asarray(center6, dtype=float).reshape(6,)
        hw = np.asarray(half_width6, dtype=float).reshape(6,)

        sx  = abs(c[0]) + hw[0]
        sy  = abs(c[1]) + hw[1]
        sz  = abs(c[2]) + hw[2]
        svx = abs(c[3]) + hw[3]
        svy = abs(c[4]) + hw[4]
        svz = abs(c[5]) + hw[5]

        n = abs(self.n)
        m = max(1e-12, abs(self.m))
        umax = abs(self.u_max)

        # drift bounds
        M1 = svx
        M2 = svy
        M3 = svz
        M4 = 3.0*(n**2)*sx + 2.0*n*svy
        M5 = 2.0*n*svx
        M6 = (n**2)*sz

        # control adds per-axis in accel channels
        Mu = umax / m

        M4_tot = M4 + Mu
        M5_tot = M5 + Mu
        M6_tot = M6 + Mu

        return float(np.sqrt(M1**2 + M2**2 + M3**2 + M4_tot**2 + M5_tot**2 + M6_tot**2))

    # ------------------------
    # Build DA dynamics
    # ------------------------
    def _cw_dynamics(self, x1, x2, x3, x4, x5, x6):
        n = float(self.n)
        # f(x)
        f1 = x4
        f2 = x5
        f3 = x6
        f4 = 3.0*(n**2)*x1 + 2.0*n*x5
        f5 = -2.0*n*x4
        f6 = -(n**2)*x3

        invm = 1.0 / float(self.m)
        # g columns (ux,uy,uz)
        g1 = [DA(0.0), DA(0.0), DA(0.0), DA(invm), DA(0.0), DA(0.0)]
        g2 = [DA(0.0), DA(0.0), DA(0.0), DA(0.0), DA(invm), DA(0.0)]
        g3 = [DA(0.0), DA(0.0), DA(0.0), DA(0.0), DA(0.0), DA(invm)]

        return (f1, f2, f3, f4, f5, f6), (g1, g2, g3)

    # ------------------------
    # Generic ICCBF construction for a given base h(x)
    # ------------------------
    def _iccbf_polys(self, x0, k1, k2, h_builder, da_order=4):
        """
        Returns numeric values and DA polynomials for:
            b2 (as 'hICCBF'), Lf b2, Lg_i b2
        with u_inf branching frozen at x0 via sign of Lg b1.
        """
        x0 = np.asarray(x0, dtype=float).reshape(6,)
        k1 = float(k1)
        k2 = float(k2)

        DA.init(int(da_order), 6)

        x10, x20, x30, x40, x50, x60 = map(float, x0.tolist())
        x1 = DA(1) + x10
        x2 = DA(2) + x20
        x3 = DA(3) + x30
        x4 = DA(4) + x40
        x5 = DA(5) + x50
        x6 = DA(6) + x60

        f, gcols = self._cw_dynamics(x1, x2, x3, x4, x5, x6)
        f1, f2, f3, f4, f5, f6 = f
        g1, g2, g3 = gcols

        # base h(x)
        h = h_builder(x1, x2, x3, x4, x5, x6)

        # ∇h
        dh = [h.deriv(i) for i in range(1, 7)]
        # Lf h
        Lfh = dh[0]*f1 + dh[1]*f2 + dh[2]*f3 + dh[3]*f4 + dh[4]*f5 + dh[5]*f6

        # ICCBF layer 1
        b1 = Lfh + k1*h
        db1 = [b1.deriv(i) for i in range(1, 7)]
        Lgb1_1 = db1[0]*g1[0] + db1[1]*g1[1] + db1[2]*g1[2] + db1[3]*g1[3] + db1[4]*g1[4] + db1[5]*g1[5]
        Lgb1_2 = db1[0]*g2[0] + db1[1]*g2[1] + db1[2]*g2[2] + db1[3]*g2[3] + db1[4]*g2[4] + db1[5]*g2[5]
        Lgb1_3 = db1[0]*g3[0] + db1[1]*g3[1] + db1[2]*g3[2] + db1[3]*g3[3] + db1[4]*g3[4] + db1[5]*g3[5]

        # Freeze branch at x0 (DA eval at zero)
        Lgb1_1_x0 = float(Lgb1_1.eval([0, 0, 0, 0, 0, 0]))
        Lgb1_2_x0 = float(Lgb1_2.eval([0, 0, 0, 0, 0, 0]))
        Lgb1_3_x0 = float(Lgb1_3.eval([0, 0, 0, 0, 0, 0]))

        umax = float(self.u_max)
        u1inf = (-umax) if (Lgb1_1_x0 > 0.0) else (umax)
        u2inf = (-umax) if (Lgb1_2_x0 > 0.0) else (umax)
        u3inf = (-umax) if (Lgb1_3_x0 > 0.0) else (umax)

        # Lf b1
        Lfb1 = db1[0]*f1 + db1[1]*f2 + db1[2]*f3 + db1[3]*f4 + db1[4]*f5 + db1[5]*f6

        # ICCBF layer 2
        b2 = Lfb1 + Lgb1_1*u1inf + Lgb1_2*u2inf + Lgb1_3*u3inf + k2*b1

        # Lf b2 and Lg b2
        db2 = [b2.deriv(i) for i in range(1, 7)]
        Lfb2 = db2[0]*f1 + db2[1]*f2 + db2[2]*f3 + db2[3]*f4 + db2[4]*f5 + db2[5]*f6
        Lgb2_1 = db2[0]*g1[0] + db2[1]*g1[1] + db2[2]*g1[2] + db2[3]*g1[3] + db2[4]*g1[4] + db2[5]*g1[5]
        Lgb2_2 = db2[0]*g2[0] + db2[1]*g2[1] + db2[2]*g2[2] + db2[3]*g2[3] + db2[4]*g2[4] + db2[5]*g2[5]
        Lgb2_3 = db2[0]*g3[0] + db2[1]*g3[1] + db2[2]*g3[2] + db2[3]*g3[3] + db2[4]*g3[4] + db2[5]*g3[5]

        # numeric values at x0
        h_val   = float(b2.eval([0, 0, 0, 0, 0, 0]))
        Lfh_val = float(Lfb2.eval([0, 0, 0, 0, 0, 0]))
        Lgh1_val = float(Lgb2_1.eval([0, 0, 0, 0, 0, 0]))
        Lgh2_val = float(Lgb2_2.eval([0, 0, 0, 0, 0, 0]))
        Lgh3_val = float(Lgb2_3.eval([0, 0, 0, 0, 0, 0]))

        vals = (Lgh1_val, Lgh2_val, Lgh3_val, Lfh_val, h_val)
        polys = {
            "hICCBF": b2,
            "LfhICCBF": Lfb2,
            "Lgh1ICCBF": Lgb2_1,
            "Lgh2ICCBF": Lgb2_2,
            "Lgh3ICCBF": Lgb2_3,
            
            # b1-level terms (for branch checks)
            "b1": b1,
            "Lgb1_1": Lgb1_1,
            "Lgb1_2": Lgb1_2,
            "Lgb1_3": Lgb1_3,

            # branch info
            "u1inf": u1inf,
            "u2inf": u2inf,
            "u3inf": u3inf,
            "branch_info": {
                "Lgb1_1_x0": Lgb1_1_x0,
                "Lgb1_2_x0": Lgb1_2_x0,
                "Lgb1_3_x0": Lgb1_3_x0,
        }
        }
        
        return vals, polys

    # ------------------------
    # Margin computation (generic)
    # ------------------------
    def _getmargin_generic(self, x0, k1, k2, hslack, tstep, half_width6, h_builder, da_order=4):
        vals, polys = self._iccbf_polys(x0, k1, k2, h_builder, da_order=da_order)

        # Lipschitz terms
        l_Lfh = self.lipschitz_bound(polys["LfhICCBF"], half_width6)
        l_Lg1 = self.lipschitz_bound(polys["Lgh1ICCBF"], half_width6)
        l_Lg2 = self.lipschitz_bound(polys["Lgh2ICCBF"], half_width6)
        l_Lg3 = self.lipschitz_bound(polys["Lgh3ICCBF"], half_width6)

        # box-friendly combination: |u_i|<=u_max => worst-case is sum_i |Lg_i| u_max
        # so we combine Lipschitz of each component by sum (conservative)
        u_max = abs(float(self.u_max))
        l_Lg_box = float(l_Lg1 + l_Lg2 + l_Lg3)

        # Lip(b2) for alpha(b2)
        l_h = self.lipschitz_bound(polys["hICCBF"], half_width6)
        l_alpha = abs(float(hslack)) * l_h

        # Delta bound
        Delta = self.delta_box_bound_L2(center6=x0, half_width6=half_width6)

        # assemble
        l2 = l_Lfh + u_max * l_Lg_box
        l1 = l2 + l_alpha
        nu = float(l1 * float(tstep) * float(Delta))
        return nu, vals

    # ------------------------
    # Public: KOZ / KIZ / SUN
    # ------------------------
    def getmargin_koz(self, x0, k1, k2, hslack, tstep, half_width6, da_order=4):
        """
        KOZ: h = (||r||^2 - rho_koz^2) / (rho_kiz^2 - rho_koz^2)
        """
        rho1 = float(self.rho_koz)
        rho2 = float(self.rho_kiz)
        alpha = 1.0 / max(1e-12, (rho2**2 - rho1**2))

        def h_builder(x1, x2, x3, x4, x5, x6):
            return DA(alpha) * (x1*x1 + x2*x2 + x3*x3 - rho1*rho1)

        return self._getmargin_generic(x0, k1, k2, hslack, tstep, half_width6, h_builder, da_order=da_order)

    def getmargin_kiz(self, x0, k1, k2, hslack, tstep, half_width6, da_order=4):
        """
        KIZ: h = (rho_kiz^2 - ||r||^2) / (rho_kiz^2 - rho_koz^2)
        """
        rho1 = float(self.rho_koz)
        rho2 = float(self.rho_kiz)
        alpha = 1.0 / max(1e-12, (rho2**2 - rho1**2))

        def h_builder(x1, x2, x3, x4, x5, x6):
            return DA(alpha) * (rho2*rho2 - (x1*x1 + x2*x2 + x3*x3))

        return self._getmargin_generic(x0, k1, k2, hslack, tstep, half_width6, h_builder, da_order=da_order)

    def getmargin_sun(self, x0, k1, k2, hslack, tstep, half_width6, rsun_unit, da_order=4):
        """
        Sun avoidance (no acos):
            require theta_b >= alpha_fov/2
            <=> cos(theta_b) <= cos(alpha_fov/2)
            with cos(theta_b) = rbhat · rsunhat, rbhat = -r/||r||.

        Define:
            h = cos(alpha/2) - (rbhat · rsunhat)
        so h>=0 is safe.
        """
        rs = np.asarray(rsun_unit, dtype=float).reshape(3,)
        rsn = np.linalg.norm(rs)
        if rsn < 1e-12:
            rs = np.array([1.0, 0.0, 0.0])
        else:
            rs = rs / rsn

        rs1, rs2, rs3 = map(float, rs.tolist())
        cth = float(np.cos(self.alpha_fov / 2.0))

        def h_builder(x1, x2, x3, x4, x5, x6):
            rnorm = (x1*x1 + x2*x2 + x3*x3).sqrt()
            # rbhat = -r/rnorm
            dot = (-x1*DA(rs1) - x2*DA(rs2) - x3*DA(rs3)) / rnorm
            return DA(cth) - dot

        return self._getmargin_generic(x0, k1, k2, hslack, tstep, half_width6, h_builder, da_order=da_order)

    # ------------------------
    # Certificate + optional branch check (add)
    # ------------------------
    def certify_local_validity_best(
        self,
        *,
        x0,
        k1,
        k2,
        hslack,
        tstep,
        half_width6,
        constraint: str,
        rsun_unit=None,
        da_order=4,
        eps_zero=1e-16,
        margin_req=0.0,
        branch_tau=1e-6,
        check_branch=False,
        verbose=False,
    ):
        """
        Local certificate for inspection ICCBF constraint.

        constraint:
            "koz" | "kiz" | "sun"
        For "sun", provide rsun_unit (3,) as in getmargin_sun.

        Computes a certified lower bound of:
            zeta(x) = (Lf b2 + hslack*b2 - nu) + u_max * sum_i |Lg_i b2|
        over x in local box.

        valid iff zeta_lb >= margin_req.
        """
        x0 = np.asarray(x0, dtype=float).reshape(6,)
        hw = np.asarray(half_width6, dtype=float).reshape(6,)

        constraint = str(constraint).lower().strip()
        if constraint not in ("koz", "kiz", "sun"):
            raise ValueError("constraint must be one of: 'koz', 'kiz', 'sun'")

        # -------- build the h_builder + nu --------
        if constraint == "koz":
            rho1 = float(self.rho_koz)
            rho2 = float(self.rho_kiz)
            alpha = 1.0 / max(1e-12, (rho2**2 - rho1**2))

            def h_builder(x1, x2, x3, x4, x5, x6):
                return DA(alpha) * (x1*x1 + x2*x2 + x3*x3 - rho1*rho1)

            nu, _ = self.getmargin_koz(x0, k1, k2, hslack, tstep, hw, da_order=da_order)

        elif constraint == "kiz":
            rho1 = float(self.rho_koz)
            rho2 = float(self.rho_kiz)
            alpha = 1.0 / max(1e-12, (rho2**2 - rho1**2))

            def h_builder(x1, x2, x3, x4, x5, x6):
                return DA(alpha) * (rho2*rho2 - (x1*x1 + x2*x2 + x3*x3))

            nu, _ = self.getmargin_kiz(x0, k1, k2, hslack, tstep, hw, da_order=da_order)

        else:  # "sun"
            rs = np.asarray(rsun_unit, dtype=float).reshape(3,)
            rsn = np.linalg.norm(rs)
            if rsn < 1e-12:
                rs = np.array([1.0, 0.0, 0.0])
            else:
                rs = rs / rsn

            rs1, rs2, rs3 = map(float, rs.tolist())
            alpha_half = float(self.alpha_fov / 2.0)

            def h_builder(x1, x2, x3, x4, x5, x6):
                rnorm = (x1*x1 + x2*x2 + x3*x3).sqrt()
                # rbhat = -r/rnorm; dot = rbhat·rsunhat
                dot = (-x1*DA(rs1) - x2*DA(rs2) - x3*DA(rs3)) / rnorm
                thetab = dot.acos()
                return thetab - DA(alpha_half)


            nu, _ = self.getmargin_sun(x0, k1, k2, hslack, tstep, hw, rs, da_order=da_order)

   
        # -------- build polynomials --------
        (_vals, polys) = self._iccbf_polys(x0, k1, k2, h_builder, da_order=da_order)

        b2_poly = polys["hICCBF"]
        Lf_poly = polys["LfhICCBF"]
        Lg1_poly = polys["Lgh1ICCBF"]
        Lg2_poly = polys["Lgh2ICCBF"]
        Lg3_poly = polys["Lgh3ICCBF"]

        # Base term: inf_x [Lf b2 + hslack*b2 - nu]
        base_poly = Lf_poly + float(hslack) * b2_poly - DA(nu)
        base_lb, base_ub = self._bound_over_box(base_poly, hw)

        # Support term for box control: u_max * sum_i |Lg_i|
        Lg1_lb, Lg1_ub = self._bound_over_box(Lg1_poly, hw)
        Lg2_lb, Lg2_ub = self._bound_over_box(Lg2_poly, hw)
        Lg3_lb, Lg3_ub = self._bound_over_box(Lg3_poly, hw)

        minabs_Lg1 = float(self.minabs_interval(Lg1_lb, Lg1_ub, eps=eps_zero))
        minabs_Lg2 = float(self.minabs_interval(Lg2_lb, Lg2_ub, eps=eps_zero))
        minabs_Lg3 = float(self.minabs_interval(Lg3_lb, Lg3_ub, eps=eps_zero))

        support_lb = abs(float(self.u_max)) * float(minabs_Lg1 + minabs_Lg2 + minabs_Lg3)

        zeta_lb = float(base_lb + support_lb)
        valid = bool(zeta_lb >= float(margin_req))

        # -------- optional branch check (sign(Lg b1) must not flip in box) --------
        branch_ok = True
        branch_margin_lb = None
        Lgb1_intervals = None

        if check_branch:
            Lgb1_1_poly = polys["Lgb1_1"]
            Lgb1_2_poly = polys["Lgb1_2"]
            Lgb1_3_poly = polys["Lgb1_3"]

            b1_1_lb, b1_1_ub = self._bound_over_box(Lgb1_1_poly, hw)
            b1_2_lb, b1_2_ub = self._bound_over_box(Lgb1_2_poly, hw)
            b1_3_lb, b1_3_ub = self._bound_over_box(Lgb1_3_poly, hw)

            Lgb1_intervals = (
                (float(b1_1_lb), float(b1_1_ub)),
                (float(b1_2_lb), float(b1_2_ub)),
                (float(b1_3_lb), float(b1_3_ub)),
            )

            crosses1 = (b1_1_lb <= eps_zero) and (b1_1_ub >= -eps_zero)
            crosses2 = (b1_2_lb <= eps_zero) and (b1_2_ub >= -eps_zero)
            crosses3 = (b1_3_lb <= eps_zero) and (b1_3_ub >= -eps_zero)

            minabs1 = self.minabs_interval(b1_1_lb, b1_1_ub, eps=eps_zero)
            minabs2 = self.minabs_interval(b1_2_lb, b1_2_ub, eps=eps_zero)
            minabs3 = self.minabs_interval(b1_3_lb, b1_3_ub, eps=eps_zero)
            branch_margin_lb = float(min(minabs1, minabs2, minabs3))

            branch_ok = (not crosses1) and (not crosses2) and (not crosses3) and (branch_margin_lb >= float(branch_tau))

            # If you want certification to REQUIRE branch stability, uncomment:
            # valid = valid and branch_ok

        if verbose:
            print(f"[cert-{constraint}] base_lb={base_lb:.3e}, support_lb={support_lb:.3e}, zeta_lb={zeta_lb:.3e}, nu={nu:.3e}")

        return {
            "constraint": constraint,
            "valid": valid,
            "zeta_lb": zeta_lb,
            "margin_req": float(margin_req),

            "base_lb": float(base_lb),
            "base_ub": float(base_ub),
            "support_lb": float(support_lb),
            "nu": float(nu),

            "Lg_interval": (
                (float(Lg1_lb), float(Lg1_ub)),
                (float(Lg2_lb), float(Lg2_ub)),
                (float(Lg3_lb), float(Lg3_ub)),
            ),
            "minabs_Lg": (float(minabs_Lg1), float(minabs_Lg2), float(minabs_Lg3)),

            "branch_ok": bool(branch_ok),
            "branch_margin_lb": None if branch_margin_lb is None else float(branch_margin_lb),
            "Lgb1_intervals": Lgb1_intervals,
            "branch_info_at_x0": polys.get("branch_info", None),
        }
