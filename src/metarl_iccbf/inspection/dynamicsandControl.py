import numpy as np
import cvxpy as cp
from scipy.linalg import expm  # for exact ZOH discretization
# from Dockingcase import DockingCase  # keep/import if you actually use it


class dynamicsAndControl:
    """
    3D JAIS inspection test case dynamics.

    Translational state (CWH in Hill frame):
        x_trans = [x, y, z, vx, vy, vz]^T

    Full state used here:
        x_full  = [x, y, z, vx, vy, vz, theta_S]^T

    Control (actions):
        u = [Fx, Fy, Fz]^T   (thrust forces in Hill-frame axes)

    Continuous-time CWH:
        ẍ = 3 n^2 x + 2 n ẏ + Fx/m
        ÿ = -2 n ẋ + Fy/m
        z̈ = -n^2 z + Fz/m

    Sun angle (discrete, from the paper):
        theta_S(k+1) = theta_S(k) - n Δt
    """

    def __init__(self, mu, n, rc, m):
        """
        Parameters (mu, r kept for compatibility but unused in CWH):
            mu : gravitational parameter (unused for linear CWH)
            n  : mean motion [rad/s]
            rc  : chief orbital radius (unused here)
            m  : deputy spacecraft mass
            om : sun angular rate; if None, set to -n so that
                 theta_S(k+1) = theta_S(k) - n Δt (JAIS Eq. 18)
        """
        self.mu = mu
        self.n = n
        self.rc = rc
        self.m = m

        # cache for discrete A(Δt), B(Δt)
        self._dt_cache = None
        self._Ad_cache = None
        self._Bd_cache = None

    # ----------------------------------------------------------------------
    # Continuous-time dynamics: f(x), g(x) for CBFs etc.
    # ----------------------------------------------------------------------
    def getfxgx(self, x):
        """
        Continuous-time CWH dynamics (3D) plus sun angle rate.

        Input:
            x : state vector of length 7 or 6:
                [x, y, z, vx, vy, vz, theta_S] or [x, y, z, vx, vy, vz]

        Returns:
            f_x : drift term (same shape as x)
            g_x : control matrix mapping u = [Fx, Fy, Fz] to ẋ
                  shape (len(x), 3)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size not in (6, 7):
            raise ValueError(
                f"Expected state of length 6 or 7, got length {x.size}."
            )

        n = self.n
        m = self.m

        # unpack translational part
        xr, yr, zr, vx, vy, vz = x[:6]

        # CWH accelerations
        ax = 3.0 * n**2 * xr + 2.0 * n * vy
        ay = -2.0 * n * vx
        az = -1.0 * n**2 * zr

        if x.size == 7:
            # sun angle dynamics: thetȧ_S = om (≈ -n to match Eq. 18)
            theta_dot = -self.n
            f_x = np.array(
                [vx, vy, vz, ax, ay, az, theta_dot],
                dtype=np.float64
            )
            g_x = (1.0 / m) * np.array(
                [
                    [0.0, 0.0, 0.0],  # ẋ
                    [0.0, 0.0, 0.0],  # ẏ
                    [0.0, 0.0, 0.0],  # ż
                    [1.0, 0.0, 0.0],  # v̇x
                    [0.0, 1.0, 0.0],  # v̇y
                    [0.0, 0.0, 1.0],  # v̇z
                    [0.0, 0.0, 0.0],  # thetȧ_S (no direct control)
                ],
                dtype=np.float64,
            )
        else:
            f_x = np.array(
                [vx, vy, vz, ax, ay, az],
                dtype=np.float64
            )
            g_x = (1.0 / m) * np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        return f_x, g_x

    # Alias, since the JAIS test case uses CWH
    def getfxgxCW(self, x):
        return self.getfxgx(x)

    # ----------------------------------------------------------------------
    # Continuous-time system matrices for translational CWH (6×6, 6×3)
    # ----------------------------------------------------------------------
    def _continuous_AB(self):
        """
        Continuous-time linear system:
            ẋ_trans = A_c x_trans + B_c u

        where:
            x_trans = [x, y, z, vx, vy, vz]^T
            u       = [Fx, Fy, Fz]^T
        """
        n = self.n
        m = self.m

        A_c = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [3.0 * n**2, 0.0, 0.0, 0.0, 2.0 * n, 0.0],
                [0.0, 0.0, 0.0, -2.0 * n, 0.0, 0.0],
                [0.0, 0.0, -1.0 * n**2, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        B_c = (1.0 / m) * np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return A_c, B_c

    # ----------------------------------------------------------------------
    # Exact ZOH discretization: A(Δt), B(Δt)
    # ----------------------------------------------------------------------
    def _discrete_AB(self, dt):
        """
        Discrete-time CW/Hill dynamics per Eq. (17) of the Safe Spacecraft Inspection paper.

        State: x = [x, y, z, xdot, ydot, zdot]
        Input: u = [Fx, Fy, Fz]  (FORCES). B includes the 1/m factor.

        x_{k+1} = A(dt) x_k + B(dt) u_k
        """
        dt = float(dt)

        if self._dt_cache == dt and self._Ad_cache is not None:
            return self._Ad_cache, self._Bd_cache

        n = float(self.n)     # mean motion [rad/s]
        m = float(self.m)     # mass [kg]

        if abs(n) < 1e-12:
            raise ValueError("Mean motion n is too close to zero; Eq. (17) CW discretisation is ill-conditioned.")

        nt = n * dt
        c = np.cos(nt)
        s = np.sin(nt)

        # --- A(dt): 6x6 ---
        A = np.zeros((6, 6), dtype=np.float64)

        A[0, 0] = 4.0 - 3.0 * c
        A[0, 3] = (1.0 / n) * s
        A[0, 4] = (2.0 / n) * (1.0 - c)

        A[1, 0] = 6.0 * (s - nt)
        A[1, 1] = 1.0
        A[1, 3] = -(2.0 / n) * (1.0 - c)
        A[1, 4] = (1.0 / n) * (4.0 * s - 3.0 * nt)

        A[2, 2] = c
        A[2, 5] = (1.0 / n) * s

        A[3, 0] = 3.0 * n * s
        A[3, 3] = c
        A[3, 4] = 2.0 * s

        A[4, 0] = -6.0 * n * (1.0 - c)
        A[4, 3] = -2.0 * s
        A[4, 4] = 4.0 * c - 3.0

        A[5, 2] = -n * s
        A[5, 5] = c

        # --- B(dt): 6x3, includes 1/m ---
        B = np.zeros((6, 3), dtype=np.float64)

        # Helpful scalars
        inv_n  = 1.0 / n
        inv_n2 = 1.0 / (n * n)

        dt_minus_s_over_n = dt - inv_n * s
        cos_minus_1 = (c - 1.0)  # note: many terms use (cos - 1)

        # Row 1
        B[0, 0] = inv_n2 * (1.0 - c)
        B[0, 1] = (2.0 * inv_n) * dt_minus_s_over_n
        # B[0, 2] = 0

        # Row 2
        B[1, 0] = (-2.0 * inv_n) * dt_minus_s_over_n
        B[1, 1] = (4.0 * inv_n2) * (1.0 - c) - 1.5 * dt * dt
        # B[1, 2] = 0

        # Row 3
        # -(1/n^2) (cos(nt) - 1) == (1/n^2)(1 - cos(nt))
        B[2, 2] = -inv_n2 * cos_minus_1

        # Row 4
        B[3, 0] = inv_n * s
        B[3, 1] = (-2.0 * inv_n) * cos_minus_1
        # B[3, 2] = 0

        # Row 5
        B[4, 0] = (2.0 * inv_n) * cos_minus_1
        B[4, 1] = (4.0 * inv_n) * s - 3.0 * dt
        # B[4, 2] = 0

        # Row 6
        B[5, 2] = inv_n * s

        B *= (1.0 / m)

        self._dt_cache = dt
        self._Ad_cache = A
        self._Bd_cache = B
        return A, B


    # ----------------------------------------------------------------------
    # Discrete-time propagation using CWH + sun dynamics (JAIS test case)
    # ----------------------------------------------------------------------
    def propwithCW(self, x, uOpt, dt):
        """
        Discrete-time propagation using the JAIS test-case model:

          Translational dynamics:
            x_trans(k+1) = A(Δt) x_trans(k) + B(Δt) u(k)

          Sun angle:
            theta_S(k+1) = theta_S(k) - n Δt    (Eq. 18)

        Inputs:
            x    : [x, y, z, vx, vy, vz, theta_S]  (or length 6 without theta_S)
            uOpt : [Fx, Fy, Fz]
            dt   : time step Δt
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        u = np.asarray(uOpt, dtype=np.float64).reshape(-1)
        if u.size != 3:
            raise ValueError(f"uOpt must have length 3 (Fx,Fy,Fz), got {uOpt!r}")

        if x.size not in (6, 7):
            raise ValueError(
                f"State must have length 6 or 7, got length {x.size}."
            )

        A_d, B_d = self._discrete_AB(dt)

        # translational part
        x_trans = x[:6]
        x_trans_next = A_d @ x_trans + B_d @ u

        if x.size == 7:
            theta_S = float(x[6])
            # Eq. (18): theta_S(t+Δt) = theta_S(t) - n Δt
            theta_S_next = theta_S - self.n* dt  # om ≈ -n
            x_next = np.concatenate([x_trans_next, [theta_S_next]])
        else:
            x_next = x_trans_next

        return x_next.astype(np.float64)

    
 
