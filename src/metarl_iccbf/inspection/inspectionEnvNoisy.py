# inspection_env.py (simplified)
# - Removes ALL "certification"/local-validity logic and related penalties
# - Keeps: episode spec loading, param randomisation, DA-based margins (nu1/nu2/nu3), ICCBF-QP, noise, reward, terminations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cvxpy as cp
from pathlib import Path

from inspection.iccbfs import ICCBF   # your sympy/symengine lambdified ICCBFs
from inspection.iccbfda import ICCBFInspectionDA
from inspection.dynamicsandControl import dynamicsAndControl
from inspection.Observation import ObservationModel


class InspectionEnv(gym.Env):
    # ================================================================
    # Init
    # ================================================================
    def __init__(self, dt = 10.0, enable_param_randomisation = False,enableNoise = False, enableCBFtunning = False, dvWeight = 10.0):
        # time
        self.tstepOriginal = float(dt)
        self.DT = float(dt)
        self.steps_done = 0
        self.MAX_STEPS = 1224  # paper setup


        self.dvWeight = float(dvWeight)
 
        self.m = 12.0            # kg
        self.mu = 3.986004418e14     # if CW uses km; dynamics class dictates actual units
   
        self.R_D = 5.0           # m
        self.R_C = 10.0          # m

        self.alpha_fov = np.deg2rad(60.0)   # rad (half-angle)
        self.R_MAX = 800.0                  # m keep-in
        self.V_MAX = 5.0                    # m/s (if used)
        self.U_MAX = 1.0                    # N max thrust magnitude (L2 ball in your notes, per-axis sat in code)

   

        # inspection discretisation
        self.N_POINTS = 100
        self.K_CLUSTERS = 4
        self.KMEANS_ITERS = 10

        # -------------------------
        # Store nominal values for sampling
        # -------------------------
        self.base_m = float(self.m)
        self.base_R_D = float(self.R_D)
        self.base_R_C = float(self.R_C)
        self.base_R_MAX = float(self.R_MAX)
        self.base_U_MAX = float(self.U_MAX)
        self.base_r_orbit = float(6771.0e3)  # m
        self.r = float(6771.0e3) 
        
        self.n = np.sqrt(self.mu / self.base_r_orbit ** 3)

        # -------------------------
        # Episode parameter randomisation config
        # -------------------------
        self.enable_param_randomisation = enable_param_randomisation
        self.frac_m = 0.10
        self.frac_R_D = 0.10
        self.frac_R_C = 0.10
        self.frac_U_MAX = 0.10
        self.frac_R_MAX = 0.10
        self.frac_r = 0.10

        def _pm(base, frac):
            return (base * (1.0 - frac), base * (1.0 + frac))

        self.pmin = {
            "m": _pm(self.base_m, self.frac_m)[0],
            "R_D": _pm(self.base_R_D, self.frac_R_D)[0],
            "R_C": _pm(self.base_R_C, self.frac_R_C)[0],
            "U_MAX": _pm(self.base_U_MAX, self.frac_U_MAX)[0],
            "r": _pm(self.base_r_orbit, self.frac_r)[0],
            "R_MAX": _pm(self.base_R_MAX, self.frac_R_MAX)[0],
        }
        self.pmax = {
            "m": _pm(self.base_m, self.frac_m)[1],
            "R_D": _pm(self.base_R_D, self.frac_R_D)[1],
            "R_C": _pm(self.base_R_C, self.frac_R_C)[1],
            "U_MAX": _pm(self.base_U_MAX, self.frac_U_MAX)[1],
            "r": _pm(self.base_r_orbit, self.frac_r)[1],
            "R_MAX": _pm(self.base_R_MAX, self.frac_R_MAX)[1],
        }

        # -------------------------
        # Noise settings
        # -------------------------
        self.enable_actuation_noise = enableNoise
        self.sigma_u_mag = 0.05            # N
        self.sigma_beta = np.deg2rad(0.1)  # rad
        self.sigma_gamma = np.deg2rad(0.1) # rad

        self.enable_state_meas_noise = enableNoise
        self.sigma_pos = 0.1       # m
        self.sigma_vel = 0.002     # m/s
        self.sigma_theta_s = 0.0   # rad

        # -------------------------
        # Derived + models
        
        self.enableCBFtunning = enableCBFtunning
        
        self._rebuild_models()
        
        self.iccbf_sym = ICCBF(
        rho1=  self.base_R_D + self.base_R_C,               # inner boundary
        rho2=float(self.R_MAX), # outer boundary
        mu=self.mu,
        n=self.n,
        m=self.m,
        alphaFOV=float(self.alpha_fov),
        )
        self.CBF1Vals, self.CBF2Vals, self.CBF3Vals = self.iccbf_sym.getICCBFs()
        
        

        # -------------------------
        # Initial distribution / cone params
        # -------------------------
        self.INIT_RANGE_MIN = 50.0
        self.INIT_RANGE_MAX = 100.0
        self.THETA_B_MIN_DEG = 40.0

        # -------------------------
        # Environment state
        # -------------------------
        self.state = np.zeros(7, dtype=np.float64)          # [x,y,z,vx,vy,vz,theta_s]
        self.inspected = np.zeros(self.N_POINTS, dtype=bool)

        # -------------------------
        # Observation/action spaces
        # -------------------------
        self.numObs = 11
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.numObs, dtype=np.float32),
            high=np.inf * np.ones(self.numObs, dtype=np.float32),
            dtype=np.float32,
        )

        if  self.enableCBFtunning:
            num_actions = 12  # [u_rl(3)] + [hslack1,hslack2,hslack3,a1,a2,b1,b2,c1,c2]
        else:
            num_actions = 3  # [u_rl(3)]
            
        self.action_space = spaces.Box(
            low=-np.ones(num_actions, dtype=np.float32),
            high=np.ones(num_actions, dtype=np.float32),
            dtype=np.float32,
        )

        self.last_u_rl = np.zeros(3, dtype=np.float64)
        self.last_u_safe = np.zeros(3, dtype=np.float64)


    def _rebuild_models(self):
        KOZ = float(self.R_C + self.R_D)

        self.iccbf_da = ICCBFInspectionDA(
                mu=self.mu,
                n=self.n,
                m=self.m,
                rho_koz=KOZ,
                rho_kiz=self.R_MAX,
                alpha_fov=self.alpha_fov,
                u_max_axis=self.U_MAX,
            )
  

        self.dynamics = dynamicsAndControl(mu=self.mu, n=self.n, rc=self.R_C, m=self.m)

    
        self.obs_model = ObservationModel.from_spherical_chief(
            radius=self.R_C,
            n_points=self.N_POINTS,
            base_rgb_value=1.0,
        )
        
        # self.iccbf_sym = ICCBF(
        # rho1=KOZ,               # inner boundary
        # rho2=float(self.R_MAX), # outer boundary
        # mu=self.mu,
        # n=self.n,
        # m=self.m,
        # alphaFOV=float(self.alpha_fov),
        # )
      
            
    def _sample_episode_params(self, rng: np.random.Generator) -> dict:
        return {k: float(rng.uniform(self.pmin[k], self.pmax[k])) for k in self.pmin.keys()}

    @staticmethod
    def _sat_axis(u: np.ndarray, umax: float) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        umax = float(umax)
        out = u.copy()
        for i in range(out.size):
            denom = max(umax, abs(out[i]))
            out[i] = (umax / denom) * out[i]
        return out

    def _apply_execution_noise(self, u_cmd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        u_cmd = np.asarray(u_cmd, dtype=float).reshape(3,)
        u_cmd = self._sat_axis(u_cmd, self.U_MAX)

        if not self.enable_actuation_noise:
            return u_cmd

        u_star = u_cmd.copy()
        u_k = float(np.linalg.norm(u_star))
        if u_k < 1e-12:
            return u_star

        beta = float(np.arcsin(np.clip(u_star[2] / u_k, -1.0, 1.0)))
        gamma = float(np.arctan2(u_star[0], u_star[1]))

        du = float(rng.normal(0.0, self.sigma_u_mag))
        db = float(rng.normal(0.0, self.sigma_beta))
        dg = float(rng.normal(0.0, self.sigma_gamma))

        u_kE = max(0.0, u_k + du)
        betaE = beta + db
        gammaE = gamma + dg

        cB, sB = np.cos(betaE), np.sin(betaE)
        sG, cG = np.sin(gammaE), np.cos(gammaE)

        u_exec = np.array([u_kE * cB * sG, u_kE * cB * cG, u_kE * sB], dtype=float)
        return self._sat_axis(u_exec, self.U_MAX)


    def _sym_u_inf_from_Lgb1(self, cbf_vals, state_vars, u_max: float):
        """
        Replicate DA branching:
            u_i_inf = -u_max if (Lgb1_i)(x0) > 0 else +u_max
        using SymPy-evaluated Lgb1_i at x0.
        """
        Lgb1_1 = float(cbf_vals.Lgb1_1_func(*state_vars))
        Lgb1_2 = float(cbf_vals.Lgb1_2_func(*state_vars))
        Lgb1_3 = float(cbf_vals.Lgb1_3_func(*state_vars))

        umax = float(u_max)
        u1inf = (-umax) if (Lgb1_1 > 0.0) else (umax)
        u2inf = (-umax) if (Lgb1_2 > 0.0) else (umax)
        u3inf = (-umax) if (Lgb1_3 > 0.0) else (umax)
        return np.array([u1inf, u2inf, u3inf], dtype=float)


    def _sym_eval_b2_terms(self, cbf_vals, *, x6, gains, u_max, rsun_unit=None):
        """
        Returns (h, Lf, Lg1, Lg2, Lg3) where h is b2(x0) (ICCBF layer-2 value),
        and (Lf, Lg) are Lie derivatives of b2 at x0.

        Critically: chooses u_inf using the sign of Lg b1 at x0 (DA-consistent).
        """
        x6 = np.asarray(x6, dtype=float).reshape(6,)
        x1, x2, x3, x4, x5, x6v = map(float, x6.tolist())

        # gains = (a1,a2,b1,b2,c1,c2) in YOUR env convention
        a1, a2, b1, b2, c1, c2 = map(float, gains)

        if rsun_unit is None:
            # matches iccbfs.py for CBF1/CBF2:
            # state_vars = [x1..x6, acoef1,acoef2,bcoef1,bcoef2,ccoef1,ccoef2]
            state_vars = [x1, x2, x3, x4, x5, x6v, a1, a2, b1, b2, c1, c2, self.m, self.R_C, self.R_D, self.r, self.R_MAX]
        else:
            rs = np.asarray(rsun_unit, dtype=float).reshape(3,)
            rsn = float(np.linalg.norm(rs))
            rs = (rs / rsn) if rsn > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)
            rs1, rs2, rs3 = map(float, rs.tolist())

            # matches iccbfs.py for CBF3:
            # state_vars = [x1..x6, rs1,rs2,rs3, acoef1,acoef2,bcoef1,bcoef2,ccoef1,ccoef2]
            state_vars = [x1, x2, x3, x4, x5, x6v, rs1, rs2, rs3, a1, a2, b1, b2, c1, c2, self.m, self.R_C, self.R_D, self.r, self.R_MAX]

        # --- DA-consistent branching for u_inf using SymPy Lg(b1) signs ---
        u_inf = self._sym_u_inf_from_Lgb1(cbf_vals, state_vars, u_max=u_max)
        u1inf, u2inf, u3inf = map(float, u_inf.tolist())

        all_vars = list(state_vars) + [u1inf, u2inf, u3inf]

        # b2-level terms (these are what your DA code returns as vals)
        h_b2  = float(cbf_vals.b2_func(*all_vars))
        Lf_b2 = float(cbf_vals.Lfb2_func(*all_vars))
        Lg1   = float(cbf_vals.Lgb2_1_func(*all_vars))
        Lg2   = float(cbf_vals.Lgb2_2_func(*all_vars))
        Lg3   = float(cbf_vals.Lgb2_3_func(*all_vars))

        return h_b2, Lf_b2, Lg1, Lg2, Lg3, u_inf


    def _noisy_measurement_state(self, x_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(x_true, dtype=float).copy()
        if not self.enable_state_meas_noise:
            return x
        x[0:3] += rng.normal(0.0, self.sigma_pos, size=3)
        x[3:6] += rng.normal(0.0, self.sigma_vel, size=3)
        x[6] += rng.normal(0.0, self.sigma_theta_s)
        x[6] = float(np.mod(x[6], 2.0 * np.pi))
        return x

    def _sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        r0_mag = float(rng.uniform(self.INIT_RANGE_MIN, self.INIT_RANGE_MAX))
        az = float(rng.uniform(0.0, 2.0 * np.pi))
        el = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))

        r0 = np.array([
            r0_mag * np.cos(el) * np.cos(az),
            r0_mag * np.cos(el) * np.sin(az),
            r0_mag * np.sin(el),
        ], dtype=float)

        v0 = np.zeros(3, dtype=float)
        theta_s0 = float(rng.uniform(0.0, 2.0 * np.pi))

        sun_dir =  self.obs_model.sun_direction(theta_s0)
        r_norm = float(np.linalg.norm(r0))
        boresight = -r0 / r_norm  # deputy -> chief

        cos_theta_b = float(np.clip(boresight.dot(sun_dir), -1.0, 1.0))
        theta_b = float(np.arccos(cos_theta_b))
        if theta_b < np.deg2rad(self.THETA_B_MIN_DEG):
            r0 = -r0

        return np.concatenate([r0, v0, [theta_s0]]).astype(np.float64)

    def _largest_cluster_direction(self) -> np.ndarray:
        unins_mask = ~self.inspected
        if not np.any(unins_mask):
            return np.zeros(3)

        pts = self.obs_model.surface_points[unins_mask]
        M = pts.shape[0]
        K = min(self.K_CLUSTERS, M)

        idx0 = self.np_random.choice(M, size=K, replace=False)
        centroids = pts[idx0].copy()

        for _ in range(self.KMEANS_ITERS):
            dists = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(K):
                mask_k = labels == k
                if np.any(mask_k):
                    centroids[k] = pts[mask_k].mean(axis=0)

        counts = np.array([(labels == k).sum() for k in range(K)])
        centroid = centroids[int(np.argmax(counts))]

        pos = self.state[:3]
        vec = centroid - pos
        norm = float(np.linalg.norm(vec))
        return np.zeros(3) if norm < 1e-9 else (vec / norm)

    def _get_obs_vector(self) -> np.ndarray:
        """
        Returns an 11-dim observation scaled (not clipped) to roughly [-1, 1]
        under nominal conditions. If the state goes outside nominal ranges,
        values may exceed [-1, 1] by design.
        """
        x_meas = self._noisy_measurement_state(self.state, self.np_random)
        x, y, z, vx, vy, vz, theta_s = map(float, x_meas)

        # counts + cluster direction
        P_i = float(self.inspected.sum())                 # [0, N_POINTS]
        P_c = self._largest_cluster_direction()           # nominally unit
        P_cx, P_cy, P_cz = map(float, P_c.tolist())

        # scales (no extra helpers)
        posS = float(800.0)                          # metres
        velS = float(10.0) 

        # angle -> [-1,1] (no clip)
        theta_s = float(np.mod(theta_s, 2.0 * np.pi))
        thetaN = 2.0 * (theta_s / (2.0 * np.pi)) - 1.0

        # inspected points -> [-1,1] (no clip)
        PiN = 2.0 * (P_i / float(self.N_POINTS)) - 1.0

        return np.array(
            [
                x / posS, y / posS, z / posS,
                vx / velS, vy / velS, vz / velS,
                thetaN, PiN,
                P_cx, P_cy, P_cz,
            ],
            dtype=np.float32,
        )

   
    def reset(self, *, seed=None, options=None):
  
        super().reset(seed=seed)
        rng = self.np_random

        # ------------------------------------------------------------
        # 1) Optional parameter randomisation (training only)
        # ------------------------------------------------------------
        if self.enable_param_randomisation:
            p = self._sample_episode_params(rng)

            self.m = float(p["m"])
            self.R_D = float(p["R_D"])
            self.R_C = float(p["R_C"])
            self.U_MAX = float(p["U_MAX"])
            self.R_MAX = float(p["R_MAX"])
            self.r = float(p["r"])
            self.n = float(np.sqrt(self.mu / self.r ** 3))

            self._rebuild_models()

        # ------------------------------------------------------------
        # 2) Sample initial state from the paper-style distribution
        # ------------------------------------------------------------
        self.state = self._sample_initial_state(rng)

        # ------------------------------------------------------------
        # 3) Reset inspection state + initial visibility
        # ------------------------------------------------------------
        self.steps_done = 0
        self.inspected = np.zeros(self.N_POINTS, dtype=bool)

        obs_dict = self.obs_model.get_observation(self._noisy_measurement_state(self.state, rng))
        self.inspected |= obs_dict["visible_mask"]

        # ------------------------------------------------------------
        # 4) Reset logs
        # ------------------------------------------------------------
        self.last_u_rl[:] = 0.0
        self.last_u_safe[:] = 0.0

        obs = self._get_obs_vector()
        info = {
            "num_inspected": int(self.inspected.sum()),
            "m": float(self.m),
            "n": float(self.n),
            "R_C": float(self.R_C),
            "R_D": float(self.R_D),
            "R_MAX": float(self.R_MAX),
            "U_MAX": float(self.U_MAX),
            "alpha_fov": float(self.alpha_fov),
            "DT": float(self.DT),
        }
        return obs, info

    # ================================================================
    # Control utilities
    def getControl(self, *, hslack1, hslack2, hslack3, a1, a2, b1, b2, c1, c2, u_rl):
        # f_x, g_x = self.dynamics.getfxgx(self.state[:6])

        # margin settings (keep as you had it)
        half_width6 = np.array([2.0, 2.0, 2.0, 0.05, 0.05, 0.05], dtype=float)
        x6 = np.asarray(self.state[:6], dtype=float)
        T = float(self.DT)

        # -------------------------
        # 1) Margins from DA (keep)
        # -------------------------
        nu1, _ = self.iccbf_da.getmargin_koz(x6, k1=a1, k2=a2, hslack=hslack1, tstep=T, half_width6=half_width6)
        nu2, _ = self.iccbf_da.getmargin_kiz(x6, k1=b1, k2=b2, hslack=hslack2, tstep=T, half_width6=half_width6)

        rsun =  self.obs_model.sun_direction(self.state[6])
        rsun = np.asarray(rsun, dtype=float)
        rsn = float(np.linalg.norm(rsun))
        rsun_unit = (rsun / rsn)
        
        nu3, _ = self.iccbf_da.getmargin_sun(
            x6, k1=c1, k2=c2, hslack=hslack3, tstep=T, half_width6=half_width6, rsun_unit=rsun_unit
        )

        # -------------------------
        # 2) SymPy coefficients (b2-level) with DA-consistent u_inf branching
        # -------------------------
        gains = (a1, a2, b1, b2, c1, c2)
        umax = float(self.U_MAX)

        h1, Lfh1, Lgh1_1, Lgh1_2, Lgh1_3, uinf1 = self._sym_eval_b2_terms(
            self.CBF1Vals, x6=x6, gains=gains, u_max=umax, rsun_unit=None
        )
        h2, Lfh2, Lgh2_1, Lgh2_2, Lgh2_3, uinf2 = self._sym_eval_b2_terms(
            self.CBF2Vals, x6=x6, gains=gains, u_max=umax, rsun_unit=None
        )
        h3, Lfh3, Lgh3_1, Lgh3_2, Lgh3_3, uinf3 = self._sym_eval_b2_terms(
            self.CBF3Vals, x6=x6, gains=gains, u_max=umax, rsun_unit=rsun_unit
        )

        # -------------------------
        # 3) Solve QP (unchanged)
        # -------------------------
        uOpt, isSolved = self.qp_optimizationICCBF(
            u_rl, 
            h1, Lfh1, Lgh1_1, Lgh1_2, Lgh1_3,
            h2, Lfh2, Lgh2_1, Lgh2_2, Lgh2_3,
            h3, Lfh3, Lgh3_1, Lgh3_2, Lgh3_3,
            self.state, hslack1, hslack2, hslack3, nu1, nu2, nu3
        )
        return uOpt, isSolved



    # ================================================================
    # Step
    # ================================================================
    def step(self, action):
        action = np.asarray(action, dtype=np.float64).ravel()

        # 1) RL thrust command
        u_cmd = np.clip(action[:3], -1.0, 1.0) * self.U_MAX
        u_rl = self._sat_axis(u_cmd, self.U_MAX)

        # 2) map remaining actions -> [0,1]
        if self.enableCBFtunning:
            idx = 3
            hslack1 = (action[idx + 0] + 1.0) / 2.0
            hslack2 = (action[idx + 1] + 1.0) / 2.0
            hslack3 = (action[idx + 2] + 1.0) / 2.0
            a1 = (action[idx + 3] + 1.0) / 2.0
            a2 = (action[idx + 4] + 1.0) / 2.0
            b1 = (action[idx + 5] + 1.0) / 2.0
            b2 = (action[idx + 6] + 1.0) / 2.0
            c1 = (action[idx + 7] + 1.0) / 2.0
            c2 = (action[idx + 8] + 1.0) / 2.0
        else:
            
            hslack1 = 0.05
            hslack2 = 0.05
            hslack3 = 0.05
            a1= 0.05
            a2= 0.05
            b1= 0.05
            b2= 0.05
            c1= 0.05
            c2= 0.05

        # 3) Safety filter (ICCBF-QP)
        u_safe, solved = self.getControl(
            hslack1=hslack1, hslack2=hslack2, hslack3=hslack3,
            a1=a1, a2=a2, b1=b1, b2=b2, c1=c1, c2=c2,
            u_rl=u_rl
        )

        if (not solved) or (u_safe is None):
            u_safe = u_rl.copy()

        u_safe = self._sat_axis(u_safe, self.U_MAX)

        # 4) Execution noise + saturation
        u_exec = self._apply_execution_noise(u_safe, self.np_random)
        u_exec = self._sat_axis(u_exec, self.U_MAX)

        self.last_u_rl = u_rl.copy()
        self.last_u_safe = u_exec.copy()

        # 5) Propagate
        self.state = self.dynamics.propwithCW(self.state, u_exec, self.DT)
        self.steps_done += 1

        # 6) Update inspection
        x_meas = self._noisy_measurement_state(self.state, self.np_random)
        obs_dict = self.obs_model.get_observation(x_meas)
        visible = obs_dict["visible_mask"]

        newly_inspected = visible & (~self.inspected)
        num_new = int(newly_inspected.sum())
        self.inspected |= visible

        # self.dvWeight = 500.0
        # 7) Reward (simple)
        reward = 0.1 * float(num_new)
        dV_proxy = np.linalg.norm(u_safe) / self.m * self.DT
        
        reward -=  self.dvWeight * dV_proxy

        # 8) Termination checks (hard checks)
        pos = self.state[:3]
        r_norm = float(np.linalg.norm(pos))
        terminated = False
        truncated = False

        # KOZ / KIZ
        if r_norm <= float(self.R_C + self.R_D):
            terminated = True
            reward -= 1.0
        if r_norm > float(self.R_MAX):
            terminated = True
            reward -= 1.0

        # Sun-avoidance hard check
        rs =  self.obs_model.sun_direction(self.state[6])
        rs = np.asarray(rs, dtype=float)
        rsn = np.linalg.norm(rs)
        rs = (rs / rsn) if rsn > 1e-12 else np.array([1.0, 0.0, 0.0])

        rbhat = -pos / max(r_norm, 1e-12)
        cos_theta_b = float(np.clip(np.dot(rbhat, rs), -1.0, 1.0))
        h_sun = float(np.cos(self.alpha_fov / 2.0) - cos_theta_b)
        if h_sun < 0.0:
            terminated = True
            reward -= 1.0

        # task completion
        if self.inspected.all():
            terminated = True

        if (self.steps_done >= self.MAX_STEPS) and (not terminated):
            truncated = True

        obs = self._get_obs_vector()
        info = {
            "num_inspected": int(self.inspected.sum()),
            "newly_inspected": num_new,
            "r_norm": r_norm,
            "h_sun": float(h_sun),
            "u_rl": self.last_u_rl.copy(),
            "u_safe": self.last_u_safe.copy(),
            "qp_solved": bool(solved),
        }
        return obs, reward, terminated, truncated, info

    # ================================================================
    # ICCBF-QP
    # ================================================================
    def qp_optimizationICCBF(
        self,
        u_rl,
        h1, Lf1h, Lg1h1, Lg1h2, Lg1h3,
        h2, Lf2h, Lg2h1, Lg2h2, Lg2h3,
        h3, Lf3h, Lg3h1, Lg3h2, Lg3h3,
        x, hSlack1, hSlack2, hSlack3, nu1, nu2, nu3
    ):
        u = cp.Variable(3)
        k = cp.Variable(nonneg=True)
        delta = cp.Variable(nonneg=True)
        gamma = cp.Variable(nonneg=True)

        cost = 10.0 * k + 10.0 * delta + 10.0 * gamma + 1e-2 * cp.sum_squares(u - u_rl)

        constraints = [
            Lf1h + Lg1h1*u[0] + Lg1h2*u[1] + Lg1h3*u[2] >= -(hSlack1 + k) * h1 + nu1,
            Lf2h + Lg2h1*u[0] + Lg2h2*u[1] + Lg2h3*u[2] >= -(hSlack2 + delta) * h2 + nu2,
            Lf3h + Lg3h1*u[0] + Lg3h2*u[1] + Lg3h3*u[2] >= -(hSlack3 + gamma) * h3 + nu3,
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True)
        except cp.error.SolverError:
            return np.zeros(3, dtype=float), False

        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
            return np.asarray(u.value, dtype=float).reshape(3,), True

        return np.zeros(3, dtype=float), False

