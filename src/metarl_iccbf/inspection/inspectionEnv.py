import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cvxpy as cp
import os
from pathlib import Path
# from my_jais_discrete_rta import MyJAISDiscreteRTA, JAISAlphaConfig

from dynamicsandControl import dynamicsAndControl
from Observation import ObservationModel
from iccbfs import ICCBF
class InspectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            dt,
            eval_mode: bool = False,
            eval_bank_path: str | None = None,
            eval_bank_size: int = 100,
            eval_bank_seed: int = 0,
            regenerate_eval_bank: bool = False,
            adversarial: bool = False,
            dv_budget_range: tuple = (0.5, 5.0),
            morl: bool = False,
            morl_coverage_threshold: float = 0.8,
        ):
  

        # time
        self.tstepOriginal = float(dt)
        self.DT = float(dt)
        self.steps_done = 0
        self.MAX_STEPS = 1224  # paper setup

        # -------------------------
        # Physical / scenario params (nominal)
        # -------------------------
        self.m = 12.0            # kg
        self.mu = 398600.0
        self.n = 0.001027        # rad/s (mean motion)
        self.R_D = 5.0           # m (deputy radius)
        self.R_C = 10.0          # m (chief radius)
        self.v0 = 0.2            # m/s (dynamic speed v0 in paper)
        self.nu1_factor = 7.5
        self.nu1 = self.nu1_factor * self.n

        self.alpha_fov = np.deg2rad(60.0)  # rad (half-angle)
        self.R_MAX = 800.0                  # m keep-in
        self.V_MAX = 5.0                    # m/s axial velocity limit
        self.U_MAX = 1.0                    # N max thrust per-axis

        self.T_fft = 100.0
        self.dt_fft = 10.0

        self.N_POINTS = 98
        self.K_CLUSTERS = 4
        self.KMEANS_ITERS = 10

        # -------------------------
        # Store nominal values for meta-RL sampling
        # -------------------------
        self.base_m = self.m
        self.base_n = self.n
        self.base_R_D = self.R_D
        self.base_R_C = self.R_C
        self.base_alpha_fov = self.alpha_fov
        self.base_R_MAX = self.R_MAX
        self.base_V_MAX = self.V_MAX
        self.base_U_MAX = self.U_MAX
        self.base_T_fft = self.T_fft
        self.base_dt_fft = self.dt_fft
        self.base_N_POINTS = self.N_POINTS
        
        self.w_dv = 0.001          # curriculum weight (paper starts at 0.001)
        self.w_min = 0.001
        self.w_max = 0.1           # paper uses 0.1 for eval (also max)
        
        KOZ = self.base_R_C +  self.base_R_D
    
        model = ICCBF( KOZ, self.base_R_MAX, self.mu, self.n ,self.m,self.alpha_fov)
        self.CBF1Vals, self.CBF2Vals , self.CBF3Vals = model.getICCBFs()
    

        # -------------------------

        # -------------------------
        # Derived parameters used in constraints/CBFs (CHANGES FOR META)
        # -------------------------
        a_max = self.U_MAX / self.m - 3.0 * (self.n ** 2) * self.R_MAX - 2.0 * self.n * self.V_MAX
        self.a_max = max(a_max, 1e-6)
        self.r_coll = self.R_D + self.R_C
        
        # init distribution / cone params
        self.INIT_RANGE_MIN = 50.0
        self.INIT_RANGE_MAX = 100.0
        self.THETA_B_MIN_DEG = 40.0

        # -------------------------
        # Dynamics + observation models
        # -------------------------
        # your dyn class should implement step(state, u_force, dt) or equivalent
        self.dynamics = dynamicsAndControl(
            mu=self.mu,
            n=self.n,
            rc=self.R_C,
            m=self.m
        )

        self.obs_model = ObservationModel.from_spherical_chief(
            radius=self.R_C,
            n_points=self.N_POINTS,
            base_rgb_value=1.0
        )

        # -------------------------
        # Environment state
        # -------------------------
        self.state = np.zeros(7, dtype=np.float64)
        self.inspected = np.zeros(self.N_POINTS, dtype=bool)

        # -------------------------
        self.numObs = 11

        # You can tighten these bounds once you finalise scaling.
        self.obslow = -np.inf * np.ones(self.numObs, dtype=np.float64)
        self.obshigh = np.inf * np.ones(self.numObs, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=self.obslow.astype(np.float32),
            high=self.obshigh.astype(np.float32),
            dtype=np.float32,
        )

        self.obsPrev = np.zeros(self.numObs, dtype=np.float64)

        num_actions = 9+3
        self.action_space = spaces.Box(
            low=-np.ones(num_actions, dtype=np.float32),
            high=np.ones(num_actions, dtype=np.float32),
            dtype=np.float32,
        )

        self.control = np.zeros(3, dtype=np.float64)   # last control (force)
        self.last_u_rl = np.zeros(3, dtype=np.float64)
        self.last_u_safe = np.zeros(3, dtype=np.float64)

        # -------------------------
        # MORL: vector reward mode
        # -------------------------
        self.morl = bool(morl)
        self.morl_coverage_threshold = float(morl_coverage_threshold)
        if self.morl:
            # reward_space: [fuel_component, safety_component]
            # fuel_component  = coverage_reward - fuel_cost  (maximise coverage, minimise ΔV)
            # safety_component = coverage_reward - safety_penalties  (maximise coverage, avoid CBF violations)
            self.reward_space = spaces.Box(
                low=np.array([-np.inf, -np.inf], dtype=np.float32),
                high=np.array([np.inf, np.inf], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )

        # -------------------------
        # Adversarial chief
        # -------------------------
        self.adversarial = bool(adversarial)
        self.dv_budget_range = (float(dv_budget_range[0]), float(dv_budget_range[1]))
        self.dv_budget = 0.0
        self.dv_per_step = 0.0
        self.dv_remaining = 0.0

        # -------------------------
        # Reward / logging placeholders
        # -------------------------
        self.reward = 0.0
        self.sigmaCounter = 0
        self.safe_so_far = True

        self.last_cbfs = np.zeros(8, dtype=np.float64)  # if you compute multiple constraints
        self.tt = np.array([0.0], dtype=np.float64)


        self._rng = np.random.default_rng()
        
        self.eval_mode = bool(eval_mode)
        self.eval_bank_path = eval_bank_path
        self.eval_bank_size = int(eval_bank_size)
        self.eval_bank_seed = int(eval_bank_seed)
        self.regenerate_eval_bank = bool(regenerate_eval_bank)

        self.eval_init_states = None  # (N, 7)
        self._eval_index = 0

        if self.eval_bank_path is not None:
            self._load_or_create_eval_bank()
    
    def _generate_eval_bank(self) -> np.ndarray:
        rng = np.random.default_rng(self.eval_bank_seed)
        bank = np.stack(
            [self._sample_initial_state(rng) for _ in range(self.eval_bank_size)],
            axis=0
        ).astype(np.float64)
        assert bank.shape == (self.eval_bank_size, 7)
        return bank

    def _load_or_create_eval_bank(self):
        p = Path(self.eval_bank_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.exists() and (not self.regenerate_eval_bank):
            data = np.load(p, allow_pickle=False)
            bank = data["init_states"]
            bank = np.asarray(bank, dtype=np.float64)
            if bank.ndim != 2 or bank.shape[1] != 7:
                raise RuntimeError(f"Bad eval bank shape {bank.shape} in {p}")
            self.eval_init_states = bank
        else:
            bank = self._generate_eval_bank()
            np.savez_compressed(p, init_states=bank)
            self.eval_init_states = bank

        self._eval_index = 0

    def set_eval_mode(self, enabled: bool):
        self.eval_mode = bool(enabled)

    def set_eval_index(self, idx: int):
        self._eval_index = int(idx)

    def _sun_direction(self, theta_s: float) -> np.ndarray:
        return ObservationModel.sun_direction(theta_s)

    def _sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        As in the paper:

        - range ∈ [50, 100] m
        - azimuth ∈ [0, 2π]
        - elevation ∈ [-π/2, π/2]
        - zero relative velocity
        - θ_S ∈ [0, 2π]
        - if θ_b < 40° (sensor at sun), flip position.
        """
        r0_mag = rng.uniform(self.INIT_RANGE_MIN, self.INIT_RANGE_MAX)
        az = rng.uniform(0.0, 2.0 * np.pi)
        el = rng.uniform(-0.5 * np.pi, 0.5 * np.pi)

        r0 = np.array([
            r0_mag * np.cos(el) * np.cos(az),
            r0_mag * np.cos(el) * np.sin(az),
            r0_mag * np.sin(el),
        ])
        v0 = np.zeros(3)
        theta_s0 = rng.uniform(0.0, 2.0 * np.pi)

        sun_dir = self._sun_direction(theta_s0)
        r_norm = np.linalg.norm(r0)
        boresight = -r0 / max(r_norm, 1e-9)   # deputy -> chief

        cos_theta_b = np.clip(boresight.dot(sun_dir), -1.0, 1.0)
        theta_b = np.arccos(cos_theta_b)

        if theta_b < np.deg2rad(self.THETA_B_MIN_DEG):
            r0 = -r0   # flip away from sun

        return np.concatenate([r0, v0, [theta_s0]])

    def _largest_cluster_direction(self) -> np.ndarray:
        unins_mask = ~self.inspected
        if not np.any(unins_mask):
            return np.zeros(3)

        pts = self.obs_model.surface_points[unins_mask]
        M = pts.shape[0]
        K = min(self.K_CLUSTERS, M)

        idx0 = np.random.choice(M, size=K, replace=False)
        centroids = pts[idx0].copy()

        for _ in range(self.KMEANS_ITERS):
            dists = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(K):
                mask_k = labels == k
                if np.any(mask_k):
                    centroids[k] = pts[mask_k].mean(axis=0)

        counts = np.array([(labels == k).sum() for k in range(K)])
        k_max = int(np.argmax(counts))
        centroid = centroids[k_max]

        pos = self.state[:3]
        vec = centroid - pos
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return np.zeros(3)
        return vec / norm

    def _get_obs_vector(self) -> np.ndarray:
        x, y, z, vx, vy, vz, theta_s = self.state
        P_i = float(self.inspected.sum())
        P_c = self._largest_cluster_direction()
        P_cx, P_cy, P_cz = P_c.tolist()

        return np.array(
            [x, y, z, vx, vy, vz, theta_s, P_i, P_cx, P_cy, P_cz],
            dtype=np.float32,
        )

    # ================================================================
    # Gymnasium API
    # ================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # --- Choose initial state ---
        if self.eval_mode:
            if self.eval_init_states is None:
                raise RuntimeError(
                    "eval_mode=True but eval_init_states is None. "
                    "Pass eval_bank_path to __init__ (and ensure it exists or can be created)."
                )

            # Allow explicit index override via options
            if options is not None and "eval_index" in options:
                self._eval_index = int(options["eval_index"])

            x0 = self.eval_init_states[self._eval_index % len(self.eval_init_states)].copy()
            self._eval_index += 1
            self.state = x0
        else:
            rng = np.random.default_rng(seed) if seed is not None else self._rng
            self.state = self._sample_initial_state(rng)

        # --- Adversarial budget ---
        if self.adversarial:
            rng = np.random.default_rng(seed) if seed is not None else self._rng
            self.dv_budget = float(rng.uniform(*self.dv_budget_range))
            self.dv_per_step = self.dv_budget / float(self.MAX_STEPS)
            self.dv_remaining = self.dv_budget
        else:
            self.dv_budget = 0.0
            self.dv_per_step = 0.0
            self.dv_remaining = 0.0

        # --- Standard reset bookkeeping ---
        self.steps_done = 0
        self.inspected = np.zeros(self.N_POINTS, dtype=bool)

        obs_dict = self.obs_model.get_observation(self.state)
        self.inspected |= obs_dict["visible_mask"]

        self.last_cbfs[:] = 0.0
        self.last_u_rl[:] = 0.0
        self.last_u_safe[:] = 0.0

        obs = self._get_obs_vector()
        info = {
            "num_inspected": int(self.inspected.sum()),
            "cbfs": self.last_cbfs.copy(),
            "u_rl": self.last_u_rl.copy(),
            "u_safe": self.last_u_safe.copy(),
            "qp_status": "reset",
            "eval_mode": bool(self.eval_mode),
            "eval_index_used": int((self._eval_index - 1) % (len(self.eval_init_states) if self.eval_init_states is not None else 1)),
        }
        return obs, info

        
    def getControl(self,action,hslack1, hslack2,hslack3, a1, b1, c1, a2, b2, c2,  u_rl):
    
        f_x,g_x = self.dynamics.getfxgx(self.state[:6])
 
        stateAndcoefs = list(self.state[:6].flatten()) + [a1, a2, b1,b2,c1,c2]

        Lgb11 = self.CBF1Vals.Lgb1_1_func(*stateAndcoefs)
        Lgb12 = self.CBF1Vals.Lgb1_2_func(*stateAndcoefs)
        Lgb13 = self.CBF1Vals.Lgb1_3_func(*stateAndcoefs)
      
        u1inf, u2inf, u3inf = self.U_MAX * np.where(
            np.array([Lgb11, Lgb12, Lgb13]) > 0.0, -1.0,  1.0
            )
            
        stateCControl = stateAndcoefs + [u1inf, u2inf,u3inf]  # [x1,...,x5, acoef1, acoef2, u1inf, u2inf]

            
        hICCBF = self.CBF1Vals.b2_func(*stateCControl)
        LfhICCBF = self.CBF1Vals.Lfb2_func(*stateCControl)
        Lgh1ICCBF = self.CBF1Vals.Lgb2_1_func(*stateCControl)
        Lgh2ICCBF = self.CBF1Vals.Lgb2_2_func(*stateCControl)
        Lgh3ICCBF = self.CBF1Vals.Lgb2_3_func(*stateCControl)
        
        CBF1_hICCBF =  hICCBF
        CBF1_LfhICCBF =  LfhICCBF
        CBF1_Lgh1ICCBF =  Lgh1ICCBF
        CBF1_Lgh2ICCBF =  Lgh2ICCBF
        CBF1_Lgh3ICCBF =  Lgh3ICCBF
        
    
        Lgb11 = self.CBF2Vals.Lgb1_1_func(*stateAndcoefs)
        Lgb12 = self.CBF2Vals.Lgb1_2_func(*stateAndcoefs)
        Lgb13 = self.CBF2Vals.Lgb1_3_func(*stateAndcoefs)
      
        u1inf2, u2inf2, u3inf2 = self.U_MAX * np.where(
            np.array([Lgb11, Lgb12, Lgb13]) > 0.0, -1.0,  1.0
            )
        
        stateCControl2 = stateAndcoefs + [u1inf2, u2inf2,u3inf2]  # [x1,...,x5, acoef1, acoef2, u1inf, u2inf]

            
        hICCBF = self.CBF2Vals.b2_func( *stateCControl2)
        LfhICCBF = self.CBF2Vals.Lfb2_func(*stateCControl2)
        Lgh1ICCBF = self.CBF2Vals.Lgb2_1_func( *stateCControl2)
        Lgh2ICCBF = self.CBF2Vals.Lgb2_2_func(*stateCControl2)
        Lgh3ICCBF = self.CBF2Vals.Lgb2_3_func(*stateCControl2)
        
        CBF2_hICCBF =  hICCBF
        CBF2_LfhICCBF = LfhICCBF
        CBF2_Lgh1ICCBF = Lgh1ICCBF
        CBF2_Lgh2ICCBF = Lgh2ICCBF
        CBF2_Lgh3ICCBF =  Lgh3ICCBF
        
         
        rsun = self._sun_direction(self.state[6])
        stateAndcoefs2 = list(self.state[:6].flatten()) + [rsun[0], rsun[1], rsun[2]] + [a1, a2, b1,b2,c1,c2]


        Lgb11 = self.CBF3Vals.Lgb1_1_func(*stateAndcoefs2)
        Lgb12 = self.CBF3Vals.Lgb1_2_func(*stateAndcoefs2)
        Lgb13 = self.CBF3Vals.Lgb1_3_func(*stateAndcoefs2)
      
        u1inf2, u2inf2, u3inf2 = self.U_MAX * np.where(
            np.array([Lgb11, Lgb12, Lgb13]) > 0.0, -1.0,  1.0
            )
        
        # get sun direction 
   
        stateCControl3 = stateAndcoefs2 + [u1inf2, u2inf2,u3inf2]  # [x1,...,x5, acoef1, acoef2, u1inf, u2inf]

            
        hICCBF = self.CBF3Vals.b2_func( *stateCControl3)
        LfhICCBF = self.CBF3Vals.Lfb2_func(*stateCControl3)
        Lgh1ICCBF = self.CBF3Vals.Lgb2_1_func( *stateCControl3)
        Lgh2ICCBF = self.CBF3Vals.Lgb2_2_func(*stateCControl3)
        Lgh3ICCBF = self.CBF3Vals.Lgb2_3_func(*stateCControl3)
        
        CBF3_hICCBF =  hICCBF
        CBF3_LfhICCBF = LfhICCBF
        CBF3_Lgh1ICCBF = Lgh1ICCBF
        CBF3_Lgh2ICCBF = Lgh2ICCBF
        CBF3_Lgh3ICCBF =  Lgh3ICCBF
        
        
   
        uOpt , isSolved= self.qp_optimizationICCBF(u_rl,f_x, g_x, CBF1_hICCBF , CBF1_LfhICCBF ,CBF1_Lgh1ICCBF , CBF1_Lgh2ICCBF ,CBF1_Lgh3ICCBF ,
                                                               CBF2_hICCBF , CBF2_LfhICCBF ,CBF2_Lgh1ICCBF , CBF2_Lgh2ICCBF , CBF2_Lgh3ICCBF ,
                                                               CBF3_hICCBF, CBF3_LfhICCBF, CBF3_Lgh1ICCBF, CBF3_Lgh2ICCBF, CBF3_Lgh3ICCBF,
                                                               self.state,hslack1 ,hslack2,hslack3)
        
        # print(uOpt)
        return uOpt, isSolved
        
        
    def step(self, action):
        # RL → thrust
        actionControl = np.asarray(action[:3], dtype=np.float64).ravel()
        actionControl = np.clip(actionControl[:3], -1.0, 1.0)
        u_rl = actionControl[:3] * self.U_MAX
        
        # u_rl = np.zeros(3, dtype=np.float64)
        
        noactionControl = 3
        
        hslack1 =(action[noactionControl]+1.0)/2.0
        hslack2 =(action[noactionControl+1]+1.0)/2.0
        hslack3 =(action[noactionControl+2]+1.0)/2.0
        a1 = (action[noactionControl+3]+1.0)/2.0
        a2 = (action[noactionControl+4]+1.0)/2.0 
        b1 = (action[noactionControl+5]+1.0)/2.0
        b2 = (action[noactionControl+6]+1.0)/2.0 
        c1 = (action[noactionControl+7]+1.0)/2.0
        c2 = (action[noactionControl+8]+1.0)/2.0

        # hslack1 = 0.5
        # hslack2 = 0.5;
        # a1 = 0.25
        # a2 = 0.25
        # b1 = 0.25
        # b2 = 0.25
        
        u_safe, isSolved = self.getControl(action,hslack1, hslack2,hslack3, a1, b1, c1, a2, b2, c2, u_rl)
        u_safe = np.clip(u_safe, -self.U_MAX, self.U_MAX)

        # CBF–QP safety filter using discrete Δh
        # u_safe, h_vals, qp_status = self.cbf_qp_filter(u_rl)
        # u_safe = u_rl
        # h_vals =  np.zeros(8, dtype=np.float64) 
        # qp_status = True;
        
        # try:
        # # u_safe = self.rta.compute_filtered_control(self.state, float(self.DT), u_rl)
        qp_status = "rta_ok"
  
        # u_safe = u_rl

        

        # self.last_cbfs = h_vals
        self.last_u_rl = u_rl
        self.last_u_safe = u_safe

        # propagate with SAFE thrust
        self.state = self.dynamics.propwithCW(self.state, u_safe, self.DT)

        # passive adversarial chief: velocity impulse away from deputy
        if self.adversarial and self.dv_remaining > 1e-9:
            r = self.state[:3]
            r_norm = float(np.linalg.norm(r))
            if r_norm > 1e-9:
                dv_k = min(self.dv_per_step, self.dv_remaining)
                self.state[3:6] += dv_k * (r / r_norm)
                self.dv_remaining -= dv_k

        self.steps_done += 1

        obs_dict = self.obs_model.get_observation(self.state)
        visible = obs_dict["visible_mask"]

        newly_inspected = visible & (~self.inspected)
        num_new = int(newly_inspected.sum())
        self.inspected |= visible

        coverage_reward = float(num_new) * 0.1

        pos = self.state[:3]
        r_norm = np.linalg.norm(pos)

        # fuel consumption
        dV_rl = (np.abs(u_rl[0]) + np.abs(u_rl[1]) + np.abs(u_rl[2])) / self.m * self.DT
        fuel_penalty = 0.02 * dV_rl

        # crash penalty
        crash_penalty = 1.0 if r_norm < self.R_D + self.R_C else 0.0

        terminated = False
        truncated = False

        # check CBF conditions h3
        rs = self._sun_direction(self.state[6])
        h3 = self.CBF3Vals.h_func(*list(self.state[:6].flatten()) + [rs[0], rs[1], rs[2]] + [a1, a2, b1, b2, c1, c2])
        cbf_penalty = 10.0 if h3 < 0.0 else 0.0

        if h3 < 0.0:
            terminated = True
        if r_norm <= self.R_D + self.R_C:
            terminated = True
        if r_norm > self.R_MAX:
            terminated = True
        if self.inspected.all():
            terminated = True

        if self.steps_done >= self.MAX_STEPS and not terminated:
            truncated = True

        obs = self._get_obs_vector()
        info = {
            "num_inspected": int(self.inspected.sum()),
            "newly_inspected": num_new,
            "r_norm": r_norm,
            "cbfs": self.last_cbfs.copy(),
            "u_rl": self.last_u_rl.copy(),
            "u_safe": self.last_u_safe.copy(),
            "qp_status": qp_status,
            "dv_budget": float(self.dv_budget),
            "dv_remaining": float(self.dv_remaining),
            "fuel_penalty": fuel_penalty,
            "safety_penalty": crash_penalty + cbf_penalty,
        }

        if self.morl:
            # Coverage ≥ threshold → truncate (success condition instead of reward component)
            coverage_frac = float(self.inspected.sum()) / self.N_POINTS
            if coverage_frac >= self.morl_coverage_threshold and not terminated:
                truncated = True
            # fuel_component:   coverage incentive − fuel cost
            # safety_component: coverage incentive − safety penalties
            r_vec = np.array(
                [coverage_reward - fuel_penalty,
                 coverage_reward - crash_penalty - cbf_penalty],
                dtype=np.float32,
            )
            return obs, r_vec, terminated, truncated, info

        # --- scalar reward (default, unchanged behaviour) ---
        reward = coverage_reward - fuel_penalty - crash_penalty - cbf_penalty
        return obs, reward, terminated, truncated, info

 
    def qp_optimizationICCBF(self,u_rl, f_x, g_x, h1,  Lf1h,Lg1h1, Lg1h2,Lg1h3, h2,  Lf2h,Lg2h1, Lg2h2,Lg2h3,   h3,  Lf3h,Lg3h1, Lg3h2,Lg3h3,
                             x, hSlack1=0.05, hSlack2 = 0.05, hSlack3=0.05):
     
        
       
        u = cp.Variable(3)
        k = cp.Variable(nonneg=True)
        delta = cp.Variable(nonneg=True)
        gamma = cp.Variable(nonneg=True)

        cost = +10.0 * k + 10.0 * delta + 10.0*gamma+ 1e-2*cp.sum_squares(u - u_rl) 
        constraints = [
            Lf1h + Lg1h1*u[0] + Lg1h2*u[1] + Lg1h3*u[2] >= -(hSlack1 + k) * h1,
            Lf2h + Lg2h1*u[0] + Lg2h2*u[1] + Lg2h3*u[2] >= -(hSlack2 + delta) * h2,
            Lf3h + Lg3h1*u[0] + Lg3h2*u[1] + Lg3h3*u[2] >= -(hSlack3 + gamma) * h3,
      
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True)
        except cp.error.SolverError:
            return np.array([0, 0,0]), False  # solver failed to run

        status = problem.status
        u_val  = getattr(u, "value", None)

        # Accept solved (or numerically acceptable) cases
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u_val is not None:
            return u_val, True

        # # Optional: accept early stop with a usable iterate
        # if status == cp.USER_LIMIT and u_val is not None:
        #     return u_val, True

        return np.array([0, 0,0]), False

