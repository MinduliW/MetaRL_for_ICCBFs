# consider also adding Lfh 
# Also scale Lgh right
import copy
import json
from typing import Optional, Dict, Any

import gymnasium as gym
from gymnasium import spaces
from math import *
# from sympy import symbols, diff, cos, sin, sqrt, Function, Matrix
import numpy as np

from scipy import integrate
from .Dockingcase import DockingCase
from .dynamicsandControl import dynamicsAndControl
from .iccbfs import ICCBF

class RLCBFcontrol(gym.Env):

    def __init__(self, dt, deterministic=False, nInitstates=10, setconst=False,
                 adversarial=False, tof=50.0, adv_omega_min=None, adv_omega_max=None,
                 adv_delta_omega=None, adv_delta_omega_max=None,
                 morl: bool = False):
        super(RLCBFcontrol, self).__init__()
      
        self.onestep = False 
        self.TOF = tof
        self.mu = 398600.0
        self.setconst = setconst
  
        self.rho = 2.4/1e3
        self.om = 0.6 * np.pi / 180.0
        self.gamma = 10.0 * np.pi / 180.0
        self.nInitstates = nInitstates
        self.m  = 1000.0
        self.r = 6771.0
        self.n = sqrt(self.mu / self.r**3)
        self.deterministic = deterministic
        self.sigmapoints = 0
        self.ncurrent = 0
        self.control = np.array([0.0, 0.0, 0.0])
        self.scalefactor = 1.0
        self.tstepOriginal = dt
        self.coneshift = 0.0

        # ---------- Store nominal values for meta-RL parameter sampling ----------
        self.base_m   = self.m
        self.base_rho = self.rho
        self.base_om  = self.om
        self.base_gamma = self.gamma
        self.base_r = self.r
        self.umax = 0.25
        # umax is defined below; base_umax will be set after that
        # ------------------------------------------------------------------------

        # ---------- Adversarial target parameters ----------
        self.adversarial = adversarial
        self.adv_omega_min = adv_omega_min if adv_omega_min is not None else 0.0 * np.pi / 180.0
        self.adv_omega_max = adv_omega_max if adv_omega_max is not None else 0.7 * np.pi / 180.0
        self.adv_delta_omega = adv_delta_omega if adv_delta_omega is not None else 0.02 * np.pi / 180.0
        self.adv_delta_omega_max = adv_delta_omega_max if adv_delta_omega_max is not None else 0.02 * np.pi / 180.0
        self.om_true = self.om
        self.om_iccbf = self.om
        # ---------------------------------------------------

        self.dockingCase = DockingCase(rho=self.rho, gamma=self.gamma)
        self.dynamics = dynamicsAndControl(mu=self.mu, n=self.n, r=self.r, m=self.m, om=self.om)
        
         # Initialize ICCBFs
        self.use_noise = True
        # (fix: these must be scalars, not tuples)
        self.pos_noise_std = 1e-4      # km   = 0.10 m
        self.vel_noise_std = 2e-6      # km/s = 2 mm/s
        self.phase_noise_std = 1.745e-3  # rad = 0.1 deg

        # actuation noise (relative; 1σ)
        self.thrust_mag_noise_frac = 0.02   # 2% magnitude noise
        self.thrust_dir_noise_std  = 1.745e-3  # rad = 0.1 deg (optional)
   
        self.iccbf = ICCBF(mu=self.mu, r=self.r, gamma=self.gamma,rho=self.rho,
            m=self.m, om=self.om, umax=self.umax)
        (
            self.b2_func,
            self.Lfb2_func,
            self.b1_func,
            self.Lgb1_1_func,
            self.Lgb1_2_func,
            self.Lgb2_1_func,
            self.Lgb2_2_func,
        ) = self.iccbf.getICCBFs()


        self.getmargin_docking = self.iccbf.getmargin

        self.validPoints = self.dockingCase.generate_evenly_spread_cone_points_2d()
        
        self.reward = 0.0
        self.tstep = dt
        
        self.numObs = 5
     
        # store nominal umax for meta-RL sampling
        self.base_umax = self.umax
    
        self.observation_space = spaces.Box(
            low=-np.ones(self.numObs),
            high=np.ones(self.numObs),
            dtype=np.float64
        )
        
        self.obsPrev = np.zeros(self.numObs)
         
        self.obslow = np.array([
            -200/1e3, -200/1e3, -20/1e3, -20/1e3, 0.0
        ])
        om_for_scaling = self.adv_omega_max if self.adversarial else self.om
        self.obshigh = np.array([
            200/1e3, 200/1e3, 20/1e3, 20/1e3,
            1.2*om_for_scaling*(self.TOF+0.5)
        ])

        num_actions = 4
        aa_lb = np.array([-1, -1, -1, -1]) 
        aa_ub = np.array([ 1,  1,  1,  1]) 
        self.aa_00 = np.ones(num_actions) * 0.35
        
        self.action_space = spaces.Box(low=aa_lb, high=aa_ub, dtype=np.float64)
        
        self.reward = 0.0
        self.sigmaCounter = 0

        # -------------------------
        # MORL: vector reward mode
        # -------------------------
        self.morl = bool(morl)
        if self.morl:
            # reward_space: [fuel_component, safety_component]
            # fuel_component:   -(fuel_ratio penalty) − terminal_distance_penalty + success_bonus
            # safety_component: -(CBF violation penalty) + success_bonus
            self.reward_space = spaces.Box(
                low=np.array([-np.inf, -np.inf], dtype=np.float32),
                high=np.array([np.inf, np.inf], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
    
        # Store empty arrays
        self.tt = np.array([0])
        self.safe_so_far = True
        
        # ---- Certificate logging ----
        self._cert_last = None          # packed dict for the most recent step
        self._cert_hist = []            # packed dicts for this episode (optional)
        self.store_cert_history = False # set True for debug runs (can be large)

        
    def getValidPoints(self):
        """Return fixed evaluation initial conditions (N, 5)."""
        return self.validPoints

    def reset(self, seed=0, postProcess=True):
        self.ncurrent = 0
        self.safe_so_far = True
        self.tstep = copy.deepcopy(self.tstepOriginal)
        

        # ---------- Meta-RL: sample new task parameters at start of episode ----------
        # mass ±10%
        mass_factor = np.random.uniform(0.9, 1.1)
        # max thrust ±5%
        thrust_factor = np.random.uniform(0.9, 1.1)
        # docking radius ±10%
        rho_factor =  np.random.uniform(0.9, 1.1)
        # rotational speed ±10%
        om_factor =  np.random.uniform(0.9, 1.1)
        
        r_factor = np.random.uniform(0.9, 1.1)
        
        gamma_factor = np.random.uniform(0.9, 1.1)
        
        

        self.m   = self.base_m   * mass_factor
        self.umax = self.base_umax * thrust_factor
        self.rho = self.base_rho * rho_factor
        self.om  = self.base_om  * om_factor
        self.r = self.base_r * r_factor
        self.gamma = self.base_gamma * gamma_factor
        
        self.n = sqrt(self.mu / self.r**3)
        
        
       
        self.iccbf = ICCBF(mu=self.mu, r=self.r, gamma=self.gamma,rho=self.rho,m = self.m, om=self.om,umax=self.umax)
        
        self.getmargin_docking = self.iccbf.getmargin
        

        # Rebuild dynamics and docking geometry with new parameters
        self.dockingCase = DockingCase(rho=self.rho, gamma=self.gamma)
        self.dynamics = dynamicsAndControl(mu=self.mu, n=self.n, r=self.r, m=self.m, om=self.om)
        self.validPoints = self.dockingCase.generate_evenly_spread_cone_points_2d()

        # ---------- Adversarial target: initialize omega state ----------
        if self.adversarial:
            self.om_true = self.om       # initial omega from meta-RL sampling
            self.om_iccbf = self.om      # no lag at t=0
            self.obshigh[4] = 1.2 * self.adv_omega_max * (self.TOF + 0.5)
        else:
            self.om_true = self.om
            self.om_iccbf = self.om
        # ---------------------------------------------------------------------------

        _MAX_IC_RETRIES = 50
        _safe_fallback = np.array([100.0/1e3, 0.0, 0.0, 0.0, 0.0])

        if self.deterministic:
            ycoord = np.linspace(
                -(90.0/1e3 - self.rho) * np.tan(self.gamma),
                (90.0/1e3 - self.rho - 1e-3) * np.tan(self.gamma),
                self.nInitstates
            )

            self.sigmaCounter = self.sigmaCounter + 1
            if self.sigmaCounter > len(ycoord):
                self.sigmaCounter = 1

            self.x0 = np.array([100.0/1e3, ycoord[self.sigmaCounter-1], 0.0, 0.0, 0.0])

            # Clamp to safe fallback if IC is infeasible
            if self.dockingCase.originalh(self.x0) < 0.0:
                import warnings
                warnings.warn("Deterministic IC in unsafe set — falling back to cone centerline")
                self.x0 = _safe_fallback.copy()
        else:
            ycoord = np.linspace(
                -(90.0/1e3 - self.rho) * np.tan(self.gamma),
                (90.0/1e3 - self.rho - 1e-3) * np.tan(self.gamma),
                self.nInitstates
            )

            accepted = False
            for _ in range(_MAX_IC_RETRIES):
                indx = np.random.randint(0, self.nInitstates - 1)
                candidate = np.array([100.0/1e3, ycoord[indx], 0.0, 0.0, 0.0])
                if self.dockingCase.originalh(candidate) >= 0.0:
                    self.x0 = candidate
                    accepted = True
                    break

            if not accepted:
                import warnings
                warnings.warn("No feasible IC found after retries — falling back to cone centerline")
                self.x0 = _safe_fallback.copy()

        self.reward = 0
        self.rewardCurrent = 0

        # Store empty arrays
        self.tt = np.array(0)

        observation = self.x0

        observation = self.scaleObservation(observation)
        self.obsPrev = observation
      
        self.Penalty = 0
        
        info = {}
        return observation, info 
    
    def _rng(self):
        # Gymnasium usually sets self.np_random in reset(seed=...)
        return getattr(self, "np_random", np.random.default_rng())

    
    def apply_control_noise(self, u_cmd):
        u = np.array(u_cmd, dtype=float).reshape(2,)
        
        rng = self._rng()
        eps = rng.normal(0.0, self.thrust_mag_noise_frac)

        if not self.use_noise:
            return np.clip(u, -self.umax, self.umax)  # or just norm clip

        # magnitude noise (relative)
        u = (1.0 + eps) * u

        # direction noise (optional)
        dtheta = rng.normal(0.0, self.thrust_dir_noise_std)
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s],
                    [s,  c]])
        u = R @ u

        # enforce admissible set ||u||2 <= umax
        un = np.linalg.norm(u)
        if un > self.umax:
            u = (self.umax / un) * u

        return u

    def _adversary_update(self, tstep):
        """Adversarial target: compute next omega using finite-difference gradient of h."""
        # Current true omega becomes stale for next step's ICCBF
        self.om_iccbf = self.om_true

        x_next = self.x0.copy()
        phi_now = x_next[4]
        x_test = x_next.copy()

        # h with omega + delta_omega
        x_test[4] = phi_now + (self.om_true + self.adv_delta_omega) * tstep
        h_plus = self.dockingCase.originalh(x_test)

        # h with omega - delta_omega
        x_test[4] = phi_now + (self.om_true - self.adv_delta_omega) * tstep
        h_minus = self.dockingCase.originalh(x_test)

        # Bang-bang: pick direction that minimizes h
        if h_minus < h_plus:
            om_new = self.om_true - self.adv_delta_omega_max
        else:
            om_new = self.om_true + self.adv_delta_omega_max

        self.om_true = float(np.clip(om_new, self.adv_omega_min, self.adv_omega_max))

    def softplus(self, x, beta=50.0):
        # stable softplus(beta*x)/beta
        z = beta * x
        return (np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)) / beta

    def validity_penalty(self,cert,
                        w_zeta=1.0,
                        w_branch=0.1,
                        branch_tau=1e-6,
                        smooth=True,
                        square=True):
        """
        Returns a nonnegative penalty. Subtract this from reward.

        - v_zeta penalises certificate margin violation.
        - v_branch penalises being near a switching surface (if you track it).
        """
        zeta_lb = float(cert.get("zeta_lb", -np.inf))
        margin_req = float(cert.get("margin_req", 0.0))

        # Main violation: want zeta_lb >= margin_req
        v_zeta = (margin_req - zeta_lb)

        if smooth:
            v_zeta = self.softplus(v_zeta)        # smooth hinge
        else:
            v_zeta = max(0.0, v_zeta)        # hinge

        if square:
            v_zeta = v_zeta * v_zeta

        # Branch term (optional but very useful)
        # Prefer margin-based version if available; fallback to boolean.
        if "branch_margin_lb" in cert:
            m = float(cert["branch_margin_lb"])
            v_branch = (branch_tau - m)
            if smooth:
                v_branch = self.softplus(v_branch)
            else:
                v_branch = max(0.0, v_branch)
            if square:
                v_branch = v_branch * v_branch
        else:
            v_branch = 0.0 if bool(cert.get("branch_ok", True)) else 1.0

        return w_zeta * v_zeta + w_branch * v_branch

    def step(self, action):
        
        acoef1 =2.0*(action[0]+1.0)/2.0
        acoef2 =2.0*(action[1]+1.0)/2.0
        hslack = 0.01 + 0.5*(action[2] + 1.0)*(1.0- 0.01)
        Lslack = 0.05 + 0.5*(action[3] + 1.0)*(1.0- 0.05)
        
        
        # hslack = 0.05
        # Lslack = 0.1
        # acoef1 = 0.25
        # acoef2 = 0.85
            
        if self.setconst:
            
            hslack = 0.05
            Lslack = 0.1
            acoef1 = 0.25
            acoef2 = 0.85
    
    
        tstepCurrent = self.tstep

        # ---------- Adversarial omega routing ----------
        if self.adversarial:
            om_for_iccbf = self.om_iccbf
            om_for_dynamics = self.om_true
        else:
            om_for_iccbf = self.om
            om_for_dynamics = self.om

        if self.tt.size == 1:
            t0 = 0.0
        else:
            t0 = np.squeeze(self.tt)[-1]

        tf = t0 + tstepCurrent
        tf = round(tf,6)
        self.tt = np.append(self.tt, tf)

        # ICCBF computation uses stale omega (in adversarial mode)
        self.dynamics.om = om_for_iccbf
        self.iccbf.om = om_for_iccbf
        f_x,g_x = self.dynamics.getfxgx(self.x0)

        stateAndcoefs = list(self.x0.flatten()) + [acoef1, acoef2, self.rho, self.m, om_for_iccbf, self.gamma, self.r]

        Lgb11 = self.Lgb1_1_func(*stateAndcoefs )
        Lgb12 = self.Lgb1_2_func(*stateAndcoefs)
        
        if Lgb11 > 0.0:
            u1inf = -self.umax
        else: 
            u1inf = self.umax
                
        
        if Lgb12 > 0.0:
            u2inf = -self.umax
        else: 
            u2inf = self.umax
        
        stateCControl = stateAndcoefs + [u1inf, u2inf]  # [x1,...,x5, acoef1, acoef2, u1inf, u2inf]

        # unpack the list so each element becomes a positional argument
        hICCBF    = self.b2_func(*stateCControl)
        LfhICCBF  = self.Lfb2_func(*stateCControl)
        Lgh1ICCBF = self.Lgb2_1_func(*stateCControl)
        Lgh2ICCBF = self.Lgb2_2_func(*stateCControl)

            
        nuMargin,vals = self.getmargin_docking(self.x0,acoef1, acoef2, hslack,self.tstep)   # new
    
        if nuMargin > 50.0:
            # print(nuMargin)
            nuMargin = 50.0
            
        # nuMargin = 0.0
            
        uOpt,kval, isSolved = self.dockingCase.qp_optimizationICCBF(
            f_x, g_x, hICCBF, LfhICCBF, Lgh1ICCBF, Lgh2ICCBF,
            self.x0, hslack, Lslack, nuMargin=nuMargin
        )
        
        # print(hslack+kval)
        
        # half_width = np.array([0.002, 0.002, 0.0005, 0.0005, 0.0002])  
        
        # cert = self.iccbf.certify_local_validity_best(
        #     x0=self.x0,
        #     a1=acoef1, a2=acoef2,
        #     hslack=hslack+kval,
        #     tstep=tstepCurrent,
        #     half_width=half_width,
        #     da_order=4,
        #     eps_zero=1e-16,
        #     margin_req=0.0,
        #     check_branch=True
        # )
        
        # packed = self._pack_cert(cert)
        # self._cert_last = packed
        # if self.store_cert_history:
        #     self._cert_hist.append(packed)
   
        uOpt = self.apply_control_noise(uOpt)
      
        if np.linalg.norm(uOpt) > self.umax: 
            self.control = uOpt/np.linalg.norm(uOpt)*self.umax
        else:
            self.control = uOpt
            
    
        # Switch to true omega for ODE propagation
        self.dynamics.om = om_for_dynamics

        ode_out = integrate.solve_ivp(fun=lambda t, y: self.dynamics.ccDynamicsV2(t, y,uOpt=self.control),
                        t_span=(0.0, tstepCurrent), y0=self.x0, method='RK45', dense_output=False,     rtol=1e-6, atol=1e-6)

        self.x0 = ode_out.y[:,-1]

        # Adversary updates omega for the next step
        if self.adversarial:
            self._adversary_update(tstepCurrent)

        originalCBF = self.dockingCase.originalh(self.x0)
        
        if originalCBF < 0.0:
            self.safe_so_far = False


        h_violation = max(0.0, -originalCBF / 100.0)       # dimensionless ~0–2

        # weight for CBF violation
        w_h = 100.0
        penalty_h = w_h * h_violation
        
        penalty_validityk = 0.0
        # print(cert)
        # if cert["zeta_lb"] < 0.0:
        #     penalty_validityk = np.abs(cert["zeta_lb"])*0.01
        # else:   
        #     penalty_validityk = 0.0
            
        # penalty_validityk = self.validity_penalty(cert,
        #                    w_zeta=50.0,      # start moderate, tune up
        #                    w_branch=5.0,
        #                    branch_tau=1e-6,
        #                    smooth=True,
        #                    square=True)

        fuel_ratio = np.linalg.norm(self.control) / self.umax   # in [0, 1]
        w_u = 0.02                                          # tune 0.1–0.5
        penalty_u = w_u * fuel_ratio

        reward = -(penalty_u + penalty_h + penalty_validityk)

        truncated = False

        info = {}
        if self.adversarial:
            info["om_true"] = self.om_true
            info["om_iccbf"] = self.om_iccbf

        np.sqrt((self.x0[0] - self.rho)**2 + self.x0[1]**2)

        observationUscl = self.x0
        observation =  self._get_observation(observationUscl)

        self.obsPrev = observation

        self.ncurrent += 1

        # Compute Lyapunov-like function (raw)
        V_raw, _ = self.dockingCase.calculate_V_and_dV(self.x0)
        # Rescale for reward shaping so it's O(1–10)
        V_scaled = V_raw / 1e6

        # Success detection: small V AND never violated CBF
        success_threshold = 0.03 / 1e6
        success_bonus = 0.01

        # Per-component MORL tracking (updated below alongside scalar reward)
        fuel_step = -penalty_u
        safety_step = -(penalty_h + penalty_validityk)

        # 1) Time horizon reached: terminal shaping
        if self.tt[-1] >= self.TOF:
            done = True

            w_V = 5.0
            # If still far from target, penalise
            if V_scaled > success_threshold:
                reward -= w_V * V_scaled
                fuel_step -= w_V * V_scaled  # distance-to-goal is a fuel/efficiency concern

            # If we achieved a nice target state AND stayed safe all along:
            if (V_scaled <= success_threshold) and self.safe_so_far:
                reward += success_bonus
                fuel_step += success_bonus
                safety_step += success_bonus
                info["success"] = True

        else:
            # 2) Early success termination: reached target before TOF
            if (V_scaled <= success_threshold) and self.safe_so_far:
                done = True
                reward += success_bonus
                fuel_step += success_bonus
                safety_step += success_bonus
                info["success"] = True

            else:
                done = False

        if np.linalg.norm(self.x0[:2]) <= 3 / 1e3:
            done = True

        if originalCBF < 0.0:
            reward = -10.0
            safety_step = -10.0  # CBF crash penalty goes entirely to safety component
            done = True

        truncated = False

        info["fuel_penalty"] = penalty_u
        info["safety_penalty"] = penalty_h + penalty_validityk + (10.0 if originalCBF < 0.0 else 0.0)

        if self.morl:
            r_vec = np.array([fuel_step, safety_step], dtype=np.float32)
            return observation, r_vec, done, truncated, info

        return observation, reward, done, truncated, info

    def _get_observation(self, x_true):
        x = np.array(x_true, dtype=float).copy()
      
        if self.use_noise:
            x[0] += self._rng().normal(0.0, self.pos_noise_std)
            x[1] += self._rng().normal(0.0, self.pos_noise_std)
            x[2] += self._rng().normal(0.0, self.vel_noise_std)
            x[3] += self._rng().normal(0.0, self.vel_noise_std)
            x[4] += self._rng().normal(0.0, self.phase_noise_std)

        # optionally clip to obslow/obshigh before scaling
        x = np.clip(x, self.obslow[:5], self.obshigh[:5])

        return self.scaleObservation(x)

    def scaleObservation(self,observation):
        obsScaled = np.zeros(len(observation))
        observation[4] = np.mod(observation[4], 2 * np.pi)
        
        
        for i in range(len(observation)):
            obsScaled[i] = 2*(observation[i] - self.obslow[i])/ (self.obshigh[i] - self.obslow[i]) -1


        return obsScaled
    
    
    
    def _pack_cert(self, cert: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert certificate dict -> flat numeric record (json/mat friendly).
        Missing fields become NaN / defaults.
        """
        def f(x, default=np.nan):
            try:
                return float(x)
            except Exception:
                return float(default)

        def b(x, default=0):
            try:
                return int(bool(x))
            except Exception:
                return int(default)

        rec = {}

        rec["valid"]       = b(cert.get("valid", False))
        rec["zeta_lb"]     = f(cert.get("zeta_lb", np.nan))
        rec["margin_req"]  = f(cert.get("margin_req", 0.0))
        rec["base_lb"]     = f(cert.get("base_lb", np.nan))
        rec["base_ub"]     = f(cert.get("base_ub", np.nan))
        rec["support_lb"]  = f(cert.get("support_lb", np.nan))
        rec["normLg_lb"]   = f(cert.get("normLg_lb", np.nan))
        rec["nu"]          = f(cert.get("nu", np.nan))

        rec["branch_ok"]        = b(cert.get("branch_ok", True))
        rec["branch_margin_lb"] = f(cert.get("branch_margin_lb", np.nan))

        # intervals (flatten)
        # Lg_interval: ((Lg1_lb,Lg1_ub),(Lg2_lb,Lg2_ub))
        Lg_int = cert.get("Lg_interval", None)
        if Lg_int is not None and len(Lg_int) == 2:
            (a1, b1), (a2, b2_) = Lg_int
            rec["Lg1_lb"], rec["Lg1_ub"] = f(a1), f(b1)
            rec["Lg2_lb"], rec["Lg2_ub"] = f(a2), f(b2_)
        else:
            rec["Lg1_lb"] = rec["Lg1_ub"] = np.nan
            rec["Lg2_lb"] = rec["Lg2_ub"] = np.nan

        # branch intervals (optional)
        Lb11 = cert.get("Lgb1_1_interval", None)
        if Lb11 is not None and len(Lb11) == 2:
            rec["Lgb1_1_lb"], rec["Lgb1_1_ub"] = f(Lb11[0]), f(Lb11[1])
        else:
            rec["Lgb1_1_lb"] = rec["Lgb1_1_ub"] = np.nan

        Lb12 = cert.get("Lgb1_2_interval", None)
        if Lb12 is not None and len(Lb12) == 2:
            rec["Lgb1_2_lb"], rec["Lgb1_2_ub"] = f(Lb12[0]), f(Lb12[1])
        else:
            rec["Lgb1_2_lb"] = rec["Lgb1_2_ub"] = np.nan

        return rec

    def get_last_certificate(self) -> Optional[Dict[str, float]]:
        """Retrieve the most recent packed cert record (or None)."""
        return getattr(self, "_cert_last", None)

    def get_episode_certificate_history(self):
        """Retrieve packed per-step history for the current episode."""
        return getattr(self, "_cert_hist", [])

    def save_episode_certificates_jsonl(self, path: str):
        """
        Save current episode certificate history as JSONL.
        Each line is one time-step cert record.
        """
        hist = self.get_episode_certificate_history()
        with open(path, "w", encoding="utf-8") as f:
            for rec in hist:
                f.write(json.dumps(rec) + "\n")

    def save_episode_certificates_npz(self, path: str):
        """
        Save current episode certificate history as a numeric NPZ (easy to load).
        """
        hist = self.get_episode_certificate_history()
        if len(hist) == 0:
            np.savez(path, empty=np.array([1], dtype=np.int32))
            return

        # turn list-of-dicts into dict-of-arrays
        keys = sorted(hist[0].keys())
        out = {k: np.array([h.get(k, np.nan) for h in hist], dtype=np.float64) for k in keys}
        np.savez(path, **out)
