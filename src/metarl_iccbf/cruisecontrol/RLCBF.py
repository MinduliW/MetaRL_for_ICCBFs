import time
import copy
import cvxpy as cp
import gymnasium as gym
from gymnasium import spaces
import random
import math
import torch as th

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Callable, Type
from scipy import integrate
from scipy.optimize import minimize 
from scipy.integrate import trapezoid
from scipy import integrate
# from iccbfs import ICCBF
from cruisecontrol.iccbftune import ICCBF
from daceypy import DA  # DACEyPy DA type (Differential Algebra)


# from ya import YA
# My Classes
# from propagations import Propagations
from cruisecontrol.supportfcn import supportFcn

class RLCBFcontrol(gym.Env):
    


    def __init__(self, dt, deterministic=False):
        super(RLCBFcontrol, self).__init__()
   
        DA.init(int(4), int(2))


        self.r_scale = 1.0   # km / LU  = 1 Re
        self.v_scale = 1.0   # km/s / LU/TU
        self.t_scale = 1.0   # s / TU
       
        self.f0 = 0.1 
        self.f1 = 5.0
        self.f2 = 0.25
        self.m  = 1650.0
        self.g0 = 9.81
        self.vmax = 24.0       # target velocity in CLF
        self.v0   = 13.89      # front car speed
        self.u    = 0.0
        
        # -------- store nominal parameters for meta-RL sampling ----------
        self.base_m    = self.m
        self.base_v0   = self.v0
        self.base_vmax = self.vmax
        # ulim defined below; base_ulim set after that
        # ---------------------------------------------------------------

        self.ICCBFs = ICCBF(self.f0, self.f1, self.f2, self.m, self.g0, self.vmax, self.v0)

        self._rng = np.random.default_rng()

        self.obslow  = np.array([-500.0,   -10.0])
        self.obshigh = np.array([ 500.0,   200.0])
        
        self.scalefactor = 1.0
        self.V_values = []
        
        self.vinit = 20.0
        self.deterministic = deterministic
        self.xinit = 100.0
 
        self.ulim = 0.25      # max thrust / accel magnitude
        self.base_ulim = self.ulim   # nominal max thrust for meta-RL
        
        self.x0rel = np.array([self.xinit, self.vinit])

        self.reward = 0.0
        
        self.pointno = 0
        self.tstep   = dt
        self.TOF     = 20.0
        
        self.nTotal   = 40
        self.ncurrent = 0
        self.nrem     = 200
        self.isOutsideICCBF = False 
       
        self.x0 = copy.deepcopy(self.x0rel)
        
        self.numObs = 2
        self.sigmaCounter = 0
    
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float64
        )
        
        num_actions = 4
        aa_lb = np.ones(num_actions) * -1.0
        aa_ub = np.ones(num_actions) *  1.0
        self.aa_00 = np.ones(num_actions) * 0.35
        
        self.use_noise = True
        # (fix: these must be scalars, not tuples)
        self.d_noise_std = 2.0      # position sensor noise
        self.v_noise_std = 0.5      # velocity sensor noise
        self.thrust_noise_std = 0.1 # actuation noise
        
        self.action_space = spaces.Box(low=aa_lb, high=aa_ub, dtype=np.float64)
        
        self.state  = copy.deepcopy(self.x0)
        self.reward = 0.0
        self.sigmaCounter = 0
    
        # Store empty arrays
        self.tt = np.array([0])
        self.yy = np.array(self.x0).reshape(-1, 1)
        self.aa = np.array(self.aa_00).reshape(-1, 1)
        self.uu = np.array([0, 0, 0]).reshape(-1, 1)
        self.dd = np.array([0, 0, 0]).reshape(-1, 1)

        # initial valid points with nominal params
        self.points = self.getValidPoints()  
        
    def calculateOriginalh(self):
        
        h = self.x0[0] - 1.8*self.x0[1]
    
        return h
    
    def interval_maxabs(self,I):
        # Try common attribute names
        for a, b in [("lb","ub"), ("lower","upper"), ("l","u"), ("inf","sup")]:
            if hasattr(I, a) and hasattr(I, b):
                return max(abs(float(getattr(I,a))), abs(float(getattr(I,b))))
        # Fallback: try indexing
        return max(abs(float(I[0])), abs(float(I[1])))

    def lipschitz_bound_bounder(self,poly, half_width):
        rd, rv = float(half_width[0]), float(half_width[1])

        # partial derivatives
        dp_d = poly.deriv(1)
        dp_v = poly.deriv(2)

        # scale to [-1,1]^2 box
        dp_d = dp_d.scaleVariable(1, rd).scaleVariable(2, rv)
        dp_v = dp_v.scaleVariable(1, rd).scaleVariable(2, rv)

        Md = self.interval_maxabs(dp_d.bound())
        Mv = self.interval_maxabs(dp_v.bound())

        return math.hypot(Md, Mv)

    def delta_box_acc_bounder(self, center, half_width, da_order=3):
        """
        Bound-based Δ for ACC:
        Δ = sup_{(d,v) in box, |u|<=ulim} ||[d_dot, v_dot]||_2
        using DA bound() instead of grid sampling.

        Uses the conservative inequality:
        sup ||[a,b]|| <= sqrt( (sup|a|)^2 + (sup|b|)^2 )
        and
        sup|v_dot| <= sup|v_dot_drift| + |g0|*ulim.
        """
        d_c, v_c = float(center[0]), float(center[1])
        rd, rv = float(half_width[0]), float(half_width[1])

        # Initialise DA if needed
        DA.init(int(da_order), 2)

        # Local variables: d = d_c + δd, v = v_c + δv
        dd = DA(1)
        dv = DA(2)
        d = d_c + dd
        v = v_c + dv

        # Drift dynamics
        d_dot = self.v0 - v
        F = self.f0 + self.f1 * v + self.f2 * (v * v)
        v_dot_drift = -(F / self.m)

        # Scale variables so δ in [-rd,rd] maps to s in [-1,1]
        d_dot_s = d_dot.scaleVariable(1, rd).scaleVariable(2, rv)
        vdr_s   = v_dot_drift.scaleVariable(1, rd).scaleVariable(2, rv)

        Md = self.interval_maxabs(d_dot_s.bound())             # sup_S |d_dot|
        Mv = self.interval_maxabs(vdr_s.bound()) + abs(self.g0) * abs(self.ulim)  # sup |v_dot|

        return float(math.hypot(Md, Mv))



    def nu_lemma2(self, T, l1, l2, Delta, eps=1e-12):
        nu = l1*T*Delta;
        return nu
        # if abs(l2) < eps:
        #     return float(l1 * Delta * T)
        # return float((l1 * Delta / l2) * np.expm1(l2 * T))

    def lipschitz_bound_from_da(self, poly, center, half_width, grid=7) -> float:
     
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
                Lmax = max(Lmax, np.sqrt(gd * gd + gv * gv))
        return Lmax


    def getmargin(self, a1, a2, hslack):
        (vals, polys) = self.getICCBFvars_dace(
            x0=self.x0,
            a1=a1, bcoef1=1.0, a2=a2, bcoef2=0.5,
            return_polys=True
        )
        (Lgh_val, Lfh_val, h_val) = vals

        Lf_poly = polys["LfhICCBF"]
        Lg_poly = polys["LghICCBF"]
        h_poly  = polys["hICCBF"]

        half_width = np.array([2.0, 2.0], dtype=float)  # δ-box widths

        # Lipschitz via DACE bounder (not sampling)
        l_Lfh = self.lipschitz_bound_bounder(Lf_poly, half_width)
        l_Lgh = self.lipschitz_bound_bounder(Lg_poly, half_width)

        # alpha(-h): alpha(s) = k*s, so Lip(alpha(-h)) = |k|*Lip(h)
        k_alpha = float(hslack)
        l_h = self.lipschitz_bound_bounder(h_poly, half_width)
        l_alpha = abs(k_alpha) * l_h

        # Δ is plant-based; compute around the physical x0 (not around [0,0])
        center_phys = np.array(self.x0, dtype=float)
        Delta = self.delta_box_acc_bounder(center_phys, half_width)

        u_max = abs(self.ulim)
        l2 = l_Lfh + l_Lgh * u_max
        l1 = l2 + l_alpha

        nu = self.nu_lemma2(self.tstep, l1, l2, Delta)
        
        # print(nu)
        return nu

    
    # def getmargin(self,a1, a2,hslack):
    #     (vals, polys) = self.getICCBFvars_dace(
    #         x0=self.x0,
    #         a1=a1, bcoef1=1.0, a2=a2, bcoef2=0.5,
    #         return_polys=True
    #     )
        
    #     (Lgh_val, Lfh_val, h_val) = vals

    #     Lf_poly = polys["LfhICCBF"]          # this is L_f h, where h := b2
    #     Lg_poly = polys["LghICCBF"]          # this is L_g h
        
    #     center = np.array([0.0,0.0])      # [d0, v0]
    #     half_width = np.array([2.0, 2.0], dtype=float)  # e.g. ±1 m, ±1 m/s (tune)
        
    #     l_Lfh = self.lipschitz_bound_from_da(Lf_poly, center, half_width, grid=11)
    #     l_Lgh = self.lipschitz_bound_from_da(Lg_poly, center, half_width, grid=11)
        
    #     k_alpha = hslack  # choose your alpha gain (same units as usual CBF tuning)
    #     h_poly = polys["hICCBF"]
    #     l_h = self.lipschitz_bound_from_da(h_poly, center, half_width, grid=11)
    #     l_alpha = abs(k_alpha) * l_h
    #     Delta = self.delta_box_acc(center, half_width, grid=11)
        
    #     u_max = abs(self.ulim)
    #     l2 = l_Lfh + l_Lgh * u_max
    #     l1 = l2 + l_alpha
  
    #     nu = self.nu_lemma2(self.tstep, l1, l2, Delta)
    #     return nu



            
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
        d = DA(1)+d0
        v = DA(2)+v0

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
        Lgb1_x0 = float(Lgb1.eval([0.0, 0.0]))
        u1inf = (-self.ulim) if (Lgb1_x0 > 0.0) else (self.ulim)

        # Lf b1 = ∇b1 · f (drift f = [d_dot, v_dot0])
        b1_d = b1.deriv(1)
        Lfb1 = b1_d * d_dot + b1_v * v_dot0

        # Freeze sign for the (potentially non-analytic) b1 power at x0
        b1_x0 = float(b1.eval([0.0, 0.0]))
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
        Lfh_val = float(LfhICCBF.eval([0.0, 0.0]))
        Lgh_val = float(LghICCBF.eval([0.0, 0.0]))
        h_val = float(hICCBF.eval([0.0, 0.0]))

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


   
    def getValidPoints(self, include_outside=True, dedupe=True):
        
        # x1 = np.arange(0.0, 121.0, 5)
        # x2 = np.arange(0.0, 25.0, 1) 
        
           
        x1 = np.arange(0.0, 121.0, 1)
        x2 = np.arange(0.0, 25.0, 0.5) 
        
        
        # x1 = np.linspace(80.0,101.0,4)
        # x2 = np.linspace(15.0, 20.0, 4) 
        

        X1, X2 = np.meshgrid(x1, x2)

        mask1 = (X1 - 1.8 * X2) >= 0.0
        mask2 = (self.v0 -X2) + 1.8/self.m*(self.f0 + self.f1 * X2 + self.f2 * (X2 ** 2)) + 1.8*self.g0*self.ulim >=0.0
     
        combined_mask = mask1  & mask2
        # Extract only those points that satisfy both conditions
        X1_valid = X1[combined_mask]
        X2_valid = X2[combined_mask]

        # Stack them into an (N,2) array
        valid_points = np.column_stack((X1_valid, X2_valid))
     
        # # also add others 
        include_outside = True
        
        if include_outside:
            outside_pts = self.pointsoutsideICCBF()
            if outside_pts.size > 0:
                valid_points = np.vstack((valid_points,outside_pts))

        # # De-duplicate and sort (optional but handy)
        # if dedupe and valid_points.size > 0:
        #     # Round to kill tiny float noise before unique
        #     vp = np.round(valid_points.astype(float), 6)
        #     vp = np.unique(vp, axis=0)
        #     # Sort by x1 then x2
        #     valid_points = vp[np.lexsort((vp[:, 1], vp[:, 0]))]


        return valid_points
    
    def reset(self, seed=0, postProcess=True):
        # DA.init(int(4), int(2))
        self.V_values = []

        # -------- Meta-RL: sample task parameters at start of episode --------
        # Uniform factors over ± ranges

        # mass ±20%  → factor in [0.8, 1.2]
        mass_factor  = np.random.uniform(0.8, 1.2)

        # front car speed v0 ±10% → [0.9, 1.1]
        v0_factor    = np.random.uniform(0.9, 1.1)

        # max thrust ulim ±20% → [0.8, 1.2]
        ulim_factor  = np.random.uniform(0.8, 1.2)

        # target speed vmax ±10% → [0.9, 1.1]
        vmax_factor  = np.random.uniform(0.9, 1.1)

        # apply to actual parameters
        self.m    = self.base_m    * mass_factor
        self.v0   = self.base_v0   * v0_factor
        self.vmax = self.base_vmax * vmax_factor
        self.ulim = self.base_ulim * ulim_factor

        # rebuild ICCBF model with new parameters
        self.ICCBFs = ICCBF(self.f0, self.f1, self.f2, self.m, self.g0, self.vmax, self.v0)

        # recompute valid points so deterministic init sees new ICCBF
        self.points = self.getValidPoints()
        # ---------------------------------------------------------------------

        if self.deterministic:

            idx = np.linspace(0, self.points.shape[0] - 1, 10, dtype=int)
            pointsSelected = self.points[idx]

            self.pointno += 1
            if self.pointno == 1:
                self.pointno += 1
            if self.pointno > 10:
                self.pointno = 0

            self.x0 = pointsSelected[self.pointno - 1]

        else:
            points = self.getValidPoints()
            idx = np.random.choice(points.shape[0])
            self.x0 = points[idx]

        X1 = self.x0[0]
        X2 = self.x0[1]

        mask1 = (X1 - 1.8 * X2)
        mask2 = (self.v0 - X2) + 1.8/self.m * (self.f0 + self.f1 * X2 + self.f2 * (X2 ** 2)) + 1.8*self.g0*0.25 

        if mask1 > 0.0 and mask2 > 0.0:
            self.isOutsideICCBF = False
        else:
            self.isOutsideICCBF = True

        self.reward = 0.0
        self.state = copy.deepcopy(self.x0)
        self.ncurrent = 0
        self.rewardCurrent = 0 

        self.tt = np.array(0)
        self.yy = np.array(self.x0).reshape(-1, 1)
        self.aa = np.array(self.aa_00).reshape(-1, 1)
        self.uu = np.array([0, 0, 0]).reshape(-1, 1)

        observation = self.scaleObservation(self.x0)

        self.Penalty = 0

        info = {}
        return observation, info


    def pointsoutsideICCBF(self):
        
        x1 = np.arange(30.0, 50.0, 1.0)
        x2 = np.arange(20.0, 25.0, 0.5) 
        

        X1, X2 = np.meshgrid(x1, x2)

        mask1 = (X1 - 1.8 * X2) >= 0.0
        
                # Dynamics pieces
        Fv  = self.f0 + self.f1 * X2 + self.f2 * (X2 ** 2)
        fx1 = self.v0 - X2
        fx2 = -(Fv / self.m)

        # h, L_f h
        h    = X1 - 1.8 * X2
        Lfb0 = fx1 + 1.8 * (Fv / self.m)  # (dh=[1,-1.8])·f

        # b1 and its derivatives
        b1       = Lfb0 + 4.0 * h
        db1_dx1  = 4.0
        db1_dx2  = -8.2 + (1.8 / self.m) * (self.f1 + 2.0 * self.f2 * X2)

        # Lg b1 with g = g0 * [0; 1]
        Lgb1 = self.g0 * db1_dx2
        
        ulim = float(self.ulim)

        if np.ndim(Lgb1) == 0:  # scalar/0-d array
            uInf = -ulim if float(Lgb1) > 0.0 else ulim
        else:  # vectorised element-wise choice
            uInf = np.where(Lgb1 > 0.0, -ulim, ulim)

        # b2 = (db1·f) + (db1·g) uInf + 7 * sqrt(|b1|)
        b2_core = db1_dx1 * fx1 + db1_dx2 * fx2
        b2 = b2_core + Lgb1 * uInf + 7.0 * np.sqrt(np.abs(b1))
        
        
        # element-wise: set b2 = 0 wherever b1 <= 0
        if np.ndim(b1) == 0:
            if b1 <= 0.0:
                b2 = 0.0
        else:
            b2 = np.where(b1 <= 0.0, 0.0, b2)
        # mask2: b2 >= 0
        mask2 = (b2 <= 0.0)
        
        combined_mask = mask1  & mask2
        # Extract only those points that satisfy both conditions
        X1_valid = X1[combined_mask]
        X2_valid = X2[combined_mask]

        # Stack them into an (N,2) array
        valid_points = np.column_stack((X1_valid, X2_valid))
        
        return valid_points
    

    def _get_observation(self, x_true):
        """
        Build the observation from the true state x_true = [d, v],
        adding sensor noise if enabled, and then scaling to [-1, 1]^2.
        """
        d_meas = x_true[0]
        v_meas = x_true[1]

        if self.use_noise:
            d_meas = d_meas + self._rng.normal(0.0, self.d_noise_std)
            v_meas = v_meas + self._rng.normal(0.0, self.v_noise_std)

        # Clip to sensor bounds before scaling
        d_meas = np.clip(d_meas, self.obslow[0], self.obshigh[0])
        v_meas = np.clip(v_meas, self.obslow[1], self.obshigh[1])

        obs_raw = np.array([d_meas, v_meas])
        return self.scaleObservation(obs_raw)
    
    def getICCBFvars(self, f,g, a1, bcoef1, a2, bcoef2):
        
        # set to original 
        # a1 = 4; 
        bcoef1 = 1.0; 
        
        # a2 = 7;
        bcoef2 = 0.5;
        
        d_dot = f[0]
        v_dot = f[1]
        
        d = self.x0[0]
        v = self.x0[1]
    
        F  = self.f0 + self.f1 * v + self.f2 * (v ** 2)
        Fv = self.f1 + 2*self.f2*v
          
        h = d - 1.8 * v 
        
        Lfh = d_dot + 1.8 * F / self.m                  # same as above
        Lgh = -1.8 * self.g0 
    
        # class-K term for h, k1(h) = a1 * sign(h) * |h|^bcoef1
        eps = 1e-8
        h_abs   = max(abs(h), eps)
        sign_h  = np.sign(h)
        h_pow1  = h_abs ** (bcoef1 - 1.0)
        
 
        h_pow2  = h_abs ** (bcoef1 - 2.0)

        k1 = a1  * (h ** bcoef1)

        # derivatives of b1 do NOT depend on u_inf (Lgh*u_inf is constant in x)
        # b1_d = ∂b1/∂d, b1_v = ∂b1/∂v, etc.
        A = 1.8 / self.m

        b1_d  = a1 * bcoef1 * h_pow1
        b1_v  = -1.0 + A * (self.f1 + 2.0 * self.f2 * v) - 1.8 * a1 * bcoef1 * h_pow1

        b1_dd = a1 * bcoef1 * (bcoef1 - 1.0) * h_pow2
        b1_dv = a1 * bcoef1 * (bcoef1 - 1.0) * h_pow2 * (-1.8)
        b1_vd = b1_dv
        b1_vv = 2.0 * A * self.f2 + (1.8 ** 2) * a1 * bcoef1 * (bcoef1 - 1.0) * h_pow2

        # L_g b1 = ∇b1 · g = g0 * ∂b1/∂v
        Lgb1 = self.g0 * b1_v

        # choose worst-case u_inf for the NEXT layer based on sign of L_g b1
        if Lgb1 > 0.0:
            u1inf = -self.ulim
        else:
            u1inf =  self.ulim

        # full b1 value (includes L_g h * u_inf term)
        b1 = Lfh + Lgh * self.ulim + k1
        

        # L_f b1 = ∇b1 · f
        Lfb1 = b1_d * d_dot + b1_v * v_dot

        # second class-K term, k2(b1) = a2 * sign(b1) * |b1|^bcoef2
        b1_abs  = max(abs(b1), eps)
        sign_b1 = np.sign(b1)
        
        if b1 > 0.0:
            k2 = a2  * (b1 ** bcoef2)
        else: 
            k2 = a2  * (b1_abs ** bcoef2)
            

       

        # derivatives of k2
        b1_pow_b2_minus1 = b1_abs ** (bcoef2 - 1.0)
        k2_d = a2 * bcoef2 * b1_pow_b2_minus1 * b1_d
        k2_v = a2 * bcoef2 * b1_pow_b2_minus1 * b1_v

        # derivatives of L_g b1
        Lgb1_d = self.g0 * b1_vd
        Lgb1_v = self.g0 * b1_vv

        # derivatives of L_f b1:
        # L_f b1 = b1_d*d_dot + b1_v*v_dot
        # using d_dot_d=0, d_dot_v=-1, v_dot_d=0, v_dot_v=-Fv/m
        Lfb1_d = b1_dd * d_dot + b1_vd * v_dot
        Lfb1_v = (
            b1_dv * d_dot
            - b1_d
            + b1_vv * v_dot
            - b1_v * Fv / self.m
        )

        # derivatives of b2 = L_f b1 + L_g b1 * u_inf + k2
        b2_d = Lfb1_d + Lgb1_d * u1inf + k2_d
        b2_v = Lfb1_v + Lgb1_v * u1inf + k2_v

        # second-layer ICCBF and its Lie derivatives
        b2      = Lfb1 + Lgb1 * u1inf + k2
        Lgh1ICCBF = self.g0 * b2_v                        # L_g b2
        LfhICCBF = b2_d * d_dot + b2_v * v_dot            # L_f b2
        hICCBF   = b2                                     # the actual ICCBF
        
        return Lgh1ICCBF, LfhICCBF,hICCBF
            

    def step(self, action):
        
        # print(action)
        # action scaling 
        action[0] = 10*(action[0] +1.0)/2.0
        action[1] = 10*( action[1] +1.0)/2.0
        
        hslack = 1.0  + 0.5*(action[2] + 1.0)*(3.0 - 1.0)
        Lslack = 8.0 +  0.5*(action[3] + 1.0)*(12.0- 8.0)

        # action[0] = 4
        # action[1] = 7
        # hslack = 2.0
        # Lslack = 8.0
        
        if self.tt.size == 1:
            t0 = 0.0
        else: 
            t0 = np.squeeze(self.tt)[-1]
            
        tf = t0 + self.tstep

        self.tt = np.append(self.tt, tf)
        
        

        # get control  
        F = self.f0 + self.f1 * self.x0[1] + self.f2 * (self.x0[1] ** 2)
        V = (self.x0[1] - self.vmax) ** 2
        
        f = np.array([self.v0-self.x0[1], -F/self.m])
        g = (self.g0) * np.array([
            [0.0],
            [1.0]
        ])


        Lgh1ICCBF, LfhICCBF,hICCBF= self.getICCBFvars( f,g, action[0], 1.0, action[1],0.5)
        # Lgh_val, Lfh_val, h_val  = self.getICCBFvars_dace(self.x0, action[0], 1.0, action[1],0.5)
        nuMargin = self.getmargin(action[0],  action[1],hslack)
        
       
        

        # hICCBF = self.b2_func(self.x0[0],self.x0[1],u1inf)

        # LfhICCBF = self.Lfb2_func(self.x0[0],self.x0[1],u1inf)
        # Lgh1ICCBF = self.Lgb2_1_func(self.x0[0],self.x0[1],u1inf)
    

    
        
        # current state (use whatever you store the state in)
        x = self.x0   # or self.state

        uOpt,deltaVal = self.qp_optimizationICCBF(V,F, f, g, hICCBF,  LfhICCBF,Lgh1ICCBF, self.x0, hslack, Lslack,nuMargin)



        u_cmd =np.clip(uOpt, -self.ulim ,  self.ulim )
        
        if self.use_noise:
            u_noisy = u_cmd + self._rng.normal(0.0, self.thrust_noise_std)
            self.u = np.clip(u_noisy, -self.ulim, self.ulim)
        else:
            self.u = u_cmd
        
        
        RelTol = 1e-8
        AbsTol = 1e-8
        
        # Integration
        ode_out = integrate.solve_ivp(fun=lambda t, y: self.ccDynamicsV2(t, y,uOpt= self.u),
                        t_span=(0.0, self.tstep), y0=self.x0, method='RK45', dense_output=False, atol=AbsTol, rtol=RelTol)
   
        xnext = ode_out.y[:,-1]
            
        originalCBF = xnext[0] - 1.8*xnext[1]
        
        penalty = 0
        
            
        if originalCBF <0.0:
            penalty += np.abs(originalCBF)*10
        else: 
            penalty = 0.0
            
        # penalty on control 
        penalty += np.abs(self.u)*2.5

        reward = -(penalty)
     
        truncated = False
        
    
        info = {}
    
  
        observation = self._get_observation(xnext)
     
        
        V = (self.x0[1] - self.vmax) ** 2
        self.V_values.append(V)
    
        self.x0 = xnext;
     
        self.reward += reward;

        self.ncurrent += 1; 
        
       
        finalRewOnly = False;
        
        
        if self.tt[-1] >= self.TOF:
            
            done = True
            
            if finalRewOnly == True:
                reward = self.reward 
                
            # if self.isOutsideICCBF == False:
                if min(self.V_values) > 10:
                    reward = reward-min(self.V_values)*50
                
        else:
            done = False
            
            if finalRewOnly == True:
                reward = 0


        reward = reward / 50.0              # keep returns O(10–100)

        if self.x0[0] < 0.0: # this is a crash
            reward = -500.0
            done = True; 
        # # if V < 1e-2:
        #     done = True
            
        # else:
        #     done = False
        
        # scale = 0.1  # or 0.01 if your raw rewards are quite big
        # reward = np.clip(reward * scale, -10.0, 10.0)

        return observation, reward, done,truncated, info
    


    def scaleObservation(self,observation):
        
    
        obsScaled = np.zeros(2)
        for i in range(2):
            obsScaled[i] = 2*(observation[i] - self.obslow[i])/ (self.obshigh[i] - self.obslow[i]) -1
            # if obsScaled[i] < -1 or obsScaled[i]  > 1:
            #    print('error') 
            #    print(i)
        
       # print(obsScaled[7:13])
        return obsScaled
    
    def unscaleObservation(self, obsScaled):
       
        observation = np.zeros_like(obsScaled)
        for i in range(len(obsScaled)):
            observation[i] = self.obslow[i] + ((obsScaled[i] + 1) * (self.obshigh[i] - self.obslow[i])) / 2.0

        return observation

    
    def qp_optimizationICCBF(self, V, F, f_x, g_x, h, Lfh, Lgh, x, hslack=2, Lslack=10,nuMargin =0.0):

        LfV = -2 * (x[1] - self.vmax) / self.m * F       # scalar
        LgV =  2 * (x[1] - self.vmax) * self.g0          # scalar

        u     = cp.Variable(1)
        k     = cp.Variable(nonneg=True)
        delta = cp.Variable(nonneg=True)

        cost = cp.sum_squares(u) + 50.0 * delta + 10.0 * k
        # print(nuMargin)

        constraints = [
            Lfh + Lgh * u >= -(hslack + k) * h + nuMargin,
            LfV + LgV * u <= -Lslack * V + delta,
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        # mosek_opts = {
        #     # conic interior-point tolerances
        #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   1e-7,
        #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   1e-7,
        #     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-7,
        #     "MSK_DPAR_INTPNT_CO_TOL_MU_RED":  1e-10,
        #     # optional: limit iterations to keep things predictable
        #     # "MSK_IPAR_INTPNT_MAX_ITERATIONS": 50,
        # }

        try:
            problem.solve(
                solver=cp.MOSEK,
                verbose=False,
                warm_start=True,     # helps a *lot* for jitter between steps
            )

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                uOpt     = float(u.value)
                deltaVal = float(delta.value)
                isSolved = True
            else:
                uOpt = 0.0
                deltaVal = 0.0
                isSolved = False

        except cp.error.SolverError:
            uOpt = 0.0
            deltaVal = 0.0
            isSolved = False

        return uOpt, deltaVal



    def ccDynamicsV2(self, t, x,uOpt):
        
        F = self.f0 + self.f1 * x[1] + self.f2 * (x[1] ** 2)
       
        xdot = np.array([self.v0 - x[1], -F / self.m + self.g0 * uOpt])
        
        return xdot
    

    def dh_dt(self, rx, ry, rz, vx, vy, vz, n, d):
        term1 = (rx**2 + ry**2 + rz**2) * (2 * n**2 * vz**2 - 2 * vx * (3 * rx * n**2 + 2 * vy * n) + 4 * n * vx * vy)
        term2 = -d**2 * (2 * n**2 * vz**2 - 2 * vx * (3 * rx * n**2 + 2 * vy * n) + 4 * n * vx * vy)
        term3 = -(vx**2 + vy**2 + vz**2) * (2 * rx * vx + 2 * ry * vy + 2 * rz * vz)
        
        return term1 + term2 + term3
    
    
 