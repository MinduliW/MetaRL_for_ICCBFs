import time
import copy
import cvxpy as cp
import gymnasium as gym
from gymnasium import spaces
import random
from math import *
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
from iccbftune import ICCBF
# from ya import YA
# My Classes
# from propagations import Propagations
from supportfcn import supportFcn

class RLCBFcontrol(gym.Env):
    


    def __init__(self, dt, deterministic=False):
        super(RLCBFcontrol, self).__init__()
   
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
        
        num_actions = 1
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
    
    def f_at(self, x: np.ndarray) -> np.ndarray:
        """Drift dynamics f(x) for cruise control. x=[d,v]."""
        d, v = float(x[0]), float(x[1])
        F = self.f0 + self.f1 * v + self.f2 * v**2
        return np.array([self.v0 - v, -(F / self.m)], dtype=float)

    def _g_at(self, x: np.ndarray) -> np.ndarray:
        """Control vector g(x) (control-affine, 2x1)."""
        return np.array([[0.0], [self.g0]], dtype=float)

  
        
    def getValidPoints(self, include_outside=True, dedupe=True):
        
        x1 = np.arange(0.0, 121.0, 1.0)
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
        


    def step(self, action):
        
        # print(action)
        # action scaling 
        uval = self.ulim*action[0]


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

        x = self.x0   # or self.state

        uOpt,deltaVal = self.qp_optimizationICCBF(V, F, x, uval, Lslack=10)



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
            
        if self.x0[1] < 0.0:
            penalty += np.abs(self.x0[1])*10
              
            
            
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
    

    
    def qp_optimizationICCBF(self, V, F, x, u_rl, Lslack=10):

        LfV = -2 * (x[1] - self.vmax) / self.m * F       # scalar
        LgV =  2 * (x[1] - self.vmax) * self.g0          # scalar

        u     = cp.Variable(1)
        k     = cp.Variable(nonneg=True)
        delta = cp.Variable(nonneg=True)

        cost = 1e5*cp.sum_squares(u-u_rl) + 50.0 * delta + 10.0 * k

        constraints = [
            LfV + LgV * u <= -Lslack * V + delta,
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

       

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
    
