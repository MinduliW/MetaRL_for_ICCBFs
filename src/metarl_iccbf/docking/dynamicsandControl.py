import numpy as np
import cvxpy as cp
from .Dockingcase import DockingCase
class dynamicsAndControl:
    def __init__(self, mu, n, r, m, om):
        """
        Parameters:
            mu: gravitational parameter
            n: mean motion
            r: orbital radius
            m: spacecraft mass
            om: angular rate for psi
        """
        self.mu = mu
        self.n = n
        self.r = r
        self.m = m
        self.om = om

        
    def getfxgx(self,x): 
        rc = np.sqrt((self.r + x[0])**2 + x[1]**2)

        term1 = self.n**2 * x[0] + 2 * self.n * x[3] + self.mu / self.r**2 - self.mu * (self.r + x[0]) / rc**3
        term2 = self.n**2 * x[1] - 2 * self.n * x[2] - self.mu * x[1] / rc**3

        f_x = np.array([x[2], x[3], term1, term2, self.om])
        g_x = (1 / self.m) * np.array([[0.0, 0.0],
                                       [0.0, 0.0],
                                       [1.0, 0.0],
                                       [0.0, 1.0],
                                       [0.0, 0.0]])
        
        
        return f_x, g_x
        
        
    def getfxgxCW(self,x):
        

        term1 = 3*self.n**2 * x[0] + 2 * self.n * x[3] 
        term2 = - 2 * self.n * x[2] 

        f_x = np.array([x[2], x[3], term1, term2, self.om])
        g_x = (1 / self.m) * np.array([[0.0, 0.0],
                                       [0.0, 0.0],
                                       [1.0, 0.0],
                                       [0.0, 1.0],
                                       [0.0, 0.0]])
        
        return f_x, g_x
    
        

    def ccDynamicsV2(self, t, x, uOpt):
      
        f_x , g_x = self.getfxgx(x)
        

        xdot = f_x + g_x @ np.array([uOpt[0], uOpt[1]])
        
        return xdot
    

    def phi_planar(self, dt):
        # ensure native floats inside STM

        nt = self.n * dt
        n = self.n
        s, c = np.sin(nt), np.cos(nt)

        Phi_rr = np.array([[4 - 3*c,         0.0],
                        [6*(s - nt),      1.0]], dtype=np.float64)

        Phi_rv = np.array([[ (1/n)*s,              (2/n)*(1 - c)],
                        [ (2/n)*(c - 1),  (1/n)*(4*s - 3*nt)]], dtype=np.float64)

        Phi_vr = np.array([[ 3*n*s,         0.0],
                        [ 6*n*(c - 1),   0.0]], dtype=np.float64)

        Phi_vv = np.array([[ c,       2*s],
                        [-2*s, 4*c - 3]], dtype=np.float64)

        return np.block([[Phi_rr, Phi_rv],
                        [Phi_vr, Phi_vv]]).astype(np.float64)

    def propwithCW(self, x, uOpt, dt):
        # sanitize types
        x    = np.asarray(x,    dtype=np.float64).ravel()
        u    = np.asarray(uOpt, dtype=np.float64).reshape(-1)
        if u.size != 2:
            raise ValueError(f"uOpt must have 2 elements, got shape {uOpt!r}")

        m    = (self.m)
        n    = (self.n)
        om   = (self.om)
        dt   = (dt)

        # compute STM and Δv for this step
        Phi  = self.phi_planar(dt)           # 4x4, float64
        dv   = (u / m) * dt                # (2,), float64

        # DO NOT mutate caller's x
        x4   = x[:4].copy()                # [px,py,vx,vy]
        x4[2:4] += dv                      # kick at start
        x4new = Phi @ x4                   # drift

        psi  = float(x[4]) + om * dt

        return np.concatenate([x4new, [psi]]).astype(np.float64)
