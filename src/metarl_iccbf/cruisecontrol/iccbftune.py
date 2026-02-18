import numpy as np
from math import pi
import symengine as se
import sympy as sp
from sympy.utilities.lambdify import lambdify


class ICCBF:
    
    def __init__(self, f0, f1, f2, m , g0 , vmax, v0):
        
        self.f0 = f0; 
        self.f1 = f1; 
        self.f2 = f2;
        self.m  = m; 
        self.g0 = 9.81; 
        self.vmax = 24.0;
        self.v0 = 13.89;

    def getICCBFs(self, a1, bcoef1, a2, bcoef2):
        
        bcoef1 = 1.0
        bcoef2  = 1.0

        d, v = se.symbols('d v')
        uInf = se.symbols('uinf')

        # Dynamics
        Fv =  self.f0 + self.f1 * v + self.f2 * v**2
        
        f = se.Matrix([self.v0-v, -Fv/self.m])
        g = (self.g0) * se.Matrix([
            [0],
            [1]
        ])

        # CLF
        # V = (v-self.vmax)**2
        # dV_dx = se.Matrix([se.diff(V, var) for var in [d,v]])

        # CBF
        h = d - 1.8*v 
        dh_dx = se.Matrix([se.diff(h, var) for var in [d ,v]])

        # Lie derivatives
        Lgb0_1 = dh_dx.dot(g.col(0))
        # Lgb0_2 = dh_dx.dot(g.col(1))
        Lfb0 = dh_dx.dot(f)

        # first class k function
        k1 = a1*h**bcoef1; 
         
              
        # ICCBF layers
        b1 = Lfb0 +Lgb0_1 *0.25 + k1;
        db1_dx = se.Matrix([se.diff(b1, var) for var in [d,v]])
        Lgb1_1 = db1_dx.dot(g.col(0))
        # Lgb1_2 = db1_dx.dot(g.col(1))


        # second class k function 
        # k2 = a2*b1**bcoef2;
        
        b2 = db1_dx.dot(f) + Lgb1_1 * uInf + a2*b1 # se.sqrt(se.sqrt(b1*b1))
        db2_dx = se.Matrix([se.diff(b2, var) for var in [d,v]])
        Lfb2 = db2_dx.dot(f)
        Lgb2_1 = db2_dx.dot(g.col(0))
        state_vars = [d,v]
        all_vars = state_vars + [uInf]

        h_func = lambdify(state_vars, h, modules='numpy')
        b2_func = lambdify(all_vars, b2, modules='numpy')
        Lfb2_func = lambdify(all_vars, Lfb2, modules='numpy')
        Lgb2_1_func = lambdify(all_vars, Lgb2_1, modules='numpy')
      
        b1_func = lambdify(state_vars, b1, modules='numpy')
        Lgb1_1_func = lambdify(state_vars, Lgb1_1, modules='numpy')
        # db2_dx_func = lambdify(state_vars, db2_dx, modules='numpy')


        
        return b2_func,Lfb2_func, b1_func, Lgb1_1_func,Lgb2_1_func

