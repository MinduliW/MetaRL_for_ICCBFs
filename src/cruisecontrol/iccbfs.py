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

        
    def build_iccbf_symbolic(f0, f1, f2, m, g0, vmax, v0):
        d, v, uInf = sp.symbols('d v uInf', real=True)

        # Dynamics
        Fv = f0 + f1*v + f2*v**2
        f = sp.Matrix([v0 - v, -Fv/m])
        g = sp.Matrix([0, g0])   # scalar input

        # Base CBF
        h = d - 1.8*v
        dh_dx = sp.Matrix([sp.diff(h, d), sp.diff(h, v)])
        Lfb0 = dh_dx.dot(f)

        # ICCBF layers (your fixed construction)
        b1 = Lfb0 + 4*h
        db1_dx = sp.Matrix([sp.diff(b1, d), sp.diff(b1, v)])
        Lgb1 = db1_dx.dot(g)

        # b2 = db1·f + Lgb1*uInf + 7*sqrt(|b1|)
        # Use abs(b1) to avoid nested sqrt(sqrt(b1*b1)) issues
        eps = sp.Float(1e-6)  # smoothing parameter (try 1e-6 to 1e-4)
        b2 = db1_dx.dot(f) + Lgb1*uInf + 7*(b1**2 + eps)**sp.Rational(1, 4)

        db2_dx = sp.Matrix([sp.diff(b2, d), sp.diff(b2, v)])
        Lfb2 = db2_dx.dot(f)
        Lgb2 = db2_dx.dot(g)   # scalar

        # H = -b2, and Lie derivatives for H
        H   = -b2
        LfH = -Lfb2
        LgH = -Lgb2

        # Analytic gradients needed for Lipschitz constants
        grad_H   = sp.Matrix([sp.diff(H, d),   sp.diff(H, v)])
        grad_LfH = sp.Matrix([sp.diff(LfH, d), sp.diff(LfH, v)])
        grad_LgH = sp.Matrix([sp.diff(LgH, d), sp.diff(LgH, v)])

        # Lambdify (vector outputs come out as arrays)
        H_fun       = lambdify((d, v, uInf), H, modules="numpy")
        LfH_fun     = lambdify((d, v, uInf), LfH, modules="numpy")
        LgH_fun     = lambdify((d, v, uInf), LgH, modules="numpy")

        grad_H_fun   = lambdify((d, v, uInf), grad_H, modules="numpy")
        grad_LfH_fun = lambdify((d, v, uInf), grad_LfH, modules="numpy")
        grad_LgH_fun = lambdify((d, v, uInf), grad_LgH, modules="numpy")

        return dict(
            H=H_fun, LfH=LfH_fun, LgH=LgH_fun,
            grad_H=grad_H_fun, grad_LfH=grad_LfH_fun, grad_LgH=grad_LgH_fun
        )


    
    def getICCBFs(self):
        

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
        V = (v-self.vmax)**2
        dV_dx = se.Matrix([se.diff(V, var) for var in [d,v]])

        # CBF
        h = d - 1.8*v 
        dh_dx = se.Matrix([se.diff(h, var) for var in [d ,v]])

        # Lie derivatives
        Lgb0_1 = dh_dx.dot(g.col(0))
        # Lgb0_2 = dh_dx.dot(g.col(1))
        Lfb0 = dh_dx.dot(f)

        # ICCBF layers
        b1 = Lfb0 + 4 * h
        db1_dx = se.Matrix([se.diff(b1, var) for var in [d,v]])
        Lgb1_1 = db1_dx.dot(g.col(0))
        # Lgb1_2 = db1_dx.dot(g.col(1))

        
        b2 = db1_dx.dot(f) + Lgb1_1 * uInf +7* se.sqrt(se.sqrt(b1*b1))
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

