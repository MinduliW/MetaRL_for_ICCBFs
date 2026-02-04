import numpy as np
from math import pi
import symengine as se
import sympy as sp
from sympy.utilities.lambdify import lambdify
from dataclasses import dataclass

def safe_acos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

@dataclass
class ICCBFFuncs:
    # Filled at runtime with callables
    h_func:      callable = None
    b1_func:     callable = None
    Lgb1_1_func: callable = None
    Lgb1_2_func: callable = None
    Lgb1_3_func: callable = None
    b2_func:     callable = None
    Lfb2_func:   callable = None
    Lgb2_1_func: callable = None
    Lgb2_2_func: callable = None
    Lgb2_3_func: callable = None
    
    # optional base-level terms
    Lfb0_func:   callable = None
    Lgb0_1_func: callable = None
    Lgb0_2_func: callable = None
    Lgb0_3_func: callable = None


class ICCBF:
    
    def __init__(self, rho1,rho2, mu, n ,m, alphaFOV):

        self.rho1 = rho1
        self.rho2 = rho2
        
        self.mu = mu
        self.n = n
        self.m = m
        self.alphaFOV = alphaFOV
 
        
  
  
    
    def getICCBFs(self):
        

        x1, x2, x3, x4, x5, x6= se.symbols('x1 x2 x3 x4 x5 x6')
        u1inf, u2inf, u3inf = se.symbols('u1inf u2inf u3inf')
        rs1, rs2,rs3 = se.symbols('rs1 rs2 rs3')
        
        #other factors 
        m, RC, RD , r, rho2 = se.symbols('m RC RD r rho2')
        
        rho1 = RC + RD;
        
        acoef1, acoef2, bcoef1,bcoef2, ccoef1,ccoef2 = se.symbols('acoef1 acoef2 bcoef1 bcoef2 ccoef1 ccoef2')

        # Parameters
        mu = self.mu
        n = se.sqrt(mu / r**3)
   

        term1 = 3.0 * n**2 * x1 + 2.0 * n * x5
        term2 =  -2.0 * n * x4 
        term3 = -1.0 * n**2 * x3

        f = se.Matrix([
            x4,     # dx/dt = vx
            x5,     # dy/dt = vy
            x6,     # dz/dt = vz
            term1,    # dvx/dt
            term2,    # dvy/dt
            term3     # dvz/dt
        ])

        # g_x: control input matrix (force over mass)
        g= (1 / m) * se.Matrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])


        

        # CBF
        alpha = 1/(rho2**2 - rho1**2); 
        
        h = alpha*(x1**2 +x2**2 + x3**2- rho1**2)
        dh_dx = se.Matrix([se.diff(h, var) for var in [x1, x2, x3, x4, x5, x6]])

        # Lie derivatives
        Lgb0_1 = dh_dx.dot(g[:, 0])
        Lgb0_2 = dh_dx.dot(g[:, 1])
        Lgb0_3 = dh_dx.dot(g[:, 2])
        Lfb0 = dh_dx.dot(f)

        # ICCBF layers
        b1 = Lfb0 + acoef1 * h
        db1_dx = se.Matrix([se.diff(b1, var) for var in [x1, x2, x3, x4, x5, x6]])
        Lgb1_1 = db1_dx.dot(g[:, 0])
        Lgb1_2 = db1_dx.dot(g[:, 1])
        Lgb1_3 = db1_dx.dot(g[:, 2])

        b2 = db1_dx.dot(f) + Lgb1_1 * u1inf + Lgb1_2 * u2inf +  Lgb1_3 * u3inf  + acoef2 * b1
        db2_dx = se.Matrix([se.diff(b2, var) for var in [x1, x2, x3, x4,x5,x6]])
        Lfb2 = db2_dx.dot(f)
        Lgb2_1 = db2_dx.dot(g[:, 0])
        Lgb2_2 = db2_dx.dot(g[:, 1])
        Lgb2_3 = db2_dx.dot(g[:, 2])
        
        state_vars = [x1, x2, x3, x4, x5, x6,acoef1,acoef2,bcoef1,bcoef2,ccoef1,ccoef2,  m, RC, RD , r, rho2]
        all_vars = state_vars + [u1inf, u2inf, u3inf]
        

        CBF1Vals = ICCBFFuncs(
                h_func      = lambdify(state_vars, h,       modules='numpy'),
                b1_func     = lambdify(state_vars, b1,      modules='numpy'),
                Lgb1_1_func = lambdify(state_vars, Lgb1_1,  modules='numpy'),
                Lgb1_2_func = lambdify(state_vars, Lgb1_2,  modules='numpy'),
                Lgb1_3_func = lambdify(state_vars, Lgb1_3,  modules='numpy'),
                b2_func     = lambdify(all_vars,   b2,      modules='numpy'),
                Lfb2_func   = lambdify(all_vars,   Lfb2,    modules='numpy'),
                Lgb2_1_func = lambdify(all_vars,   Lgb2_1,  modules='numpy'),
                Lgb2_2_func = lambdify(all_vars,   Lgb2_2,  modules='numpy'),
                Lgb2_3_func = lambdify(all_vars,   Lgb2_3,  modules='numpy'),
                # optional base-level
                Lfb0_func   = lambdify(state_vars, Lfb0,    modules='numpy'),
                Lgb0_1_func = lambdify(state_vars, Lgb0_1,  modules='numpy'),
                Lgb0_2_func = lambdify(state_vars, Lgb0_2,  modules='numpy'),
                Lgb0_3_func = lambdify(state_vars, Lgb0_3,  modules='numpy')
            )
     
     
     
          
        # CBF
        h = alpha*(rho2**2 - x1**2 -x2**2 -x3**2)
        dh_dx = se.Matrix([se.diff(h, var) for var in [x1, x2, x3, x4, x5, x6]])

        # Lie derivatives
        Lgb0_1 = dh_dx.dot(g[:, 0])
        Lgb0_2 = dh_dx.dot(g[:, 1])
        Lgb0_3 = dh_dx.dot(g[:, 2])
        Lfb0 = dh_dx.dot(f)

        # ICCBF layers
        b1 = Lfb0 + bcoef1 * h
        db1_dx = se.Matrix([se.diff(b1, var) for var in [x1, x2, x3, x4, x5, x6]])
        Lgb1_1 = db1_dx.dot(g[:, 0])
        Lgb1_2 = db1_dx.dot(g[:, 1])
        Lgb1_3 = db1_dx.dot(g[:, 2])

        b2 = db1_dx.dot(f) + Lgb1_1 * u1inf + Lgb1_2 * u2inf +  Lgb1_3 * u3inf  + bcoef2 * b1
        db2_dx = se.Matrix([se.diff(b2, var) for var in [x1, x2, x3, x4,x5,x6]])
        Lfb2 = db2_dx.dot(f)
        Lgb2_1 = db2_dx.dot(g[:, 0])
        Lgb2_2 = db2_dx.dot(g[:, 1])
        Lgb2_3 = db2_dx.dot(g[:, 2])
        
        state_vars = [x1, x2, x3, x4, x5, x6,acoef1,acoef2,bcoef1,bcoef2,ccoef1,ccoef2, m, RC, RD , r, rho2]
        all_vars = state_vars + [u1inf, u2inf, u3inf]
       
        
        CBF2Vals = ICCBFFuncs(
                h_func      = lambdify(state_vars, h,       modules='numpy'),
                b1_func     = lambdify(state_vars, b1,      modules='numpy'),
                Lgb1_1_func = lambdify(state_vars, Lgb1_1,  modules='numpy'),
                Lgb1_2_func = lambdify(state_vars, Lgb1_2,  modules='numpy'),
                Lgb1_3_func = lambdify(state_vars, Lgb1_3,  modules='numpy'),
                b2_func     = lambdify(all_vars,   b2,      modules='numpy'),
                Lfb2_func   = lambdify(all_vars,   Lfb2,    modules='numpy'),
                Lgb2_1_func = lambdify(all_vars,   Lgb2_1,  modules='numpy'),
                Lgb2_2_func = lambdify(all_vars,   Lgb2_2,  modules='numpy'),
                Lgb2_3_func = lambdify(all_vars,   Lgb2_3,  modules='numpy'),
                # optional base-level
                Lfb0_func   = lambdify(state_vars, Lfb0,    modules='numpy'),
                Lgb0_1_func = lambdify(state_vars, Lgb0_1,  modules='numpy'),
                Lgb0_2_func = lambdify(state_vars, Lgb0_2,  modules='numpy'),
                Lgb0_3_func = lambdify(state_vars, Lgb0_3,  modules='numpy')
            )
     
     
        # LOS constraint 
        rbhat = -se.Matrix([x1, x2, x3]) / se.sqrt(x1**2 + x2**2 + x3**2)

        # unit sun vector
        rshat = se.Matrix([rs1, rs2, rs3]) / se.sqrt(rs1**2 + rs2**2 + rs3**2)

        # angle between them
        thetab = se.acos(rbhat.dot(rshat))

        hLOS = thetab - self.alphaFOV/2;
        dh_dx = se.Matrix([se.diff(hLOS, var) for var in [x1, x2, x3, x4, x5, x6]])
        Lgb0_1 = dh_dx.dot(g[:, 0])
        Lgb0_2 = dh_dx.dot(g[:, 1])
        Lgb0_3 = dh_dx.dot(g[:, 2])
        Lfb0 = dh_dx.dot(f)

        # ICCBF layers
        b1 = Lfb0 + ccoef1 * hLOS
        db1_dx = se.Matrix([se.diff(b1, var) for var in [x1, x2, x3, x4, x5, x6]])
        Lgb1_1 = db1_dx.dot(g[:, 0])
        Lgb1_2 = db1_dx.dot(g[:, 1])
        Lgb1_3 = db1_dx.dot(g[:, 2])

        b2 = db1_dx.dot(f) + Lgb1_1 * u1inf + Lgb1_2 * u2inf +  Lgb1_3 * u3inf  + ccoef2 * b1
        db2_dx = se.Matrix([se.diff(b2, var) for var in [x1, x2, x3, x4,x5,x6]])
        Lfb2 = db2_dx.dot(f)
        Lgb2_1 = db2_dx.dot(g[:, 0])
        Lgb2_2 = db2_dx.dot(g[:, 1])
        Lgb2_3 = db2_dx.dot(g[:, 2])

        state_vars = [x1, x2, x3, x4, x5, x6, rs1, rs2, rs3, acoef1,acoef2,bcoef1,bcoef2,ccoef1,ccoef2, m, RC, RD , r, rho2]
        all_vars = state_vars + [u1inf, u2inf, u3inf]
       
        custom = {"acos": safe_acos}
      
        CBF3Vals = ICCBFFuncs(
                h_func      = lambdify(state_vars, hLOS,     modules=[custom, "numpy"]),
                b1_func     = lambdify(state_vars, b1,      modules='numpy'),
                Lgb1_1_func = lambdify(state_vars, Lgb1_1,  modules='numpy'),
                Lgb1_2_func = lambdify(state_vars, Lgb1_2,  modules='numpy'),
                Lgb1_3_func = lambdify(state_vars, Lgb1_3,  modules='numpy'),
                b2_func     = lambdify(all_vars,   b2,      modules='numpy'),
                Lfb2_func   = lambdify(all_vars,   Lfb2,    modules='numpy'),
                Lgb2_1_func = lambdify(all_vars,   Lgb2_1,  modules='numpy'),
                Lgb2_2_func = lambdify(all_vars,   Lgb2_2,  modules='numpy'),
                Lgb2_3_func = lambdify(all_vars,   Lgb2_3,  modules='numpy'),
                # optional base-level
                Lfb0_func   = lambdify(state_vars, Lfb0,    modules='numpy'),
                Lgb0_1_func = lambdify(state_vars, Lgb0_1,  modules='numpy'),
                Lgb0_2_func = lambdify(state_vars, Lgb0_2,  modules='numpy'),
                Lgb0_3_func = lambdify(state_vars, Lgb0_3,  modules='numpy')
            )
     

     

        
        return CBF1Vals, CBF2Vals,CBF3Vals

 