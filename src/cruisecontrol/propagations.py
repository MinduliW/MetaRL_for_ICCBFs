import numpy as np
from daceypy import DA, array
from daceypy.op import cos, sin, sqr, sqrt, vnorm, atan2,asin
from math import pi 
import copy
from transformations import Transformations
from observations import Observations

class Propagations:
    
    def diffx(x:array,t:float, mu) -> array:
        
        xdot = array.zeros(6)
        xdot[-1] = sqrt(mu/x[0]**3)
    
        return xdot
    
    def nonlinearRel(x, t, ac, mu):
        
        xdot = array.zeros(6)
        nc = sqrt(mu/ac**3)
        
        xdot[0] = x[3]
        xdot[1] = x[4]
        xdot[2] = x[5]
        
        denom = ((x[0]+ac)**2 + x[1]**2 + x[2]**2)**1.5
        
        xdot[3] = 2*nc*x[4] + nc*nc*x[0] + nc*nc*ac - mu*(x[0]+ac)/denom
        xdot[4] = -2*nc*x[3]+nc*nc*x[1] - mu*x[1]/denom
        xdot[5] = -mu*x[2]/denom
        
        return xdot
        
        

    def CW(x, omeg, t):
        
        r0 = x[0:3]
        rdot0 = x[3:6]
        x0 = r0[0]
        y0 = r0[1]
        z0 = r0[2]
        xdot0 = rdot0[0]
        ydot0 = rdot0[1]
        zdot0 = rdot0[2]
        
        xt = (4*x0 + (2*ydot0)/omeg)+(xdot0/omeg)*sin(omeg*t)-(3*x0+(2*ydot0)/omeg)*cos(omeg*t)
        yt = (y0 - (2*xdot0)/omeg)+((2*xdot0)/omeg)*cos(omeg*t)+(6*x0 + (4*ydot0)/omeg)*sin(omeg*t)-(6*omeg*x0+3*ydot0)*t
        zt = z0*cos(omeg*t)+(zdot0/omeg)*sin(omeg*t)
        
        xdott = (3*omeg*x0+2*ydot0)*sin(omeg*t)+xdot0*cos(omeg*t)
        ydott = (6*omeg*x0+4*ydot0)*cos(omeg*t)-2*xdot0*sin(omeg*t)-(6*omeg*x0+3*ydot0)
        zdott = zdot0*cos(omeg*t)-z0*omeg*sin(omeg*t)
        
        
        return array([xt, yt, zt,xdott,ydott,zdott])
    
    

    def propCoe(coe, tstep,mu = 1.0):
    
        coe_new = copy.deepcopy(coe)
        
        # get MA
        M = Transformations.f2M(coe_new[-1], coe_new[1])
        
        n_coe = sqrt(mu/coe_new[0]**3)
        M_new = M+ n_coe*tstep
        
        f_new,_  = Transformations.Kepler(coe_new[1],M_new)
        
        coe_new[-1] = f_new
        
        return coe_new
    
    def propViacoe(xrv, tvec, mu):
        
        xcoe = Transformations.rv2coe(xrv, mu)
        MA0 = Transformations.f2M(xcoe[-1], xcoe[1])
        n = np.sqrt(mu / xcoe[0]**3)
        x2rv = np.zeros((len(tvec), 6))
        
        for i in range(len(tvec)):
            MAnew = (MA0 + n * tvec[i])
            MAnew = MAnew%(2*pi)
            TAnew,_ = Transformations.Kepler(xcoe[1],MAnew)
            xcoenew = np.concatenate((xcoe[:5], [TAnew]))
            x2rv[i,:] = Transformations.coe2rv(xcoenew, mu)
        
        return x2rv



    
    def propCoeDA(coe, tstep,mu = 1.0):
    
        coe_new = copy.deepcopy(coe)
        
        # get MA
        M = Transformations.f2MDA(coe_new[-1], coe_new[1])
        
        n_coe = sqrt(mu/coe_new[0]**3)
        M_new = M+ n_coe*tstep
        
        f_new,_  = Transformations.KeplerDA(coe_new[1],M_new)
        
        coe_new[-1] = f_new
        
        return coe_new
    
    def diffEqObsLin(x:array,t:float,n_cheif) -> array:
            
            
        xdot = array.zeros(42)    
        #print(t)
        x0RL = x[:6]
        
        #aPropagations.propcw
        
        xRL = Propagations.CW(x0RL,n_cheif,t)
        
        y = Observation.getObsRL(xRL)
        
        # print(y.cons())
        dy_dx0 =np.matrix(y.linear())
        dy_dx0T = np.matrix(dy_dx0.transpose()); 

        H=np.matmul(dy_dx0T,dy_dx0)
        
       
        
        k = 0; 
        for i in range(6):
            for j in range(6):
                xdot[k] = H[i,j] + 0*DA(1)
                k = k+1
                
        
        return xdot
    
    def diffEqObsnewobs(x:array,t:float, mu,coe_cheif0, coe_dep0) -> array:
        
        coe_cheif = copy.deepcopy(coe_cheif0)
        coe_cheif = Propagations.propCoeDA(coe_cheif, t,mu)
        #print(n_cheif*t)
        coe_dep = copy.deepcopy(coe_dep0)
        coe_dep = Propagations.propCoeDA(coe_dep, t,mu)
        
        cart_cheif = Transformations.coe2rvDA(coe_cheif)
        cart_dep = Transformations.coe2rvDA(coe_dep)
        
        x = Transformations.Cart2RelDA(cart_cheif,cart_dep)
        
        obsDA = x[:3]/vnorm(x[:3].cons());
        obsbal = x[:3].cons()/vnorm(x[:3].cons());
        
        obs =obsDA - obsbal
        
        # rconshat = x[:3].cons()/vnorm(x[:3].cons())
        
        # rdotrconshat = x[:3]-x[:3].dot(rconshat)*rconshat
        
        xdot = 0*DA(1)
        for i in range(3):
            xdot = xdot + obsDA[i]*obsbal[i] 
        
        #xdot = xdot/vnorm(x[:3].cons())**2
        
              
              
        
        #[r - r dot r.cons.hat]^t*[r - r dot r.cons.hat] 
            
        # xdot = 0*DA(1)
        # for i in range(3):
        #     xdot = xdot + obsDA[i]*obsbal[i] 
              
        return xdot

    def diffEqObsnlws(x:array,t:float, ac, mu) -> array:
        
        print(t)
            
        xdot = array.zeros(42)
        
        xdot[:6] = Propagations.nonlinearRel(x[:6], t, ac, mu)
        
        #print(xdot[:6].cons())
        y = Observation.getObsRL(x[:6])
        
        # print(y.cons())
        dy_dx0 =np.matrix(y.linear())
        dy_dx0T = np.matrix(dy_dx0.transpose()); 

        H=np.matmul(dy_dx0T,dy_dx0)
        

        k = 0; 
        for i in range(6):
            for j in range(6):
                xdot[k+6] = H[i,j] + 0*DA(1)
                k = k+1
                
        
        return xdot


          
    def diffEqObs(x:array,t:float, mu,coe_cheif0, coe_dep0) -> array:
                
                
         #print(t)
        #print(x[0].cons())
        #print( coe_cheif0[-1].cons())
        coe_cheif = copy.deepcopy(coe_cheif0)
        coe_cheif = Propagations.propCoe(coe_cheif, t,mu)
        #print(n_cheif*t)
        coe_dep = copy.deepcopy(coe_dep0)
        coe_dep = Propagations.propCoeDA(coe_dep, t,mu)
        
        xdot = array.zeros(36)
        
        cart_cheif = Transformations.coe2rv(coe_cheif)
        cart_dep = Transformations.coe2rvDA(coe_dep)
        
        
        y = Observation.getObs(cart_cheif, cart_dep)
        
        # print(y.cons())
        dy_dx0 =np.matrix(y.linear())
        dy_dx0T = np.matrix(dy_dx0.transpose()); 

        H=np.matmul(dy_dx0T,dy_dx0)
        

        k = 0; 
        for i in range(6):
            for j in range(6):
                xdot[k] = H[i,j] + 0*DA(1)
                k = k+1
                
        
        return xdot
