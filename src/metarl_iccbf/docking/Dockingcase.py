import numpy as np
import cvxpy as cp

class DockingCase:
    
    
    def __init__(self, rho, gamma):
        self.rho = rho
        self.gamma = gamma  # ensure gamma is in radians
        
        # self.hslack = hslack

    def generate_evenly_spread_cone_points_2d(self, n_points=1000, r_max=100.0):
        points = []
        for _ in range(n_points):
            r = r_max
            theta = np.random.uniform(-self.gamma, self.gamma)
            x = (r + self.rho) * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, 0.0, 0.0, 0.0])
        return np.array(points)

    def originalh(self, x):
        rcp = np.array([
            x[0] - (self.rho ) * np.cos(x[4]),
            x[1] - (self.rho) * np.sin(x[4])
        ])
        ehat = np.array([np.cos(x[4]), np.sin(x[4])])
        h_raw = np.dot(rcp, ehat) / np.linalg.norm(rcp) - np.cos(self.gamma)
        return  100*h_raw

        

    def calculate_V_and_dV(self, x):
        px, py, vx, vy, psi = x
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        term1 = vx + (px - self.rho * cos_psi) / 10.0
        term2 = vy + (py - self.rho * sin_psi) / 10.0
        V_val = term1**2 + term2**2

        dV_dx_val = np.array([
            2 * term1 / 10.0,
            2 * term2 / 10.0,
            2 * term1,
            2 * term2,
            (2 * term1 * self.rho * sin_psi / 10.0) - (2 * term2 * self.rho * cos_psi /10.0)
        ])

        return 1e6*V_val,  1e6*dV_dx_val


    def qp_optimizationorm(self, f_x, g_x, x):
        V, dV_dx_val = self.calculate_V_and_dV(x)
        LgV = dV_dx_val @ g_x
        LfV = dV_dx_val @ f_x

        u = cp.Variable(2)
        delta = cp.Variable()

        cost = cp.sum_squares(u) + 50.0 * delta
        constraints = [
            LfV + LgV @ u <= -0.1 * V + delta,
            delta >= 0.0
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value, True
        except cp.error.SolverError:
            pass

        return np.zeros(2), False

    def qp_optimization(self, f_x, g_x, h, dh_dx_np, x,hslack,Lslack):
        V, dV_dx_val = self.calculate_V_and_dV(x)
        LgV = dV_dx_val @ g_x
        LfV = dV_dx_val @ f_x
        Lgh = dh_dx_np @ g_x
        Lfh = dh_dx_np @ f_x

        u = cp.Variable(2)
        # k = cp.Variable(nonneg=True)
        delta = cp.Variable(nonneg=True)

        # print(hslack)
        # print(Lslack)
        cost = cp.sum_squares(u) + 50.0 * delta
        constraints = [
            Lfh + Lgh @ u >= -(hslack) * h,
            LfV + LgV @ u <= -Lslack * V + delta,
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value, True
        except cp.error.SolverError:
            pass

        return np.zeros(2), False
    
    
    def qp_optimizationICCBF2(self, f_x, g_x, h,  Lfh,Lgh1, Lgh2, x, hslack=0.05, Lslack = 0.1, nuMargin=0.0):
        V, dV_dx_val = self.calculate_V_and_dV(x)
        LgV = dV_dx_val @ g_x
        LfV = dV_dx_val @ f_x
        
    
        u = cp.Variable(2)
        delta = cp.Variable(nonneg=True)
        


        cost =  cp.sum_squares(u) + 50.0 * delta
        constraints = [
            Lfh + Lgh1*u[0] + Lgh2*u[1] >= -(hslack) * h + nuMargin,
            LfV + LgV @ u <= -Lslack * V + delta,
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.MOSEK,  verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value, True
        except cp.error.SolverError:
            pass

        return np.array([0.0, 0.0]), False
    
    
    def qp_optimizationICCBF(self, f_x, g_x, h,  Lfh,Lgh1, Lgh2, x, hslack=0.05, Lslack = 0.1, nuMargin=0.0):
        V, dV_dx_val = self.calculate_V_and_dV(x)
        LgV = dV_dx_val @ g_x
        LfV = dV_dx_val @ f_x
        
       
        u = cp.Variable(2)
        delta = cp.Variable(nonneg=True)
        k = cp.Variable(nonneg=True)
        
        
        # mosek_opts = {
        #     # conic interior-point tolerances
        #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   1e-7,
        #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   1e-7,
        #     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-7,
        #     "MSK_DPAR_INTPNT_CO_TOL_MU_RED":  1e-10,
        #     # optional: limit iterations to keep things predictable
        #     # "MSK_IPAR_INTPNT_MAX_ITERATIONS": 50,
        # }


        cost =  cp.sum_squares(u) + 50.0 * delta+ 10.0*k
        constraints = [
            Lfh + Lgh1*u[0] + Lgh2*u[1] >= -(hslack+k) * h + nuMargin,
            LfV + LgV @ u <= -Lslack * V + delta,
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.MOSEK,  verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value, k.value, True
        except cp.error.SolverError:
            pass

        return np.array([0.0, 0.0]), 0.0, False
    

    
