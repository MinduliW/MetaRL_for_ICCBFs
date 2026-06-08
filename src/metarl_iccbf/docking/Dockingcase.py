import numpy as np
import cvxpy as cp

class DockingCase:
    
    
    def __init__(self, rho, gamma):
        self.rho = rho
        self.gamma = gamma  # ensure gamma is in radians

        # -- Parametric QP (built once, re-solved with warm_start) --
        self._qp_Lgh1_p  = cp.Parameter()       # CBF coupling, component 1
        self._qp_Lgh2_p  = cp.Parameter()       # CBF coupling, component 2
        self._qp_h_p     = cp.Parameter()       # ICCBF value (multiplies k)
        self._qp_rhs_cbf = cp.Parameter()       # nuMargin - Lfh - hslack*h
        self._qp_LgV1_p  = cp.Parameter()       # CLF coupling, component 1
        self._qp_LgV2_p  = cp.Parameter()       # CLF coupling, component 2
        self._qp_rhs_clf = cp.Parameter()       # -Lslack*V - LfV

        self._qp_u     = cp.Variable(2)
        self._qp_k     = cp.Variable(nonneg=True)
        self._qp_delta = cp.Variable(nonneg=True)

        _cost = cp.sum_squares(self._qp_u) + 50.0 * self._qp_delta + 10.0 * self._qp_k
        _constraints = [
            # CBF: Lgh1*u[0] + Lgh2*u[1] + h*k >= rhs_cbf
            self._qp_Lgh1_p * self._qp_u[0]
            + self._qp_Lgh2_p * self._qp_u[1]
            + self._qp_h_p * self._qp_k
            >= self._qp_rhs_cbf,
            # CLF: LgV1*u[0] + LgV2*u[1] - delta <= rhs_clf
            self._qp_LgV1_p * self._qp_u[0]
            + self._qp_LgV2_p * self._qp_u[1]
            - self._qp_delta
            <= self._qp_rhs_clf,
        ]
        self._qp_problem = cp.Problem(cp.Minimize(_cost), _constraints)

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
    
    
    def qp_optimizationICCBF(self, f_x, g_x, h, Lfh, Lgh1, Lgh2, x, hslack=0.05, Lslack=0.1, nuMargin=0.0):
        V, dV_dx_val = self.calculate_V_and_dV(x)
        LgV = dV_dx_val @ g_x   # shape (2,)
        LfV = dV_dx_val @ f_x   # scalar

        # Update parameter values (no problem rebuild)
        self._qp_Lgh1_p.value  = float(Lgh1)
        self._qp_Lgh2_p.value  = float(Lgh2)
        self._qp_h_p.value     = float(h)
        self._qp_rhs_cbf.value = float(nuMargin - Lfh - hslack * h)
        self._qp_LgV1_p.value  = float(LgV[0])
        self._qp_LgV2_p.value  = float(LgV[1])
        self._qp_rhs_clf.value = float(-Lslack * V - LfV)

        try:
            self._qp_problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True)
            if self._qp_problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                u_val = self._qp_u.value if self._qp_u.value is not None else np.zeros(2)
                k_val = float(self._qp_k.value) if self._qp_k.value is not None else 0.0
                return np.asarray(u_val).flatten(), k_val, True
        except cp.error.SolverError:
            pass

        return np.array([0.0, 0.0]), 0.0, False
    

    
