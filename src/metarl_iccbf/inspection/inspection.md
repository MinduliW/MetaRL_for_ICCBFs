What's actually in the code

propwithCW at dynamicsandControl.py:266 does:


x_trans_next = A_d @ x_trans + B_d @ u
theta_S_next = theta_S - n*dt
So the real equation is:
$$x_{k+1} = \underbrace{A_d(T)}{\phi(T)} x_k + \underbrace{B_d(T)}{\Gamma(T)} u_k, \quad \theta_{S,k+1} = \theta_{S,k} - nT$$

$u_k = [F_x, F_y, F_z]^T$ — forces in Newtons, not $\Delta v$
$B_d$ already has the $1/m$ baked in (line 255)
Chief is at the origin, stationary — no adversarial logic currently exists
Explaining the $x_{k+1}$ equation

$\phi(T)$ is the exact zero-order-hold (ZOH) discretization of the Clohessy-Wiltshire-Hill (CWH) equations. The CWH continuous dynamics are:
$$\ddot{x} = 3n^2 x + 2n\dot{y} + F_x/m$$
$$\ddot{y} = -2n\dot{x} + F_y/m$$
$$\ddot{z} = -n^2 z + F_z/m$$

These describe relative motion of the deputy w.r.t. the chief in the Hill (LVLH) frame. The $2n\dot{y}$, $-2n\dot{x}$ terms are Coriolis; $3n^2 x$ is the tidal/radial gradient. Discretizing this exactly over one step $T$ gives the 6×6 $A_d$ at line 193.

Sign issue for your adversarial formulation

Since $\mathbf{r} = r_{dep} - r_{chief}$, the chief's thrust $u_{chief}$ enters as:
$$x_{k+1} = \phi(T)x_k + \Gamma(T)u_{dep,k} - \Gamma(T)u_{chief,k}$$

So it's minus, not plus. Also note: your $d_k$ is written as a $\Delta v$ (m/s), but $\Gamma(T)$ maps forces (N). You'd need to either:

Convert: $u_{chief} = m \cdot \Delta v_{max} / T$ (if applied as constant force over step), or
Use a separate "velocity impulse" mapping (add $\Delta v$ directly to velocity states)
The simplest implementation for a velocity-impulse adversary would be to add $[0,0,0, d_{x}, d_{y}, d_{z}]^T$ directly to the state after propagation, bypassing $\Gamma(T)$ entirely.

Because the CWH dynamics are a linear ODE: $\dot{x} = Ax + Bu$.

The exact solution over one time step $T$ (with $u$ held constant — ZOH) is:

$$x(T) = \underbrace{e^{AT}}{\phi(T)}, x_0 + \underbrace{\left(\int_0^T e^{A\tau},d\tau\right)B}{\Gamma(T)}, u$$

So $\phi(T) = e^{AT}$ is the matrix exponential — it answers: "if I apply no control and just let orbital mechanics evolve for $T$ seconds, where does the deputy end up?"

It captures all the natural dynamics in one matrix multiply:

The $3n^2 x$ tidal acceleration (deputy drifts radially due to gravity gradient)
The Coriolis terms $2n\dot{y}$, $-2n\dot{x}$ (orbital frame is rotating)
The out-of-plane oscillation at frequency $n$
Without $\phi(T) x_k$, you'd just be ignoring that the spacecraft is in orbit and treating it like it's floating in empty space. The $\phi(T)$ term is the "free drift" — what happens for free — and $\Gamma(T) u_k$ is the correction your thruster applies on top of that.

Relative-frame sign convention (from the code)

state[:3] = $\mathbf{r} = r_{dep} - r_{chief}$, so $\mathbf{r}$ points from chief to deputy. The chief thrusting away from the deputy means it moves in direction $-\hat{\mathbf{r}}$ in absolute space, which appears in the relative frame as:

$$\Delta \mathbf{r} = \underbrace{-\Delta r_{chief}}{+\Delta v{max}\hat{\mathbf{r}}}$$

So the disturbance increases $||\mathbf{r}||$ — correct behavior.

Cleanest implementation — velocity impulse

Skip the $\Gamma(T)$ debate entirely. After normal propagation, add directly to velocities:

$$x_{k+1} = A_d x_k + B_d u_{dep,k} + \underbrace{\begin{bmatrix}\mathbf{0}_3 \ \Delta v_k \hat{\mathbf{r}}k\end{bmatrix}}{d_k}$$

In propwithCW terms:


x_next = A_d @ x_trans + B_d @ u_dep           # normal step
r = x_trans[:3]
r_hat = r / np.linalg.norm(r)
x_next[3:6] += dv_max * r_hat                  # chief retreats
Budget tracking


dv_budget_remaining  # initialized from U[0.5, 5] m/s each episode
dv_per_step = dv_total / MAX_STEPS             # uniform spend
dv_max = min(dv_per_step, dv_budget_remaining)
dv_budget_remaining -= dv_max
Uniform spend is the most natural for a "passive adversary." A smarter adversary could save budget and spend it at critical moments (e.g., when deputy is close).

What changes in the env

reset(): sample self.dv_budget = rng.uniform(0.5, 5.0) and compute self.dv_per_step
step(): after propwithCW, apply the impulse above before updating self.state
info: add dv_budget_remaining so it's observable/loggable

