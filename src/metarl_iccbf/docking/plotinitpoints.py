import numpy as np
import matplotlib.pyplot as plt

spec_path = "docking_episode_spec_N5000_seed123.npz"
with np.load(spec_path) as d:
    x0s = d["x0s"]  # (N,5)
    rho_vec = d["rho_vec"] if "rho_vec" in d.files else None
    gamma_vec = d["gamma_vec"] if "gamma_vec" in d.files else None
    h0_vec = d["h0_vec"] if "h0_vec" in d.files else None

# Position (convert km -> m for readability)
x_m = x0s[:, 0] * 1e3
y_m = x0s[:, 1] * 1e3

plt.figure()
plt.scatter(x_m, y_m, s=6, alpha=0.4)
plt.axis("equal")
plt.xlabel("x0 [m]")
plt.ylabel("y0 [m]")
plt.title("Initial position cloud")
plt.grid(True)
plt.show()

# Optional: colour by inner-safe margin h0 (if saved)
if h0_vec is not None:
    plt.figure()
    sc = plt.scatter(x_m, y_m, s=6, alpha=0.6, c=h0_vec)
    plt.axis("equal")
    plt.xlabel("x0 [m]")
    plt.ylabel("y0 [m]")
    plt.title("Initial positions coloured by h0 (inner-safe margin)")
    plt.grid(True)
    plt.colorbar(sc, label="h0 at x0")
    plt.show()

# Optional: plot (y-range) implied by rho/gamma at x = x_init (visual sanity check)
if (rho_vec is not None) and (gamma_vec is not None):
    tg = np.tan(gamma_vec)
    y_low  = -((90.0/1e3 - rho_vec)        * tg) * 1e3  # m
    y_high =  ((90.0/1e3 - rho_vec - 1e-3) * tg) * 1e3  # m

    plt.figure()
    plt.scatter(np.arange(len(y_low)), y_low, s=2, alpha=0.3, label="y_low")
    plt.scatter(np.arange(len(y_high)), y_high, s=2, alpha=0.3, label="y_high")
    plt.xlabel("episode index")
    plt.ylabel("y bound [m]")
    plt.title("Per-episode y-bounds implied by (rho, gamma)")
    plt.grid(True)
    plt.legend()
    plt.show()
