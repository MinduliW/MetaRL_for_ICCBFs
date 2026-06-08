"""
3-D diagram of the spacecraft inspection environment.

Shows:
  r_s          : distance from chief to sun
  R_C          : chief body radius (transparent sphere at origin)
  alpha_fov    : camera FOV cone from chaser pointing toward chief (half-angle = alpha_fov/2)
  r_b, theta_b : bearing vector chaser→chief and angle to sun
  insp_pts     : subset of Fibonacci-distributed inspection points on chief
  sun_beam     : parallel illumination rays from sun direction onto chief
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

BG     = "#ffffff"
DARK   = "#1a1a1a"
RED    = "#c0392b"
BLUE   = "#2471a3"
ORANGE = "#e67e22"
GRAY   = "#bbbbbb"
GOLD   = "#f1c40f"

# ── env constants ─────────────────────────────────────────────────────────────
R_C       = 10.0              # chief radius [m]
# alpha_fov in the env is the full FOV angle; half-angle = alpha_fov/2 is used in the constraint
# h_sun = theta_b - alpha_fov/2 >= 0  →  keepout half-angle = 30°
ALPHA_FOV = np.deg2rad(60.0)  # full FOV angle [rad]

# ── schematic layout ──────────────────────────────────────────────────────────
S_chief  = 0.18                            # R_C sphere display radius
S_chaser = np.array([0.68, 0.52, 0.36])   # chaser position

THETA_S = np.deg2rad(40.0)
r_sun   = np.array([np.cos(THETA_S), np.sin(THETA_S), 0.0])
sun_pos = 1.45 * r_sun          # schematic sun position in orbit plane

# boresight: unit vector from chaser toward chief (origin)
boresight = -S_chaser / np.linalg.norm(S_chaser)

# ── helpers ───────────────────────────────────────────────────────────────────

def _sphere_mesh(r, n=28):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    return (r * np.outer(np.cos(u), np.sin(v)),
            r * np.outer(np.sin(u), np.sin(v)),
            r * np.outer(np.ones(n), np.cos(v)))


def draw_sphere(ax, center, r, color, alpha=0.10, wire_alpha=0.15, n=28):
    x, y, z = _sphere_mesh(r, n)
    ax.plot_surface(x + center[0], y + center[1], z + center[2],
                    color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(x + center[0], y + center[1], z + center[2],
                      color=color, alpha=wire_alpha,
                      rstride=n // 5, cstride=n // 5, linewidth=0.4)


def draw_cone(ax, apex, axis, half_angle, length, color, alpha=0.10, n=50):
    axis = axis / np.linalg.norm(axis)
    perp = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    v1 = np.cross(axis, perp)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)
    phi = np.linspace(0, 2 * np.pi, n)
    r_base = length * np.tan(half_angle)
    base = (apex[:, None] + length * axis[:, None]
            + r_base * np.outer(v1, np.cos(phi))
            + r_base * np.outer(v2, np.sin(phi)))
    tris = [[apex, base[:, i], base[:, i + 1]] for i in range(n - 1)]
    ax.add_collection3d(Poly3DCollection(tris, alpha=alpha,
                                         facecolor=color, edgecolor="none"))
    ax.plot(base[0], base[1], base[2], color=color, lw=0.9, alpha=0.5)
    for k in [0, n // 4, n // 2, 3 * n // 4]:
        ax.plot([apex[0], base[0, k]], [apex[1], base[1, k]], [apex[2], base[2, k]],
                color=color, lw=0.7, alpha=0.35, linestyle="--")


def arr(ax, start, end, color, lw=1.5):
    d = np.asarray(end) - np.asarray(start)
    ax.quiver(*start, *d, color=color, lw=lw, arrow_length_ratio=0.20)


def lbl(ax, pos, txt, color=DARK, fs=10, **kw):
    ax.text(*pos, txt, color=color, fontsize=fs, **kw)


def _fibonacci_sphere(n_points):
    """Fibonacci (golden angle) sampling on unit sphere."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n_points, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / n_points
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * i
    return np.stack([r_xy * np.cos(theta), r_xy * np.sin(theta), z], axis=1)


def draw_sun_beam(ax, sun_dir, chief_radius, color=GOLD, alpha=0.35, n_rays=7):
    """Draw parallel illumination rays from sun direction onto chief sphere.

    Rays are arranged in a disk perpendicular to the sun direction,
    filling the silhouette of the chief (radius = chief_radius in schematic units).
    """
    sd = np.asarray(sun_dir, dtype=float)
    sd /= np.linalg.norm(sd)

    # Two orthogonal vectors spanning the plane ⊥ to sun direction
    perp = np.array([0., 0., 1.]) if abs(sd[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(sd, perp)
    u /= np.linalg.norm(u)
    v = np.cross(sd, u)
    v /= np.linalg.norm(v)

    # Ray offsets: concentric rings filling the chief disk
    offsets = []
    offsets.append(np.zeros(3))                   # centre ray
    for ring_r in [0.40, 0.75]:                   # two rings at 40 % and 75 % of radius
        for k in range(n_rays - 1):
            ang = 2 * np.pi * k / (n_rays - 1)
            offsets.append(chief_radius * ring_r * (np.cos(ang) * u + np.sin(ang) * v))

    ray_len   = 0.60    # schematic length of each ray segment
    ray_start = 1.20    # how far back from chief centre rays begin (along +sun_dir)

    for off in offsets:
        p0 = chief_radius * sd * ray_start + off
        p1 = p0 - sd * ray_len               # ray travels toward chief (−sun_dir)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=color, lw=1.0, alpha=alpha, linestyle="-")


# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 7), facecolor=BG)
ax  = fig.add_subplot(111, projection="3d", facecolor=BG)
ax.set_xlim(-1.7, 1.7)
ax.set_ylim(-1.7, 1.7)
ax.set_zlim(-1.2, 1.2)
ax.set_box_aspect([1, 1, 0.75])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#eeeeee")
ax.grid(False)

# ── R_C: chief body sphere ────────────────────────────────────────────────────
draw_sphere(ax, np.zeros(3), S_chief, RED, alpha=0.08, wire_alpha=0.16)
ax.scatter([0], [0], [0], color=DARK, s=18, zorder=5, depthshade=False)


# ── chaser ────────────────────────────────────────────────────────────────────
cx, cy, cz = S_chaser
ax.scatter([cx], [cy], [cz], color=BLUE, s=50, zorder=6, depthshade=False)

# ── r_b arrow: chaser → chief, labeled with r_b and theta_b arc ───────────────
arr(ax, [cx, cy, cz], [0., 0., 0.], BLUE, lw=1.6)
rb_mid = np.array([cx, cy, cz]) * 0.5
lbl(ax, rb_mid + np.array([-0.06, -0.06, 0.10]), r"$\mathbf{r}_b$", BLUE, fs=10)

# theta_b: angle at chaser between direction-to-chief and direction-to-sun
b_hat = boresight / np.linalg.norm(boresight)   # chaser → chief
s_hat = r_sun / np.linalg.norm(r_sun)           # toward sun (global)
cos_tb = np.clip(np.dot(b_hat, s_hat), -1.0, 1.0)
theta_b_val = np.arccos(cos_tb)

# dashed line from chaser in sun direction (anchors the arc visually)
sun_leg_len = 0.38
chaser_pt = np.array([cx, cy, cz])
ax.plot([cx, cx + sun_leg_len * s_hat[0]],
        [cy, cy + sun_leg_len * s_hat[1]],
        [cz, cz + sun_leg_len * s_hat[2]],
        color=ORANGE, lw=1.0, linestyle="--", alpha=0.7)

# arc sweeping between the two legs (slerp)
arc_r = 0.20
arc_pts = np.array([
    chaser_pt + arc_r * (
        np.sin((1 - t) * theta_b_val) / np.sin(theta_b_val) * b_hat +
        np.sin(t * theta_b_val) / np.sin(theta_b_val) * s_hat
    )
    for t in np.linspace(0, 1, 40)
])
ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2],
        color=ORANGE, lw=1.2, alpha=0.8)

# label at arc midpoint
mid_dir = b_hat + s_hat
mid_dir /= np.linalg.norm(mid_dir)
lbl(ax, chaser_pt + (arc_r + 0.07) * mid_dir, r"$\theta_b$", ORANGE, fs=10)

# ── alpha_fov cone: from chaser toward chief, half-angle = alpha_fov/2 = 30° ──
cone_len = np.linalg.norm(S_chaser) * 0.92   # just reaches the chief sphere
draw_cone(ax, np.array([cx, cy, cz]),
          boresight, ALPHA_FOV / 2.0, cone_len,
          BLUE, alpha=0.06)
# label at the cone rim
rim_dir = np.cross(boresight, np.array([0., 0., 1.]))
rim_dir /= np.linalg.norm(rim_dir)
rim_pt = (np.array([cx, cy, cz])
          + cone_len * boresight
          + cone_len * np.tan(ALPHA_FOV / 2.0) * rim_dir)
lbl(ax, rim_pt + np.array([0.0, 0.0, 0.08]),
    r"$\alpha_{fov}/2$", BLUE, fs=9)

# ── inspection points on chief surface ───────────────────────────────────────
N_VIZ_PTS = 20   # show a sparse subset for clarity
fib_pts = _fibonacci_sphere(N_VIZ_PTS) * S_chief
# split into illuminated (faces sun) vs shadowed
lit_mask = fib_pts @ r_sun > 0
ax.scatter(fib_pts[lit_mask, 0], fib_pts[lit_mask, 1], fib_pts[lit_mask, 2],
           color=GOLD, s=14, zorder=7, depthshade=False, alpha=0.9,
           label="insp pt (lit)")
ax.scatter(fib_pts[~lit_mask, 0], fib_pts[~lit_mask, 1], fib_pts[~lit_mask, 2],
           color=GRAY, s=14, zorder=7, depthshade=False, alpha=0.7,
           label="insp pt (shadow)")

# ── sun beam: parallel illumination rays onto chief ───────────────────────────
draw_sun_beam(ax, r_sun, S_chief, color=GOLD, alpha=0.30, n_rays=6)

# ── sun sphere + r_s label ────────────────────────────────────────────────────
xs, ys, zs = _sphere_mesh(0.09, n=22)
ax.plot_surface(xs + sun_pos[0], ys + sun_pos[1], zs + sun_pos[2],
                color=ORANGE, alpha=0.92, linewidth=0, zorder=7)
lbl(ax, sun_pos + np.array([0.0, 0.0, 0.16]), "Sun", ORANGE, fs=9, ha="center")

# r_s: line from origin to sun with double-ended tick feel
ax.plot([0, sun_pos[0]], [0, sun_pos[1]], [0, sun_pos[2]],
        color=ORANGE, lw=0.9, linestyle="--", alpha=0.55)
lbl(ax, sun_pos * 0.52 + np.array([0., 0., -0.10]),
    r"$r_s$", ORANGE, fs=10)

# ── save ──────────────────────────────────────────────────────────────────────
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspection_geometry.png")
plt.tight_layout()
plt.savefig(_out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {_out}")
