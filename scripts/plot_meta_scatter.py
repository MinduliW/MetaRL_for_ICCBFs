#!/usr/bin/env python3
"""
Meta-performance scatter plot: Safety % vs Task Completion % across all three scenarios.

Each task uses a distinct marker shape; each method uses a distinct color.
For Cruise Control, task completion = normalized fuel efficiency (lower fuel → higher score).

Usage:
    python scripts/plot_meta_scatter.py
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "outputs/figures/meta_scatter.png"

METHOD_COLORS = {
    "LSTM+PPO":   "#1f77b4",
    "GRU+PPO":    "#ff7f0e",
    "Mamba2+PPO": "#2ca02c",
}

# ── Data ─────────────────────────────────────────────────────────────────────
_cc_fuels = {"LSTM+PPO": 4.016, "GRU+PPO": 3.293, "Mamba2+PPO": 2.622}
_cc_min, _cc_max = min(_cc_fuels.values()), max(_cc_fuels.values())
def _cc_task(m): return (_cc_max - _cc_fuels[m]) / (_cc_max - _cc_min) * 100

# Each entry: (task_key, method, safety, task_completion, display_x, display_y)
# display_x/y apply visual offsets to separate overlapping points without
# changing what they represent — labels point back with a line.
POINTS = [
    # CC (triangle) — display_y = true safety; x is already spread naturally
    ("CC",   "LSTM+PPO",   98.8, _cc_task("LSTM+PPO"),  _cc_task("LSTM+PPO"),  98.8),
    ("CC",   "GRU+PPO",    98.8, _cc_task("GRU+PPO"),   _cc_task("GRU+PPO"),   98.8),
    ("CC",   "Mamba2+PPO", 98.9, _cc_task("Mamba2+PPO"),_cc_task("Mamba2+PPO"),98.9),
    # Docking (circle) — LSTM/GRU both at x=0; nudge x only, y = true safety
    ("Dock", "LSTM+PPO",   98.3,  0.0,  -1.5, 98.3),
    ("Dock", "GRU+PPO",    98.3,  0.0,   1.5, 98.3),
    ("Dock", "Mamba2+PPO", 97.9, 95.0,  95.0, 97.9),
    # Inspection (square) — y = true safety; all three x near 97-100, nudge
    ("Insp", "LSTM+PPO",   94.4, 97.68, 95.0, 94.4),
    ("Insp", "GRU+PPO",    67.8, 98.11, 97.0, 67.8),
    ("Insp", "Mamba2+PPO", 99.4, 99.55, 99.5, 99.4),
]

TASK_MARKERS = {"CC": "^", "Dock": "o", "Insp": "s"}

# Label positions (lx, ly) and alignment — chosen to avoid any overlap.
# Each label is connected to its point with a thin line via arrowprops.
LABELS = {
    # (task, method): (label_x, label_y, ha, va)
    ("CC",   "LSTM+PPO"):   (  15,   100,  "left",   "center"),
    ("CC",   "GRU+PPO"):    ( 44,   96,  "center", "top"),
    ("CC",   "Mamba2+PPO"): ( 102,   90,  "right",  "top"),
    ("Dock", "LSTM+PPO"):   ( 7,   90,  "right",  "center"),
    ("Dock", "GRU+PPO"):    (  4,   94,  "left",   "center"),
    ("Dock", "Mamba2+PPO"): ( 82,   92,  "right",  "center"),
    ("Insp", "LSTM+PPO"):   ( 73,   95,  "right",  "bottom"),
    ("Insp", "GRU+PPO"):    ( 88,   63,  "right",  "top"),
    ("Insp", "Mamba2+PPO"): ( 88,   101, "right",  "bottom"),
}

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 10,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.set_facecolor("white")
ax.spines[["top", "right"]].set_visible(False)

# ── Scatter points ────────────────────────────────────────────────────────────
for task, method, safety, task_val, dx, dy in POINTS:
    ax.scatter(dx, dy,
               marker=TASK_MARKERS[task],
               color=METHOD_COLORS[method],
               s=130, edgecolors="white", linewidths=0.6, zorder=3)

# ── Annotations ───────────────────────────────────────────────────────────────
for task, method, safety, task_val, dx, dy in POINTS:
    lx, ly, ha, va = LABELS[(task, method)]
    arch = method.split("+")[0]   # LSTM / GRU / Mamba2
    color = METHOD_COLORS[method]
    ax.annotate(
        arch,
        xy=(dx, dy),
        xytext=(lx, ly),
        fontsize=11,
        color=color,
        ha=ha, va=va,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
        zorder=4,
    )

# ── Legends — placed outside the axes ────────────────────────────────────────
method_handles = [
    mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o", linestyle="None",
                  markersize=8, label=m)
    for m in METHOD_COLORS
]
task_handles = [
    mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                  markersize=8, label=lbl)
    for mk, lbl in [("^", "Cruise Control (norm. fuel eff.)"),
                    ("o", "Docking (dock rate %)"),
                    ("s", "Inspection (points inspected %)")]
]

ax.legend(handles=task_handles, title="Task  (x-axis metric)",
          loc="center left", bbox_to_anchor=(1.08, 0.5),
          framealpha=0.95, edgecolor="lightgray")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Task Completion [%]")
ax.set_ylabel("Safety (CBF) [%]")
ax.set_xlim(-5, 107)
ax.set_ylim(55, 105)

fig.tight_layout()
fig.subplots_adjust(right=0.65)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
