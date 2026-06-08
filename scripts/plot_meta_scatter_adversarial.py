#!/usr/bin/env python3
"""
Meta-performance scatter plot (adversarial): Safety % vs Task Completion %
for Docking and Inspection under adversarial disturbances (PPO variants).

No adversarial CC data exists, so only two tasks are shown.

Usage:
    python scripts/plot_meta_scatter_adversarial.py
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "outputs/figures/meta_scatter_adversarial.png"

METHOD_COLORS = {
    "LSTM+PPO":   "#1f77b4",
    "GRU+PPO":    "#ff7f0e",
    "Mamba2+PPO": "#2ca02c",
}

# ── Data ─────────────────────────────────────────────────────────────────────
# Adversarial Docking: safety %, dock %
# Adversarial Inspection: safety %, points inspected %
#
# Each entry: (task_key, method, safety, task_val, display_x, display_y)
# display_x/y are visual positions to avoid overlap; labels connect via lines.
POINTS = [
    # Docking (circle) — display_y always = true safety; only x nudged if needed
    ("Dock", "LSTM+PPO",   95.6, 95.1,  95.1, 95.6),
    ("Dock", "GRU+PPO",    88.3, 74.3,  74.3, 88.3),
    ("Dock", "Mamba2+PPO", 95.4, 95.2,  95.2, 95.4),
    # Inspection (square) — x values nearly identical so nudge x slightly
    ("Insp", "LSTM+PPO",   97.2, 89.45,  89.45, 97.2),
    ("Insp", "GRU+PPO",    91.0, 98.79,  98.8, 91.0),
    ("Insp", "Mamba2+PPO", 98.0, 98.74,  98.74, 98.0),
]

TASK_MARKERS = {"Dock": "o", "Insp": "s"}

LABELS = {
    # (task, method): (label_x, label_y, ha, va)
    ("Dock", "LSTM+PPO"):   ( 92,   96,  "right",  "top"),
    ("Dock", "GRU+PPO"):    ( 70,   88,  "center", "top"),
    ("Dock", "Mamba2+PPO"): ( 98,   96,  "left",   "center"),
    ("Insp", "LSTM+PPO"):   ( 88,   98,  "right",  "bottom"),
    ("Insp", "GRU+PPO"):    ( 99,   90,  "left",   "center"),
    ("Insp", "Mamba2+PPO"): ( 97,   99,  "right",  "center"),
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
    arch = method.split("+")[0]
    color = METHOD_COLORS[method]
    ax.annotate(
        arch,
        xy=(dx, dy),
        xytext=(lx, ly),
        fontsize=12,
        color=color,
        ha=ha, va=va,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
        zorder=4,
    )

# ── Legend ────────────────────────────────────────────────────────────────────
# task_handles = [
#     mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
#                   markersize=8, label=lbl)
#     for mk, lbl in [("o", "Docking (dock rate %)"),
#                     ("s", "Inspection (points inspected %)")]
# ]

# ax.legend(handles=task_handles, title="Task  (x-axis metric)",
#           loc="center left", bbox_to_anchor=(1.08, 0.5),
#           framealpha=0.95, edgecolor="lightgray")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Task Completion [%]")
ax.set_ylabel("Safety (CBF) [%]")
ax.set_xlim(55, 100)
ax.set_ylim(82, 100)

fig.tight_layout()
fig.subplots_adjust(right=0.65)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
