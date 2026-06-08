#!/usr/bin/env python3
"""
Generate inspection comparison plots:
  1) LSTM+ICCBF  vs  LSTM (RL-only)           -> 5x2 figure
  2) LSTM+ICCBF  vs  LSTM (RL-only) vs Mamba2+ICCBF  -> 5x3 figure

Uses the existing plot_inspection_results.plot_inspection_threeway for the
3-column case, and a lightly adapted 2-column variant for the head-to-head.
"""

from __future__ import annotations
import os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.io import loadmat

# ── reuse helpers from the project plotting module ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
from metarl_iccbf.inspection.plot_inspection_results import (
    plot_inspection_threeway,
    _get_field, _get_dt, _get_actions, _get_num_inspected,
    _get_hsun, _get_RC_RD_RMAX, _get_meta_params,
    _crop_kstop, _total_dv, _log_color_mapper, _latex_table,
)

# ── paths ───────────────────────────────────────────────────────────────
LSTM_ICCBF  = "outputs/data/inspection/Inspection_RNNRL.mat"
LSTM_RLONLY = "ResultsEval/inspection/inspection_eval_RNN_rl_only_parallel.mat"
MAMBA_ICCBF = "ResultsEval/inspection/inspection_eval_MAMBA_iccbf_parallel.mat"

OUT_DIR = "outputs/figs"
os.makedirs(OUT_DIR, exist_ok=True)


def _plot_kiz_annulus(ax, Rlo, Rhi):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(
        Rhi * np.cos(theta), Rhi * np.sin(theta),
        Rlo * np.cos(theta), Rlo * np.sin(theta),
        color="lightyellow", edgecolor="goldenrod", linewidth=0.6, alpha=0.45,
    )


def _prepare(M, dt_fallback=1.0, target_inspected=100.0, tol_inspected=1e-9):
    """Prepare per-model derived arrays (same logic as plot_inspection_threeway)."""
    states = np.asarray(_get_field(M, "states"), dtype=float).squeeze()
    N, T, D = states.shape
    X = states[:, :, :6]

    U = np.asarray(_get_actions(M), dtype=float).squeeze()
    if U.shape[0] != N and U.shape[1] == N:
        U = np.transpose(U, (1, 0, 2))
    # For ICCBF policies, actions may be 12D (3 thrust + 9 CBF params).
    # Only keep the first 3 columns (thrust) for trajectory colouring.
    if U.ndim == 3 and U.shape[2] > 3:
        U = U[:, :, :3]

    steps_taken = np.asarray(_get_field(M, "steps_taken")).reshape(-1).astype(int)
    hsun = _get_hsun(M, N, T)
    nins = _get_num_inspected(M, N, T)
    R_C, R_D, R_MAX = _get_RC_RD_RMAX(M, N)
    kStop, is_unsafe = _crop_kstop(X, steps_taken, R_C, R_D, R_MAX, hsun)

    m_vec = None
    meta, names = _get_meta_params(M)
    if meta is not None and names and "m" in [s.strip() for s in names]:
        m_vec = meta[:, [s.strip() for s in names].index("m")]
    else:
        mv = _get_field(M, "m_vec", default=None)
        if mv is not None and np.size(mv) > 0:
            mv = np.asarray(mv, dtype=float).reshape(-1)
            if mv.size == N:
                m_vec = mv

    dt_i = _get_dt(M, default=dt_fallback)

    # Prefer uOptmag (actual applied ||u_safe||) for DV computation when
    # the actions array stores raw RL outputs instead of the CBF-filtered
    # control.  This is the case for the parallel evaluator .mat files.
    uOptmag = _get_field(M, "uOptmag", default=None)
    if uOptmag is not None and np.size(uOptmag) > 0:
        uOptmag = np.asarray(uOptmag, dtype=float).squeeze()  # (N, T)
        Utot_all = np.zeros(N, dtype=float)
        for i in range(N):
            k = int(kStop[i])
            mag_sum = float(np.sum(uOptmag[i, :k]))
            if m_vec is not None:
                Utot_all[i] = mag_sum / m_vec[i] * dt_i
            else:
                Utot_all[i] = mag_sum * dt_i
    else:
        Utot_all = _total_dv(U, kStop, dt_i, m_vec=m_vec)

    if np.all(np.isnan(nins)):
        is_success = np.zeros(N, dtype=bool)
    else:
        is_success = np.array([
            nins[i, int(kStop[i]) - 1] >= (target_inspected - tol_inspected)
            for i in range(N)
        ])

    return {
        "M": M, "X": X, "U": U, "N": N, "T": T, "dt": dt_i,
        "steps_taken": steps_taken, "kStop": kStop.astype(int),
        "is_unsafe": is_unsafe, "hsun": hsun, "nins": nins,
        "R_C": R_C, "R_D": R_D, "R_MAX": R_MAX,
        "Utot_all": Utot_all, "is_success": is_success,
    }


def _print_summary(per, model_names):
    """Print a quick summary table to stdout."""
    hdr = f"{'Metric':<30s}"
    for n in model_names:
        hdr += f"  {n:>22s}"
    print(hdr)
    print("-" * len(hdr))

    # DV
    row = f"{'DV mean +/- std (m/s)':<30s}"
    for P in per:
        u = P["Utot_all"]
        row += f"  {np.mean(u):>8.2f} +/- {np.std(u):<8.2f}"
    print(row)

    # Inspection
    row = f"{'Inspection mean (%)':<30s}"
    for P in per:
        nins = P["nins"]
        kStop = P["kStop"]
        fi = np.array([nins[i, kStop[i]-1] for i in range(P["N"])])
        row += f"  {np.mean(fi):>22.2f}"
    print(row)

    # Task success
    row = f"{'Task success (%)':<30s}"
    for P in per:
        row += f"  {np.mean(P['is_success'])*100:>22.1f}"
    print(row)

    # Safety
    row = f"{'Safety (%)':<30s}"
    for P in per:
        row += f"  {(1 - np.mean(P['is_unsafe']))*100:>22.1f}"
    print(row)
    print()


def plot_n_way(
    mat_paths: list[str],
    model_names: list[str],
    save_path: str,
    *,
    stride_traj: int = 5,
    stride_ts: int = 5,
    Nsucc_plot: int = 500,
    Nfail_plot: int = 500,
    rng_seed: int = 1,
    target_inspected: float = 100.0,
    kiz_nominal: float = 800.0,
    kiz_band: float = 0.10,
):
    """Generalised N-column version of plot_inspection_threeway."""
    ncols = len(mat_paths)
    models = [loadmat(p) for p in mat_paths]
    per = [_prepare(M, target_inspected=target_inspected) for M in models]

    _print_summary(per, model_names)

    # global colour scale
    succ_utots_for_scale = []
    for col, P in enumerate(per):
        rng2 = np.random.default_rng(rng_seed + 100 * (col + 1))
        idx_succ = np.where(P["is_success"])[0]
        if idx_succ.size > 0:
            pick = rng2.choice(idx_succ, min(Nsucc_plot, idx_succ.size), replace=False)
            succ_utots_for_scale.extend(P["Utot_all"][pick].tolist())

    if not succ_utots_for_scale:
        umin_global, umax_global = 0.0, 1.0
    else:
        umin_global = float(np.percentile(succ_utots_for_scale, 5))
        umax_global = float(np.percentile(succ_utots_for_scale, 95))
        if umax_global <= umin_global + 1e-12:
            umax_global = umin_global + 1.0

    cmap = get_cmap("turbo")
    colorOf = _log_color_mapper(
        np.array(succ_utots_for_scale, dtype=float), umin_global, umax_global, cmap
    )

    # figure
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
    fig_w = 4.3 * ncols + 0.6
    fig = plt.figure(figsize=(fig_w, 10.3), dpi=150)
    gs = fig.add_gridspec(5, ncols, wspace=0.28, hspace=0.35)

    RkizLo = (1.0 - kiz_band) * kiz_nominal
    RkizHi = (1.0 + kiz_band) * kiz_nominal

    for col, (P, name) in enumerate(zip(per, model_names)):
        X, U, dt_i = P["X"], P["U"], P["dt"]
        N, T = P["N"], P["T"]
        kStop = P["kStop"]
        nins = P["nins"]
        hsun = P["hsun"]

        rnorm = np.linalg.norm(X[:, :, :3], axis=2)
        KOZ = (P["R_C"] + P["R_D"]).reshape(-1, 1)
        hKOZ = rnorm - KOZ
        hKIZ = P["R_MAX"].reshape(-1, 1) - rnorm

        rng2 = np.random.default_rng(rng_seed + 100 * (col + 1))
        idx_succ = np.where(P["is_success"])[0]
        idx_fail = np.where(~P["is_success"])[0]
        idx_succ_plot = (
            rng2.choice(idx_succ, min(Nsucc_plot, idx_succ.size), replace=False)
            if idx_succ.size else np.array([], dtype=int)
        )
        idx_fail_plot = (
            rng2.choice(idx_fail, min(Nfail_plot, idx_fail.size), replace=False)
            if idx_fail.size else np.array([], dtype=int)
        )
        idx_plot = np.concatenate([idx_fail_plot, idx_succ_plot])
        Utot_plot = P["Utot_all"][idx_plot] if idx_plot.size else np.array([], dtype=float)

        tidx_traj = np.arange(0, T, stride_traj, dtype=int)
        tidx_ts = np.arange(0, T, stride_ts, dtype=int)

        # row 1: XY trajectory
        ax1 = fig.add_subplot(gs[0, col])
        ax1.grid(True)
        for j, ii in enumerate(idx_plot):
            ii = int(ii)
            kEnd = int(kStop[ii])
            ti = tidx_traj[tidx_traj < kEnd]
            if ti.size < 2:
                continue
            ls = "--" if ii in idx_fail_plot else "-"
            lw = 0.9 if ii in idx_fail_plot else 0.6
            ax1.plot(X[ii, ti, 0], X[ii, ti, 1], ls=ls, lw=lw, color=colorOf(Utot_plot[j]))
        _plot_kiz_annulus(ax1, RkizLo, RkizHi)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("x [m]")
        if col == 0:
            ax1.set_ylabel("y [m]")
        ax1.set_title(name)
        nFail = int(np.sum(~P["is_success"]))
        nSucc = int(np.sum(P["is_success"]))
        ax1.plot([], [], "k--", lw=0.9, label=f"Fail ({nFail})")
        ax1.plot([], [], "k-", lw=0.6, label=f"Success ({nSucc})")
        ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False, fontsize=8)

        # helper for time-series rows
        def plot_h(row, H, ylabel):
            ax = fig.add_subplot(gs[row, col])
            ax.grid(True)
            vals = []
            for ii in idx_plot:
                kEnd = int(kStop[int(ii)])
                ti = tidx_ts[tidx_ts < kEnd]
                if ti.size >= 2:
                    vals.append(H[int(ii), ti])
            if vals:
                vv = np.concatenate(vals)
                ylo, yhi = np.percentile(vv, [1, 99]).tolist()
                pad = 0.08 * (yhi - ylo + 1e-12)
                ylo -= pad; yhi += pad
            else:
                ylo, yhi = -1.0, 1.0
            ylo = min(ylo, -0.1); yhi = max(yhi, 0.1)

            for j, ii in enumerate(idx_plot):
                ii = int(ii)
                kEnd = int(kStop[ii])
                ti = tidx_ts[tidx_ts < kEnd]
                if ti.size < 2:
                    continue
                tt = ti * dt_i
                ax.plot(tt, np.maximum(H[ii, ti], ylo), lw=0.7, color=colorOf(Utot_plot[j]))
            ax.axhline(0.0, ls="--", lw=0.8, color="k")
            ax.set_ylim([ylo, yhi])
            if col == 0:
                ax.set_ylabel(ylabel)
            return ax

        # row 2: hKOZ
        plot_h(1, hKOZ, r"$h_{\mathrm{KOZ}}(t)$")

        # row 3: hKIZ
        plot_h(2, hKIZ, r"$h_{\mathrm{KIZ}}(t)$")

        # row 4: hSUN
        ax4 = fig.add_subplot(gs[3, col])
        if np.all(np.isnan(hsun)):
            ax4.axis("off")
            ax4.text(0.05, 0.6, r"$h_{\mathrm{SUN}}$ n/a", transform=ax4.transAxes)
        else:
            plot_h(3, hsun, r"$h_{\mathrm{SUN}}(t)$")

        # row 5: inspection %
        ax5 = fig.add_subplot(gs[4, col])
        if np.all(np.isnan(nins)):
            ax5.axis("off")
        else:
            ax5.grid(True)
            for j, ii in enumerate(idx_plot):
                ii = int(ii)
                kEnd = int(kStop[ii])
                ti = tidx_ts[tidx_ts < kEnd]
                if ti.size < 2:
                    continue
                ax5.plot(ti * dt_i, nins[ii, ti], lw=0.9, color=colorOf(Utot_plot[j]))
            ax5.axhline(target_inspected, ls=":", lw=0.8, color="k")
            ax5.set_ylim([0, 105])
            if col == 0:
                ax5.set_ylabel("Inspected [%]")
            ax5.set_xlabel("t [s]")

    # colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=umin_global, vmax=umax_global)
    cax = fig.add_axes([0.93, 0.07, 0.012, 0.88])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$\Delta v$ [m/s] (log-mapped)")

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Plot 1: LSTM+ICCBF  vs  LSTM (RL-only)
    print("=" * 60)
    print("Plot 1: LSTM+ICCBF  vs  LSTM (RL-only)")
    print("=" * 60)
    plot_n_way(
        mat_paths=[LSTM_ICCBF, LSTM_RLONLY],
        model_names=["LSTM + ICCBF", "LSTM (RL-only)"],
        save_path=os.path.join(OUT_DIR, "inspection_LSTM_iccbf_vs_rlonly.png"),
    )

    # Plot 2: all three architectures
    print("=" * 60)
    print("Plot 2: LSTM+ICCBF  vs  LSTM (RL-only)  vs  Mamba2+ICCBF")
    print("=" * 60)
    plot_n_way(
        mat_paths=[LSTM_ICCBF, LSTM_RLONLY, MAMBA_ICCBF],
        model_names=["LSTM + ICCBF", "LSTM (RL-only)", "Mamba2 + ICCBF"],
        save_path=os.path.join(OUT_DIR, "inspection_all_architectures.png"),
    )
