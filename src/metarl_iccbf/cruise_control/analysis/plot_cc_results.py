# src/metarl_iccbf/cruise_control/analysis/plot_cc_results.py
"""
Cruise-control plotting utilities (Python port of your MATLAB visualisation).

What it does
------------
Given 3 .mat result files (ICCBF, MLP-tuned, RNN-tuned), this module:
  1) Loads each .mat (scipy.io.loadmat).
  2) Applies viability filtering using a "max-brake" open-loop test computed on
     the baseline file’s initial-condition ordering (so all 3 models are sliced identically).
  3) Creates a 3x3 figure:
       Row 1: (x,v) trajectories colour-coded by total thrust (shared colour scale)
       Row 2: h(t) time-series (cropped; y=0 line)
       Row 3: V(t) time-series if present
  4) Creates a violin plot of total thrust over ALL viable episodes.
  5) Returns a dict with useful arrays and a LaTeX table string.

Usage (in a notebook)
---------------------
from metarl_iccbf.cruise_control.analysis.plot_cc_results import plot_cruisecontrol_threeway

out = plot_cruisecontrol_threeway(
    baseline_mat="level4/NN_noRL_FIXEDSPEC.mat",
    mlp_mat="level4/NN_PPO_FIXEDSPEC.mat",
    rnn_mat="level4/MarginNoiseMetaICCBFRNN_FIXEDSPEC.mat",
    model_names=("ICCBF","MLP-tuned ICCBF","RNN-tuned ICCBF"),
    stride_traj=10,
    sampling_mode="linspace",
    Nsamp_ts=160,
    Nsucc_plot=1000,
    Nfail_plot=60,
    seed=1,
    save_prefix=None,   # or "figs/cc_level4"
)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# -----------------------------
# Loading helpers
# -----------------------------
def load_mat(path: str) -> Dict[str, Any]:
    """Load a MATLAB .mat as a flat dict; drop MATLAB metadata keys."""
    D = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in D.items() if not k.startswith("__")}


def _as_2d_h_or_v(arr: Any) -> Optional[np.ndarray]:
    """Convert hs/Vs to shape (N, Nt) when possible."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0]
    if a.ndim == 2:
        return a
    if a.ndim == 1:
        return a[None, :]
    return a


def _get_tvec(M: Dict[str, Any], Nt: int) -> Tuple[np.ndarray, float]:
    if "tvec_full" in M and np.size(M["tvec_full"]) >= 2:
        t = np.asarray(M["tvec_full"]).reshape(-1)
    else:
        t = np.arange(Nt, dtype=float)
    dt = float(t[1] - t[0]) if t.size >= 2 else 1.0
    return t, dt


# -----------------------------
# Viability test (max brake)
# -----------------------------
def simulate_max_brake(x0: np.ndarray, T: float = 40.0, dt: float = 0.1) -> bool:
    """
    Returns True if safe under max-brake open-loop (viability test), else False.
    x0 = [d, v]
    """
    f0, f1, f2 = 0.1, 5.0, 0.25
    m, g0 = 1650.0, 9.81
    v0_lead = 13.89

    x = np.array(x0, dtype=float).reshape(2)
    t = 0.0
    while t < T:
        u = -0.25  # full brake
        F = f0 + f1 * x[1] + f2 * (x[1] ** 2)
        xdot = np.array([v0_lead - x[1], -F / m + g0 * u])
        x = x + dt * xdot
        t += dt
        h = x[0] - 1.8 * x[1]
        if h < 0.0:
            return False
    return True


def slice_model_by_episode(M: Dict[str, Any], idx_keep: np.ndarray, n_ref: int) -> Dict[str, Any]:
    """Slice every ndarray whose first dimension matches n_ref."""
    out = dict(M)
    for k, v in M.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_ref:
            out[k] = v[idx_keep, ...]
    return out


# -----------------------------
# Time-series sampling / truncation
# -----------------------------
def get_tidx_ts(k_end: int, stride_ts: int, sampling_mode: str, Nsamp_ts: int) -> np.ndarray:
    k_end = max(2, int(k_end))
    mode = sampling_mode.lower()
    if mode == "stride":
        tidx = np.arange(0, k_end, stride_ts, dtype=int)
    else:
        Ns = min(Nsamp_ts, k_end)
        tidx = np.unique(np.clip(np.round(np.linspace(0, k_end - 1, Ns)), 0, k_end - 1).astype(int))
    if tidx.size < 2:
        tidx = np.array([0, k_end - 1], dtype=int)
    return tidx


def compute_kstop(M: Dict[str, Any], i: int, Nt: int, k_fail_i: int, is_fail_i: bool) -> int:
    """
    True end index for plotting (removes trailing zero padding).
    - If failure: stop at first failure (k_fail_i)
    - Else: stop at last non-padded sample inferred from nonzero state/u/h/V.
    Returns k_stop in 1..Nt (MATLAB-like count, not Python index).
    """
    eps0 = 1e-10
    k_stop = Nt

    X = np.asarray(M["states"])[i, :, :]
    maskX = np.any(np.abs(X) > eps0, axis=1)
    if np.any(maskX):
        kX = int(np.where(maskX)[0][-1] + 1)
        k_stop = min(k_stop, kX)

    if "hs" in M:
        H = _as_2d_h_or_v(M["hs"])
        if H is not None:
            maskH = np.abs(H[i, :]) > eps0
            if np.any(maskH):
                kH = int(np.where(maskH)[0][-1] + 1)
                k_stop = max(k_stop, kH)

    if "Vs" in M:
        V = _as_2d_h_or_v(M["Vs"])
        if V is not None:
            maskV = np.abs(V[i, :]) > eps0
            if np.any(maskV):
                kV = int(np.where(maskV)[0][-1] + 1)
                k_stop = max(k_stop, kV)

    if "uOpts" in M:
        Uall = np.asarray(M["uOpts"])
        Ui = Uall[i]
        if Ui.ndim == 1:
            maskU = np.abs(Ui) > eps0
        else:
            maskU = np.any(np.abs(Ui) > eps0, axis=-1)
        if np.any(maskU):
            kU = int(np.where(maskU)[0][-1] + 1)
            k_stop = max(k_stop, kU)

    elif "actionStore" in M:
        Aall = np.asarray(M["actionStore"])
        Ai = Aall[i]
        if Ai.ndim == 1:
            maskA = np.abs(Ai) > eps0
        else:
            maskA = np.any(np.abs(Ai) > eps0, axis=-1)
        if np.any(maskA):
            kA = int(np.where(maskA)[0][-1] + 1)
            k_stop = max(k_stop, kA)

    k_stop = max(2, min(Nt, k_stop))

    if is_fail_i:
        k_stop = min(k_stop, max(2, min(Nt, int(k_fail_i))))

    return int(k_stop)


def compute_total_thrust(M: Dict[str, Any], i: int, k_end: int, dt: float) -> float:
    """
    Integral of ||u|| dt. Uses uOpts if present; else uses actionStore[:,0] proxy.
    Robust to uOpts stored as:
      - (N, Nt)          scalar control
      - (N, Nt, 1)       scalar control with singleton dim
      - (N, Nt, du)      vector control
    """
    if "uOpts" in M:
        Uall = np.asarray(M["uOpts"])
        Ui = Uall[i, :k_end]  # (k_end,) or (k_end,du)
        if Ui.ndim == 1:
            return float(np.sum(np.abs(Ui)) * dt)
        per_step = np.sum(np.abs(Ui), axis=-1)
        return float(np.sum(per_step) * dt)

    if "actionStore" in M:
        Aall = np.asarray(M["actionStore"])
        Ai = Aall[i, :k_end]
        if Ai.ndim == 1:
            a1 = Ai
        else:
            a1 = Ai[..., 0]
        return float(np.sum(np.abs(a1)) * dt)

    return float("nan")


# -----------------------------
# Summary table (LaTeX)
# -----------------------------
def thrust_totals_table_latex(
    uTotals: np.ndarray,
    model_names: Sequence[str],
    safety_rates: Optional[Sequence[float]] = None,
) -> str:
    """
    Produce a compact LaTeX table string with mean/std/median/q25/q75/safety%/N.
    uTotals: shape (n_models, n_eps), may contain NaNs.
    safety_rates: optional sequence of safety rates in [0, 100] (one per model).
    """
    rows = []
    for j, name in enumerate(model_names):
        x = np.asarray(uTotals[j, :], dtype=float)
        x = x[np.isfinite(x)]
        sr = float(safety_rates[j]) if safety_rates is not None else float("nan")
        if x.size == 0:
            rows.append((name, np.nan, np.nan, np.nan, np.nan, np.nan, sr, 0))
            continue
        rows.append((
            name,
            float(np.mean(x)),
            float(np.std(x)),
            float(np.median(x)),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 75)),
            sr,
            int(x.size),
        ))

    # Manual LaTeX (no pandas dependency)
    lines = []
    if safety_rates is not None:
        lines.append(r"\begin{tabular}{lrrrrrrl}")
        lines.append(r"\hline")
        lines.append(r"Method & Mean & Std & Median & Q1 & Q3 & Safety \% & $N$ \\")
        lines.append(r"\hline")
        for (name, mu, sd, med, q1, q3, sr, n) in rows:
            sr_str = f"{sr:.1f}" if np.isfinite(sr) else "--"
            if n == 0:
                lines.append(fr"{name} & -- & -- & -- & -- & -- & {sr_str} & 0 \\")
            else:
                lines.append(fr"{name} & {mu:.4g} & {sd:.4g} & {med:.4g} & {q1:.4g} & {q3:.4g} & {sr_str} & {n:d} \\")
    else:
        lines.append(r"\begin{tabular}{lrrrrrr}")
        lines.append(r"\hline")
        lines.append(r"Method & Mean & Std & Median & Q1 & Q3 & $N$ \\")
        lines.append(r"\hline")
        for (name, mu, sd, med, q1, q3, _sr, n) in rows:
            if n == 0:
                lines.append(fr"{name} & -- & -- & -- & -- & -- & 0 \\")
            else:
                lines.append(fr"{name} & {mu:.4g} & {sd:.4g} & {med:.4g} & {q1:.4g} & {q3:.4g} & {n:d} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# -----------------------------
# Main public function
# -----------------------------
def plot_cruisecontrol_threeway(
    baseline_mat: str,
    mlp_mat: str,
    rnn_mat: str,
    model_names: Sequence[str] = ("ICCBF", "MLP-tuned ICCBF", "RNN-tuned ICCBF"),
    **kwargs,
) -> Dict[str, Any]:
    """Legacy 3-model wrapper. Delegates to :func:`plot_cruisecontrol`."""
    return plot_cruisecontrol(
        mat_paths=[baseline_mat, mlp_mat, rnn_mat],
        model_names=model_names,
        **kwargs,
    )


def plot_cruisecontrol(
    mat_paths: Sequence[str],
    model_names: Optional[Sequence[str]] = None,
    stride_traj: int = 10,
    stride_ts: int = 2,
    sampling_mode: str = "linspace",  # "stride" or "linspace"
    Nsamp_ts: int = 160,
    Nsucc_plot: int = 1000,
    Nfail_plot: int = 60,
    seed: int = 1,
    linex_max: float = 160.0,
    v_max_plot: float = 30.0,
    h_ylim: Tuple[float, float] = (0.0, 150.0),
    V_ylim: Tuple[float, float] = (0.0, 700.0),
    cmap_name: str = "turbo",
    save_prefix: Optional[str] = None,
    skip_viability_filter: bool = False,
) -> Dict[str, Any]:
    """
    Create the MATLAB-style cruise-control figure + violin plot + LaTeX table
    for an arbitrary number of models.

    Parameters
    ----------
    mat_paths : sequence of str
        Paths to .mat result files.  The first is treated as the baseline
        (used for viability filtering).
    model_names : sequence of str, optional
        Display names for each model.  Defaults to ``("Model 0", "Model 1", ...)``.
    skip_viability_filter : bool, optional
        If True, skip the max-brake open-loop viability filter and treat all
        episodes as viable.  Use this when impossible ICs have already been
        removed from the episode bank (default: False).

    Returns
    -------
    out : dict
      Keys include:
        - "models_sliced": list of dicts after viability filtering
        - "viable_idx": indices kept
        - "uTotals_full": (n_models, n_viable) array of total thrust
        - "latex_table": str
        - "fig_main": matplotlib Figure (3 x n_models)
        - "fig_violin": matplotlib Figure (violin)
    """
    n_models = len(mat_paths)
    if model_names is None:
        model_names = tuple(f"Model {i}" for i in range(n_models))
    assert len(model_names) == n_models, "model_names length must match mat_paths"

    models = [load_mat(p) for p in mat_paths]

    # --- viability filtering based on baseline ordering ---
    if "states" not in models[0]:
        raise KeyError("Baseline .mat must contain 'states' for viability filtering.")

    states0 = np.asarray(models[0]["states"])
    if states0.ndim != 3 or states0.shape[-1] < 2:
        raise ValueError(f"Expected baseline states shape (N,Nt,2+). Got {states0.shape}")

    nEpisodes = states0.shape[0]

    if skip_viability_filter:
        idx_keep = np.arange(nEpisodes)
        models_sliced = list(models)
        nEp_viable = nEpisodes
        print(f"Viability filter skipped. Using all {nEpisodes} episodes.")
    else:
        viable_mask = np.zeros((nEpisodes,), dtype=bool)
        for i in range(nEpisodes):
            x0 = states0[i, 0, 0:2]  # [d, v]
            viable_mask[i] = simulate_max_brake(x0)

        idx_keep = np.where(viable_mask)[0]
        models_sliced = [slice_model_by_episode(M, idx_keep, nEpisodes) for M in models]
        nEp_viable = int(np.asarray(models_sliced[0]["states"]).shape[0])

        print(f"Total episodes: {nEpisodes}, viable episodes: {idx_keep.size}")
        print(f"Viable episodes AFTER slicing: {nEp_viable}")

    # --- per-model preprocessing and global colour scaling (success-based robust percentiles) ---
    cmap = plt.get_cmap(cmap_name, 256)
    all_u_succ = []

    per = []
    for j, M in enumerate(models_sliced):
        X = np.asarray(M["states"])
        Ntraj, Nt, _ = X.shape
        t, dt = _get_tvec(M, Nt)

        Hraw = _as_2d_h_or_v(M.get("hs", None))
        Vraw = _as_2d_h_or_v(M.get("Vs", None))

        # failure detection via first time h < 0
        isFail = np.zeros((Ntraj,), dtype=bool)
        kFail = np.full((Ntraj,), Nt, dtype=int)
        if Hraw is not None:
            for i in range(Ntraj):
                neg = np.where(Hraw[i, :] < 0.0)[0]
                if neg.size > 0:
                    isFail[i] = True
                    kFail[i] = int(neg[0] + 1)  # MATLAB-like count

        idxFail = np.where(isFail)[0]
        idxSucc = np.where(~isFail)[0]

        rng = np.random.default_rng(seed + 100 * (j + 1))
        # choose plotted subset
        nFail_full = idxFail.size
        nSucc_full = idxSucc.size

        if nFail_full > 0:
            idxFail_plot = rng.choice(idxFail, size=min(Nfail_plot, nFail_full), replace=False)
        else:
            idxFail_plot = np.array([], dtype=int)

        if nSucc_full > 0:
            idxSucc_plot = rng.choice(idxSucc, size=min(Nsucc_plot, nSucc_full), replace=False)
        else:
            idxSucc_plot = np.array([], dtype=int)

        idxPlot = np.concatenate([idxSucc_plot, idxFail_plot]).astype(int)

        # kStop for all trajectories
        kStop = np.zeros((Ntraj,), dtype=int)
        for i in range(Ntraj):
            kStop[i] = compute_kstop(M, i, Nt, int(kFail[i]), bool(isFail[i]))

        # Utot for plotted
        Utot_plot = np.zeros((idxPlot.size,), dtype=float)
        for jj, ii in enumerate(idxPlot):
            Utot_plot[jj] = compute_total_thrust(M, int(ii), int(kStop[ii]), float(dt))

        # collect successes for global scale
        succ_mask_plot = ~isFail[idxPlot]
        all_u_succ.append(Utot_plot[succ_mask_plot])

        per.append(dict(
            M=M, X=X, t=t, dt=dt, Hraw=Hraw, Vraw=Vraw,
            isFail=isFail, kFail=kFail, kStop=kStop,
            idxPlot=idxPlot, Utot_plot=Utot_plot
        ))

    all_u_succ = np.concatenate([x for x in all_u_succ if x.size > 0], axis=0) if any(x.size > 0 for x in all_u_succ) else np.array([])
    if all_u_succ.size == 0:
        # fallback: use all plotted
        all_u = np.concatenate([p["Utot_plot"] for p in per if p["Utot_plot"].size > 0], axis=0)
        all_u_succ = all_u

    umin = float(np.percentile(all_u_succ, 5))
    umax = float(np.percentile(all_u_succ, 95))
    if abs(umax - umin) < 1e-12:
        umax = umin + 1.0

    def color_of(uval: float) -> Tuple[float, float, float, float]:
        u = min(max(uval, umin), umax)
        alpha = (u - umin) / (umax - umin)
        idx = int(np.floor(alpha * (cmap.N - 1)))
        return cmap(idx)

    # reference constraint line: v = x/1.8
    linex = np.linspace(0.0, linex_max, 200)
    liney = linex / 1.8

    # --------------------------------
    # Paper-quality rcParams
    # --------------------------------
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 12,
        "lines.linewidth": 0.7,
    })

    # Main 3 x n_models figure
    fig_main = plt.figure(figsize=(5.5 * n_models, 12), dpi=300, facecolor="white")
    axes = np.empty((3, n_models), dtype=object)

    for j in range(n_models):
        ax1 = fig_main.add_subplot(3, n_models, j + 1)
        ax2 = fig_main.add_subplot(3, n_models, j + 1 + n_models)
        ax3 = fig_main.add_subplot(3, n_models, j + 1 + 2 * n_models)
        axes[:, j] = [ax1, ax2, ax3]

        P = per[j]
        M = P["M"]
        X = P["X"]
        t = P["t"]
        Hraw = P["Hraw"]
        Vraw = P["Vraw"]
        isFail = P["isFail"]
        kStop = P["kStop"]
        kFail = P["kFail"]
        idxPlot = P["idxPlot"]
        Utot_plot = P["Utot_plot"]

        Ntraj, Nt, _ = X.shape

        # ---- Row 1: (x,v) trajectories ----
        ax1.grid(True, alpha=0.25)
        ax1.plot(linex, liney, "k--", linewidth=1.0)
        tidx_traj_all = np.arange(0, Nt, stride_traj, dtype=int)

        for jj, ii in enumerate(idxPlot):
            k_end = int(kStop[ii])
            tidx = tidx_traj_all[tidx_traj_all < k_end]
            if tidx.size < 2:
                continue
            col = color_of(float(Utot_plot[jj]))
            ls = "--" if bool(isFail[ii]) else "-"
            lw = 0.9 if bool(isFail[ii]) else 0.7
            ax1.plot(X[ii, tidx, 0], X[ii, tidx, 1], linestyle=ls, linewidth=lw, color=col)

        ax1.set_title(str(model_names[j]))
        ax1.set_xlabel("x [m]")
        if j == 0:
            ax1.set_ylabel("v [m/s]")
        ax1.set_xlim(0, linex_max)
        ax1.set_ylim(0, v_max_plot)

        # ---- Row 2: h(t) ----
        ax2.grid(True, alpha=0.25)
        if Hraw is None:
            ax2.axis("off")
            ax2.text(0.1, 0.6, "h(t) not available", transform=ax2.transAxes)
        else:
            for jj, ii in enumerate(idxPlot):
                k_end = int(min(kStop[ii], kFail[ii]))
                tidx = get_tidx_ts(k_end, stride_ts, sampling_mode, Nsamp_ts)
                if tidx.size < 2:
                    continue
                col = color_of(float(Utot_plot[jj]))
                ls = "--" if bool(isFail[ii]) else "-"
                lw = 0.9 if bool(isFail[ii]) else 0.7
                ax2.plot(t[tidx], Hraw[ii, tidx], linestyle=ls, linewidth=lw, color=col)

                if bool(isFail[ii]):
                    kf = int(kFail[ii])
                    if 1 <= kf <= k_end:
                        ax2.plot(t[kf - 1], Hraw[ii, kf - 1], "o", markersize=3, color=col, markerfacecolor=col)

            ax2.axhline(0.0, color="k", linestyle="--", linewidth=0.9)
            if j == 0:
                ax2.set_ylabel("h(t)")
            ax2.set_ylim(*h_ylim)

        # ---- Row 3: V(t) ----
        ax3.grid(True, alpha=0.25)
        if Vraw is None:
            ax3.axis("off")
            ax3.text(0.1, 0.6, "V(t) not available", transform=ax3.transAxes)
        else:
            for jj, ii in enumerate(idxPlot):
                k_end = int(kStop[ii])
                tidx = get_tidx_ts(k_end, stride_ts, sampling_mode, Nsamp_ts)
                if tidx.size < 2:
                    continue
                col = color_of(float(Utot_plot[jj]))
                ls = "--" if bool(isFail[ii]) else "-"
                lw = 0.9 if bool(isFail[ii]) else 0.7
                ax3.plot(t[tidx], Vraw[ii, tidx], linestyle=ls, linewidth=lw, color=col)

            if j == 0:
                ax3.set_ylabel("V(t)")
            ax3.set_xlabel("t [s]")
            ax3.set_ylim(*V_ylim)

        # consistent colormap scale (for the colourbar host)
        for a in (ax1, ax2, ax3):
            a.set_facecolor("white")

    # shared horizontal colourbar (like MATLAB)
    cax = fig_main.add_axes([0.25, 0.05, 0.50, 0.02])
    norm = plt.Normalize(vmin=umin, vmax=umax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(r"Total thrust  $\int \|u\|\,dt$")

    fig_main.tight_layout(rect=[0, 0.06, 1, 1])

    if save_prefix:
        fig_main.savefig(f"{save_prefix}_main.png", dpi=300, bbox_inches="tight")

    # -----------------------------
    # Violin plot + LaTeX table over ALL viable episodes
    # -----------------------------
    uTotals_full = np.full((n_models, nEp_viable), np.nan, dtype=float)
    safety_rates: List[float] = []

    for j in range(n_models):
        M = models_sliced[j]
        X = np.asarray(M["states"])
        Ntraj, Nt, _ = X.shape
        t, dt = _get_tvec(M, Nt)

        Hraw = _as_2d_h_or_v(M.get("hs", None))
        isFail = np.zeros((Ntraj,), dtype=bool)
        kFail = np.full((Ntraj,), Nt, dtype=int)
        if Hraw is not None:
            for i in range(Ntraj):
                neg = np.where(Hraw[i, :] < 0.0)[0]
                if neg.size > 0:
                    isFail[i] = True
                    kFail[i] = int(neg[0] + 1)

        n_safe = int(np.sum(~isFail))
        safety_rates.append(100.0 * n_safe / Ntraj if Ntraj > 0 else 0.0)
        print(f"  [{model_names[j]}] N_safe={n_safe}, N_MC={Ntraj}, safety={safety_rates[-1]:.1f}%")

        kStop = np.zeros((Ntraj,), dtype=int)
        for i in range(Ntraj):
            kStop[i] = compute_kstop(M, i, Nt, int(kFail[i]), bool(isFail[i]))

        u_all = np.zeros((Ntraj,), dtype=float)
        for i in range(Ntraj):
            u_all[i] = compute_total_thrust(M, i, int(kStop[i]), float(dt))

        uTotals_full[j, :Ntraj] = u_all

    fig_violin = plt.figure(figsize=(max(4, 2.5 * n_models), 3.5), dpi=300, facecolor="white")
    axv = fig_violin.add_subplot(1, 1, 1)
    axv.set_facecolor("white")
    axv.grid(True, alpha=0.25)

    data = [uTotals_full[j, np.isfinite(uTotals_full[j, :])] for j in range(n_models)]
    parts = axv.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    axv.set_xticks(range(1, n_models + 1))
    axv.set_xticklabels(list(model_names))
    axv.set_ylabel("Total thrust  ∫||u|| dt")

    fig_violin.tight_layout()

    if save_prefix:
        fig_violin.savefig(f"{save_prefix}_violin.png", dpi=300, bbox_inches="tight")

    latex_table = thrust_totals_table_latex(uTotals_full, model_names, safety_rates=safety_rates)

    return dict(
        models_sliced=models_sliced,
        viable_idx=idx_keep,
        uTotals_full=uTotals_full,
        safety_rates=safety_rates,
        latex_table=latex_table,
        fig_main=fig_main,
        fig_violin=fig_violin,
        colour_scale=(umin, umax),
    )
