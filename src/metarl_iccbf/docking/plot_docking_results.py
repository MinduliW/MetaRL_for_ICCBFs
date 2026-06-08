"""plot_docking_results.py

Docking plotting utilities (Python port of your MATLAB rotating-docking visualisation).

Usage
-----
from plot_docking_results import plot_docking_comparison

out = plot_docking_comparison(
    mat_paths=[
        "Docking_noRL.mat",
        "Docking_NNfinal.mat",
        "Docking_RNN.mat",
        "Docking_Mamba2.mat",
    ],
    model_names=["ICCBF", "MLP-tuned", "RNN-tuned", "Mamba2-tuned"],
    save_prefix="figs/docking",
)

print(out["latex_table"])

What it does
------------
Given N .mat result files, this module:
  1) Loads each .mat (scipy.io.loadmat).
  2) Computes success/failure from `hs` (barrier violation = hs < 0).
  3) Builds an Nx3 figure:
       Row 1: XY trajectories with rotating cone overlay (start + end).
       Row 2: h(t) (cropped; y=0 reference)
       Row 3: V(t) if present (cropped)
  4) Builds a violin/box-style plot (matplotlib) of total ΔV [m/s] for all episodes.
  5) Returns a dict with arrays + a LaTeX table string.

Expected .mat fields (flexible)
-------------------------------
Required:
  - states: (N, Nt, nx)

Optional (used if available):
  - hs: (N, Nt) or (N, Nt, 1)   (barrier value, success if never < 0)
  - Vs: (N, Nt) or (N, Nt, 1)   (Lyapunov)
  - uOpts / actions / u_safe / u_rl / actionStore: control history
  - tvec_full: (Nt,)
  - steps_taken: (N,)  (used for truncation; else infer last-active)
  - dones: (N, Nt)  (fallback failure marking)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_mat(path: str) -> Dict[str, Any]:
    """Load a MATLAB .mat as a flat dict; drop MATLAB metadata keys."""
    D = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in D.items() if not k.startswith("__")}


def _as_2d(arr: Any) -> Optional[np.ndarray]:
    """Convert hs/Vs/etc to shape (N, Nt) when possible."""
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


def _get_control_array(M: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return control array as (N, Nt, du) if possible."""
    for key in ("uOpts", "actions", "u_safe", "u_rl", "actionStore"):
        if key in M and M[key] is not None:
            U = np.asarray(M[key])
            if U.ndim == 2:
                return U[:, :, None]
            if U.ndim == 3:
                return U
    return None


def _infer_failures(M: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Infer failure episodes and first failure indices.

    Returns:
      is_fail: (N,) bool
      k_fail:  (N,) first index (1..Nt) where failure occurs; Nt if never fails

    Only uses ``hs < 0`` (barrier violation) as a failure signal.  The
    ``dones`` array records episode *termination* (success or failure)
    and is therefore not a reliable failure indicator.
    """
    X = np.asarray(M["states"])
    N, Nt = X.shape[0], X.shape[1]

    H = _as_2d(M.get("hs", None))
    if H is not None and H.shape[0] == N and H.shape[1] == Nt:
        is_fail = np.any(H < 0.0, axis=1)
        k_fail = np.full(N, Nt, dtype=int)
        for i in range(N):
            idx = np.where(H[i, :] < 0.0)[0]
            if idx.size:
                k_fail[i] = int(idx[0] + 1)
        return is_fail, k_fail

    return np.zeros(N, dtype=bool), np.full(N, Nt, dtype=int)


def _infer_docking_success(M: Dict[str, Any], is_fail: np.ndarray, lenvec: int) -> np.ndarray:
    """Return is_docked bool array (True = successfully docked).

    Uses the ``docked`` field saved by eval_parallel if present.
    Falls back to inferring from early termination: steps_taken < lenvec
    AND no barrier violation (is_fail == False).
    """
    if "docked" in M:
        return np.asarray(M["docked"]).reshape(-1).astype(bool)
    if "steps_taken" in M:
        st = np.asarray(M["steps_taken"]).reshape(-1)
        return (st < lenvec) & (~is_fail)
    return ~is_fail  # fallback


def _compute_kstop(M: Dict[str, Any], i: int, Nt: int, is_fail_i: bool, k_fail_i: int) -> int:
    """Plot stop index (1..Nt)."""
    eps0 = 1e-10

    if "steps_taken" in M and np.size(M["steps_taken"]) >= (i + 1):
        k_stop = int(np.asarray(M["steps_taken"]).reshape(-1)[i])
        k_stop = max(2, min(Nt, k_stop))
    else:
        k_stop = Nt

        X = np.asarray(M["states"])[i, :, :]
        maskX = np.any(np.abs(X) > eps0, axis=1)
        if np.any(maskX):
            k_stop = int(np.where(maskX)[0][-1] + 1)

        H = _as_2d(M.get("hs", None))
        if H is not None and H.shape[1] == Nt:
            maskH = np.abs(H[i, :]) > eps0
            if np.any(maskH):
                k_stop = max(k_stop, int(np.where(maskH)[0][-1] + 1))

        V = _as_2d(M.get("Vs", None))
        if V is not None and V.shape[1] == Nt:
            maskV = np.abs(V[i, :]) > eps0
            if np.any(maskV):
                k_stop = max(k_stop, int(np.where(maskV)[0][-1] + 1))

        U = _get_control_array(M)
        if U is not None and U.shape[1] == Nt:
            maskU = np.any(np.abs(U[i, :, :]) > eps0, axis=1)
            if np.any(maskU):
                k_stop = max(k_stop, int(np.where(maskU)[0][-1] + 1))

        k_stop = max(2, min(Nt, k_stop))

    if is_fail_i:
        k_stop = min(k_stop, max(2, min(Nt, int(k_fail_i))))

    return int(k_stop)


def _total_thrust(M: Dict[str, Any], i: int, k_stop: int, dt: float, m: float = 1.0) -> float:
    """Return total ΔV [m/s] for episode *i*.

    Dynamics use km-based units (mu in km³/s²), so control/m gives km/s².
    Multiplying by 1000 converts the final km/s result to m/s.

    Prefers the precomputed ``uTotal`` field (present in all eval .mat
    files) over recomputing from the control array, which may contain
    RL actions rather than physical controls.
    """
    if "uTotal" in M:
        ut = np.asarray(M["uTotal"]).reshape(-1)
        if ut.shape[0] > i:
            return float(ut[i]) / m * 1000.0
    U = _get_control_array(M)
    if U is None:
        return float("nan")
    Ui = U[i, :k_stop, :]
    return float(np.sum(np.linalg.norm(Ui, axis=1)) * dt) / m * 1000.0


def _plot_cones(ax: plt.Axes, R: float, alpha0: float, alphaT: float, half_angle: float) -> None:
    th = np.linspace(-half_angle, half_angle, 200)

    # Start cone
    ax.plot([0, R * np.cos(alpha0 - half_angle)], [0, R * np.sin(alpha0 - half_angle)], linestyle="--", linewidth=1.0)
    ax.plot([0, R * np.cos(alpha0 + half_angle)], [0, R * np.sin(alpha0 + half_angle)], linestyle="--", linewidth=1.0)
    ax.plot(R * np.cos(alpha0 + th), R * np.sin(alpha0 + th), linestyle=":", linewidth=1.0)

    # End cone
    ax.plot([0, R * np.cos(alphaT - half_angle)], [0, R * np.sin(alphaT - half_angle)], linestyle="--", linewidth=1.0)
    ax.plot([0, R * np.cos(alphaT + half_angle)], [0, R * np.sin(alphaT + half_angle)], linestyle="--", linewidth=1.0)
    ax.plot(R * np.cos(alphaT + th), R * np.sin(alphaT + th), linestyle=":", linewidth=1.0)


def _violin_box(ax: plt.Axes, data: List[np.ndarray], labels: Sequence[str]) -> None:
    clean = [d[~np.isnan(d)] for d in data]
    positions = np.arange(1, len(labels) + 1)
    # violinplot requires at least one data point per dataset; skip empty ones
    nonempty_data = [c for c in clean if c.size > 0]
    nonempty_pos = [p for c, p in zip(clean, positions) if c.size > 0]
    if nonempty_data:
        ax.violinplot(nonempty_data, positions=nonempty_pos,
                      showmeans=False, showmedians=True, showextrema=False)
    # boxplot also needs non-empty data per position; replace empty with NaN placeholder
    box_data = [c if c.size > 0 else np.array([float("nan")]) for c in clean]
    ax.boxplot(box_data, positions=positions, widths=0.2, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def _latex_table_from_arrays(
    u_totals: List[np.ndarray],
    names: Sequence[str],
    safety_rates: Optional[Sequence[float]] = None,
    dock_rates: Optional[Sequence[float]] = None,
    u_totals_docked: Optional[List[np.ndarray]] = None,
) -> str:
    rows = []
    for j, (name, u) in enumerate(zip(names, u_totals)):
        v = u[~np.isnan(u)]
        if v.size == 0:
            mu = sd = med = q25 = q75 = float("nan")
        else:
            mu = float(np.mean(v))
            sd = float(np.std(v))
            med = float(np.median(v))
            q25, q75 = np.percentile(v, [25, 75]).astype(float)
        sr = float(safety_rates[j]) if safety_rates is not None else float("nan")
        dr = float(dock_rates[j]) if dock_rates is not None else float("nan")
        # Mean fuel conditioned on successful docking
        if u_totals_docked is not None:
            ud = u_totals_docked[j]
            ud = ud[~np.isnan(ud)]
            mu_docked = float(np.mean(ud)) if ud.size > 0 else float("nan")
        else:
            mu_docked = float("nan")
        rows.append((name, mu, sd, med, q25, q75, sr, dr, mu_docked))

    has_safety = safety_rates is not None
    has_dock = dock_rates is not None
    has_docked_fuel = u_totals_docked is not None

    # Build column spec and header dynamically
    col_spec = "lccccc"
    header = r"Method & Mean & Std & Median & 25\% & 75\%"
    if has_safety:
        col_spec += "c"
        header += r" & Safety \%"
    if has_dock:
        col_spec += "c"
        header += r" & Dock \%"
    if has_docked_fuel:
        col_spec += "c"
        header += r" & $\Delta V$\,|\,Docked [m/s]"
    header += r" \\"

    lines = [
        fr"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for (name, mu, sd, med, q25, q75, sr, dr, mu_d) in rows:
        row = fr"{name} & {mu:.3f} & {sd:.3f} & {med:.3f} & {q25:.3f} & {q75:.3f}"
        if has_safety:
            row += f" & {sr:.1f}" if np.isfinite(sr) else " & --"
        if has_dock:
            row += f" & {dr:.1f}" if np.isfinite(dr) else " & --"
        if has_docked_fuel:
            row += f" & {mu_d:.3f}" if np.isfinite(mu_d) else " & --"
        row += r" \\"
        lines.append(row)
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def plot_docking_comparison(
    mat_paths: Sequence[str],
    model_names: Optional[Sequence[str]] = None,
    save_prefix: Optional[str] = None,
    stride_traj: int = 5,
    stride_ts: int = 1,
    Nsucc_plot: int = 1000,
    Nfail_plot: int = 40,
    seed: int = 1,
    TOF: float = 50.0,
    omega_deg: float = 0.6,
    cone_half_deg: float = 10.0,
    alpha0_deg: float = 0.0,
) -> Dict[str, Any]:
    """N-way docking comparison (grid + violin + latex table).

    Parameters
    ----------
    mat_paths : sequence of str
        Paths to .mat result files (any number >= 1).
    model_names : sequence of str or None
        Display names; defaults to ``["Model 0", "Model 1", ...]``.
    """
    n_models = len(mat_paths)
    if model_names is None:
        model_names = [f"Model {i}" for i in range(n_models)]
    assert len(model_names) == n_models

    Ms = [load_mat(p) for p in mat_paths]

    # Reference episode count from first file
    Nref = np.asarray(Ms[0]["states"]).shape[0]

    # Per-model time vectors and Nt (may differ, e.g. 101 vs 100)
    tvecs: List[np.ndarray] = []
    dts: List[float] = []
    Nts: List[int] = []
    for M in Ms:
        Nt_m = np.asarray(M["states"]).shape[1]
        Nts.append(Nt_m)
        t_m, dt_m = _get_tvec(M, Nt_m)
        tvecs.append(t_m)
        dts.append(dt_m)

    half_angle = np.deg2rad(cone_half_deg)
    alpha0 = np.deg2rad(alpha0_deg)
    alphaT = np.deg2rad(alpha0_deg + omega_deg * TOF)

    # Cone radius based on all methods so geometry matches
    rmax = 0.0
    for M in Ms:
        X = np.asarray(M["states"])
        r = np.sqrt(X[:, :, 0] ** 2 + X[:, :, 1] ** 2)
        rmax = max(rmax, float(np.max(r)))
    cone_R = 1.05 * rmax if rmax > 0 else 1.0

    is_fail_list, k_fail_list, k_stop_list, u_totals_list = [], [], [], []
    is_docked_list = []
    safety_rates = []
    dock_rates = []
    u_totals_docked_list = []

    for idx_m, M in enumerate(Ms):
        X = np.asarray(M["states"])
        N_m, Nt_m = X.shape[0], X.shape[1]
        if N_m != Nref:
            raise ValueError(
                f"Model {idx_m} has N={N_m} episodes, expected {Nref}."
            )

        is_fail, k_fail = _infer_failures(M)
        is_docked = _infer_docking_success(M, is_fail, Nts[idx_m])
        k_stop = np.array([
            _compute_kstop(M, i, Nt_m, bool(is_fail[i]), int(k_fail[i]))
            for i in range(Nref)
        ], dtype=int)
        m_arr = np.asarray(M.get("m_vec", np.ones(Nref)), dtype=float).reshape(-1)
        if m_arr.size != Nref:
            m_arr = np.ones(Nref)
        u_totals = np.array([
            _total_thrust(M, i, int(k_stop[i]), dts[idx_m], m=float(m_arr[i]))
            for i in range(Nref)
        ], dtype=float)

        # Fuel conditioned on successful docking (NaN for non-docked episodes)
        u_docked = np.where(is_docked, u_totals, float("nan"))

        is_fail_list.append(is_fail)
        k_fail_list.append(k_fail)
        k_stop_list.append(k_stop)
        u_totals_list.append(u_totals)
        is_docked_list.append(is_docked)
        safety_rates.append(float(np.mean(~is_fail)))
        dock_rates.append(float(np.mean(is_docked)))
        u_totals_docked_list.append(u_docked)

    # Global colour scale: success-only across all methods
    succ_u = []
    for u, f in zip(u_totals_list, is_fail_list):
        if np.any(~f):
            succ_u.append(u[~f])
    succ_u_all = np.concatenate(succ_u) if succ_u else np.array([])

    if succ_u_all.size == 0:
        umin, umax = 0.0, 1.0
    else:
        umin = float(np.nanpercentile(succ_u_all, 5))
        umax = float(np.nanpercentile(succ_u_all, 95))
        if not np.isfinite(umin):
            umin = 0.0
        if not np.isfinite(umax) or umax <= umin + 1e-12:
            umax = umin + 1.0

    cmap = plt.get_cmap("turbo")

    def color_of(uval: float):
        uclamp = min(max(float(uval), umin), umax)
        a = (uclamp - umin) / (umax - umin) if umax > umin else 0.5
        return cmap(a)

    rng = np.random.default_rng(seed)

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

    fig = plt.figure(figsize=(5.5 * n_models, 9), dpi=300)
    fig.patch.set_facecolor("white")

    for col, (M, name, is_fail, k_fail, k_stop, u_totals) in enumerate(
        zip(Ms, model_names, is_fail_list, k_fail_list, k_stop_list, u_totals_list),
        start=1,
    ):
        Nt_m = Nts[col - 1]
        t = tvecs[col - 1]

        idx_succ = np.where(~is_fail)[0]
        idx_fail = np.where(is_fail)[0]
        idx_s = (
            rng.choice(idx_succ, size=min(Nsucc_plot, idx_succ.size), replace=False)
            if idx_succ.size else np.array([], dtype=int)
        )
        idx_f = (
            rng.choice(idx_fail, size=min(Nfail_plot, idx_fail.size), replace=False)
            if idx_fail.size else np.array([], dtype=int)
        )
        idx_plot = np.concatenate([idx_s, idx_f])

        X = np.asarray(M["states"])
        H = _as_2d(M.get("hs", None))
        V = _as_2d(M.get("Vs", None))

        # Row 1: XY
        ax1 = fig.add_subplot(3, n_models, col)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor("white")

        for ii in idx_plot:
            k_end_traj = int(k_fail[ii]) if is_fail[ii] else Nt_m
            tidx = np.arange(0, k_end_traj, stride_traj, dtype=int)
            if tidx.size < 2:
                continue
            ax1.plot(
                X[ii, tidx, 0],
                X[ii, tidx, 1],
                linestyle="--" if is_fail[ii] else "-",
                linewidth=0.9 if is_fail[ii] else 0.6,
                color=color_of(u_totals[ii]),
            )

        _plot_cones(ax1, cone_R, alpha0, alphaT, half_angle)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title(name)
        ax1.set_xlabel("x [km]")
        if col == 1:
            ax1.set_ylabel("y [km]")


        # Row 2: h(t)
        ax2 = fig.add_subplot(3, n_models, col + n_models)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("white")
        if H is None:
            ax2.axis("off")
            ax2.text(0.1, 0.6, "h(t) not available", transform=ax2.transAxes)
        else:
            vals = []
            for ii in idx_plot:
                k_end = int(k_stop[ii])
                tidx = np.arange(0, k_end, stride_ts, dtype=int)
                if tidx.size >= 2:
                    vals.append(H[ii, tidx])
            vals = np.concatenate(vals) if vals else np.array([0.0])
            ylo, yhi = np.percentile(vals, [1, 99]).astype(float)
            pad = 0.08 * (yhi - ylo + 1e-12)
            ylo = min(ylo - pad, -0.1)
            yhi = max(yhi + pad, 0.1)

            for ii in idx_plot:
                k_end = int(k_stop[ii])
                tidx = np.arange(0, k_end, stride_ts, dtype=int)
                if tidx.size < 2:
                    continue
                tt = t[tidx]
                hseg = H[ii, tidx]
                ax2.plot(
                    tt, np.maximum(hseg, ylo),
                    linewidth=0.9 if is_fail[ii] else 0.6,
                    color=color_of(u_totals[ii]),
                )

            ax2.axhline(0.0, linestyle="--", linewidth=0.8)
            ax2.set_ylim([ylo, yhi])
            ax2.set_ylabel("h(t)" if col == 1 else "")

        # Row 3: V(t)
        ax3 = fig.add_subplot(3, n_models, col + 2 * n_models)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor("white")
        if V is None:
            ax3.axis("off")
            ax3.text(0.1, 0.6, "V(t) not available", transform=ax3.transAxes)
        else:
            for ii in idx_plot:
                k_end = int(k_stop[ii])
                tidx = np.arange(0, k_end, stride_ts, dtype=int)
                if tidx.size < 2:
                    continue
                tt = t[tidx]
                ax3.plot(
                    tt, V[ii, tidx],
                    linewidth=0.9 if is_fail[ii] else 0.6,
                    color=color_of(u_totals[ii]),
                )
            ax3.set_ylabel("V(t)" if col == 1 else "")
            ax3.set_xlabel("t [s]")

    fig.subplots_adjust(hspace=0.75)

    # Global colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=umin, vmax=umax))
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$\Delta V$ [m/s]")

    # Violin plot: all episodes (top) + docked episodes only (bottom)
    fig_v = plt.figure(figsize=(max(4, 2.5 * n_models), 5), dpi=300)
    fig_v.patch.set_facecolor("white")
    axv1 = fig_v.add_subplot(2, 1, 1)
    axv1.set_facecolor("white")
    axv1.grid(True, axis="y", alpha=0.3)
    _violin_box(axv1, u_totals_list, model_names)
    axv1.set_ylabel(r"$\Delta V$ [m/s]")
    axv1.set_title(r"$\Delta V$ [m/s] — all episodes")

    axv2 = fig_v.add_subplot(2, 1, 2)
    axv2.set_facecolor("white")
    axv2.grid(True, axis="y", alpha=0.3)
    _violin_box(axv2, u_totals_docked_list, model_names)
    axv2.set_ylabel(r"$\Delta V$ [m/s]")
    axv2.set_title(r"$\Delta V$ [m/s] — docked episodes only")
    fig_v.tight_layout()

    safety_rates_pct = [sr * 100.0 for sr in safety_rates]
    dock_rates_pct = [dr * 100.0 for dr in dock_rates]
    latex_table = _latex_table_from_arrays(
        u_totals_list,
        model_names,
        safety_rates=safety_rates_pct,
        dock_rates=dock_rates_pct,
        u_totals_docked=u_totals_docked_list,
    )

    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_grid.png", dpi=300, bbox_inches="tight")
        fig_v.savefig(f"{save_prefix}_violin.png", dpi=300, bbox_inches="tight")

    return {
        "latex_table": latex_table,
        "u_totals": u_totals_list,
        "u_totals_docked": u_totals_docked_list,
        "safety_rates": safety_rates_pct,   # barrier h(t) >= 0 throughout, % in [0, 100]
        "dock_rates": dock_rates_pct,        # task completion (V <= threshold at term), % in [0, 100]
        "colour_scale": (umin, umax),
        "figs": {"main": fig, "violin": fig_v},
    }


def plot_docking_threeway(
    baseline_mat: str,
    mlp_mat: str,
    rnn_mat: str,
    save_prefix: Optional[str] = None,
    model_names: Tuple[str, str, str] = ("ICCBF", "MLP-tuned ICCBF", "RNN-tuned ICCBF"),
    **kwargs,
) -> Dict[str, Any]:
    """Legacy 3-way wrapper around :func:`plot_docking_comparison`."""
    return plot_docking_comparison(
        mat_paths=[baseline_mat, mlp_mat, rnn_mat],
        model_names=list(model_names),
        save_prefix=save_prefix,
        **kwargs,
    )