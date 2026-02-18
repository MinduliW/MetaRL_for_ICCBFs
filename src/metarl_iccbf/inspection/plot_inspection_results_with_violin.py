"""
inspection/plot_inspection_results.py

Python analogue of your MATLAB inspection visualisation script:
- 5x3 main figure (columns = methods):
  Row 1: XY trajectory spaghetti + KIZ annulus (R_MAX_nominal ± 10% by default)
  Row 2: h_KOZ(t) = ||r|| - (R_C + R_D)
  Row 3: h_KIZ(t) = R_MAX - ||r||
  Row 4: h_SUN(t) if available in .mat (>=0 safe)
  Row 5: Inspected [%] over time if available (target = 100)

Colouring: per-episode total thrust integral (task-success-only global scale), log-mapped.
Cropping: steps_taken, and optionally first time any h < 0 (if h_sun is available; KOZ/KIZ computed).
Task success: inspected reaches 100 at end (kStop).

Also outputs LaTeX tables:
- Total thrust across ALL episodes (mean±std, percentiles)
- Final inspected points across ALL episodes

Public API:
    from inspection.plot_inspection_results import plot_inspection_threeway
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.io import loadmat

# ----------------------------- helpers -----------------------------

def _total_dv(U: np.ndarray, kStop: np.ndarray, dt: float, m_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """
    If m_vec is provided, interpret U as FORCE [N] and return Δv = ∫ ||u||/m dt.
    If m_vec is None, interpret U as ACCEL [m/s^2] and return Δv = ∫ ||u|| dt.

    U:     (N,T,nu)
    kStop: (N,) 1..T
    m_vec: (N,) or None
    """
    N = U.shape[0]
    out = np.zeros((N,), dtype=float)
    for i in range(N):
        k = int(kStop[i])
        seg = U[i, :k, :]
        mag = np.linalg.norm(seg, axis=1)
        if m_vec is not None:
            out[i] = float(np.sum(mag / float(m_vec[i])) * dt)
        else:
            out[i] = float(np.sum(mag) * dt)
    return out


def _to_str_list(x) -> List[str]:
    """
    Robust conversion of MATLAB-loaded string arrays to a Python list[str].

    Handles:
      - object arrays like (1,P) where each entry is a 1-element ndarray(['R_C'])
      - MATLAB char matrices (n, m) of single characters
      - normal numpy string arrays
    """
    if x is None:
        return []

    # already a python sequence
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]

    if isinstance(x, np.ndarray):
        arr = x

        # MATLAB cellstr often loads as dtype=object, shape (1,P)
        if arr.dtype == object:
            out = []
            for e in arr.reshape(-1):
                # unwrap nested arrays
                if isinstance(e, np.ndarray):
                    ee = np.squeeze(e)
                    if ee.size == 1:
                        out.append(str(ee.item()).strip())
                    else:
                        # could be char matrix inside a cell
                        if ee.ndim == 2 and ee.dtype.kind in ("U", "S"):
                            out.append("".join(ee.reshape(-1)).strip())
                        else:
                            out.append(str(ee).strip())
                else:
                    out.append(str(e).strip())
            return out

        # MATLAB char matrix: array of single characters
        if arr.ndim == 2 and arr.dtype.kind in ("U", "S") and arr.dtype.itemsize <= 4:
            return ["".join(row.tolist()).strip() for row in arr]

        # normal string array
        if arr.dtype.kind in ("U", "S"):
            return [str(v).strip() for v in arr.reshape(-1)]

        # fallback
        try:
            return [str(v).strip() for v in arr.reshape(-1)]
        except Exception:
            return [str(arr)]

    return [str(x)]


def _get_field(M: Dict[str, Any], *names: str, default=None):
    for n in names:
        if n in M:
            return M[n]
    return default


def _squeeze2(a):
    if a is None:
        return None
    a = np.asarray(a)
    return np.squeeze(a)


def _ensure_T_dim(X: np.ndarray, want_last: int) -> np.ndarray:
    """
    Ensure X is (N,T,dim). Some MATLAB saves can come as (T,N,dim) depending on how saved.
    We detect by looking for the time dimension.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N,T,dim). Got shape={X.shape}")
    if X.shape[1] == want_last:
        return X
    # common transpose: (T,N,dim)
    if X.shape[0] == want_last:
        return np.transpose(X, (1, 0, 2))
    return X


def _get_dt(M: Dict[str, Any], default: float = 1.0) -> float:
    dt = _get_field(M, "dt", default=None)
    if dt is None:
        return float(default)
    dt = float(np.asarray(dt).reshape(-1)[0])
    return dt


def _get_actions(M: Dict[str, Any]) -> np.ndarray:
    for k in ("u_safe", "u_rl", "uOpts", "actions"):
        if k in M and M[k] is not None and np.size(M[k]) > 0:
            return np.asarray(M[k], dtype=float)
    raise KeyError("No control field found (expected one of u_safe, u_rl, uOpts, actions).")


def _get_num_inspected(M: Dict[str, Any], N: int, T: int) -> np.ndarray:
    A = _get_field(M, "num_inspected", default=None)
    if A is None or np.size(A) == 0:
        return np.full((N, T), np.nan, dtype=float)
    A = np.asarray(A, dtype=float)
    A = np.squeeze(A)
    if A.ndim == 3:
        A = A[:, :, 0]
    # handle (T,N) case
    if A.shape == (T, N):
        A = A.T
    if A.shape != (N, T):
        return np.full((N, T), np.nan, dtype=float)
    return A


def _get_hsun(M: Dict[str, Any], N: int, T: int) -> np.ndarray:
    A = _get_field(M, "h_sun", "hSUN", "HSUN", default=None)
    if A is None or np.size(A) == 0:
        return np.full((N, T), np.nan, dtype=float)
    A = np.asarray(A, dtype=float)
    A = np.squeeze(A)
    if A.ndim == 3:
        A = A[:, :, 0]
    if A.shape == (T, N):
        A = A.T
    if A.shape != (N, T):
        return np.full((N, T), np.nan, dtype=float)
    return A


def _get_meta_params(M: Dict[str, Any]) -> Tuple[Optional[np.ndarray], List[str]]:
    meta = _get_field(M, "meta_params", default=None)
    names = _get_field(M, "meta_param_names", default=None)
    if meta is None or names is None:
        return None, []
    meta = np.asarray(meta, dtype=float)
    names = _to_str_list(names)
    return meta, names


def _get_param_vectors_fallback(M: Dict[str, Any], N: int) -> Dict[str, Optional[np.ndarray]]:
    """
    Fallback if meta_params is absent or incomplete.

    We try (in order):
      1) Per-episode vectors saved by Python evaluator: R_C_vec, R_D_vec, R_MAX_vec
      2) Scalars saved in .mat: R_C, R_D, R_MAX (broadcast to length N)
      3) Alternative common scalar names: RC, RD, Rmax, R_MAX0, etc. (broadcast)
    """
    def vec(name: str) -> Optional[np.ndarray]:
        v = _get_field(M, name, default=None)
        if v is None or np.size(v) == 0:
            return None
        v = np.asarray(v, dtype=float).reshape(-1)
        return v if v.shape[0] == N else None

    def scalar(names: Sequence[str]) -> Optional[float]:
        for nm in names:
            v = _get_field(M, nm, default=None)
            if v is None or np.size(v) == 0:
                continue
            arr = np.asarray(v, dtype=float).reshape(-1)
            if arr.size == 1 and np.isfinite(arr[0]):
                return float(arr[0])
        return None

    # (1) vectors
    R_C_v = vec("R_C_vec")
    R_D_v = vec("R_D_vec")
    R_M_v = vec("R_MAX_vec")

    # (2-3) broadcast scalars if vectors missing
    if R_C_v is None:
        s = scalar(["R_C", "RC", "Rchief", "R_C0", "R_C_nom"])
        if s is not None:
            R_C_v = np.full((N,), s, dtype=float)
    if R_D_v is None:
        s = scalar(["R_D", "RD", "Rdeputy", "R_D0", "R_D_nom"])
        if s is not None:
            R_D_v = np.full((N,), s, dtype=float)
    if R_M_v is None:
        s = scalar(["R_MAX", "Rmax", "R_MAX0", "R_MAX_nom", "R_kiz", "Rkiz"])
        if s is not None:
            R_M_v = np.full((N,), s, dtype=float)

    return {"R_C": R_C_v, "R_D": R_D_v, "R_MAX": R_M_v}


def _get_RC_RD_RMAX(M: Dict[str, Any], N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve per-episode R_C, R_D, R_MAX.

    Preferred source:
      - meta_params (N,P) with meta_param_names containing R_C, R_D, R_MAX

    Fallbacks:
      - R_*_vec (length N) or broadcastable scalars stored in the .mat
    """
    meta, names = _get_meta_params(M)

    if meta is not None and len(names) > 0:
        # normalise names (strip spaces)
        names_norm = [s.strip() for s in names]
        def idx(key: str) -> Optional[int]:
            return names_norm.index(key) if key in names_norm else None

        iRC = idx("R_C")
        iRD = idx("R_D")
        iRM = idx("R_MAX")

        if iRC is not None and iRD is not None and iRM is not None:
            return meta[:, iRC], meta[:, iRD], meta[:, iRM]

    fb = _get_param_vectors_fallback(M, N)
    if fb["R_C"] is not None and fb["R_D"] is not None and fb["R_MAX"] is not None:
        return fb["R_C"], fb["R_D"], fb["R_MAX"]

    # last-resort: if we have meta but names are weird, print what we found
    raise KeyError(
        "Could not find R_C/R_D/R_MAX. "
        "Expected meta_params+meta_param_names containing {'R_C','R_D','R_MAX'} "
        "or fields R_C_vec/R_D_vec/R_MAX_vec (or scalars R_C/R_D/R_MAX)."
    )



def _crop_kstop(states: np.ndarray,
                steps_taken: np.ndarray,
                R_C: np.ndarray,
                R_D: np.ndarray,
                R_MAX: np.ndarray,
                h_sun: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine safety-crop index kStop per episode:
      kStop = min(steps_taken, first_k where any of (hKOZ,hKIZ,hSUN) < 0)
    Returns (kStop, is_unsafe_fail).
    """
    N, T, _ = states.shape
    pos = states[:, :, 0:3]
    rnorm = np.linalg.norm(pos, axis=2)

    KOZ = (R_C + R_D).reshape(-1, 1)  # (N,1)
    hKOZ = rnorm - KOZ
    hKIZ = R_MAX.reshape(-1, 1) - rnorm

    # if h_sun missing => ignore
    if np.all(np.isnan(h_sun)):
        hSUN = np.full_like(hKOZ, np.inf)
    else:
        hSUN = np.where(np.isnan(h_sun), np.inf, h_sun)

    kStop = np.clip(steps_taken.astype(int), 1, T).copy()
    is_unsafe = np.zeros((N,), dtype=bool)
    kUnsafe = np.full((N,), T, dtype=int)

    for i in range(N):
        kEnd = kStop[i]
        h1 = hKOZ[i, :kEnd]
        h2 = hKIZ[i, :kEnd]
        h3 = hSUN[i, :kEnd]
        bad = np.where((h1 < 0) | (h2 < 0) | (h3 < 0))[0]
        if bad.size > 0:
            is_unsafe[i] = True
            kUnsafe[i] = int(bad[0] + 1)  # 1-based step index
            kStop[i] = min(kStop[i], kUnsafe[i])

    return kStop, is_unsafe


def _total_thrust(U: np.ndarray, kStop: np.ndarray, dt: float) -> np.ndarray:
    """
    U: (N,T,3)
    kStop: (N,) 1..T
    Returns Utot (N,)
    """
    N = U.shape[0]
    out = np.zeros((N,), dtype=float)
    for i in range(N):
        k = int(kStop[i])
        seg = U[i, :k, :]
        out[i] = float(np.sum(np.linalg.norm(seg, axis=1)) * dt)
    return out


def _latex_table(values_by_method: Sequence[np.ndarray],
                 names: Sequence[str],
                 caption: str,
                 label: str,
                 fmt: str = "{:.2f}") -> str:
    lines = []
    lines.append(r"\begin{table}[hbt!]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lll}")
    lines.append(r"Case & [$\mu$ $\pm$ $\sigma$] & [$Q_1, Q_2, Q_3, P_{99}$] \\ \hline")

    for name, v in zip(names, values_by_method):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            lines.append(rf"{name} & -- & -- \\")
            continue
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        q1, q2, q3, p99 = np.percentile(v, [25, 50, 75, 99]).tolist()
        lines.append(
            rf"{name} & {fmt.format(mu)} $\pm$ {fmt.format(sd)} & "
            rf"[{fmt.format(q1)}, {fmt.format(q2)}, {fmt.format(q3)}, {fmt.format(p99)}] \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _log_color_mapper(values: np.ndarray, vmin: float, vmax: float, cmap):
    """
    Returns function mapping a scalar value -> RGBA from cmap using log10 scaling.
    """
    vmin = max(float(vmin), 0.0)
    vmax = float(vmax)
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1.0

    eps_u = max(1e-12, 1e-3 * vmax)
    Lmin = np.log10(vmin + eps_u)
    Lmax = np.log10(vmax + eps_u)
    if abs(Lmax - Lmin) < 1e-12:
        Lmax = Lmin + 1.0

    def color(u):
        u = float(np.clip(u, vmin, vmax))
        t = (np.log10(u + eps_u) - Lmin) / (Lmax - Lmin)
        t = float(np.clip(t, 0.0, 1.0))
        return cmap(t)

    return color


def _plot_dv_score_violin(
    dv_by_method: Sequence[np.ndarray],
    score_by_method: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Make a dual-axis violin plot similar to your MATLAB figure:
      - Left axis (blue): Total Δv (log scale)
      - Right axis (red): Inspection score (%) (linear)
    We plot both at each category with small horizontal offsets.
    """
    nM = len(names)
    pos = np.arange(1, nM + 1, dtype=float)

    fig, axL = plt.subplots(figsize=(13.8, 5.2), dpi=120)
    axL.set_facecolor("white")
    fig.patch.set_facecolor("white")
    axL.grid(True, alpha=0.25)

    # Offsets and styling
    offL = -0.18
    offR = +0.18
    blue = (0.0, 0.35, 0.90)
    red = (0.90, 0.25, 0.25)

    # ---------- Left: Δv ----------
    dv_clean = []
    for v in dv_by_method:
        vv = np.asarray(v, dtype=float)
        vv = vv[np.isfinite(vv) & (vv > 0)]
        dv_clean.append(vv)

    vp = axL.violinplot(dv_clean, positions=pos + offL, widths=0.55, showmeans=False, showextrema=False, showmedians=False)
    for b in vp["bodies"]:
        b.set_facecolor(blue)
        b.set_edgecolor("none")
        b.set_alpha(0.18)

    # Scatter + median/IQR (quick “box” overlay)
    for i, vv in enumerate(dv_clean, start=1):
        if vv.size == 0:
            continue
        x = (i + offL) + 0.06 * np.random.default_rng(1234 + i).standard_normal(vv.size)
        axL.scatter(x, vv, s=6, color=blue, alpha=0.65, linewidths=0)

        q1, q2, q3 = np.percentile(vv, [25, 50, 75])
        axL.plot([i + offL - 0.18, i + offL + 0.18], [q2, q2], color=blue, linewidth=2.0)
        axL.plot([i + offL, i + offL], [q1, q3], color="k", linewidth=1.2, alpha=0.85)

    axL.set_yscale("log")
    axL.set_ylabel(r"Total $\Delta v$ [m/s]", color=blue)
    axL.tick_params(axis="y", colors=blue)

    # ---------- Right: inspection score ----------
    axR = axL.twinx()

    score_clean = []
    for v in score_by_method:
        vv = np.asarray(v, dtype=float)
        vv = vv[np.isfinite(vv)]
        score_clean.append(vv)

    vp2 = axR.violinplot(score_clean, positions=pos + offR, widths=0.55, showmeans=False, showextrema=False, showmedians=False)
    for b in vp2["bodies"]:
        b.set_facecolor(red)
        b.set_edgecolor("none")
        b.set_alpha(0.22)

    for i, vv in enumerate(score_clean, start=1):
        if vv.size == 0:
            continue
        x = (i + offR) + 0.06 * np.random.default_rng(4321 + i).standard_normal(vv.size)
        axR.scatter(x, vv, s=14, facecolors="none", edgecolors=red, alpha=0.75, linewidths=0.8)

        q1, q2, q3 = np.percentile(vv, [25, 50, 75])
        axR.plot([i + offR - 0.18, i + offR + 0.18], [q2, q2], color=red, linewidth=2.0)
        axR.plot([i + offR, i + offR], [q1, q3], color="k", linewidth=1.2, alpha=0.85)

    axR.set_ylabel("Inspection Score (%)", color=red)
    axR.tick_params(axis="y", colors=red)
    axR.set_ylim(10, 100)

    # X-axis categories
    axL.set_xlim(0.5, nM + 0.5)
    axL.set_xticks(pos)
    axL.set_xticklabels(list(names))
    if title:
        axL.set_title(title)

    fig.tight_layout()

    if save_path:
        import os
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


# ----------------------------- main plotting -----------------------------

def _plot_kiz_annulus(ax, Rlo: float, Rhi: float):
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(Rlo*np.cos(th), Rlo*np.sin(th), linestyle=":", linewidth=1.0)
    ax.plot(Rhi*np.cos(th), Rhi*np.sin(th), linestyle=":", linewidth=1.0)


def plot_inspection_threeway(
    *,
    baseline_mat: str,
    mlp_mat: str,
    rnn_mat: str,
    save_prefix: Optional[str] = None,
    model_names: Sequence[str] = ("ICCBF", "MLP-tuned ICCBF", "RNN-tuned ICCBF"),
    stride_traj: int = 5,
    stride_ts: int = 5,
    Nsucc_plot: int = 100,
    Nfail_plot: int = 10,
    rng_seed: int = 1,
    target_inspected: float = 100.0,
    tol_inspected: float = 1e-9,
    kiz_nominal: float = 800.0,
    kiz_band: float = 0.10,
) -> Dict[str, Any]:
    """
    Loads three .mat files and produces:
      - main 5x3 plot
      - returns summary + LaTeX tables
    """
    paths = [baseline_mat, mlp_mat, rnn_mat]
    models = [loadmat(p) for p in paths]

    # Infer dt/TOF per model; use first model for time axis if present
    dts = [_get_dt(M, default=1.0) for M in models]
    dt = float(dts[0])

    # Prepare per-model derived arrays
    per = []
    for M in models:
        states = np.asarray(_get_field(M, "states", default=None), dtype=float)
        if states is None or states.size == 0:
            raise KeyError("Missing 'states' in .mat (expected states: N x T x 7 or N x T x 6).")
        states = np.asarray(states, dtype=float)
        states = np.squeeze(states)
        if states.ndim != 3:
            raise ValueError(f"'states' must be 3D. Got shape={states.shape}")

        N, T, D = states.shape
        if D < 6:
            raise ValueError(f"Expected at least 6 state dims (r,v). Got D={D}")
        # keep only first 6 to match MATLAB convention, but allow D=7
        X = states[:, :, :6]

        U = _get_actions(M)
        U = np.asarray(U, dtype=float)
        U = np.squeeze(U)
        if U.ndim != 3:
            raise ValueError(f"Controls must be 3D (N,T,3). Got shape={U.shape}")
        # harmonise possible transposes
        if U.shape[0] != N and U.shape[1] == N:
            U = np.transpose(U, (1, 0, 2))
        if U.shape[0] != N or U.shape[1] != T:
            raise ValueError(f"Control shape must match states (N,T,*). states={states.shape}, U={U.shape}")

        steps_taken = np.asarray(_get_field(M, "steps_taken", default=None)).reshape(-1)
        if steps_taken.size != N:
            raise ValueError(f"steps_taken must be length N={N}. Got {steps_taken.size}")

        hsun = _get_hsun(M, N, T)
        nins = _get_num_inspected(M, N, T)

        R_C, R_D, R_MAX = _get_RC_RD_RMAX(M, N)

        kStop, is_unsafe = _crop_kstop(X, steps_taken, R_C, R_D, R_MAX, hsun)
        Utot_all = _total_thrust(U, kStop, _get_dt(M, default=dt))

        # Task success based on inspected points at kStop
        if np.all(np.isnan(nins)):
            is_success = np.zeros((N,), dtype=bool)
        else:
            is_success = np.array([nins[i, int(kStop[i])-1] >= (target_inspected - tol_inspected) for i in range(N)])

        per.append({
            "M": M,
            "X": X,
            "U": U,
            "N": N,
            "T": T,
            "dt": _get_dt(M, default=dt),
            "steps_taken": steps_taken.astype(int),
            "kStop": kStop.astype(int),
            "is_unsafe": is_unsafe,
            "hsun": hsun,
            "nins": nins,
            "R_C": R_C,
            "R_D": R_D,
            "R_MAX": R_MAX,
            "Utot_all": Utot_all,
            "is_success": is_success,
        })

    # ---------------- global colour scale (task-success only, sampled like MATLAB) ----------------
    rng = np.random.default_rng(rng_seed)
    succ_utots_for_scale = []
    for col, P in enumerate(per, start=1):
        idx_succ = np.where(P["is_success"])[0]
        idx_fail = np.where(~P["is_success"])[0]

        # sample like MATLAB: independent per method seed shift
        rng2 = np.random.default_rng(rng_seed + 100*col)

        if idx_succ.size > 0:
            pick = rng2.choice(idx_succ, size=min(Nsucc_plot, idx_succ.size), replace=False)
            succ_utots_for_scale.extend(P["Utot_all"][pick].tolist())

    if len(succ_utots_for_scale) == 0:
        umin_global, umax_global = 0.0, 1.0
    else:
        umin_global = float(np.percentile(succ_utots_for_scale, 5))
        umax_global = float(np.percentile(succ_utots_for_scale, 95))
        if umax_global <= umin_global + 1e-12:
            umax_global = umin_global + 1.0

    cmap = get_cmap("turbo")
    colorOf = _log_color_mapper(np.array(succ_utots_for_scale, dtype=float), umin_global, umax_global, cmap)

    # ---------------- plot ----------------
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
    })

    fig = plt.figure(figsize=(12.8, 10.3), dpi=110)
    gs = fig.add_gridspec(5, 3, wspace=0.28, hspace=0.35)

    RkizLo = (1.0 - kiz_band) * kiz_nominal
    RkizHi = (1.0 + kiz_band) * kiz_nominal

    for col, (P, name) in enumerate(zip(per, model_names)):
        X, U, dt_i = P["X"], P["U"], P["dt"]
        N, T = P["N"], P["T"]
        kStop = P["kStop"]
        nins = P["nins"]
        hsun = P["hsun"]

        # derived hKOZ/hKIZ
        rnorm = np.linalg.norm(X[:, :, 0:3], axis=2)
        KOZ = (P["R_C"] + P["R_D"]).reshape(-1, 1)
        hKOZ = rnorm - KOZ
        hKIZ = P["R_MAX"].reshape(-1, 1) - rnorm

        # choose rollouts to draw
        rng2 = np.random.default_rng(rng_seed + 100*(col+1))
        idx_succ = np.where(P["is_success"])[0]
        idx_fail = np.where(~P["is_success"])[0]
        idx_succ_plot = rng2.choice(idx_succ, size=min(Nsucc_plot, idx_succ.size), replace=False) if idx_succ.size else np.array([], dtype=int)
        idx_fail_plot = rng2.choice(idx_fail, size=min(Nfail_plot, idx_fail.size), replace=False) if idx_fail.size else np.array([], dtype=int)
        idx_plot = np.concatenate([idx_succ_plot, idx_fail_plot])

        # per-rollout Utot for plotted rollouts
        Utot_plot = P["Utot_all"][idx_plot] if idx_plot.size else np.array([], dtype=float)

        # time indices
        tidx_traj_all = np.arange(0, T, stride_traj, dtype=int)
        tidx_ts_all = np.arange(0, T, stride_ts, dtype=int)

        # -------- row 1: XY spaghetti + kiz band --------
        ax1 = fig.add_subplot(gs[0, col])
        ax1.grid(True)
        for j, ii in enumerate(idx_plot):
            ii = int(ii)
            kEnd = int(kStop[ii])
            tidx = tidx_traj_all[tidx_traj_all < kEnd]
            if tidx.size < 2:
                continue
            ls = "--" if (ii in idx_fail_plot) else "-"
            lw = 0.9 if (ii in idx_fail_plot) else 0.6
            ax1.plot(X[ii, tidx, 0], X[ii, tidx, 1], linestyle=ls, linewidth=lw, color=colorOf(Utot_plot[j]))
        _plot_kiz_annulus(ax1, RkizLo, RkizHi)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_title(name)

        # legend counts
        nFail_full = int(np.sum(~P["is_success"]))
        nSucc_full = int(np.sum(P["is_success"]))
        ax1.plot([], [], "k--", linewidth=0.9, label=f"Task fail ({nFail_full})")
        ax1.plot([], [], "k-",  linewidth=0.6, label=f"Task success ({nSucc_full})")
        ax1.plot([], [], "k:",  linewidth=1.0, label=f"KIZ band ({kiz_nominal:g} ± {100*kiz_band:g}%)")
        ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1, frameon=False)

        # -------- helper for rows 2-4 --------
        def plot_h(ax, H, ylabel):
            ax.grid(True)
            # y-limits from percentile across plotted traces
            vals = []
            for ii in idx_plot:
                kEnd = int(kStop[int(ii)])
                tidx = tidx_ts_all[tidx_ts_all < kEnd]
                if tidx.size >= 2:
                    vals.append(H[int(ii), tidx])
            if len(vals) == 0:
                ylo, yhi = -1.0, 1.0
            else:
                vv = np.concatenate([v.ravel() for v in vals])
                ylo, yhi = np.percentile(vv, [1, 99]).tolist()
                pad = 0.08 * (yhi - ylo + 1e-12)
                ylo -= pad
                yhi += pad
            ylo = min(ylo, -0.1)
            yhi = max(yhi,  0.1)

            for j, ii in enumerate(idx_plot):
                ii = int(ii)
                kEnd = int(kStop[ii])
                tidx = tidx_ts_all[tidx_ts_all < kEnd]
                if tidx.size < 2:
                    continue
                tt = tidx * dt_i
                Hseg = H[ii, tidx]
                # clip for display like MATLAB
                Hdisp = np.maximum(Hseg, ylo)
                ax.plot(tt, Hdisp, linewidth=0.7, color=colorOf(Utot_plot[j]))
            ax.axhline(0.0, linestyle="--", linewidth=0.8, color="k")
            ax.set_ylim([ylo, yhi])
            ax.set_ylabel(ylabel)

        # -------- row 2: hKOZ --------
        ax2 = fig.add_subplot(gs[1, col])
        plot_h(ax2, hKOZ, r"$h_{\mathrm{KOZ}}(t)$")
        if col == 0:
            ax2.set_xlabel("t [s]")

        # -------- row 3: hKIZ --------
        ax3 = fig.add_subplot(gs[2, col])
        plot_h(ax3, hKIZ, r"$h_{\mathrm{KIZ}}(t)$")
        if col == 0:
            ax3.set_xlabel("t [s]")

        # -------- row 4: hSUN --------
        ax4 = fig.add_subplot(gs[3, col])
        if np.all(np.isnan(hsun)):
            ax4.axis("off")
            ax4.text(0.05, 0.6, r"$h_{\mathrm{SUN}}(t)$ not available", transform=ax4.transAxes)
        else:
            plot_h(ax4, hsun, r"$h_{\mathrm{SUN}}(t)$")
            if col == 0:
                ax4.set_xlabel("t [s]")

        # -------- row 5: inspected % --------
        ax5 = fig.add_subplot(gs[4, col])
        if np.all(np.isnan(nins)):
            ax5.axis("off")
            ax5.text(0.05, 0.6, r"num\_inspected not available", transform=ax5.transAxes)
        else:
            ax5.grid(True)
            for j, ii in enumerate(idx_plot):
                ii = int(ii)
                kEnd = int(kStop[ii])
                tidx = tidx_ts_all[tidx_ts_all < kEnd]
                if tidx.size < 2:
                    continue
                tt = tidx * dt_i
                ax5.plot(tt, nins[ii, tidx], linewidth=0.9, color=colorOf(Utot_plot[j]))
            ax5.axhline(target_inspected, linestyle=":", linewidth=0.8, color="k")
            ax5.set_ylim([0, max(105, target_inspected + 5)])
            ax5.set_ylabel("Inspected [%]")
            ax5.set_xlabel("t [s]")
            ax5.set_title("Inspection progress")

    # global colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=umin_global, vmax=umax_global)
    cax = fig.add_axes([0.93, 0.07, 0.012, 0.88])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"Total $\Delta v$ $\int \|u\|\,dt$ (task-success-scaled, log-mapped)")

    fig_paths = {}

    if save_prefix is not None:
        import os
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True) if os.path.dirname(save_prefix) else None
        main_path = f"{save_prefix}_main.png"
        fig.savefig(main_path, bbox_inches="tight")
        fig_paths["main"] = main_path

    # ---------------- summary stats + latex ----------------
    uTotals_full = [P["Utot_all"] for P in per]

    # final inspected points (not safety-cropped; mimic MATLAB compute_final_inspected_all_inspection)
    nFinal_full = []
    for P in per:
        nins = P["nins"]
        if np.all(np.isnan(nins)):
            nFinal_full.append(np.full((P["N"],), np.nan))
        else:
            vals = np.zeros((P["N"],), dtype=float)
            for i in range(P["N"]):
                kEnd = int(np.clip(P["steps_taken"][i], 1, P["T"]))
                vals[i] = nins[i, kEnd-1]
            nFinal_full.append(vals)

    latex_thrust = _latex_table(uTotals_full, model_names, "Inspection Thrust Consumption", "InspectionTT")
    latex_pts = _latex_table(nFinal_full, model_names, "Final Number of Points Inspected", "InspectionPtsFinal")

    success_rates = [float(np.mean(P["is_success"])) for P in per]

    
    # ---------------- dual-axis violin: total Δv vs inspection score ----------------
    # Define inspection score as final inspected percentage at episode end (NOT safety-cropped),
    # matching the MATLAB-style "Inspection Score (%)" distribution plot.
    score_full = nFinal_full

    fig_violin = _plot_dv_score_violin(
        dv_by_method=uTotals_full,
        score_by_method=score_full,
        names=model_names,
        title="Total $\Delta v$ and Inspection Score",
        save_path=(f"{save_prefix}_dv_score.png" if save_prefix is not None else None),
    )

    out = {
            "fig": fig,
            "fig_violin": fig_violin,
            "fig_paths": fig_paths,
            "fig_paths_violin": ({"dv_score": f"{save_prefix}_dv_score.png"} if save_prefix is not None else {}),
            "latex_table": latex_thrust + "\n\n" + latex_pts,
            "latex_table_thrust": latex_thrust,
            "latex_table_points": latex_pts,
            "success_rates": success_rates,
            "umin_global": umin_global,
            "umax_global": umax_global,
        }
    return out
