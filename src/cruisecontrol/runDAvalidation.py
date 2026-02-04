
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def plot_cruisecontrol_da_bound_violins(
    *,
    baseline_mat: str,
    mlp_mat: str,
    rnn_mat: str,
    save_prefix: Optional[str] = None,   # e.g. "figs/cc_da_level4" (saves PNG); None => no save
    data_dir: Optional[str] = None,      # default: ../src/data/cruise_control (relative to this file)
    # --- validation settings (defaults match your CLI script) ---
    hw: Tuple[float, float] = (2.0, 2.0),
    stride: int = 5,
    da_order: int = 4,
    M: int = 10,
    seed: int = 0,
    eps_zero: float = 1e-16,
    actions_unscaled: bool = False,
    use_param_estimates: bool = True,
    v0_nom: float = 13.89,
    m_nom: float = 1650.0,
    ulim_nom: float = 0.25,
    f0: float = 0.1,
    f1: float = 5.0,
    f2: float = 0.25,
    g0: float = 9.81,
    # --- plotting settings ---
    which: str = "boundary",              # "boundary" or "centre"
    abs_vals: bool = True,                # match your nonnegative slack-style plot
    clip_quantiles: Optional[Tuple[float, float]] = (0.0, 0.995),
    figsize: Tuple[float, float] = (14.5, 4.6),
) -> Dict[str, Any]:
    """
    Runs DA-bound Monte-Carlo validation (creates mc_bounds_slack_data.mat) for
    baseline/MLP/RNN cruise-control rollouts, then produces clean violin plots.

    Output folder (default):
      ../src/data/cruise_control/mc_bounds_da/<method>/

    Returns:
      dict with:
        - slack_mat_paths: generated slack mat files
        - fig, axs
        - latex_table: min slack summary (signed)
        - stats: per-method per-term violations/min slack
        - saved: paths to PNG if save_prefix is set
    """
    # Import validator only when called
    from cruisecontrol.validate_mc_da_bounds_boundary_cruise import validate_mc_boundary

    # Resolve default data directory
    if data_dir is None:
        here = Path(__file__).resolve()
        data_dir = str((here.parents[2] / "src" / "data" / "cruise_control").resolve())
    data_dir_p = Path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)

    which = which.lower().strip()
    if which not in ("boundary", "centre"):
        raise ValueError("which must be 'boundary' or 'centre'")

    # ---------- run validator for each method ----------
    def _run_one(method_name: str, mat_path: str) -> str:
        out_dir = data_dir_p / "mc_bounds_da" / method_name
        out_dir.mkdir(parents=True, exist_ok=True)

        _, mat_out = validate_mc_boundary(
            mat_path=str(mat_path),
            out_dir=str(out_dir),
            half_width=np.array(hw, dtype=float),
            da_order=int(da_order),
            stride=int(stride),
            M=int(M),
            seed=int(seed),
            eps_zero=float(eps_zero),
            actions_unscaled=bool(actions_unscaled),
            use_param_estimates=bool(use_param_estimates),
            v0_nom=float(v0_nom),
            m_nom=float(m_nom),
            ulim_nom=float(ulim_nom),
            f0=float(f0),
            f1=float(f1),
            f2=float(f2),
            g0=float(g0),
        )
        return str(mat_out)

    slack_paths = {
        "ICCBF": _run_one("baseline_iccbf", baseline_mat),
        "MLP-tuned ICCBF": _run_one("stage1_mlp", mlp_mat),
        "RNN-tuned ICCBF": _run_one("stage1_rnn", rnn_mat),
    }

    # ---------- robust slack loader ----------
    def _load_slacks(mat_slack_path: str) -> Dict[str, np.ndarray]:
        D = sio.loadmat(mat_slack_path, squeeze_me=True, struct_as_record=False)
        suffix = "boundary" if which == "boundary" else "centre"

        # New names produced by your script
        new = {
            "b2": f"slack_h_{suffix}",
            "Lf": f"slack_Lfh_{suffix}",
            "Lg": f"slack_Lgh_{suffix}",
        }
        # Old fallbacks
        old = {
            "b2": f"slack_b2_{suffix}",
            "Lf": f"slack_Lfb2_{suffix}",
            "Lg": f"slack_Lgb2_{suffix}",
        }

        keys = {}
        for k, nm in new.items():
            if nm in D:
                keys[k] = nm
        for k, nm in old.items():
            if k not in keys and nm in D:
                keys[k] = nm

        miss = [k for k in ("b2", "Lf", "Lg") if k not in keys]
        if miss:
            raise KeyError(
                f"{mat_slack_path} missing {miss}. "
                f"Expected: {list(new.values())} (or fallbacks {list(old.values())})."
            )

        out = {}
        for k in ("b2", "Lf", "Lg"):
            x = np.asarray(D[keys[k]], dtype=float).ravel()
            out[k] = x[np.isfinite(x)]
        return out

    raw = {m: _load_slacks(p) for m, p in slack_paths.items()}

    # ---------- stats (signed slack for violations) ----------
    def _stats(x: np.ndarray) -> Dict[str, float]:
        x = x[np.isfinite(x)]
        n = int(x.size)
        nv = int(np.sum(x < 0.0))
        mn = float(np.min(x)) if n else float("nan")
        return {"N": n, "violations": nv, "min_slack": mn}

    stats = {m: {k: _stats(v) for k, v in sl.items()} for m, sl in raw.items()}

    # ---------- latex table: min slacks ----------
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Method & $\\min s(b_2)$ & $\\min s(L_f b_2)$ & $\\min s(L_g b_2)$ \\\\",
        "\\hline",
    ]
    for m in ("ICCBF", "MLP-tuned ICCBF", "RNN-tuned ICCBF"):
        lines.append(
            f"{m} & {stats[m]['b2']['min_slack']:+.2e} & {stats[m]['Lf']['min_slack']:+.2e} & {stats[m]['Lg']['min_slack']:+.2e} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    latex_table = "\n".join(lines)

    # ---------- plot: 3 subplots, pure violins ----------
    def _prep(x: np.ndarray) -> np.ndarray:
        x = x[np.isfinite(x)]
        if abs_vals:
            x = np.abs(x)
        if clip_quantiles is not None and x.size > 10:
            ql, qu = clip_quantiles
            lo, hi = np.quantile(x, [ql, qu])
            x = np.clip(x, lo, hi)
        return x

    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
    methods = ["ICCBF", "MLP-tuned ICCBF", "RNN-tuned ICCBF"]
    xticklabels = [r"$b_2$", r"$L_f\,b_2$", r"$L_g\,b_2$"]

    for ax, m in zip(axs, methods):
        data = [_prep(raw[m]["b2"]), _prep(raw[m]["Lf"]), _prep(raw[m]["Lg"])]
        parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True, widths=0.82)
        for body in parts["bodies"]:
            body.set_alpha(0.25)
        for k in ["cbars", "cmins", "cmaxes", "cmeans"]:
            if k in parts and parts[k] is not None:
                parts[k].set_linewidth(1.0)

        ax.set_title(m, fontsize=14)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.grid(True, alpha=0.25)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

    axs[0].set_ylabel(r"$|s_t^{\min}|$" if abs_vals else r"$s_t^{\min}$", fontsize=16)
    fig.tight_layout()

    saved = {}
    if save_prefix is not None:
        save_prefix = str(save_prefix)
        Path(os.path.dirname(save_prefix) or ".").mkdir(parents=True, exist_ok=True)
        fig_path = f"{save_prefix}_da_bounds_violins.png"
        fig.savefig(fig_path, dpi=250, bbox_inches="tight")
        saved["violin_png"] = fig_path

    # Notebook-friendly summary
    print("=== Cruise DA-bound MC validation (violin) ===")
    print(f"data_dir: {data_dir_p}")
    for m in methods:
        s = stats[m]
        print(f"\n[{m}]")
        print(f"  b2: min={s['b2']['min_slack']:+.3e}, viol={s['b2']['violations']}/{s['b2']['N']}")
        print(f"  Lf: min={s['Lf']['min_slack']:+.3e}, viol={s['Lf']['violations']}/{s['Lf']['N']}")
        print(f"  Lg: min={s['Lg']['min_slack']:+.3e}, viol={s['Lg']['violations']}/{s['Lg']['N']}")

    return {
        "data_dir": str(data_dir_p),
        "slack_mat_paths": slack_paths,
        "stats": stats,
        "latex_table": latex_table,
        "fig": fig,
        "axs": axs,
        "saved": saved,
    }
