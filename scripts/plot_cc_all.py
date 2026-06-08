#!/usr/bin/env python3
"""
Generate cruise-control comparison plots and LaTeX table for all architectures × algorithms.

Usage (after running scripts/run_cc_eval_all.sh):
    python scripts/plot_cc_all.py

Or with explicit .mat paths:
    python scripts/plot_cc_all.py \
        --lstm-ppo   outputs/data/cruise_control/LSTMsol.mat \
        --lstm-sac   outputs/eval/cruise_control/<LSTMSAC_...>.mat \
        --gru-ppo    outputs/eval/cruise_control/<GRU_PPO_...>.mat \
        --gru-sac    outputs/eval/cruise_control/<GRU_SAC_...>.mat \
        --mamba2-ppo outputs/eval/cruise_control/<Mamba2_PPO_...>.mat \
        --mamba2-sac outputs/eval/cruise_control/<Mamba2_SAC_...>.mat \
        --out-prefix outputs/figures/cc_all

Outputs:
    <out-prefix>_main.png   — trajectory / h(t) / V(t) plots (3 x N_models)
    <out-prefix>_violin.png — violin plot of total thrust
    <out-prefix>_table.tex  — LaTeX table with fuel stats + safety %
"""

from __future__ import annotations

import argparse
import os
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _latest_mat(pattern: str) -> str | None:
    """Return the most recently modified .mat matching glob pattern, or None."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _auto_find_mats(eval_dir: str) -> dict[str, str | None]:
    """Auto-discover the latest .mat for each model in eval_dir."""
    d = eval_dir.rstrip("/")
    return {
        "lstm_ppo":   _latest_mat(f"{d}/*LSTMTuned*"),
        "lstm_sac":   _latest_mat(f"{d}/*LSTMSAC*"),
        "gru_ppo":    _latest_mat(f"{d}/*GRUTuned*") or _latest_mat(f"{d}/*GRU_PPO*"),
        "gru_sac":    _latest_mat(f"{d}/*GRUSAC*"),
        "mamba2_ppo": _latest_mat(f"{d}/*Mamba2Tuned*") or _latest_mat(f"{d}/*Mamba2_PPO*"),
        "mamba2_sac": _latest_mat(f"{d}/*Mamba2SAC*"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lstm-ppo",   default=None)
    parser.add_argument("--lstm-sac",   default=None)
    parser.add_argument("--gru-ppo",    default=None)
    parser.add_argument("--gru-sac",    default=None)
    parser.add_argument("--mamba2-ppo", default=None)
    parser.add_argument("--mamba2-sac", default=None)
    parser.add_argument("--eval-dir",   default=str(REPO_ROOT / "outputs/eval/cruise_control"),
                        help="Directory to auto-discover .mat files when explicit paths are not given")
    parser.add_argument("--out-prefix", default=str(REPO_ROOT / "outputs/figures/cc_all"))
    parser.add_argument("--no-viability-filter", dest="skip_viability_filter",
                        action="store_true", default=True,
                        help="Skip max-brake viability filter (default: True; ICs already clean)")
    parser.add_argument("--viability-filter", dest="skip_viability_filter",
                        action="store_false",
                        help="Re-enable max-brake viability filter")
    parser.add_argument(
        "--group", choices=["ppo", "sac", "all"], default="all",
        help="Filter models: ppo = LSTM/GRU/Mamba2+PPO only, sac = +SAC only, "
             "all = all 6 models (default). PPO/SAC give 3-column paper-friendly figures."
    )
    args = parser.parse_args()

    # Auto-discover missing paths from eval_dir
    auto = _auto_find_mats(args.eval_dir)
    if args.lstm_ppo   is None: args.lstm_ppo   = auto["lstm_ppo"] or str(REPO_ROOT / "outputs/data/cruise_control/LSTMsol.mat")
    if args.lstm_sac   is None: args.lstm_sac   = auto["lstm_sac"]
    if args.gru_ppo    is None: args.gru_ppo    = auto["gru_ppo"]
    if args.gru_sac    is None: args.gru_sac    = auto["gru_sac"]
    if args.mamba2_ppo is None: args.mamba2_ppo = auto["mamba2_ppo"]
    if args.mamba2_sac is None: args.mamba2_sac = auto["mamba2_sac"]

    # Build ordered list of (label, path) — skip missing
    candidates = [
        ("LSTM+PPO",   args.lstm_ppo),
        ("LSTM+SAC",   args.lstm_sac),
        ("GRU+PPO",    args.gru_ppo),
        ("GRU+SAC",    args.gru_sac),
        ("Mamba2+PPO", args.mamba2_ppo),
        ("Mamba2+SAC", args.mamba2_sac),
    ]
    # Filter by algorithm group for paper-friendly 3-column figures
    if args.group == "ppo":
        candidates = [(n, p) for (n, p) in candidates if "+PPO" in n]
    elif args.group == "sac":
        candidates = [(n, p) for (n, p) in candidates if "+SAC" in n]
    available = [(name, path) for (name, path) in candidates if path and os.path.isfile(path)]

    if not available:
        print("ERROR: No .mat files found. Run scripts/run_cc_eval_all.sh first, or pass paths explicitly.")
        sys.exit(1)

    print(f"Models to plot ({len(available)}):")
    for name, path in available:
        print(f"  {name:12s}  {path}")

    missing = [(name, path) for (name, path) in candidates if not (path and os.path.isfile(str(path)))]
    if missing:
        print(f"\nWarning: {len(missing)} model(s) not found and will be skipped:")
        for name, path in missing:
            print(f"  {name:12s}  {path or '(not provided)'}")

    mat_paths  = [p for _, p in available]
    model_names = [n for n, _ in available]

    # Ensure output directory exists
    out_prefix = args.out_prefix
    if args.group != "all":
        out_prefix = f"{out_prefix}_{args.group}"
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    # Import here so REPO_ROOT adjustment isn't needed at module level
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from metarl_iccbf.cruise_control.analysis.plot_cc_results import plot_cruisecontrol

    print(f"\nRunning plot_cruisecontrol (skip_viability_filter={args.skip_viability_filter}) ...")
    out = plot_cruisecontrol(
        mat_paths=mat_paths,
        model_names=model_names,
        skip_viability_filter=args.skip_viability_filter,
        save_prefix=out_prefix,
    )

    # Save LaTeX table
    table_path = f"{out_prefix}_table.tex"
    with open(table_path, "w") as f:
        f.write(out["latex_table"])
    print(f"\nLaTeX table saved to: {table_path}")
    print("\n--- LaTeX table ---")
    print(out["latex_table"])

    # Print safety rates summary
    print("\n--- Safety rates ---")
    for name, sr in zip(model_names, out["safety_rates"]):
        print(f"  {name:12s}: {sr:.1f}%")


if __name__ == "__main__":
    main()
