"""CLI entry point for docking evaluation.

Usage::

    # Serial evaluation
    python -m metarl_iccbf.docking.evaluation.cli \\
        --model-path outputs/docking/.../best_model.zip \\
        --policy-type MAMBA

    # Parallel evaluation
    python -m metarl_iccbf.docking.evaluation.cli \\
        --model-path outputs/docking/.../best_model.zip \\
        --policy-type MAMBA \\
        --parallel \\
        --ics-path src/metarl_iccbf/docking/T2docking_episode_spec_N5000_seed123.npz \\
        --num-workers 8
"""

import argparse
from datetime import datetime
from pathlib import Path


def _auto_mat_name(model_path: str, mode: str) -> str:
    """Generate a .mat filename that includes the model name and a timestamp.

    E.g. ``eval_Mamba2TunedICCBF_Docking_20260226_091304_parallel_20260315_163012.mat``
    """
    model_name = Path(model_path).resolve().parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{model_name}_{mode}_{timestamp}.mat"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a docking policy.")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model (best_model.zip).",
    )
    parser.add_argument(
        "--policy-type", choices=["MLP", "RNN", "GRU", "LSTM", "MAMBA"], default="MLP",
        help="Policy type (default: MLP).",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Use parallel evaluation (requires --ics-path).",
    )
    parser.add_argument(
        "--ics-path", type=str, default=None,
        help="Path to .npz episode bank (required for --parallel).",
    )
    parser.add_argument("--dt", type=float, default=0.5, help="Time step (default: 0.5).")
    parser.add_argument("--tof", type=float, default=50.0, help="Time of flight (default: 50.0).")
    parser.add_argument("--out-dir", type=str, default="ResultsEval", help="Output directory.")
    parser.add_argument("--out-mat", type=str, default=None, help="Output .mat filename.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting (serial only).")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers (default: 8).")
    parser.add_argument("--num-chunks", type=int, default=32, help="Parallel chunks (default: 32).")
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm used for training (default: ppo)",
    )
    parser.add_argument("--no-da", action="store_true", help="Skip DA.init().")
    parser.add_argument(
        "--adversarial", action="store_true",
        help="Enable adversarial target rotation in evaluation.",
    )
    args = parser.parse_args()

    if args.parallel:
        if args.ics_path is None:
            parser.error("--ics-path is required when using --parallel")

        from metarl_iccbf.docking.evaluation.eval_parallel import (
            evaluate_parallel,
            EvalConfig,
        )

        out_mat = args.out_mat or _auto_mat_name(args.model_path, "parallel")
        cfg = EvalConfig(
            ics_path=args.ics_path,
            model_path=args.model_path,
            policy_type=args.policy_type,
            algo=args.algo,
            dt=args.dt,
            TOF=args.tof,
            init_da=not args.no_da,
            adversarial=args.adversarial,
            n_workers=args.num_workers,
            n_chunks=args.num_chunks,
            out_dir=args.out_dir,
            out_mat=out_mat,
        )
        evaluate_parallel(cfg)
    else:
        from metarl_iccbf.docking.evaluation.eval_serial import evaluate_serial

        evaluate_serial(
            model_path=args.model_path,
            policy_type=args.policy_type,
            algo=args.algo,
            dt=args.dt,
            TOF=args.tof,
            out_dir=args.out_dir,
            out_mat=args.out_mat,
            plot=not args.no_plot,
            init_da=not args.no_da,
        )


if __name__ == "__main__":
    main()
