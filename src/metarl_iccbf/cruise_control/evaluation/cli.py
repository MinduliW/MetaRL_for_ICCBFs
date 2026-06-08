"""Simple CLI for cruise-control evaluation."""

import argparse
from datetime import datetime
from pathlib import Path

from metarl_iccbf.cruise_control.evaluation.eval_serial import evaluate_serial
from metarl_iccbf.cruise_control.evaluation.eval_parallel import evaluate_parallel, EvalConfig


def _auto_mat_name(model_path: str, mode: str) -> str:
    """Generate a .mat filename that includes the model name and a timestamp.

    E.g. ``eval_Mamba2TunedICCBF_CruiseControl_20260225_084139_serial_20260225_163012.mat``
    """
    # Extract the training-run name from the model path's parent directory
    model_name = Path(model_path).resolve().parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{model_name}_{mode}_{timestamp}.mat"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a cruise-control policy.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model (best_model.zip or best_model)",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm used for training (default: ppo)",
    )
    parser.add_argument(
        "--policy-type",
        choices=["MLP", "RNN", "GRU", "MAMBA"],
        default="MLP",
        help="Policy architecture (default: MLP)",
    )
    parser.add_argument(
        "--env-type",
        choices=["iccbf", "rl_only"],
        default="iccbf",
        help="Environment type (default: iccbf)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel evaluation (requires --ics-path)",
    )
    parser.add_argument(
        "--ics-path",
        type=str,
        default=None,
        help="Path to initial conditions file (.npy or .npz). Required for --parallel.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step (default: 0.1)",
    )
    parser.add_argument(
        "--tof",
        type=float,
        default=40.0,
        help="Total time-of-flight (default: 40.0)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ResultsEval",
        help="Output directory (default: ResultsEval)",
    )
    parser.add_argument(
        "--out-mat",
        type=str,
        default=None,
        help="Output .mat filename. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting (serial mode only)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (parallel mode only, default: 8)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=32,
        help="Number of chunks for parallel evaluation (default: 32)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,  # Will auto-select based on policy type
        help="Device to use (cuda or cpu). Auto-selects based on policy type if not specified.",
    )
    args = parser.parse_args()

    if args.parallel:
        if args.ics_path is None:
            parser.error("--ics-path is required when using --parallel")

        # Generate default output filename if not provided
        out_mat = args.out_mat or _auto_mat_name(args.model_path, "parallel")

        # Auto-select device based on policy type if not specified
        device = args.device
        if device is None:
            device = "cuda" if args.policy_type == "MAMBA" else "cpu"

        cfg = EvalConfig(
            ics_path=args.ics_path,
            model_path=args.model_path,
            dt=args.dt,
            TOF=args.tof,
            policy_type=args.policy_type,
            env_type=args.env_type,
            algo=args.algo,
            device=device,
            n_workers=args.num_workers,
            n_chunks=args.num_chunks,
            out_dir=args.out_dir,
            out_mat=out_mat,
        )
        evaluate_parallel(cfg)
    else:
        out_mat = args.out_mat or _auto_mat_name(args.model_path, "serial")
        evaluate_serial(
            model_path=args.model_path,
            policy_type=args.policy_type,
            env_type=args.env_type,
            algo=args.algo,
            dt=args.dt,
            TOF=args.tof,
            out_dir=args.out_dir,
            out_mat=out_mat,
            plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
