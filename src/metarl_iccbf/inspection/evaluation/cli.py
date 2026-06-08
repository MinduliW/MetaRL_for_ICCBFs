"""CLI entry point for inspection evaluation.

Usage::

    # Serial evaluation
    metarl-eval-inspection --model-path path/to/best_model.zip --policy-type RNN

    # Parallel evaluation (requires episode bank)
    metarl-eval-inspection --model-path path/to/best_model.zip --policy-type MAMBA \\
        --parallel --ics-path episode_bank.npz --device cuda
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from metarl_iccbf.inspection.evaluation.eval_serial import evaluate_serial
from metarl_iccbf.inspection.evaluation.eval_parallel import (
    EvalConfig,
    evaluate_parallel,
)


def _auto_mat_name(model_path: str, mode: str) -> str:
    """Generate a .mat filename that includes the model name and a timestamp."""
    model_name = Path(model_path).resolve().parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{model_name}_{mode}_{timestamp}.mat"


def main():
    parser = argparse.ArgumentParser(description="Evaluate an inspection policy.")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model (.zip)",
    )
    parser.add_argument(
        "--policy-type",
        choices=["MLP", "RNN", "GRU", "LSTM", "MAMBA"],
        default="MLP",
        help="Policy architecture (default: MLP)",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm used for training (default: ppo)",
    )
    parser.add_argument(
        "--env-type",
        choices=["iccbf", "rl_only"],
        default="iccbf",
        help="'iccbf' for CBF tuning, 'rl_only' for thrust-only. Default: iccbf",
    )

    # Serial-specific
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Number of episodes for serial eval (default: 10)",
    )

    # Parallel-specific
    parser.add_argument("--parallel", action="store_true", help="Use parallel evaluation")
    parser.add_argument(
        "--ics-path", type=str, default=None,
        help="Path to .npz episode bank (required for --parallel)",
    )
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Randomly sample this many episodes from the bank (default: all)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="RNG seed for episode subsampling (default: 42)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=16)
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip evaluation, just merge existing chunk files")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)

    # Shared
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--tof", type=float, default=3.4 * 3600.0, help="Time of flight in seconds")
    parser.add_argument("--dv-weight", type=float, default=10.0)
    parser.add_argument("--no-param-randomisation", action="store_true")
    parser.add_argument("--enable-noise", action="store_true")
    parser.add_argument("--out-dir", type=str, default="ResultsEval/inspection")
    parser.add_argument("--out-mat", type=str, default=None)
    parser.add_argument("--adversarial", action="store_true",
                        help="Enable adversarial chief perturbation during eval")
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()

    if args.parallel:
        if args.ics_path is None:
            parser.error("--ics-path is required when using --parallel")

        device = args.device or ("cuda" if args.policy_type == "MAMBA" else "cpu")
        out_mat = args.out_mat or _auto_mat_name(args.model_path, "parallel")

        cfg = EvalConfig(
            ics_path=args.ics_path,
            model_path=args.model_path,
            policy_type=args.policy_type,
            algo=args.algo,
            dt=args.dt,
            TOF=args.tof,
            enableCBFtunning=(args.env_type == "iccbf"),
            num_episodes=args.num_episodes,
            sample_seed=args.sample_seed,
            dvWeight=args.dv_weight,
            enable_param_randomisation=not args.no_param_randomisation,
            enableNoise=args.enable_noise,
            adversarial=args.adversarial,
            device=device,
            n_workers=args.num_workers,
            n_chunks=args.num_chunks,
            out_dir=args.out_dir,
            out_mat=out_mat,
            merge_only=args.merge_only,
        )
        evaluate_parallel(cfg)
    else:
        out_mat = args.out_mat or _auto_mat_name(args.model_path, "serial")

        evaluate_serial(
            args.model_path,
            policy_type=args.policy_type,
            algo=args.algo,
            env_type=args.env_type,
            dt=args.dt,
            n_episodes=args.n_episodes,
            enable_param_randomisation=not args.no_param_randomisation,
            enableNoise=args.enable_noise,
            dvWeight=args.dv_weight,
            out_mat=out_mat,
            out_dir=args.out_dir,
            plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
