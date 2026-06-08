"""CLI entry point for inspection training.

Usage::

    metarl-train-inspection --policy-type MAMBA --num-env 28 --total-episodes 100000
    metarl-train-inspection --policy-type RNN --num-env 64
    metarl-train-inspection --policy-type MLP --env-type rl_only --num-env 64
"""

from __future__ import annotations

import argparse

from metarl_iccbf.inspection.training.train import train_inspection


def main():
    parser = argparse.ArgumentParser(description="Train an inspection policy.")
    parser.add_argument(
        "--policy-type",
        choices=["MLP", "RNN", "GRU", "MAMBA", "LSTM"],
        default="RNN",
        help="Policy architecture (default: RNN)",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm (default: ppo)",
    )
    parser.add_argument(
        "--env-type",
        choices=["iccbf", "rl_only"],
        default="iccbf",
        help="'iccbf' for CBF tuning (12D action), 'rl_only' for thrust-only (3D action). Default: iccbf",
    )
    parser.add_argument("--num-env", type=int, default=64, help="Parallel environments (default: 64)")
    parser.add_argument("--total-episodes", type=int, default=100_000, help="Total training episodes (default: 100000)")
    parser.add_argument("--dt", type=float, default=10.0, help="Timestep in seconds (default: 10.0)")
    parser.add_argument("--dv-weight", type=float, default=10.0, help="Fuel cost weight (default: 10.0)")
    parser.add_argument("--no-param-randomisation", action="store_true", help="Disable episodic parameter randomisation")
    parser.add_argument("--enable-noise", action="store_true", help="Enable actuation/measurement noise")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name. Omit to disable.")
    parser.add_argument("--training-name", type=str, default=None, help="Custom run name. Auto-generated if omitted.")
    parser.add_argument("--load", type=str, nargs="?", const=True, help="Resume from best_model.zip in run directory, or from a custom path if provided.")
    parser.add_argument("--wandb-resume-id", type=str, default=None, help="W&B run ID to resume (e.g. 'qx46k9ol'). Requires --wandb-project.")
    parser.add_argument("--adversarial", action="store_true", help="Enable passive adversarial chief (hidden ΔV budget).")
    parser.add_argument("--dv-budget-min", type=float, default=0.5, help="Min adversarial ΔV budget in m/s (default: 0.5)")
    parser.add_argument("--dv-budget-max", type=float, default=5.0, help="Max adversarial ΔV budget in m/s (default: 5.0)")


    parser.add_argument("--morl", action="store_true", help="Enable MORL")
    parser.add_argument("--morl-arch", choices=["concat", "multi_body"], default="concat")
    parser.add_argument("--morl-n-objectives", type=int, default=2, help="Number of MORL objectives (default: 2)")
    parser.add_argument("--morl-fixed-obj", type=int, default=0, help="Objective index with fixed weight: 0=coverage, 1=fuel/time, 2=safety (default: 0)")
    parser.add_argument("--morl-w-fixed", type=float, default=0.6, help="Fixed weight for --morl-fixed-obj (default: 0.6)")
    parser.add_argument(
        "--morl-objective-set",
        choices=["fuel", "time"],
        default="fuel",
        help="Objective vector contents: 'fuel' = [coverage, -fuel_norm, safety] (default), 'time' = [coverage, -1/MAX_STEPS, safety]",
    )
    parser.add_argument(
        "--fixed-ic",
        action="store_true",
        help="Pin the initial state to [95,0,0,0,0,0,π/2] (near outer init range). Reduces Pareto-eval variance.",
    )
    parser.add_argument("--mdmm-entropy", action="store_true")
    parser.add_argument("--mdmm-H-target", type=float, default=1.0)
    parser.add_argument("--mdmm-H-decay", type=float, default=0.0)
    parser.add_argument("--popart", action="store_true")
    parser.add_argument("--popart-beta", type=float, default=3e-4)

    # SAC-specific args
    parser.add_argument(
        "--target-entropy", type=str, default=None,
        help='SAC target entropy: "auto" or float e.g. "-6.0" (default: auto)',
    )
    parser.add_argument(
        "--ent-coef-min", type=float, default=None,
        help="SAC ent_coef floor (default: from config = 0.05)",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=None,
        help="SAC random exploration steps before training (default: from config)",
    )
    parser.add_argument(
        "--gradient-steps", type=int, default=None,
        help="SAC gradient steps per update (default: from config)",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None,
        help="SAC replay buffer capacity (default: from config)",
    )
    parser.add_argument(
        "--chunk-len", type=int, default=None,
        help="SAC training sequence length (default: from config)",
    )
    parser.add_argument(
        "--train-freq", type=int, default=None,
        help="SAC env steps between gradient updates (default: from config)",
    )

    args = parser.parse_args()

    if args.fixed_ic:
        import math
        import numpy as np
        fixed_ic_arr = np.array([95.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0], dtype=np.float64)
    else:
        fixed_ic_arr = None

    train_inspection(
        policy_type=args.policy_type,
        algo=args.algo,
        env_type=args.env_type,
        num_env=args.num_env,
        total_episodes=args.total_episodes,
        dt=args.dt,
        dvWeight=args.dv_weight,
        enable_param_randomisation=not args.no_param_randomisation,
        enableNoise=args.enable_noise,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_resume_id=args.wandb_resume_id,
        training_name=args.training_name,
        trainLoad=args.load,
        adversarial=args.adversarial,
        morl=args.morl,
        morl_arch=args.morl_arch,
        morl_n_objectives=args.morl_n_objectives,
        morl_fixed_obj=args.morl_fixed_obj,
        morl_w_fixed=args.morl_w_fixed,
        morl_objective_set=args.morl_objective_set,
        fixed_ic=fixed_ic_arr,
        mdmm_entropy=args.mdmm_entropy,
        mdmm_H_target=args.mdmm_H_target,
        mdmm_H_decay=args.mdmm_H_decay,
        popart=args.popart,
        popart_beta=args.popart_beta,

        dv_budget_range=(args.dv_budget_min, args.dv_budget_max),
        target_entropy=args.target_entropy,
        ent_coef_min=args.ent_coef_min,
        learning_starts=args.learning_starts,
        gradient_steps=args.gradient_steps,
        buffer_size=args.buffer_size,
        chunk_len=args.chunk_len,
        train_freq=args.train_freq,
    )


if __name__ == "__main__":
    main()
