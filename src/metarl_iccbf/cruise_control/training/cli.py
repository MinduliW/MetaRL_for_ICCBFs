"""Simple CLI for cruise-control training."""

import argparse

from metarl_iccbf.cruise_control.training.train import train_cruise_control


def main():
    parser = argparse.ArgumentParser(description="Train a cruise-control policy.")
    parser.add_argument(
        "--policy-type",
        choices=["MLP", "RNN", "MAMBA"],
        default="RNN",
        help="Policy architecture (default: RNN)",
    )
    parser.add_argument(
        "--num-env",
        type=int,
        default=32,
        help="Number of parallel environments (default: 32)",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=100_000,
        help="Total training episodes (default: 100000)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name. Omit to disable wandb logging.",
    )
    args = parser.parse_args()

    train_cruise_control(
        policy_type=args.policy_type,
        num_env=args.num_env,
        total_episodes=args.total_episodes,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
