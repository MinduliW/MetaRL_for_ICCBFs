"""CLI entry point for docking Mamba2 training.

Usage::

    python -m metarl_iccbf.docking.training.cli --num-env 28 --total-episodes 200000
"""

from metarl_iccbf.docking.training.train import main

if __name__ == "__main__":
    main()
