"""
Subject-Independent Evaluation — Leave-One-Subject-Out Cross-Validation
========================================================================
This is now a thin wrapper around train_efficient.py's LOSO mode.

Usage:
    python subject_independent_eval.py
    python subject_independent_eval.py --data_dir /path/to/data/

Or use train_efficient.py directly:
    python train_efficient.py --mode loso

Paper reference: Subject-Independent accuracy = 74.48%
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_efficient import (
    DEFAULT_CONFIG,
    get_device,
    run_loso,
    seed_everything,
)


def main():
    import argparse
    p = argparse.ArgumentParser(description="LOSO Subject-Independent Evaluation")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=32)
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["loso_epochs"] = args.epochs
    cfg["seed"] = args.seed
    if args.data_dir:
        cfg["data_dir"] = args.data_dir

    device = get_device()
    seed_everything(cfg["seed"])
    run_loso(cfg, device)


if __name__ == "__main__":
    main()
