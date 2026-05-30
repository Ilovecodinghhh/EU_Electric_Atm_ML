"""
Run repeatable ST-GCN seed/config experiments.

Default behavior is a dry print of commands. Add --execute to run them.
"""

import argparse
import os
import subprocess
import sys


SEEDS = [11, 22, 33, 44, 55]
CONFIGS = {
    "compact_h32_b2": {"hidden_dim": 32, "n_blocks": 2, "dropout": 0.3},
    "full_h64_b3": {"hidden_dim": 64, "n_blocks": 3, "dropout": 0.3},
}


def command_for(seed, name, cfg, args):
    output_dir = os.path.join(args.output_dir, f"{name}_seed{seed}")
    return [
        sys.executable,
        "train_stgcn.py",
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--n_blocks", str(cfg["n_blocks"]),
        "--dropout", str(cfg["dropout"]),
        "--seed", str(seed),
        "--output_dir", output_dir,
    ]


def main():
    parser = argparse.ArgumentParser(description="Run ST-GCN seed/config sweep")
    parser.add_argument("--output_dir", default=os.path.join("quant_output", "stgcn_sweeps"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for config_name, cfg in CONFIGS.items():
        for seed in SEEDS:
            cmd = command_for(seed, config_name, cfg, args)
            print(" ".join(cmd))
            if args.execute:
                env = os.environ.copy()
                env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
                env.setdefault("PYTHONIOENCODING", "utf-8")
                subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
