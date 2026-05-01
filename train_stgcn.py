"""
ST-GCN Training Script.

Loads preprocessed tensors from build_tensor.py, builds DataLoaders,
trains the ST-GCN model, and evaluates against naive baseline.

Usage:
    python train_stgcn.py [--device cuda] [--epochs 100] [--batch_size 16]
                          [--cluster_gcn] [--dry_run]
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from stgcn_model import STGCNModel, HybridLoss, STGCNTrainer, ClusterGCNSampler

# ─── Config ─────────────────────────────────────────────────────────
TENSOR_DIR = "tensor_output"
WINDOW_SIZE = 168
HORIZON = 24
STRIDE = 6
N_FEATURES = 16
N_NODES = 100
# ────────────────────────────────────────────────────────────────────


class STGCNWindowDataset(Dataset):
    """Sliding window dataset reading from memmap."""
    def __init__(self, tensor_memmap, target, start_idx, end_idx,
                 window_size=168, horizon=24, stride=6):
        self.tensor = tensor_memmap
        self.target = target
        self.window_size = window_size
        self.horizon = horizon

        max_start = end_idx - window_size - horizon + 1
        min_start = max(start_idx, 0)
        self.indices = []
        for i in range(min_start, max_start + 1, stride):
            y_idx = i + window_size - 1 + horizon
            if y_idx < len(target) and np.isfinite(target[y_idx]):
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        X = np.array(self.tensor[t:t + self.window_size], dtype=np.float32)
        y_idx = t + self.window_size - 1 + self.horizon
        y = self.target[y_idx]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--cluster_gcn", action="store_true",
                        help="Enable ClusterGCN subgraph sampling")
    parser.add_argument("--n_sample_clusters", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 2 epochs with small batches for testing")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # ── Load metadata ──────────────────────────────────────────────
    with open(os.path.join(TENSOR_DIR, "tensor_meta.json")) as f:
        meta = json.load(f)
    T_total, N, F = meta["shape"]
    print(f"Tensor: ({T_total}, {N}, {F})")

    # ── Load tensors ───────────────────────────────────────────────
    print("Loading feature tensor (memmap)...")
    tensor = np.memmap(os.path.join(TENSOR_DIR, "feature_tensor.npy"),
                       dtype=np.float32, mode="r",
                       shape=(T_total, N, F))

    target = np.load(os.path.join(TENSOR_DIR, "target_vector.npy"))
    print(f"Target: {target.shape}, valid: {np.isfinite(target).sum()}")

    # ── Load adjacency ─────────────────────────────────────────────
    A_norm = torch.load("adjacency_matrix.pt", weights_only=False)
    print(f"Adjacency: {A_norm.shape}, nnz={A_norm._nnz()}")

    # ── Split indices ──────────────────────────────────────────────
    train_end_idx = meta["n_train"] - 1
    val_start_idx = meta["n_train"]
    val_end_idx = meta["n_train"] + meta["n_val"] - 1
    test_start_idx = meta["n_train"] + meta["n_val"]

    # ── Build Datasets ─────────────────────────────────────────────
    train_ds = STGCNWindowDataset(tensor, target, 0, train_end_idx,
                                  WINDOW_SIZE, HORIZON, STRIDE)
    val_ds = STGCNWindowDataset(tensor, target, val_start_idx, val_end_idx,
                                WINDOW_SIZE, HORIZON, STRIDE)
    test_ds = STGCNWindowDataset(tensor, target, test_start_idx, T_total - 1,
                                 WINDOW_SIZE, HORIZON, STRIDE)

    print(f"\nDatasets — Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Test: {len(test_ds)}")

    bs = 4 if args.dry_run else args.batch_size
    epochs = 2 if args.dry_run else args.epochs

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=2, pin_memory=(args.device == "cuda"),
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                             num_workers=2, pin_memory=(args.device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=(args.device == "cuda"))

    # ── Model ──────────────────────────────────────────────────────
    model = STGCNModel(
        n_features=F,
        n_nodes=N,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        n_output=3,
        dropout=args.dropout
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    # ── ClusterGCN (optional) ──────────────────────────────────────
    cluster_sampler = None
    if args.cluster_gcn:
        import pandas as pd
        nodes = pd.read_csv("processed_nodes.csv")
        cluster_sampler = ClusterGCNSampler(
            nodes["physical_cluster"].values, A_norm,
            n_sample_clusters=args.n_sample_clusters
        )
        print(f"ClusterGCN enabled: sampling {args.n_sample_clusters} clusters/batch")

    # ── Trainer ────────────────────────────────────────────────────
    trainer = STGCNTrainer(
        model=model,
        A_norm=A_norm,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=1e-4,
        epochs=epochs,
        warmup_epochs=5,
        patience=args.patience,
        lambda_l1=0.1,
        device=args.device,
        use_cluster_gcn=args.cluster_gcn,
        cluster_sampler=cluster_sampler
    )

    # ── Naive Baseline ─────────────────────────────────────────────
    naive_mse = trainer.benchmark_naive(test_loader)
    print(f"\nNaive Baseline (predict 0): MSE = {naive_mse:.6f}")

    # ── Train ──────────────────────────────────────────────────────
    test_loss = trainer.train()

    # ── Compare ────────────────────────────────────────────────────
    improvement = (1 - test_loss / naive_mse) * 100
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Naive MSE:     {naive_mse:.6f}")
    print(f"  Model Loss:    {test_loss:.6f}")
    print(f"  Improvement:   {improvement:+.2f}%")
    if improvement > 5:
        print(f"  ✓ Model beats naive by >{5}%!")
    else:
        print(f"  ✗ Model does not beat naive by 5% (needs more tuning)")
    print(f"{'='*60}")

    # Save results
    results = {
        "naive_mse": float(naive_mse),
        "test_loss": float(test_loss),
        "improvement_pct": float(improvement),
        "best_val_loss": float(trainer.best_val_loss),
        "epochs_run": len(trainer.history["train_loss"]),
        "n_params": n_params,
        "config": vars(args)
    }
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to training_results.json")


if __name__ == "__main__":
    main()
