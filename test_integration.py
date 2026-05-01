"""
Integration test: verify full pipeline with real data but tiny window (T=24).
Tests forward + backward + loss on actual preprocessed tensors.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from stgcn_model import STGCNModel, HybridLoss, get_cosine_warmup_scheduler

TENSOR_DIR = "tensor_output"

# Load metadata
with open(os.path.join(TENSOR_DIR, "tensor_meta.json")) as f:
    meta = json.load(f)
T_total, N, F = meta["shape"]

# Load real tensors
tensor = np.memmap(os.path.join(TENSOR_DIR, "feature_tensor.npy"),
                   dtype=np.float32, mode="r", shape=(T_total, N, F))
target = np.load(os.path.join(TENSOR_DIR, "target_vector.npy"))
A_norm = torch.load("adjacency_matrix.pt", weights_only=False)

print(f"Tensor: ({T_total}, {N}, {F})")
print(f"Adjacency: {A_norm.shape}, nnz={A_norm._nnz()}")

# Tiny dataset: just 5 windows of size 24 from beginning of data
TINY_WINDOW = 24
TINY_HORIZON = 24
samples = []
for start in range(0, 300, 60):
    y_idx = start + TINY_WINDOW - 1 + TINY_HORIZON
    if y_idx < T_total and np.isfinite(target[y_idx]):
        X = np.array(tensor[start:start + TINY_WINDOW], dtype=np.float32)
        y = target[y_idx]
        samples.append((torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)))

print(f"Tiny samples: {len(samples)}")

# Stack into batch
X_batch = torch.stack([s[0] for s in samples])  # (B, 24, 100, 16)
y_batch = torch.stack([s[1] for s in samples])   # (B,)
print(f"X: {X_batch.shape}, y: {y_batch.shape}")

# Model
model = STGCNModel(n_features=F, n_nodes=N, hidden_dim=32,
                   n_blocks=2, n_output=3, dropout=0.1)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# Forward
model.train()
pred = model(X_batch, A_norm)
print(f"Pred shape: {pred.shape}")

# Loss
criterion = HybridLoss(0.1)
loss = criterion(pred[:, 0], y_batch)  # PC1 only
print(f"Loss: {loss.item():.6f}")

# Backward
loss.backward()
grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
print(f"Gradient norms — min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}")
assert all(np.isfinite(grad_norms)), "Non-finite gradients!"

# Optimizer step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer.step()

# Verify weights changed
pred2 = model(X_batch, A_norm)
assert not torch.allclose(pred, pred2), "Weights didn't update"

# Quick 3-step train loop
print("\nMini training loop (3 steps)...")
for step in range(3):
    optimizer.zero_grad()
    p = model(X_batch, A_norm)
    l = criterion(p[:, 0], y_batch)
    l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"  Step {step+1}: loss={l.item():.6f}")

# Verify feature ranges from real data
print(f"\nFeature stats from real data (first window):")
x0 = X_batch[0]  # (24, 100, 16)
feat_names = ["ssr", "t2m", "u100", "v100", "price",
              "pc1", "pc2", "pc3", "is_mkt", "cap",
              "lat", "lon", "sin_h", "cos_h", "sin_m", "cos_m"]
for i, name in enumerate(feat_names):
    vals = x0[:, :, i]
    print(f"  [{i:2d}] {name:8s}  mean={vals.mean():+7.3f}  "
          f"std={vals.std():.3f}  range=[{vals.min():.3f}, {vals.max():.3f}]")

print("\n✅ Integration test PASSED — real data, real model, real gradients")
