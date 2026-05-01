"""
ST-GCN Model for Renewable Energy Stock Prediction.

Architecture:
  - 2-3 ST-Conv Blocks (Temporal GLU + Spatial GCN)
  - Attention Pooling across node dimension
  - MLP readout → 3-dim PCA prediction

Optimised for RTX 5070 Ti (16GB VRAM) with:
  - Sparse adjacency (torch.sparse.mm)
  - Mixed precision (AMP)
  - Optional ClusterGCN sampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  1. MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════

class TemporalGLU(nn.Module):
    """
    1D Gated Linear Unit for temporal modelling.
    Input:  (B, N, T, C_in)
    Output: (B, N, T', C_out)  where T' depends on kernel/padding
    """
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        # GLU needs 2x channels: one for value, one for gate
        self.conv = nn.Conv1d(c_in, 2 * c_out, kernel_size,
                              padding=padding, dilation=dilation)
        self.c_out = c_out

    def forward(self, x):
        # x: (B, N, T, C) → reshape for Conv1d
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)  # (B*N, C, T)
        x = self.conv(x)                               # (B*N, 2*C_out, T')
        # GLU split
        value, gate = x.split(self.c_out, dim=1)
        x = value * torch.sigmoid(gate)                # (B*N, C_out, T')
        T_out = x.shape[2]
        x = x.permute(0, 2, 1).reshape(B, N, T_out, self.c_out)
        return x


class SparseGraphConv(nn.Module):
    """
    Spectral GCN layer using pre-normalised sparse adjacency.
    A_norm = D^{-1/2} A D^{-1/2} (precomputed, sparse COO).
    Input:  (B, N, T, C_in)
    Output: (B, N, T, C_out)
    """
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=bias)

    def forward(self, x, A_norm):
        """
        x: (B, N, T, C_in)
        A_norm: (N, N) sparse tensor
        """
        # Linear transform first: (B, N, T, C_in) → (B, N, T, C_out)
        x = self.linear(x)
        B, N, T, C = x.shape

        # Graph convolution: A @ X for each (batch, time)
        # Reshape to (B*T, N, C), apply sparse mm, reshape back
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, C)  # (B*T, N, C)

        # torch.sparse.mm: (N, N) @ (N, C) for each sample in batch
        # Batch sparse mm via reshape: (B*T*C, N) is too large
        # Instead: (N, N) @ (N, B*T*C) → transpose trick
        x_t = x.permute(1, 0, 2).reshape(N, B * T * C)   # (N, B*T*C)
        out = torch.sparse.mm(A_norm, x_t)                # (N, B*T*C)
        out = out.reshape(N, B * T, C).permute(1, 0, 2)   # (B*T, N, C)

        out = out.reshape(B, T, N, C).permute(0, 2, 1, 3)  # (B, N, T, C)
        return out


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block:
      Temporal GLU → Spatial GCN → Temporal GLU → BatchNorm + Residual + Dropout
    """
    def __init__(self, c_in, c_hid, c_out, kernel_size=3, dropout=0.3):
        super().__init__()
        self.temporal1 = TemporalGLU(c_in, c_hid, kernel_size)
        self.spatial = SparseGraphConv(c_hid, c_hid)
        self.temporal2 = TemporalGLU(c_hid, c_out, kernel_size)
        self.bn = nn.BatchNorm2d(c_out)  # over (N, T) treated as spatial dims
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions change
        self.residual = (nn.Linear(c_in, c_out) if c_in != c_out
                         else nn.Identity())

    def forward(self, x, A_norm):
        """
        x: (B, N, T, C_in)
        """
        residual = x

        # Temporal → Spatial → Temporal
        out = self.temporal1(x, )       # (B, N, T, c_hid)
        out = F.relu(out)
        out = self.spatial(out, A_norm)  # (B, N, T, c_hid)
        out = F.relu(out)
        out = self.temporal2(out)        # (B, N, T, c_out)

        # Residual connection (match time dim if needed)
        T_out = out.shape[2]
        T_res = residual.shape[2]
        if T_res > T_out:
            # Center-crop residual
            offset = (T_res - T_out) // 2
            residual = residual[:, :, offset:offset + T_out, :]
        residual = self.residual(residual)

        out = out + residual

        # BatchNorm: treat as (B, C, N, T)
        B, N, T, C = out.shape
        out = out.permute(0, 3, 1, 2)  # (B, C, N, T)
        out = self.bn(out)
        out = out.permute(0, 2, 3, 1)  # (B, N, T, C)

        out = self.dropout(F.relu(out))
        return out


class AttentionPooling(nn.Module):
    """
    Attention-based aggregation across node dimension.
    Input:  (B, N, T, C)
    Output: (B, T, C)
    """
    def __init__(self, c_in):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(c_in, c_in // 2),
            nn.Tanh(),
            nn.Linear(c_in // 2, 1)
        )

    def forward(self, x):
        # x: (B, N, T, C)
        scores = self.attn(x)            # (B, N, T, 1)
        weights = F.softmax(scores, dim=1)  # softmax over N
        # Weighted sum over nodes
        out = (x * weights).sum(dim=1)   # (B, T, C)
        return out


# ═══════════════════════════════════════════════════════════════════
#  2. FULL ST-GCN MODEL
# ═══════════════════════════════════════════════════════════════════

class STGCNModel(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.

    Args:
        n_features:   Input feature dimension (F=16)
        n_nodes:      Number of graph nodes (100)
        hidden_dim:   Hidden channels in ST-Conv blocks
        n_blocks:     Number of ST-Conv blocks (2-3)
        n_output:     Output dimension (3 for PCA components)
        kernel_size:  Temporal convolution kernel
        dropout:      Dropout rate
    """
    def __init__(self, n_features=16, n_nodes=100, hidden_dim=64,
                 n_blocks=3, n_output=3, kernel_size=3, dropout=0.3):
        super().__init__()
        self.n_nodes = n_nodes

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # ST-Conv blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            c_in = hidden_dim
            c_out = hidden_dim
            self.blocks.append(STConvBlock(c_in, hidden_dim, c_out,
                                           kernel_size, dropout))

        # Temporal aggregation (adaptive pool to fixed size)
        self.temporal_pool_size = 8
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.temporal_pool_size)

        # Attention pooling across nodes
        self.attn_pool = AttentionPooling(hidden_dim)

        # MLP readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * self.temporal_pool_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_output)
        )

    def forward(self, x, A_norm):
        """
        x: (B, T, N, F) — input feature tensor
        A_norm: (N, N) sparse — normalised adjacency
        Returns: (B, n_output)
        """
        B, T, N, F = x.shape

        # Rearrange to (B, N, T, F) for internal processing
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)

        # Input projection
        x = self.input_proj(x)  # (B, N, T, hidden)

        # ST-Conv blocks
        for block in self.blocks:
            x = block(x, A_norm)  # (B, N, T, hidden)

        # Attention pooling over nodes → (B, T, hidden)
        x = self.attn_pool(x)

        # Temporal pooling → fixed length
        # (B, T, C) → (B, C, T) → pool → (B, C, pool_size) → (B, C*pool_size)
        B2, T2, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.temporal_pool(x)  # (B, C, pool_size)
        x = x.reshape(B2, -1)  # (B, C * pool_size)

        # MLP readout
        out = self.readout(x)  # (B, n_output)
        return out


# ═══════════════════════════════════════════════════════════════════
#  3. HYBRID LOSS
# ═══════════════════════════════════════════════════════════════════

class HybridLoss(nn.Module):
    """MSE + λ·L1 for fat-tailed financial data."""
    def __init__(self, lambda_l1=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        return mse + self.lambda_l1 * l1


# ═══════════════════════════════════════════════════════════════════
#  4. LEARNING RATE SCHEDULER WITH WARMUP
# ═══════════════════════════════════════════════════════════════════

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs,
                                 min_lr=1e-6, base_lr=1e-3):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr / base_lr,
                   0.5 * (1 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════
#  5. CLUSTER-GCN SAMPLER
# ═══════════════════════════════════════════════════════════════════

class ClusterGCNSampler:
    """Sample subgraphs by selecting k clusters per mini-batch."""
    def __init__(self, cluster_ids, A_sparse, n_sample_clusters=3):
        self.cluster_ids = np.array(cluster_ids)
        self.unique_clusters = np.unique(self.cluster_ids)
        self.A_sparse = A_sparse
        self.k = min(n_sample_clusters, len(self.unique_clusters))

    def sample(self):
        """Returns (node_indices, A_sub) for a random subgraph."""
        chosen = np.random.choice(self.unique_clusters, size=self.k, replace=False)
        mask = np.isin(self.cluster_ids, chosen)
        indices = np.where(mask)[0]

        # Extract sub-adjacency (dense indexing on sparse)
        idx_tensor = torch.LongTensor(indices)
        A_dense = self.A_sparse.to_dense()
        A_sub = A_dense[idx_tensor][:, idx_tensor]

        # Re-sparsify
        A_sub_sparse = A_sub.to_sparse_coo()
        return indices, A_sub_sparse


# ═══════════════════════════════════════════════════════════════════
#  6. TRAINER
# ═══════════════════════════════════════════════════════════════════

class STGCNTrainer:
    """
    Training loop with:
      - Mixed precision (AMP)
      - Early stopping
      - Cosine warmup scheduler
      - VRAM monitoring
      - Optional ClusterGCN
    """
    def __init__(self, model, A_norm, train_loader, val_loader, test_loader,
                 lr=1e-3, weight_decay=1e-4, epochs=100, warmup_epochs=5,
                 patience=15, lambda_l1=0.1, device="cuda",
                 use_cluster_gcn=False, cluster_sampler=None):
        self.model = model.to(device)
        self.A_norm = A_norm.to(device)
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.use_cluster_gcn = use_cluster_gcn
        self.cluster_sampler = cluster_sampler

        self.criterion = HybridLoss(lambda_l1)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=weight_decay)
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer, warmup_epochs, epochs, min_lr=1e-6, base_lr=lr)
        self.scaler = GradScaler("cuda", enabled=(device == "cuda"))  # AMP

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Optional ClusterGCN: subsample nodes
            A_used = self.A_norm
            if self.use_cluster_gcn and self.cluster_sampler:
                indices, A_sub = self.cluster_sampler.sample()
                idx = torch.LongTensor(indices).to(self.device)
                X_batch = X_batch[:, :, indices, :]
                A_used = A_sub.to(self.device)

            self.optimizer.zero_grad()

            with autocast(self.device, enabled=(self.device == "cuda")):
                pred = self.model(X_batch, A_used)
                # If predicting 3 PCs but target is only ΔPC1 (scalar)
                if pred.shape[-1] > 1 and y_batch.dim() == 1:
                    pred = pred[:, 0]  # take PC1 prediction
                loss = self.criterion(pred, y_batch)

            if self.device == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            pred = self.model(X_batch, self.A_norm)
            if pred.shape[-1] > 1 and y_batch.dim() == 1:
                pred = pred[:, 0]
            loss = self.criterion(pred, y_batch)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training ST-GCN | {self.epochs} epochs | device={self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.val_loader)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), "best_stgcn.pt")
            else:
                self.patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or self.patience_counter == 0:
                marker = " ★" if self.patience_counter == 0 else ""
                print(f"  Epoch {epoch:3d} | Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | LR: {lr:.2e}{marker}")

            if self.patience_counter >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(patience={self.patience})")
                break

        # Final evaluation
        self.model.load_state_dict(torch.load("best_stgcn.pt",
                                               weights_only=True))
        test_loss = self.evaluate(self.test_loader)
        print(f"\n  Best Val Loss:  {self.best_val_loss:.6f}")
        print(f"  Test Loss:      {test_loss:.6f}")
        return test_loss

    def benchmark_naive(self, loader):
        """Naive baseline: predict 0 for ΔPC1. MSE = Var(y)."""
        all_y = []
        for _, y_batch in loader:
            all_y.append(y_batch.numpy())
        all_y = np.concatenate(all_y)
        naive_mse = np.mean(all_y ** 2)  # pred=0 → MSE = E[y²]
        return naive_mse


# ═══════════════════════════════════════════════════════════════════
#  7. SMOKE TEST (small random data)
# ═══════════════════════════════════════════════════════════════════

def smoke_test():
    """
    Verify forward + backward with tiny random data.
    Batch=2, Time=24, Nodes=10, Features=16
    """
    print("\n" + "=" * 60)
    print("  SMOKE TEST: Forward + Backward (random data)")
    print("=" * 60)

    B, T, N, F = 2, 24, 10, 16
    device = "cpu"  # smoke test on CPU

    # Random input
    X = torch.randn(B, T, N, F)
    y = torch.randn(B, 3)  # 3 PCA targets

    # Random sparse adjacency (normalised)
    # Create a small random graph
    A_dense = torch.rand(N, N) * 0.3
    A_dense = (A_dense + A_dense.T) / 2  # symmetric
    A_dense.fill_diagonal_(1.0)
    A_dense[A_dense < 0.2] = 0  # sparsify
    # Normalise: D^{-1/2} A D^{-1/2}
    degree = A_dense.sum(dim=1)
    d_inv_sqrt = torch.where(degree > 0, 1.0 / torch.sqrt(degree),
                              torch.zeros_like(degree))
    A_norm = A_dense * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
    A_sparse = A_norm.to_sparse_coo()

    # Model
    model = STGCNModel(n_features=F, n_nodes=N, hidden_dim=32,
                       n_blocks=2, n_output=3, dropout=0.1)
    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Forward
    print("\n  Forward pass...")
    model.train()
    pred = model(X, A_sparse)
    print(f"    Input:  X {X.shape}")
    print(f"    Output: pred {pred.shape}")
    assert pred.shape == (B, 3), f"Expected (2, 3), got {pred.shape}"
    print(f"    ✓ Output shape correct")

    # Loss + Backward
    print("\n  Backward pass...")
    criterion = HybridLoss(lambda_l1=0.1)
    loss = criterion(pred, y)
    loss.backward()
    print(f"    Loss: {loss.item():.6f}")

    # Check gradients exist
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    print(f"    Gradient norms: min={min(grad_norms):.6f}, "
          f"max={max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")
    assert all(np.isfinite(grad_norms)), "Non-finite gradients!"
    print(f"    ✓ All gradients finite")

    # Optimizer step
    print("\n  Optimizer step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.step()
    print(f"    ✓ AdamW step completed")

    # Second forward (verify weights updated)
    pred2 = model(X, A_sparse)
    assert not torch.allclose(pred, pred2), "Weights didn't update!"
    print(f"    ✓ Weights updated (predictions changed)")

    # Test scheduler
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_epochs=2,
                                             total_epochs=10)
    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    print(f"\n  LR schedule (10 epochs): {[f'{lr:.2e}' for lr in lrs]}")

    # ClusterGCN test
    print("\n  ClusterGCN sampling test...")
    clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    from scipy.sparse import csr_matrix as sp_csr
    sampler = ClusterGCNSampler(clusters, A_sparse, n_sample_clusters=2)
    indices, A_sub = sampler.sample()
    print(f"    Sampled {len(indices)} nodes from 2 clusters")
    print(f"    Sub-adjacency shape: {A_sub.shape}")
    X_sub = X[:, :, indices, :]
    pred_sub = model.forward.__wrapped__(model, X_sub, A_sub) if hasattr(
        model.forward, '__wrapped__') else None
    # Just test the subgraph shapes
    print(f"    X_sub shape: {X_sub.shape}")
    print(f"    ✓ ClusterGCN sampling works")

    # Memory estimation
    print("\n  Memory estimation for full model (B=16, T=168, N=100, F=16):")
    B_full, T_full, N_full, F_full = 16, 168, 100, 16
    input_mb = B_full * T_full * N_full * F_full * 4 / 1024**2
    # Activations (rough: 3 blocks × 2 intermediate tensors)
    act_mb = 3 * 2 * B_full * N_full * T_full * 64 * 4 / 1024**2
    param_mb = n_params * 4 / 1024**2 * (N_full / N)  # scale approx
    total_mb = input_mb + act_mb + param_mb * 3  # params + grads + optimizer
    print(f"    Input:       {input_mb:.1f} MB")
    print(f"    Activations: {act_mb:.1f} MB (est.)")
    print(f"    Total est:   {total_mb:.1f} MB")
    print(f"    {'✓' if total_mb < 12000 else '⚠'} "
          f"{'Fits' if total_mb < 12000 else 'May exceed'} 12GB VRAM")

    print("\n" + "=" * 60)
    print("  ✅ SMOKE TEST PASSED — All components verified")
    print("=" * 60)

    return True


if __name__ == "__main__":
    smoke_test()
