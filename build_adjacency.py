"""
Build Sparse Adjacency Matrix for ST-GCN from clustered wind farm nodes.

Produces:
  - adjacency_matrix.npz      (scipy sparse CSR)
  - adjacency_matrix.pt       (PyTorch sparse COO tensor)
  - adjacency_heatmap.png     (sparsity pattern visualisation)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Parameters ─────────────────────────────────────────────────────
DIST_THRESHOLD = 150.0   # km — inter-cluster connection cutoff
SIGMA = 50.0             # km — Gaussian decay parameter
SPARSIFY_THRESHOLD = 0.1 # weights below this → 0
EARTH_RADIUS_KM = 6371.0
# ────────────────────────────────────────────────────────────────────

# 1. Load processed nodes
df = pd.read_csv("processed_nodes.csv")
N = len(df)
print(f"Nodes: {N}")

coords = df[["latitude", "longitude"]].values
clusters = df["physical_cluster"].values

# 2. Haversine distance matrix (all pairs)
def haversine_matrix(coords_deg):
    """Compute NxN haversine distance matrix in km."""
    coords_rad = np.radians(coords_deg)
    lat = coords_rad[:, 0]
    lon = coords_rad[:, 1]
    
    # Pairwise differences
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c

print("Computing haversine distance matrix...")
dist_matrix = haversine_matrix(coords)

# 3. Build adjacency matrix
print("Building adjacency matrix...")
A = np.zeros((N, N), dtype=np.float32)

# Same-cluster mask
cluster_matrix = clusters[:, None] == clusters[None, :]

# Rule 1 (Intra-cluster): w_ij = 1.0
A[cluster_matrix] = 1.0

# Rule 2 (Inter-cluster, within threshold): w_ij = exp(-d/sigma)
inter_mask = (~cluster_matrix) & (dist_matrix < DIST_THRESHOLD)
A[inter_mask] = np.exp(-dist_matrix[inter_mask] / SIGMA).astype(np.float32)

# Rule 3 (Self-loop): w_ii = 1.0
np.fill_diagonal(A, 1.0)

# Sparsify: remove tiny weights
A[A < SPARSIFY_THRESHOLD] = 0.0

# 4. Check density
nnz = np.count_nonzero(A)
density = nnz / (N * N)
print(f"\n{'='*55}")
print(f"  Matrix shape   : {N} × {N}")
print(f"  Non-zero elems : {nnz}")
print(f"  Density        : {density*100:.2f}%")
print(f"{'='*55}")

if density > 0.10:
    print(f"  ⚠ Density {density*100:.2f}% > 10%! dist_threshold={DIST_THRESHOLD} km is too large.")
    print(f"    Consider reducing to ~100 km or less.")
else:
    print(f"  ✓ Density {density*100:.2f}% ≤ 10% — good for 5070 Ti VRAM.")

# 5. Laplacian normalisation: A_norm = D^{-1/2} A D^{-1/2}
print("\nApplying symmetric normalisation (Laplacian)...")
degree = np.array(A.sum(axis=1)).flatten()
# Numerical stability: avoid divide-by-zero
d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0).astype(np.float32)

# D^{-1/2} * A * D^{-1/2}
A_norm = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

# 6. Convert to sparse CSR
A_sparse = csr_matrix(A_norm)
print(f"  Sparse CSR: {A_sparse.nnz} stored values, "
      f"{A_sparse.data.nbytes / 1024:.1f} KB data")

# Save scipy sparse
from scipy.sparse import save_npz
save_npz("adjacency_matrix.npz", A_sparse)
print("  ✓ Saved adjacency_matrix.npz")

# 7. Convert to PyTorch sparse COO tensor
try:
    import torch
    coo = A_sparse.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    A_torch = torch.sparse_coo_tensor(indices, values, size=(N, N))
    torch.save(A_torch, "adjacency_matrix.pt")
    print("  ✓ Saved adjacency_matrix.pt (PyTorch sparse COO)")
except ImportError:
    print("  ⚠ PyTorch not installed — skipped .pt export.")
    print("    Install with: pip install torch")

# 8. Visualisation — sparsity heatmap
print("\nGenerating heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sort nodes by cluster for block-diagonal visibility
sort_idx = np.argsort(clusters)
A_sorted = A_norm[sort_idx][:, sort_idx]

# Left: raw sparsity pattern
ax = axes[0]
ax.spy(csr_matrix(A_sorted), markersize=0.8, color="navy", alpha=0.6)
ax.set_title(f"Sparsity Pattern (density={density*100:.2f}%)", fontsize=11)
ax.set_xlabel("Node index (sorted by cluster)")
ax.set_ylabel("Node index (sorted by cluster)")

# Right: weighted heatmap
ax = axes[1]
im = ax.imshow(A_sorted, cmap="hot_r", aspect="equal", interpolation="nearest")
ax.set_title("Adjacency Weights (normalised)", fontsize=11)
ax.set_xlabel("Node index (sorted by cluster)")
ax.set_ylabel("Node index (sorted by cluster)")
plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle(f"ST-GCN Adjacency Matrix — {N} nodes, {nnz} edges, "
             f"threshold={DIST_THRESHOLD} km", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("adjacency_heatmap.png", dpi=150)
print("✓ Saved adjacency_heatmap.png")
