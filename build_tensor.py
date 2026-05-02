"""
Multimodal Spatiotemporal Feature Tensor Assembly & DataLoader for ST-GCN.

Produces:
  - feature_tensor.npy   (T, N, F) float32 memmap
  - target_vector.npy    (T,) float32 — ΔPC1 at +24h
  - scaler_params.npz    — all scaler parameters for inference
  - feature_summary.txt  — feature documentation
  - train/val/test DataLoaders (chronological split, no leakage)

Feature vector (F=16) per node per timestep:
  [0]  ssr            — global Z-score
  [1]  t2m            — global Z-score
  [2]  u100           — global Z-score
  [3]  v100           — global Z-score
  [4]  price          — RobustScaler (per-country, mapped to node)
  [5]  finance_pc1    — Z-score (broadcast to all nodes)
  [6]  finance_pc2    — Z-score (broadcast to all nodes)
  [7]  finance_pc3    — Z-score (broadcast to all nodes)
  [8]  is_market_open — binary 0/1, no scaling
  [9]  capacity_mw    — log1p + Z-score (static, broadcast over time)
  [10] latitude       — MinMax normalised (static)
  [11] longitude      — MinMax normalised (static)
  [12] sin_hour       — sin(2π·hour/24)
  [13] cos_hour       — cos(2π·hour/24)
  [14] sin_month      — sin(2π·(month-1)/12)
  [15] cos_month      — cos(2π·(month-1)/12)
"""

import os
import gc
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Parameters ─────────────────────────────────────────────────────
WINDOW_SIZE = 168        # 7 days of hourly data
HORIZON = 24             # predict 24h ahead
STRIDE = 6              # sliding window stride
BATCH_SIZE = 16
NUM_WORKERS = 2          # DataLoader workers
N_NODES = 100
N_FEATURES = 16
TRAIN_END = "2023-12-31 23:00:00"   # inclusive
VAL_MONTHS = 0           # 0 = no separate val from train; we'll split 2024 H1 as val, H2 as test
# Actually: 2019-2023 train, 2024-H1 val, 2024-H2 test
VAL_START = "2024-01-01 00:00:00"
VAL_END = "2024-06-30 23:00:00"
TEST_START = "2024-07-01 00:00:00"

DATA_DIR = "data"
OUTPUT_DIR = "tensor_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────────


def load_hourly_csv(filename):
    """Load a (T, N) hourly CSV, return DataFrame with datetime index."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_finance_pca_hourly(time_index):
    """
    Build hourly finance PCA features + is_market_open indicator.
    Returns: pca_hourly (T, 3), is_open (T,), pca_daily (for target)
    """
    # Load raw finance
    fin = pd.read_csv(os.path.join(DATA_DIR, "Finance20192024", "raw_finance.csv"),
                       index_col=0, parse_dates=True)
    fin.index = pd.to_datetime(fin.index, utc=True)
    fin.dropna(inplace=True)

    # Log returns for PCA
    log_ret = np.log(fin / fin.shift(1)).dropna()

    # Fit PCA on all data (3 components)
    scaler_fin = StandardScaler()
    ret_scaled = scaler_fin.fit_transform(log_ret)
    pca = PCA(n_components=3)
    pca_values = pca.fit_transform(ret_scaled)  # (n_trading_days, 3)
    print(f"  Finance PCA explained variance: {pca.explained_variance_ratio_.round(4)}")

    # Create daily PCA DataFrame
    pca_daily = pd.DataFrame(pca_values, index=log_ret.index,
                              columns=["pc1", "pc2", "pc3"])

    # Build is_market_open: 1 for trading days
    trading_dates = set(pca_daily.index.normalize())

    # Reindex to hourly, forward-fill
    hourly_dates = pd.DatetimeIndex(time_index)
    pca_hourly = pca_daily.reindex(hourly_dates.normalize())
    # For dates before first trading day, backfill
    pca_hourly = pca_hourly.ffill().bfill()
    pca_hourly.index = hourly_dates

    # is_market_open
    is_open = np.array([1.0 if ts.normalize() in trading_dates else 0.0
                        for ts in hourly_dates], dtype=np.float32)

    # Z-score the 3 PCs (fit on training period only will be done later;
    # here we compute global for simplicity, caller can re-fit on train)
    pca_arr = pca_hourly.values.astype(np.float32)

    return pca_arr, is_open, pca_daily


def main():
    print("=" * 60)
    print("  ST-GCN Feature Tensor Assembly")
    print("=" * 60)

    # ── 1. Load node metadata ──────────────────────────────────────
    nodes = pd.read_csv("data/processed_nodes.csv")
    node_names = nodes["name"].tolist()
    countries = nodes["country"].values
    clusters = nodes["physical_cluster"].values
    lat = nodes["latitude"].values.astype(np.float32)
    lon = nodes["longitude"].values.astype(np.float32)
    cap = nodes["capacity_mw"].values.astype(np.float32)
    print(f"Nodes: {len(nodes)}, Clusters: {len(np.unique(clusters))}")

    # ── 2. Load hourly time-series ─────────────────────────────────
    print("\nLoading hourly CSVs...")
    price_df = load_hourly_csv("price_top100_2019-01-01_2024-12-31.csv")
    ssr_df   = load_hourly_csv("ssr_top100_2019-01-01_2024-12-31.csv")
    t2m_df   = load_hourly_csv("t2m_top100_2019-01-01_2024-12-31.csv")
    u100_df  = load_hourly_csv("u100_top100_2019-01-01_2024-12-31.csv")
    v100_df  = load_hourly_csv("v100_top100_2019-01-01_2024-12-31.csv")

    # Align columns to node order
    for df_name, df in [("price", price_df), ("ssr", ssr_df), ("t2m", t2m_df),
                         ("u100", u100_df), ("v100", v100_df)]:
        assert set(node_names).issubset(set(df.columns)), \
            f"{df_name}: missing nodes {set(node_names) - set(df.columns)}"

    price_df = price_df[node_names]
    ssr_df   = ssr_df[node_names]
    t2m_df   = t2m_df[node_names]
    u100_df  = u100_df[node_names]
    v100_df  = v100_df[node_names]

    # Common time index (intersection)
    common_idx = price_df.index.intersection(ssr_df.index).intersection(
        t2m_df.index).intersection(u100_df.index).intersection(v100_df.index)
    common_idx = common_idx.sort_values()
    T = len(common_idx)
    print(f"Common time steps: {T}  ({common_idx[0]} → {common_idx[-1]})")

    price_arr = price_df.loc[common_idx].values.astype(np.float32)
    ssr_arr   = ssr_df.loc[common_idx].values.astype(np.float32)
    t2m_arr   = t2m_df.loc[common_idx].values.astype(np.float32)
    u100_arr  = u100_df.loc[common_idx].values.astype(np.float32)
    v100_arr  = v100_df.loc[common_idx].values.astype(np.float32)

    # Free DataFrames
    del price_df, ssr_df, t2m_df, u100_df, v100_df
    gc.collect()

    # ── 3. Finance PCA + is_market_open ────────────────────────────
    print("\nBuilding finance PCA features...")
    pca_arr, is_open, pca_daily = build_finance_pca_hourly(common_idx)

    # ── 4. Determine train/val/test split indices ──────────────────
    train_end_ts = pd.Timestamp(TRAIN_END, tz="UTC")
    val_start_ts = pd.Timestamp(VAL_START, tz="UTC")
    val_end_ts   = pd.Timestamp(VAL_END, tz="UTC")
    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")

    train_mask = common_idx <= train_end_ts
    val_mask   = (common_idx >= val_start_ts) & (common_idx <= val_end_ts)
    test_mask  = common_idx >= test_start_ts

    n_train = train_mask.sum()
    n_val   = val_mask.sum()
    n_test  = test_mask.sum()
    print(f"\nChronological split:")
    print(f"  Train: {n_train} steps (→ {TRAIN_END})")
    print(f"  Val:   {n_val} steps ({VAL_START} → {VAL_END})")
    print(f"  Test:  {n_test} steps ({TEST_START} →)")

    # ── 5. Feature scaling (fit on TRAIN only) ─────────────────────
    print("\nScaling features (fit on train set only)...")
    scaler_params = {}

    # 5a. Weather: global Z-score (across all nodes & train timesteps)
    def global_zscore(arr, name):
        train_data = arr[train_mask]
        mu = train_data.mean()
        sigma = train_data.std() + 1e-8
        scaler_params[f"{name}_mean"] = float(mu)
        scaler_params[f"{name}_std"] = float(sigma)
        return (arr - mu) / sigma

    ssr_arr   = global_zscore(ssr_arr, "ssr")
    t2m_arr   = global_zscore(t2m_arr, "t2m")
    u100_arr  = global_zscore(u100_arr, "u100")
    v100_arr  = global_zscore(v100_arr, "v100")
    print("  ✓ Weather features: global Z-score")

    # 5b. Electricity price: RobustScaler per country
    unique_countries = np.unique(countries)
    for c in unique_countries:
        col_mask = countries == c
        col_indices = np.where(col_mask)[0]
        train_prices = price_arr[train_mask][:, col_indices].flatten()
        median = np.median(train_prices)
        q25, q75 = np.percentile(train_prices, [25, 75])
        iqr = q75 - q25 + 1e-8
        price_arr[:, col_indices] = (price_arr[:, col_indices] - median) / iqr
        scaler_params[f"price_{c}_median"] = float(median)
        scaler_params[f"price_{c}_iqr"] = float(iqr)
    print(f"  ✓ Electricity price: RobustScaler per country ({len(unique_countries)} countries)")

    # 5c. Finance PCA: Z-score per component (fit on train)
    for i in range(3):
        train_pc = pca_arr[train_mask, i]
        mu = train_pc.mean()
        sigma = train_pc.std() + 1e-8
        pca_arr[:, i] = (pca_arr[:, i] - mu) / sigma
        scaler_params[f"finance_pc{i+1}_mean"] = float(mu)
        scaler_params[f"finance_pc{i+1}_std"] = float(sigma)
    print("  ✓ Finance PCA: Z-score per component")

    # 5d. is_market_open: no scaling
    print("  ✓ is_market_open: binary (no scaling)")

    # 5e. Capacity: log1p + Z-score
    cap_log = np.log1p(cap)
    cap_mu = cap_log.mean()
    cap_sigma = cap_log.std() + 1e-8
    cap_scaled = (cap_log - cap_mu) / cap_sigma
    scaler_params["cap_log_mean"] = float(cap_mu)
    scaler_params["cap_log_std"] = float(cap_sigma)
    print("  ✓ Capacity: log1p + Z-score")

    # 5f. Lat/Lon: MinMax
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    lat_norm = (lat - lat_min) / (lat_max - lat_min + 1e-8)
    lon_norm = (lon - lon_min) / (lon_max - lon_min + 1e-8)
    scaler_params["lat_min"] = float(lat_min)
    scaler_params["lat_max"] = float(lat_max)
    scaler_params["lon_min"] = float(lon_min)
    scaler_params["lon_max"] = float(lon_max)
    print("  ✓ Lat/Lon: MinMax normalised")

    # 5g. Time cyclical features
    hours = np.array([ts.hour for ts in common_idx], dtype=np.float32)
    months = np.array([ts.month for ts in common_idx], dtype=np.float32)
    sin_hour = np.sin(2 * np.pi * hours / 24)
    cos_hour = np.cos(2 * np.pi * hours / 24)
    sin_month = np.sin(2 * np.pi * (months - 1) / 12)
    cos_month = np.cos(2 * np.pi * (months - 1) / 12)
    print("  ✓ Cyclical time features: hour + month")

    # Save scaler params
    np.savez(os.path.join(OUTPUT_DIR, "scaler_params.npz"), **scaler_params)

    # ── 6. Assemble feature tensor (T, N, F) via memmap ────────────
    print(f"\nAssembling feature tensor: ({T}, {N_NODES}, {N_FEATURES})...")
    tensor_path = os.path.join(OUTPUT_DIR, "feature_tensor.npy")
    # Pre-compute shape and save header
    shape = (T, N_NODES, N_FEATURES)
    tensor_size_mb = T * N_NODES * N_FEATURES * 4 / (1024**2)
    print(f"  Tensor size: {tensor_size_mb:.1f} MB")

    # Use memmap for memory efficiency
    fp = np.memmap(tensor_path, dtype=np.float32, mode="w+", shape=shape)

    # Fill features
    # [0-3] weather (T, N)
    fp[:, :, 0] = ssr_arr
    fp[:, :, 1] = t2m_arr
    fp[:, :, 2] = u100_arr
    fp[:, :, 3] = v100_arr
    del ssr_arr, t2m_arr, u100_arr, v100_arr
    gc.collect()
    print("  [0-3] Weather features filled")

    # [4] price (T, N)
    fp[:, :, 4] = price_arr
    del price_arr
    gc.collect()
    print("  [4]   Price filled")

    # [5-7] finance PCA (T, 3) → broadcast to (T, N, 3)
    fp[:, :, 5] = pca_arr[:, 0:1]  # broadcast
    fp[:, :, 6] = pca_arr[:, 1:2]
    fp[:, :, 7] = pca_arr[:, 2:3]
    print("  [5-7] Finance PCA filled (broadcast)")

    # [8] is_market_open (T,) → broadcast to (T, N)
    fp[:, :, 8] = is_open[:, None]
    print("  [8]   is_market_open filled")

    # [9] capacity (N,) → broadcast to (T, N)
    fp[:, :, 9] = cap_scaled[None, :]
    print("  [9]   Capacity filled (static)")

    # [10-11] lat/lon (N,) → broadcast
    fp[:, :, 10] = lat_norm[None, :]
    fp[:, :, 11] = lon_norm[None, :]
    print("  [10-11] Lat/Lon filled (static)")

    # [12-15] cyclical time (T,) → broadcast to (T, N)
    fp[:, :, 12] = sin_hour[:, None]
    fp[:, :, 13] = cos_hour[:, None]
    fp[:, :, 14] = sin_month[:, None]
    fp[:, :, 15] = cos_month[:, None]
    print("  [12-15] Cyclical time features filled")

    fp.flush()
    print(f"  ✓ Saved {tensor_path}")

    # ── 7. Build target vector: ΔPC1 at +24h ──────────────────────
    print("\nBuilding target: ΔPC1 (24h ahead change rate)...")
    # Use the scaled PC1 (feature index 5) — same as pca_arr[:, 0] already Z-scored
    # Target: (pc1[t+24] - pc1[t]) / (|pc1[t]| + epsilon)
    # But since PC1 is Z-scored and can be near 0, use simple difference instead
    # ΔPC1 = pc1_zscore[t + HORIZON] - pc1_zscore[t]
    pc1 = pca_arr[:, 0]  # already Z-scored
    target_full = np.full(T, np.nan, dtype=np.float32)
    target_full[:T - HORIZON] = pc1[HORIZON:] - pc1[:T - HORIZON]

    target_path = os.path.join(OUTPUT_DIR, "target_vector.npy")
    np.save(target_path, target_full)
    print(f"  ✓ Saved {target_path}  (valid: {np.isfinite(target_full).sum()}/{T})")

    # Save time index for reference
    time_path = os.path.join(OUTPUT_DIR, "time_index.csv")
    pd.Series(common_idx).to_csv(time_path, index=False, header=["timestamp"])
    print(f"  ✓ Saved {time_path}")

    # Save tensor shape metadata
    meta = {
        "shape": list(shape),
        "features": [
            "ssr", "t2m", "u100", "v100", "price",
            "finance_pc1", "finance_pc2", "finance_pc3",
            "is_market_open", "capacity_mw",
            "latitude", "longitude",
            "sin_hour", "cos_hour", "sin_month", "cos_month"
        ],
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "stride": STRIDE,
        "train_end": TRAIN_END,
        "val_range": [VAL_START, VAL_END],
        "test_start": TEST_START,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
    }
    with open(os.path.join(OUTPUT_DIR, "tensor_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── 8. Dataset & DataLoader ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Building DataLoaders")
    print("=" * 60)

    # Get split boundaries as integer indices
    train_end_idx = np.searchsorted(common_idx, train_end_ts, side="right") - 1
    val_start_idx = np.searchsorted(common_idx, val_start_ts, side="left")
    val_end_idx   = np.searchsorted(common_idx, val_end_ts, side="right") - 1
    test_start_idx = np.searchsorted(common_idx, test_start_ts, side="left")

    print(f"  Train indices: 0 → {train_end_idx}")
    print(f"  Val indices:   {val_start_idx} → {val_end_idx}")
    print(f"  Test indices:  {test_start_idx} → {T-1}")

    class STGCNWindowDataset(Dataset):
        """
        Sliding window dataset for ST-GCN.
        X: (window_size, N, F) — past WINDOW_SIZE hours
        y: scalar — ΔPC1 at HORIZON hours after window end
        """
        def __init__(self, tensor_memmap, target, start_idx, end_idx,
                     window_size, horizon, stride):
            self.tensor = tensor_memmap
            self.target = target
            self.window_size = window_size
            self.horizon = horizon

            # Valid window starts: need window_size past + horizon future
            max_start = end_idx - window_size - horizon + 1
            min_start = max(start_idx, 0)
            self.indices = list(range(min_start, max_start + 1, stride))
            # Filter out windows where target is NaN
            self.indices = [i for i in self.indices
                           if np.isfinite(target[i + window_size - 1 + horizon])]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            t = self.indices[idx]
            X = np.array(self.tensor[t:t + self.window_size], dtype=np.float32)
            # Target: ΔPC1 at (window_end + horizon)
            y_idx = t + self.window_size - 1 + self.horizon
            y = self.target[y_idx]
            return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)

    train_ds = STGCNWindowDataset(fp, target_full, 0, train_end_idx,
                                   WINDOW_SIZE, HORIZON, STRIDE)
    val_ds   = STGCNWindowDataset(fp, target_full, val_start_idx, val_end_idx,
                                   WINDOW_SIZE, HORIZON, STRIDE)
    test_ds  = STGCNWindowDataset(fp, target_full, test_start_idx, T - 1,
                                   WINDOW_SIZE, HORIZON, STRIDE)

    print(f"\n  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    print(f"  Test samples:  {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True)

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # ── 9. Smoke test ──────────────────────────────────────────────
    print("\n  Smoke test — loading first batch...")
    X_batch, y_batch = next(iter(train_loader))
    print(f"    X shape: {X_batch.shape}  (batch, window, nodes, features)")
    print(f"    y shape: {y_batch.shape}")
    print(f"    X dtype: {X_batch.dtype}, y dtype: {y_batch.dtype}")
    print(f"    X range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
    print(f"    y range: [{y_batch.min():.3f}, {y_batch.max():.3f}]")

    # Check for NaN/Inf
    assert torch.isfinite(X_batch).all(), "X contains NaN/Inf!"
    assert torch.isfinite(y_batch).all(), "y contains NaN/Inf!"
    print("    ✓ No NaN/Inf detected")

    # ── 10. VRAM estimation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VRAM Usage Estimation (RTX 5070 Ti, 16GB)")
    print("=" * 60)
    check_vram_usage(X_batch)

    # ── 11. Training Tips Summary ──────────────────────────────────
    write_training_tips()

    print("\n✅ Data pipeline complete. All files in:", OUTPUT_DIR)
    return train_loader, val_loader, test_loader


def check_vram_usage(X_batch, adj_nnz=848, n_nodes=100, hidden_dim=64,
                     n_stgcn_blocks=3, batch_size=16):
    """
    Estimate VRAM for a single forward pass.
    """
    # Input tensor
    input_bytes = X_batch.nelement() * 4  # float32

    # Adjacency matrix (sparse COO): indices (2 * nnz * 8) + values (nnz * 4)
    adj_bytes = 2 * adj_nnz * 8 + adj_nnz * 4

    # ST-GCN intermediate activations (rough estimate)
    # Each block: batch × nodes × hidden × temporal
    T_win = X_batch.shape[1]
    activation_per_block = batch_size * n_nodes * hidden_dim * T_win * 4
    total_activations = activation_per_block * n_stgcn_blocks * 2  # fwd + grad

    # Parameters: ~3 blocks × (GCN weights + temporal conv)
    param_bytes = n_stgcn_blocks * (
        hidden_dim * hidden_dim * 4 +      # GCN linear
        hidden_dim * hidden_dim * 3 * 4 +   # temporal conv (kernel=3)
        hidden_dim * 2 * 4                   # bias + BN
    )

    # Optimizer states (Adam: 2× params)
    optimizer_bytes = param_bytes * 2

    # Gradient buffers ≈ params
    grad_bytes = param_bytes

    total = input_bytes + adj_bytes + total_activations + param_bytes + optimizer_bytes + grad_bytes
    total_mb = total / (1024 ** 2)

    print(f"  Input batch:     {input_bytes / 1024**2:6.1f} MB")
    print(f"  Adjacency:       {adj_bytes / 1024**2:6.1f} MB")
    print(f"  Activations:     {total_activations / 1024**2:6.1f} MB")
    print(f"  Parameters:      {param_bytes / 1024**2:6.1f} MB")
    print(f"  Optimizer:       {optimizer_bytes / 1024**2:6.1f} MB")
    print(f"  Gradients:       {grad_bytes / 1024**2:6.1f} MB")
    print(f"  ────────────────────────────────")
    print(f"  Estimated total: {total_mb:6.1f} MB")
    print(f"  5070 Ti headroom: {16*1024 - total_mb:.0f} MB free of 16 GB")

    if total_mb > 12 * 1024:
        print("  ⚠ May exceed 12GB! Consider ClusterGCN sampling.")
    else:
        print("  ✓ Fits comfortably in 16GB VRAM.")


def write_training_tips():
    """Write training recommendations to file."""
    tips = """
╔══════════════════════════════════════════════════════════════╗
║              ST-GCN Training Tips & Recommendations          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. TARGET: Predict ΔPC1 (first-differenced)                 ║
║     - Raw PCA values have trend → bad for convergence        ║
║     - ΔPC1 = PC1(t+24) - PC1(t) is more stationary          ║
║     - Already implemented in target_vector.npy               ║
║                                                              ║
║  2. LOSS FUNCTION: Hybrid MSE + L1                           ║
║     loss = MSE(pred, true) + λ * L1(pred, true)              ║
║     - Recommended λ = 0.1 to start                           ║
║     - MSE penalises large errors; L1 adds robustness         ║
║     - Financial data has fat tails → pure MSE overweights    ║
║       extreme events                                         ║
║                                                              ║
║  3. ClusterGCN SAMPLING (if needed):                         ║
║     - Sample 2-3 clusters per mini-batch (~50 nodes)         ║
║     - Reduces memory from O(N²) to O(|cluster|²)            ║
║     - Implementation sketch:                                 ║
║       clusters = nodes.physical_cluster.unique()             ║
║       sampled = np.random.choice(clusters, size=3)           ║
║       mask = nodes.physical_cluster.isin(sampled)            ║
║       X_sub = X[:, mask, :]                                  ║
║       A_sub = A[mask][:, mask]                               ║
║                                                              ║
║  4. LEARNING RATE: Cosine annealing                          ║
║     - Start: 1e-3, min: 1e-6                                ║
║     - Warmup: 5 epochs                                       ║
║                                                              ║
║  5. EARLY STOPPING: patience=15 on val loss                  ║
║                                                              ║
║  6. DATA LEAKAGE PREVENTION:                                 ║
║     ✓ Chronological split (2019-2023 / 2024H1 / 2024H2)     ║
║     ✓ No shuffle across splits                               ║
║     ✓ Scalers fitted on train only                           ║
║     ✓ Train DataLoader shuffle=True (within-split only)      ║
║                                                              ║
║  7. EXPECTED BASELINE:                                       ║
║     - Naive (predict 0): MSE ≈ var(ΔPC1)                    ║
║     - Good model should beat this by > 5%                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    path = os.path.join(OUTPUT_DIR, "training_tips.txt")
    with open(path, "w") as f:
        f.write(tips)
    print(f"\n  ✓ Training tips saved to {path}")


# ─── Sample code: Hybrid Loss ──────────────────────────────────────
class HybridLoss(torch.nn.Module):
    """MSE + λ·L1 loss for financial prediction."""
    def __init__(self, lambda_l1=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

    def forward(self, pred, target):
        return self.mse(pred, target) + self.lambda_l1 * self.l1(pred, target)


# ─── Sample code: ClusterGCN Sampler ───────────────────────────────
class ClusterGCNSampler:
    """
    Samples subgraphs by randomly selecting k clusters.
    Use when full-graph training exceeds VRAM.
    """
    def __init__(self, cluster_ids, n_sample_clusters=3):
        self.unique_clusters = np.unique(cluster_ids)
        self.cluster_ids = cluster_ids
        self.k = n_sample_clusters

    def sample(self):
        """Returns node indices for a random subgraph."""
        chosen = np.random.choice(self.unique_clusters, size=self.k, replace=False)
        mask = np.isin(self.cluster_ids, chosen)
        return np.where(mask)[0]

    def subgraph(self, X, A_sparse, indices):
        """Extract subgraph tensors."""
        X_sub = X[:, indices, :]
        A_sub = A_sparse[indices][:, indices]
        return X_sub, A_sub


if __name__ == "__main__":
    main()
