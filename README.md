# 🌬️ EU Wind Energy → Stock Market Prediction (ST-GCN)

A Spatio-Temporal Graph Convolutional Network that predicts renewable energy stock fluctuations using European wind farm meteorological data, electricity prices, and financial indicators.

## 🎯 Project Overview

**Research Question:** Can spatiotemporal patterns in European wind energy production predict movements in renewable energy stocks?

**Approach:** Build a graph of 100 major European wind farms, capture hourly weather/power signals propagating across the network (modelling the westerly jet stream physics), and predict 24-hour-ahead changes in a portfolio of wind energy stocks (Ørsted, Vestas, Iberdrola, EDPR).

**Key Insight:** The model learns that weather fluctuations accumulate as "latent energy" during market-closed hours (`is_market_open=0`), then manifest as price movements at market open — bridging the 24/7 physical world with the 5/7 financial world.

## 🏗️ Architecture

```
Input: (B, 168, 100, 16) — 7 days × 100 nodes × 16 features
         │
    Input Projection (16 → 64)
         │
    ┌────┴────┐
    │ ST-Conv Block ×3 │
    │  ├─ Temporal GLU (Gated Linear Unit)
    │  ├─ Sparse Graph Conv (Ã = D⁻½AD⁻½)
    │  ├─ Temporal GLU
    │  └─ BatchNorm + Residual + Dropout
    └────┬────┘
         │
    Attention Pooling (100 nodes → weighted sum)
         │
    Temporal Pool → MLP Readout
         │
Output: (B, 3) — ΔPC1, ΔPC2, ΔPC3 predictions
```

## 📊 Feature Vector (16 dimensions per node per hour)

| # | Feature | Scaling | Source |
|---|---------|---------|--------|
| 0-3 | SSR, T2M, U100, V100 | Global Z-score | ERA5 meteorological |
| 4 | Electricity price | RobustScaler (per country) | Day-ahead markets |
| 5-7 | Finance PC1-PC3 | Z-score (broadcast) | Stock PCA |
| 8 | is_market_open | Binary (no scaling) | Trading calendar |
| 9 | Capacity (MW) | log1p + Z-score | Static node metadata |
| 10-11 | Latitude, Longitude | MinMax | Static node metadata |
| 12-13 | sin/cos(hour) | Cyclical | Time encoding |
| 14-15 | sin/cos(month) | Cyclical | Time encoding |

## 📁 Project Structure

```
├── README.md                    # This file
├── spatial_clustering.py        # DBSCAN spatial clustering (haversine)
├── finance_pca.py               # PCA on wind energy stocks
├── build_adjacency.py           # Sparse adjacency matrix construction
├── build_tensor.py              # Feature tensor assembly & DataLoader
├── stgcn_model.py               # ST-GCN model + Trainer + utilities
├── train_stgcn.py               # Training script (full pipeline)
├── test_integration.py          # Integration test (real data verification)
├── NodeLoader.py                # Original node data loader
│
├── asset_nodes.csv              # 865 wind farm nodes (raw)
├── processed_nodes.csv          # Top-100 nodes + cluster IDs
├── adjacency_matrix.npz         # Sparse CSR adjacency (scipy)
├── adjacency_matrix.pt          # Sparse COO adjacency (PyTorch)
│
├── data/
│   ├── Finance20192024/
│   │   └── raw_finance.csv      # Daily stock prices (2019-2024)
│   ├── price_top100_*.csv       # Hourly electricity prices
│   ├── ssr_top100_*.csv         # Surface solar radiation
│   ├── t2m_top100_*.csv         # 2m temperature
│   ├── u100_top100_*.csv        # 100m U-wind component
│   └── v100_top100_*.csv        # 100m V-wind component
│
├── tensor_output/
│   ├── feature_tensor.npy       # (52585, 100, 16) memmap — 321 MB
│   ├── target_vector.npy        # ΔPC1 at +24h
│   ├── scaler_params.npz        # All scaler parameters
│   ├── tensor_meta.json         # Shape & split metadata
│   └── training_tips.txt        # Training recommendations
│
├── cluster_map.png              # Spatial clustering visualisation
├── adjacency_heatmap.png        # Adjacency matrix sparsity pattern
└── finance_pca_scree.png        # PCA explained variance plot
```

## 🔬 Pipeline Steps (built 2025-05-01)

### Step 1: Spatial Clustering
- **Algorithm:** DBSCAN with haversine distance (eps=65 km, min_samples=3)
- **Result:** 100 top-capacity wind farms → 39 physical clusters
- **Purpose:** Optimise graph structure; capture mesoscale weather system coherence

### Step 2: Finance PCA
- **Input:** 4 wind energy stocks (ORSTED, VWS, IBE, EDPR), daily 2019–2024
- **Result:** PC1 explains **62.17%**, top-3 explain **90.38%** (>85% ✓)
- **Interpretation:** PC1 = sector momentum; PC2 = North/South Europe divergence

### Step 3: Adjacency Matrix
- **Intra-cluster:** Weight = 1.0 (full connection within physical clusters)
- **Inter-cluster:** Weight = exp(−d/σ) if d < 150 km
- **Self-loops:** 1.0
- **Normalisation:** Symmetric Laplacian (D⁻½AD⁻½)
- **Density:** 8.48% (sparse enough for efficient computation)

### Step 4: Feature Tensor Assembly
- **Shape:** (52,585 hours, 100 nodes, 16 features)
- **Split:** Train 2019–2023 | Val 2024-H1 | Test 2024-H2
- **Zero leakage:** Scalers fit on train only; chronological split; no cross-time shuffle
- **Target:** ΔPC1 = PC1(t+24) − PC1(t) (first-differenced for stationarity)

### Step 5: ST-GCN Model
- **Parameters:** ~199K (lightweight)
- **VRAM estimate:** ~415 MB per batch (fits easily in 16 GB)
- **Loss:** Hybrid MSE + 0.1×L1 (robust to financial fat tails)
- **Optimiser:** AdamW + Cosine warmup (5 epochs → 100 epochs)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install torch numpy pandas scikit-learn scipy matplotlib

# 2. Generate feature tensor (run once, ~2 min)
python build_tensor.py

# 3. Train on GPU
python train_stgcn.py --device cuda --epochs 100 --batch_size 16

# 4. With ClusterGCN (if VRAM limited)
python train_stgcn.py --device cuda --cluster_gcn --n_sample_clusters 3

# 5. Dry run (2 epochs, verify setup)
python train_stgcn.py --device cuda --dry_run
```

## 🧪 Validation

```bash
# Smoke test (random data, verifies architecture)
python -c "from stgcn_model import smoke_test; smoke_test()"

# Integration test (real data, verifies full pipeline)
python test_integration.py
```

## 📈 Baseline & Success Criteria

| Metric | Value |
|--------|-------|
| Naive baseline (predict 0) | MSE = Var(ΔPC1) |
| Success threshold | Beat naive by ≥ 5% |
| Evaluation | Test set (2024-H2) only |

## ⚡ Hardware Requirements

- **Minimum:** Any GPU with 4+ GB VRAM (with ClusterGCN)
- **Recommended:** RTX 5070 Ti (16 GB) — batch_size=16, full graph
- **CPU fallback:** Works but very slow (use `--dry_run` for testing)

## 📝 Key Design Decisions

1. **Global Z-score for weather** — Preserves geographic intensity differences (Scotland wind > Germany wind)
2. **RobustScaler for electricity prices** — Handles 2022 energy crisis spikes (500+ €/MWh)
3. **is_market_open flag** — Lets the model learn accumulation during closed markets
4. **Haversine distance** — Accurate great-circle distance, not Euclidean
5. **DBSCAN clustering** — Identifies non-spherical coastal/mountain wind farm clusters
6. **Attention pooling** — Learns which nodes matter most for financial prediction
7. **Memmap tensors** — 321 MB tensor doesn't need to fit in RAM

## 📄 License

Research project — academic use.

---

*Built with ST-GCN, PyTorch, scikit-learn, and ERA5 reanalysis data.*
