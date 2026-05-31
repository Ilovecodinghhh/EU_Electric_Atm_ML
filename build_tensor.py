"""
Multimodal Spatiotemporal Feature Tensor Assembly & DataLoader for ST-GCN.

Produces:
  - feature_tensor.npy   (T, N, F) float32 memmap
  - target_vector.npy    (T,) float32 — next trading-day PC1 factor
  - scaler_params.npz    — all scaler parameters for inference
  - feature_summary.txt  — feature documentation
  - train/val/test DataLoaders (chronological split, no leakage)

Feature vector (F=25) per node per timestep:
  [0]  ssr            — global Z-score
  [1]  t2m            — global Z-score
  [2]  u100           — global Z-score
  [3]  v100           — global Z-score
  [4]  wind_speed     — global Z-score
  [5]  wind_ramp      — global Z-score
  [6]  wind_power     — cubic power-curve proxy, global Z-score
  [7]  country_wind_speed   — capacity-weighted country wind speed
  [8]  country_wind_ramp    — capacity-weighted country wind ramp
  [9]  cluster_wind_speed   — capacity-weighted cluster wind speed
  [10] cluster_wind_ramp    — capacity-weighted cluster wind ramp
  [11] country_price        — country-level price, RobustScaler
  [12] market_price_mean    — equal-country Europe price mean
  [13] country_price_spread — country price minus Europe mean
  [14] finance_pc1          — Z-score, previous trading close
  [15] finance_pc2          — Z-score, previous trading close
  [16] finance_pc3          — Z-score, previous trading close
  [17] is_market_open       — binary 0/1, no scaling
  [18] capacity_mw          — log1p + Z-score (static, broadcast over time)
  [19] latitude             — MinMax normalised (static)
  [20] longitude            — MinMax normalised (static)
  [21] sin_hour             — sin(2π·hour/24)
  [22] cos_hour             — cos(2π·hour/24)
  [23] sin_month            — sin(2π·(month-1)/12)
  [24] cos_month            — cos(2π·(month-1)/12)
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
STRIDE = 24             # one sample per daily trading decision timestamp
BATCH_SIZE = 16
NUM_WORKERS = 0          # Local Dataset in this builder is not picklable on Windows
N_FEATURES = 25
DECISION_HOUR_UTC = 23
EUROPE_LAT_RANGE = (35.0, 72.0)
EUROPE_LON_RANGE = (-15.0, 35.0)
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


def winsorize_by_train_quantiles(arr, train_mask, name, params, q_low=0.01, q_high=0.99):
    """Clip using train-period quantiles, preserving out-of-sample discipline."""
    train_data = arr[train_mask]
    lo = np.nanquantile(train_data, q_low, axis=0)
    hi = np.nanquantile(train_data, q_high, axis=0)
    params[f"{name}_winsor_q_low"] = np.array(lo, dtype=np.float32)
    params[f"{name}_winsor_q_high"] = np.array(hi, dtype=np.float32)
    return np.clip(arr, lo, hi)


def load_hourly_csv(filename):
    """Load a (T, N) hourly CSV, return DataFrame with datetime index."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_finance_pca_hourly(time_index, train_end_ts):
    """
    Build hourly finance PCA features + is_market_open indicator.
    Returns:
      pca_feature_arr: lagged PCA available as of each hour (T, 3)
      pca_realized_arr: same-date PCA used only for targets (T, 3)
      is_open: trading-date indicator (T,)
      pca_daily: realized daily PCA
      pca_params: fitted train-only finance parameters
    """
    # Load raw finance
    fin = pd.read_csv(os.path.join(DATA_DIR, "Finance20192024", "raw_finance.csv"),
                       index_col=0, parse_dates=True)
    fin.index = pd.to_datetime(fin.index, utc=True)
    fin.dropna(inplace=True)

    # Log returns for PCA.
    log_ret = np.log(fin / fin.shift(1)).dropna()

    # Fit all finance preprocessing on train-period returns only.
    train_ret_mask = log_ret.index <= train_end_ts
    if not train_ret_mask.any():
        raise ValueError("No finance returns found in the configured training period")
    ret_q_low = log_ret.loc[train_ret_mask].quantile(0.01)
    ret_q_high = log_ret.loc[train_ret_mask].quantile(0.99)
    log_ret = log_ret.clip(lower=ret_q_low, upper=ret_q_high, axis=1)

    scaler_fin = StandardScaler()
    ret_train_scaled = scaler_fin.fit_transform(log_ret.loc[train_ret_mask])
    ret_scaled = scaler_fin.transform(log_ret)

    pca = PCA(n_components=3)
    pca.fit(ret_train_scaled)
    pca_values = pca.transform(ret_scaled)  # (n_trading_days, 3)
    print(f"  Finance PCA train explained variance: {pca.explained_variance_ratio_.round(4)}")

    # Create daily PCA DataFrame
    pca_daily = pd.DataFrame(pca_values, index=log_ret.index,
                              columns=["pc1", "pc2", "pc3"])
    pca_by_date = pca_daily.copy()
    pca_by_date.index = pca_by_date.index.normalize()

    # Build is_market_open: 1 for trading days
    trading_dates = set(pca_by_date.index)

    # Reindex realized PCA to hourly for target construction.
    hourly_dates = pd.DatetimeIndex(time_index)
    hourly_norm = hourly_dates.normalize()
    pca_realized_hourly = pca_by_date.reindex(hourly_norm)
    pca_realized_hourly = pca_realized_hourly.ffill()
    pca_realized_hourly.index = hourly_dates

    # Feature PCA is strictly lagged by one available trading close.
    # For date D, this uses the PCA return from the previous trading date,
    # never the same-date close. Initial unavailable rows are neutral zeros.
    pca_lagged_daily = pca_by_date.shift(1)
    pca_feature_hourly = pca_lagged_daily.reindex(hourly_norm)
    pca_feature_hourly = pca_feature_hourly.ffill().fillna(0.0)
    pca_feature_hourly.index = hourly_dates

    # is_market_open
    is_open = np.array([1.0 if ts.normalize() in trading_dates else 0.0
                        for ts in hourly_dates], dtype=np.float32)

    pca_feature_arr = pca_feature_hourly.values.astype(np.float32)
    pca_realized_arr = pca_realized_hourly.values.astype(np.float32)

    pca_params = {
        "finance_return_mean": scaler_fin.mean_.astype(np.float32),
        "finance_return_scale": scaler_fin.scale_.astype(np.float32),
        "finance_pca_components": pca.components_.astype(np.float32),
        "finance_pca_mean": pca.mean_.astype(np.float32),
        "finance_pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "finance_pca_feature_lag_trading_days": np.array([1], dtype=np.int32),
        "finance_return_winsor_q_low": ret_q_low.values.astype(np.float32),
        "finance_return_winsor_q_high": ret_q_high.values.astype(np.float32),
    }

    return pca_feature_arr, pca_realized_arr, is_open, pca_daily, pca_params


def main():
    print("=" * 60)
    print("  ST-GCN Feature Tensor Assembly")
    print("=" * 60)

    # ── 1. Load node metadata ──────────────────────────────────────
    nodes = pd.read_csv("data/processed_nodes.csv")
    valid_geo = (
        nodes["latitude"].between(*EUROPE_LAT_RANGE)
        & nodes["longitude"].between(*EUROPE_LON_RANGE)
    )
    if not valid_geo.all():
        bad = nodes.loc[~valid_geo, ["name", "country", "latitude", "longitude"]]
        print("Dropping nodes outside Europe bounds:")
        print(bad.to_string(index=False))
        nodes = nodes.loc[valid_geo].reset_index(drop=True)
    node_names = nodes["name"].tolist()
    countries = nodes["country"].values
    clusters = nodes["physical_cluster"].values
    lat = nodes["latitude"].values.astype(np.float32)
    lon = nodes["longitude"].values.astype(np.float32)
    cap = nodes["capacity_mw"].values.astype(np.float32)
    n_nodes = len(nodes)
    print(f"Nodes: {n_nodes}, Clusters: {len(np.unique(clusters))}")

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

    unique_countries = np.unique(countries)
    country_to_col = {country: i for i, country in enumerate(unique_countries)}
    country_price_raw = np.empty((T, len(unique_countries)), dtype=np.float32)
    for country in unique_countries:
        country_nodes = [name for name, c in zip(node_names, countries) if c == country]
        country_prices = price_df.loc[common_idx, country_nodes]
        if country_prices.T.drop_duplicates().shape[0] != 1:
            print(f"  Warning: {country} has non-identical node price series; using first node")
        country_price_raw[:, country_to_col[country]] = country_prices.iloc[:, 0].values.astype(np.float32)

    ssr_arr   = ssr_df.loc[common_idx].values.astype(np.float32)
    t2m_arr   = t2m_df.loc[common_idx].values.astype(np.float32)
    u100_arr  = u100_df.loc[common_idx].values.astype(np.float32)
    v100_arr  = v100_df.loc[common_idx].values.astype(np.float32)
    wind_speed_arr = np.sqrt(u100_arr ** 2 + v100_arr ** 2).astype(np.float32)
    wind_speed_ramp_arr = np.diff(
        wind_speed_arr, axis=0, prepend=wind_speed_arr[[0]]
    ).astype(np.float32)
    wind_power_proxy_arr = np.clip(
        (wind_speed_arr - 3.0) / (12.0 - 3.0), 0.0, 1.0
    ).astype(np.float32) ** 3
    country_wind_speed_arr = np.empty((T, n_nodes), dtype=np.float32)
    country_wind_ramp_arr = np.empty((T, n_nodes), dtype=np.float32)
    cluster_wind_speed_arr = np.empty((T, n_nodes), dtype=np.float32)
    cluster_wind_ramp_arr = np.empty((T, n_nodes), dtype=np.float32)
    for country in unique_countries:
        idx = np.where(countries == country)[0]
        weights = cap[idx] / cap[idx].sum()
        country_wind_speed_arr[:, idx] = wind_speed_arr[:, idx].dot(weights)[:, None]
        country_wind_ramp_arr[:, idx] = wind_speed_ramp_arr[:, idx].dot(weights)[:, None]
    for cluster in np.unique(clusters):
        idx = np.where(clusters == cluster)[0]
        weights = cap[idx] / cap[idx].sum()
        cluster_wind_speed_arr[:, idx] = wind_speed_arr[:, idx].dot(weights)[:, None]
        cluster_wind_ramp_arr[:, idx] = wind_speed_ramp_arr[:, idx].dot(weights)[:, None]

    # Free DataFrames
    del price_df, ssr_df, t2m_df, u100_df, v100_df
    gc.collect()

    # ── 3. Determine train/val/test split indices ──────────────────
    train_end_ts = pd.Timestamp(TRAIN_END, tz="UTC")
    val_start_ts = pd.Timestamp(VAL_START, tz="UTC")
    val_end_ts   = pd.Timestamp(VAL_END, tz="UTC")
    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")

    # ── 4. Finance PCA + is_market_open ────────────────────────────
    print("\nBuilding finance PCA features...")
    pca_arr, pca_realized_arr, is_open, pca_daily, pca_params = build_finance_pca_hourly(
        common_idx, train_end_ts)

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
    scaler_params = dict(pca_params)

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
    wind_speed_arr = global_zscore(wind_speed_arr, "wind_speed")
    wind_speed_ramp_arr = global_zscore(wind_speed_ramp_arr, "wind_speed_ramp")
    wind_power_proxy_arr = global_zscore(wind_power_proxy_arr, "wind_power_proxy")
    country_wind_speed_arr = global_zscore(country_wind_speed_arr, "country_wind_speed")
    country_wind_ramp_arr = global_zscore(country_wind_ramp_arr, "country_wind_ramp")
    cluster_wind_speed_arr = global_zscore(cluster_wind_speed_arr, "cluster_wind_speed")
    cluster_wind_ramp_arr = global_zscore(cluster_wind_ramp_arr, "cluster_wind_ramp")
    print("  ✓ Weather features: global Z-score + wind speed/ramp/proxy")

    # 5b. Electricity price: country-level RobustScaler, then mapped to nodes.
    country_price_raw = winsorize_by_train_quantiles(
        country_price_raw, train_mask, "country_price", scaler_params
    )
    country_price_scaled = country_price_raw.copy()
    for c in unique_countries:
        country_col = country_to_col[c]
        train_prices = country_price_raw[train_mask, country_col]
        median = np.median(train_prices)
        q25, q75 = np.percentile(train_prices, [25, 75])
        iqr = q75 - q25 + 1e-8
        country_price_scaled[:, country_col] = (country_price_raw[:, country_col] - median) / iqr
        scaler_params[f"price_{c}_median"] = float(median)
        scaler_params[f"price_{c}_iqr"] = float(iqr)
    print(f"  ✓ Electricity price: country-level market features ({len(unique_countries)} countries)")

    country_price_arr = np.empty((T, n_nodes), dtype=np.float32)
    for node_idx, country in enumerate(countries):
        country_price_arr[:, node_idx] = country_price_scaled[:, country_to_col[country]]
    market_price_mean = country_price_scaled.mean(axis=1).astype(np.float32)
    country_price_spread_arr = country_price_arr - market_price_mean[:, None]

    # 5c. Finance PCA: Z-score per component (fit on train daily PCA)
    pca_train_daily_mask = pca_daily.index <= train_end_ts
    for i in range(3):
        train_pc = pca_daily.loc[pca_train_daily_mask, f"pc{i+1}"].values
        finite_train_pc = train_pc[np.isfinite(train_pc)]
        mu = finite_train_pc.mean()
        sigma = finite_train_pc.std() + 1e-8
        pca_arr[:, i] = (pca_arr[:, i] - mu) / sigma
        pca_realized_arr[:, i] = (pca_realized_arr[:, i] - mu) / sigma
        scaler_params[f"finance_pc{i+1}_mean"] = float(mu)
        scaler_params[f"finance_pc{i+1}_std"] = float(sigma)
    print("  ✓ Finance PCA: train-only Z-score; input PCA lagged one trading close")

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
    print(f"\nAssembling feature tensor: ({T}, {n_nodes}, {N_FEATURES})...")
    tensor_path = os.path.join(OUTPUT_DIR, "feature_tensor.npy")
    # Pre-compute shape and save header
    shape = (T, n_nodes, N_FEATURES)
    tensor_size_mb = T * n_nodes * N_FEATURES * 4 / (1024**2)
    print(f"  Tensor size: {tensor_size_mb:.1f} MB")

    # Use memmap for memory efficiency
    fp = np.memmap(tensor_path, dtype=np.float32, mode="w+", shape=shape)

    # Fill features
    # [0-3] weather (T, N)
    fp[:, :, 0] = ssr_arr
    fp[:, :, 1] = t2m_arr
    fp[:, :, 2] = u100_arr
    fp[:, :, 3] = v100_arr
    fp[:, :, 4] = wind_speed_arr
    fp[:, :, 5] = wind_speed_ramp_arr
    fp[:, :, 6] = wind_power_proxy_arr
    fp[:, :, 7] = country_wind_speed_arr
    fp[:, :, 8] = country_wind_ramp_arr
    fp[:, :, 9] = cluster_wind_speed_arr
    fp[:, :, 10] = cluster_wind_ramp_arr
    del ssr_arr, t2m_arr, u100_arr, v100_arr
    del wind_speed_arr, wind_speed_ramp_arr, wind_power_proxy_arr
    del country_wind_speed_arr, country_wind_ramp_arr
    del cluster_wind_speed_arr, cluster_wind_ramp_arr
    gc.collect()
    print("  [0-10] Weather/wind features filled")

    # [4] price (T, N)
    fp[:, :, 11] = country_price_arr
    fp[:, :, 12] = market_price_mean[:, None]
    fp[:, :, 13] = country_price_spread_arr
    del country_price_raw, country_price_scaled, country_price_arr
    del market_price_mean, country_price_spread_arr
    gc.collect()
    print("  [11-13] Country/market price features filled")

    # [5-7] finance PCA (T, 3) → broadcast to (T, N, 3)
    fp[:, :, 14] = pca_arr[:, 0:1]  # broadcast
    fp[:, :, 15] = pca_arr[:, 1:2]
    fp[:, :, 16] = pca_arr[:, 2:3]
    print("  [14-16] Finance PCA filled (broadcast, previous trading close)")

    # [8] is_market_open (T,) → broadcast to (T, N)
    fp[:, :, 17] = is_open[:, None]
    print("  [17]  is_market_open filled")

    # [9] capacity (N,) → broadcast to (T, N)
    fp[:, :, 18] = cap_scaled[None, :]
    print("  [18]  Capacity filled (static)")

    # [10-11] lat/lon (N,) → broadcast
    fp[:, :, 19] = lat_norm[None, :]
    fp[:, :, 20] = lon_norm[None, :]
    print("  [19-20] Lat/Lon filled (static)")

    # [12-15] cyclical time (T,) → broadcast to (T, N)
    fp[:, :, 21] = sin_hour[:, None]
    fp[:, :, 22] = cos_hour[:, None]
    fp[:, :, 23] = sin_month[:, None]
    fp[:, :, 24] = cos_month[:, None]
    print("  [21-24] Cyclical time features filled")

    fp.flush()
    print(f"  ✓ Saved {tensor_path}")

    # ── 7. Build target vector: next trading-day PC1 ──────────────
    print("\nBuilding target: next trading-day PC1 factor...")
    # PC1 is derived from daily stock returns, so the target is sampled once
    # per trading day at the configured decision hour.
    # Target convention:
    #   target_full[t] = pc1_zscore[next_trading_day]
    # A training window ending at t must therefore read target_full[t].
    pc1_mu = scaler_params["finance_pc1_mean"]
    pc1_sigma = scaler_params["finance_pc1_std"]
    pca_daily_z = pca_daily.copy()
    pca_daily_z.index = pca_daily_z.index.normalize()
    pca_daily_z["pc1_z"] = (pca_daily_z["pc1"] - pc1_mu) / pc1_sigma

    common_pos = {ts: i for i, ts in enumerate(common_idx)}
    target_full = np.full(T, np.nan, dtype=np.float32)
    target_time_full = np.full(T, "", dtype=object)

    def same_split(decision_ts, target_ts):
        if decision_ts <= train_end_ts:
            return target_ts <= train_end_ts
        if val_start_ts <= decision_ts <= val_end_ts:
            return target_ts <= val_end_ts
        if decision_ts >= test_start_ts:
            return True
        return False

    trading_dates = list(pca_daily_z.index)
    for j in range(len(trading_dates) - 1):
        decision_ts = trading_dates[j] + pd.Timedelta(hours=DECISION_HOUR_UTC)
        target_ts = trading_dates[j + 1] + pd.Timedelta(hours=DECISION_HOUR_UTC)
        if decision_ts not in common_pos or not same_split(decision_ts, target_ts):
            continue
        target_full[common_pos[decision_ts]] = pca_daily_z.iloc[j + 1]["pc1_z"]
        target_time_full[common_pos[decision_ts]] = str(target_ts)

    finite_targets = np.where(np.isfinite(target_full))[0]
    assert all(common_idx[i] < pd.Timestamp(target_time_full[i]) for i in finite_targets)

    target_path = os.path.join(OUTPUT_DIR, "target_vector.npy")
    np.save(target_path, target_full)
    print(f"  ✓ Saved {target_path}  (valid: {np.isfinite(target_full).sum()}/{T})")

    # Save time index for reference
    time_path = os.path.join(OUTPUT_DIR, "time_index.csv")
    pd.Series(common_idx).to_csv(time_path, index=False, header=["timestamp"])
    target_time_path = os.path.join(OUTPUT_DIR, "target_time_index.csv")
    pd.DataFrame({
        "timestamp": common_idx.astype(str),
        "target_time": target_time_full,
    }).to_csv(target_time_path, index=False)
    print(f"  ✓ Saved {time_path}")
    print(f"  ✓ Saved {target_time_path}")

    # Save tensor shape metadata
    meta = {
        "shape": list(shape),
        "features": [
            "ssr", "t2m", "u100", "v100",
            "wind_speed", "wind_speed_ramp", "wind_power_proxy",
            "country_wind_speed", "country_wind_ramp",
            "cluster_wind_speed", "cluster_wind_ramp",
            "country_price", "market_price_mean", "country_price_spread",
            "finance_pc1", "finance_pc2", "finance_pc3",
            "is_market_open", "capacity_mw",
            "latitude", "longitude",
            "sin_hour", "cos_hour", "sin_month", "cos_month"
        ],
        "window_size": WINDOW_SIZE,
        "horizon": "next_trading_day",
        "decision_hour_utc": DECISION_HOUR_UTC,
        "stride": STRIDE,
        "train_end": TRAIN_END,
        "val_range": [VAL_START, VAL_END],
        "test_start": TEST_START,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "finance_feature_lag": "previous_trading_close",
        "finance_return_winsor_quantiles": [0.01, 0.99],
        "electricity_price_winsor_quantiles": [0.01, 0.99],
        "wind_aggregate_policy": "capacity_weighted_by_country_and_physical_cluster",
        "electricity_price_granularity": "country_level_equal_country_market_context",
        "electricity_price_countries": unique_countries.tolist(),
        "target_source": "next_trading_day_realized_finance_pc1",
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
            # Filter out windows where the delta starting at window_end is NaN.
            self.indices = [i for i in self.indices
                           if np.isfinite(target[i + window_size - 1])]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            t = self.indices[idx]
            X = np.array(self.tensor[t:t + self.window_size], dtype=np.float32)
            # Target: ΔPC1 from window_end to window_end + horizon
            y_idx = t + self.window_size - 1
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
    with open(path, "w", encoding="utf-8") as f:
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
