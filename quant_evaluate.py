"""
Quant research evaluation for the EU wind/finance project.

This script is deliberately model-agnostic. It builds timestamp-clean
walk-forward tabular samples from raw data, runs simple baselines and
ablations, evaluates prediction quality and a PC1 proxy backtest, and writes
fold-level artifacts under quant_output/.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler


DATA_DIR = "data"
OUTPUT_DIR = "quant_output"
WINDOW_SIZE = 168
STRIDE = 6
TX_COST_BPS = 10.0
RANDOM_SEED = 11

FOLDS = [
    {
        "name": "fold_2019_2021_to_2023",
        "train_end": "2021-12-31 23:59:59",
        "val_start": "2022-01-01 00:00:00",
        "val_end": "2022-12-31 23:59:59",
        "test_start": "2023-01-01 00:00:00",
        "test_end": "2023-12-31 23:59:59",
    },
    {
        "name": "fold_2019_2022_to_2023h2",
        "train_end": "2022-12-31 23:59:59",
        "val_start": "2023-01-01 00:00:00",
        "val_end": "2023-06-30 23:59:59",
        "test_start": "2023-07-01 00:00:00",
        "test_end": "2023-12-31 23:59:59",
    },
    {
        "name": "fold_2019_2023_to_2024h2",
        "train_end": "2023-12-31 23:59:59",
        "val_start": "2024-01-01 00:00:00",
        "val_end": "2024-06-30 23:59:59",
        "test_start": "2024-07-01 00:00:00",
        "test_end": "2024-12-31 23:59:59",
    },
]


@dataclass
class RawData:
    common_index: pd.DatetimeIndex
    weather_price: pd.DataFrame
    finance: pd.DataFrame


def utc_ts(value):
    return pd.Timestamp(value, tz="UTC")


def load_hourly_csv(filename, node_names):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[node_names]


def load_raw_data():
    nodes = pd.read_csv(os.path.join(DATA_DIR, "processed_nodes.csv"))
    node_names = nodes["name"].tolist()
    frames = {
        "price": load_hourly_csv("price_top100_2019-01-01_2024-12-31.csv", node_names),
        "ssr": load_hourly_csv("ssr_top100_2019-01-01_2024-12-31.csv", node_names),
        "t2m": load_hourly_csv("t2m_top100_2019-01-01_2024-12-31.csv", node_names),
        "u100": load_hourly_csv("u100_top100_2019-01-01_2024-12-31.csv", node_names),
        "v100": load_hourly_csv("v100_top100_2019-01-01_2024-12-31.csv", node_names),
    }
    common = frames["price"].index
    for df in frames.values():
        common = common.intersection(df.index)
    common = common.sort_values()

    agg = pd.DataFrame(index=common)
    for name, df in frames.items():
        values = df.loc[common].astype(np.float32)
        agg[f"{name}_node_mean"] = values.mean(axis=1)
        agg[f"{name}_node_std"] = values.std(axis=1)

    finance = pd.read_csv(
        os.path.join(DATA_DIR, "Finance20192024", "raw_finance.csv"),
        index_col=0,
        parse_dates=True,
    )
    finance.index = pd.to_datetime(finance.index, utc=True)
    finance = finance.dropna()
    return RawData(common, agg, finance)


def fit_fold_finance(finance, train_end):
    log_ret = np.log(finance / finance.shift(1)).dropna()
    train_mask = log_ret.index <= train_end
    if train_mask.sum() < 30:
        raise ValueError(f"Too few finance observations before {train_end}")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(log_ret.loc[train_mask])
    all_scaled = scaler.transform(log_ret)

    pca = PCA(n_components=3)
    pca.fit(train_scaled)
    pc = pd.DataFrame(
        pca.transform(all_scaled),
        index=log_ret.index,
        columns=["pc1", "pc2", "pc3"],
    )

    weights = pd.Series(pca.components_[0], index=log_ret.columns)
    weights = weights / weights.abs().sum()
    basket_return = log_ret.dot(weights)

    daily = pc.copy()
    daily["pc1_change"] = daily["pc1"].diff()
    daily["target_pc1_next"] = daily["pc1"].shift(-1) - daily["pc1"]
    daily["basket_return_next"] = basket_return.shift(-1)
    daily["target_time"] = daily.index.to_series().shift(-1) + pd.Timedelta(hours=23)
    daily["finance_feature_time"] = daily.index.to_series().shift(1) + pd.Timedelta(hours=23)
    daily["lag_pc1"] = daily["pc1"].shift(1)
    daily["lag_pc2"] = daily["pc2"].shift(1)
    daily["lag_pc3"] = daily["pc3"].shift(1)
    daily["prev_pc1_change"] = daily["pc1_change"].shift(1)
    daily["rolling5_pc1_change"] = daily["pc1_change"].shift(1).rolling(5, min_periods=1).mean()
    daily = daily.dropna(subset=["target_pc1_next", "basket_return_next", "target_time"])

    return daily, weights, {
        "columns": list(log_ret.columns),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_basket_weights": weights.to_dict(),
        "train_end": str(train_end),
    }


def rolling_hourly_features(hourly):
    out = pd.DataFrame(index=hourly.index)
    for col in hourly.columns:
        roll = hourly[col].rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE)
        out[f"{col}_win_mean"] = roll.mean()
        out[f"{col}_win_std"] = roll.std()
        out[f"{col}_last"] = hourly[col]
    out["sin_hour"] = np.sin(2 * np.pi * hourly.index.hour / 24)
    out["cos_hour"] = np.cos(2 * np.pi * hourly.index.hour / 24)
    out["sin_month"] = np.sin(2 * np.pi * (hourly.index.month - 1) / 12)
    out["cos_month"] = np.cos(2 * np.pi * (hourly.index.month - 1) / 12)
    return out


def build_fold_samples(raw, fold):
    train_end = utc_ts(fold["train_end"])
    val_start = utc_ts(fold["val_start"])
    val_end = utc_ts(fold["val_end"])
    test_start = utc_ts(fold["test_start"])
    test_end = utc_ts(fold["test_end"])

    hourly = raw.weather_price.copy()
    train_mask = hourly.index <= train_end
    for col in [c for c in hourly.columns if c.startswith(("ssr", "t2m", "u100", "v100"))]:
        mu = hourly.loc[train_mask, col].mean()
        sigma = hourly.loc[train_mask, col].std() + 1e-8
        hourly[col] = (hourly[col] - mu) / sigma
    for col in [c for c in hourly.columns if c.startswith("price")]:
        med = hourly.loc[train_mask, col].median()
        iqr = hourly.loc[train_mask, col].quantile(0.75) - hourly.loc[train_mask, col].quantile(0.25) + 1e-8
        hourly[col] = (hourly[col] - med) / iqr

    hourly_features = rolling_hourly_features(hourly)
    daily, weights, pca_meta = fit_fold_finance(raw.finance, train_end)

    sample_rows = []
    eligible_ends = hourly_features.index[
        (hourly_features.index.hour == 23)
        & hourly_features.notna().all(axis=1)
    ]
    for window_end in eligible_ends:
        day = window_end.normalize()
        if day not in daily.index:
            continue
        drow = daily.loc[day]
        target_time = drow["target_time"]
        if not (window_end < target_time):
            continue
        row = hourly_features.loc[window_end].to_dict()
        row.update({
            "window_end": window_end,
            "target_time": target_time,
            "finance_feature_time": drow["finance_feature_time"],
            "y": float(drow["target_pc1_next"]),
            "basket_return_next": float(drow["basket_return_next"]),
            "lag_pc1": float(drow.get("lag_pc1", 0.0) if pd.notna(drow.get("lag_pc1", np.nan)) else 0.0),
            "lag_pc2": float(drow.get("lag_pc2", 0.0) if pd.notna(drow.get("lag_pc2", np.nan)) else 0.0),
            "lag_pc3": float(drow.get("lag_pc3", 0.0) if pd.notna(drow.get("lag_pc3", np.nan)) else 0.0),
            "prev_pc1_change": float(drow.get("prev_pc1_change", 0.0) if pd.notna(drow.get("prev_pc1_change", np.nan)) else 0.0),
            "rolling5_pc1_change": float(drow.get("rolling5_pc1_change", 0.0) if pd.notna(drow.get("rolling5_pc1_change", np.nan)) else 0.0),
        })
        sample_rows.append(row)

    samples = pd.DataFrame(sample_rows).sort_values("window_end").reset_index(drop=True)
    samples["split"] = "unused"
    samples.loc[samples["target_time"] <= train_end, "split"] = "train"
    samples.loc[(samples["target_time"] >= val_start) & (samples["target_time"] <= val_end), "split"] = "val"
    samples.loc[(samples["target_time"] >= test_start) & (samples["target_time"] <= test_end), "split"] = "test"

    if not (samples["window_end"] < samples["target_time"]).all():
        raise AssertionError("Found sample with target_time not after window_end")
    available = samples["finance_feature_time"].isna() | (samples["finance_feature_time"] < samples["window_end"])
    if not available.all():
        raise AssertionError("Found finance feature timestamp not before window_end")

    return samples, pca_meta


def feature_sets(columns):
    weather = [c for c in columns if c.startswith(("ssr_", "t2m_", "u100_", "v100_"))]
    electricity = [c for c in columns if c.startswith("price_")]
    finance = ["lag_pc1", "lag_pc2", "lag_pc3", "prev_pc1_change", "rolling5_pc1_change"]
    time_cols = ["sin_hour", "cos_hour", "sin_month", "cos_month"]
    return {
        "ridge_full": weather + electricity + finance + time_cols,
        "ridge_finance_only": finance,
        "ridge_weather_only": weather + time_cols,
        "ridge_electricity_only": electricity + time_cols,
        "ridge_weather_electricity": weather + electricity + time_cols,
        "ridge_no_finance": weather + electricity + time_cols,
        "linear_full": weather + electricity + finance + time_cols,
        "gradient_boosting_full": weather + electricity + finance + time_cols,
    }


def fit_predict_models(samples):
    train = samples[samples["split"] == "train"].copy()
    test = samples[samples["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Each fold needs non-empty train and test samples")

    preds = []
    base = test[["window_end", "target_time", "y", "basket_return_next"]].copy()

    def add_prediction(name, pred):
        df = base.copy()
        df["model"] = name
        df["y_pred"] = np.asarray(pred, dtype=float)
        preds.append(df)

    add_prediction("zero_change", np.zeros(len(test)))
    add_prediction("previous_pc1_change", test["prev_pc1_change"].values)
    add_prediction("rolling5_mean", test["rolling5_pc1_change"].values)

    sets = feature_sets(samples.columns)
    for name, cols in sets.items():
        if name == "linear_full":
            model = LinearRegression()
        elif name == "gradient_boosting_full":
            model = GradientBoostingRegressor(random_state=RANDOM_SEED)
        else:
            model = Ridge(alpha=1.0)
        model.fit(train[cols].fillna(0.0), train["y"])
        add_prediction(name, model.predict(test[cols].fillna(0.0)))

    return pd.concat(preds, ignore_index=True)


def max_drawdown(returns):
    if len(returns) == 0:
        return np.nan
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity - peak))


def information_coefficient(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan, np.nan
    ic = float(np.corrcoef(y_true, y_pred)[0, 1])
    denom = max(1e-12, 1 - ic**2)
    t_stat = ic * math.sqrt((len(y_true) - 2) / denom)
    return ic, float(t_stat)


def bootstrap_ci(values, fn, n_boot=500, seed=RANDOM_SEED):
    values = np.asarray(values)
    if len(values) < 2:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        stats.append(fn(values[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def metrics_for_predictions(df, tx_cost_bps=TX_COST_BPS):
    rows = []
    cost_rate = tx_cost_bps / 10000.0
    for model, g in df.groupby("model"):
        g = g.sort_values("target_time")
        y = g["y"].to_numpy(float)
        pred = g["y_pred"].to_numpy(float)
        err = pred - y
        signal = np.sign(pred)
        gross = signal * g["basket_return_next"].to_numpy(float)
        turnover = np.abs(np.diff(np.r_[0.0, signal]))
        net = gross - cost_rate * turnover
        ic, ic_t = information_coefficient(y, pred)
        sharpe = np.nan
        if np.std(net) > 0:
            sharpe = float(np.mean(net) / np.std(net) * math.sqrt(252))
        rows.append({
            "model": model,
            "n": int(len(g)),
            "mse": float(np.mean(err**2)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(y))),
            "ic": ic,
            "ic_t_stat": ic_t,
            "annualized_return": float(np.mean(net) * 252),
            "annualized_volatility": float(np.std(net) * math.sqrt(252)),
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(net),
            "mean_turnover": float(np.mean(turnover)),
            "gross_pnl": float(np.sum(gross)),
            "net_pnl": float(np.sum(net)),
            "tx_cost_bps": float(tx_cost_bps),
            "mse_ci95": bootstrap_ci(err, lambda x: np.mean(x**2)),
            "ic_ci95": bootstrap_ci(np.column_stack([y, pred]), lambda x: np.corrcoef(x[:, 0], x[:, 1])[0, 1] if np.std(x[:, 0]) > 0 and np.std(x[:, 1]) > 0 else np.nan),
            "sharpe_ci95": bootstrap_ci(net, lambda x: np.mean(x) / np.std(x) * math.sqrt(252) if np.std(x) > 0 else np.nan),
            "directional_accuracy_ci95": bootstrap_ci(np.column_stack([y, pred]), lambda x: np.mean(np.sign(x[:, 1]) == np.sign(x[:, 0]))),
        })
    return pd.DataFrame(rows).sort_values("mse")


def dm_comparison(predictions, metrics):
    baseline_models = [m for m in metrics["model"] if m != "gradient_boosting_full"]
    best_baseline = metrics[metrics["model"].isin(baseline_models)].sort_values("mse").iloc[0]["model"]
    base = predictions[predictions["model"] == best_baseline][["target_time", "y", "y_pred"]].rename(columns={"y_pred": "base_pred"})
    rows = []
    for model in predictions["model"].unique():
        if model == best_baseline:
            continue
        g = predictions[predictions["model"] == model][["target_time", "y_pred"]]
        merged = base.merge(g, on="target_time", how="inner")
        if len(merged) < 3:
            continue
        d = (merged["y"] - merged["y_pred"]) ** 2 - (merged["y"] - merged["base_pred"]) ** 2
        se = d.std(ddof=1) / math.sqrt(len(d)) if d.std(ddof=1) > 0 else np.nan
        t_stat = float(d.mean() / se) if se and not np.isnan(se) else np.nan
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2))) if not np.isnan(t_stat) else np.nan
        rows.append({
            "model": model,
            "best_baseline": best_baseline,
            "mean_loss_diff_model_minus_baseline": float(d.mean()),
            "dm_t_stat": t_stat,
            "normal_approx_p_value": p_value,
        })
    return pd.DataFrame(rows)


def regime_label(ts):
    ts = pd.Timestamp(ts)
    if ts < utc_ts("2022-01-01"):
        return "pre_2022"
    if ts < utc_ts("2023-01-01"):
        return "energy_crisis_2022"
    if ts < utc_ts("2024-01-01"):
        return "post_2022"
    return "test_2024"


def regime_metrics(predictions):
    tmp = predictions.copy()
    tmp["regime"] = tmp["target_time"].map(regime_label)
    rows = []
    for (model, regime), g in tmp.groupby(["model", "regime"]):
        m = metrics_for_predictions(g)
        if not m.empty:
            row = m.iloc[0].to_dict()
            row["model"] = model
            row["regime"] = regime
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_stgcn_predictions(path, out_dir):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["window_end", "target_time"])
    df = df.rename(columns={"y_true": "y", "y_pred": "y_pred"})
    df["basket_return_next"] = df["y"]
    metrics = metrics_for_predictions(df[["window_end", "target_time", "y", "basket_return_next", "model", "y_pred"]])
    output_path = os.path.join(out_dir, "stgcn_prediction_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(orient="records"), f, indent=2)
    return output_path


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    raw = load_raw_data()
    all_metrics = []
    all_regimes = []
    all_dm = []
    pca_meta = {}

    for fold in FOLDS:
        print(f"\n=== {fold['name']} ===")
        samples, meta = build_fold_samples(raw, fold)
        pca_meta[fold["name"]] = meta
        predictions = fit_predict_models(samples)
        fold_dir = os.path.join(args.output_dir, fold["name"])
        os.makedirs(fold_dir, exist_ok=True)
        samples.to_csv(os.path.join(fold_dir, "samples.csv"), index=False)
        predictions.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        metrics = metrics_for_predictions(predictions, args.tx_cost_bps)
        metrics["fold"] = fold["name"]
        metrics.to_json(os.path.join(fold_dir, "metrics.json"), orient="records", indent=2)
        metrics.to_csv(os.path.join(fold_dir, "metrics.csv"), index=False)
        all_metrics.append(metrics)

        regimes = regime_metrics(predictions)
        regimes["fold"] = fold["name"]
        regimes.to_csv(os.path.join(fold_dir, "regime_metrics.csv"), index=False)
        all_regimes.append(regimes)

        dm = dm_comparison(predictions, metrics)
        dm["fold"] = fold["name"]
        dm.to_csv(os.path.join(fold_dir, "dm_comparison.csv"), index=False)
        all_dm.append(dm)
        print(metrics[["model", "mse", "directional_accuracy", "sharpe", "net_pnl"]].to_string(index=False))

    summary = pd.concat(all_metrics, ignore_index=True)
    summary.to_csv(os.path.join(args.output_dir, "summary_metrics.csv"), index=False)
    pd.concat(all_regimes, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "summary_regime_metrics.csv"), index=False)
    pd.concat(all_dm, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "summary_dm_comparison.csv"), index=False)

    stgcn_metrics_path = evaluate_stgcn_predictions(args.stgcn_predictions, args.output_dir)

    report = {
        "folds": FOLDS,
        "tx_cost_bps": args.tx_cost_bps,
        "pca_meta": pca_meta,
        "outputs": {
            "summary_metrics": os.path.join(args.output_dir, "summary_metrics.csv"),
            "summary_regime_metrics": os.path.join(args.output_dir, "summary_regime_metrics.csv"),
            "summary_dm_comparison": os.path.join(args.output_dir, "summary_dm_comparison.csv"),
            "stgcn_prediction_metrics": stgcn_metrics_path,
        },
    }
    with open(os.path.join(args.output_dir, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward quant evaluation")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--tx_cost_bps", type=float, default=TX_COST_BPS)
    parser.add_argument("--stgcn_predictions", default=os.path.join(OUTPUT_DIR, "stgcn_predictions.csv"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
