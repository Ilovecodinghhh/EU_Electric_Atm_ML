"""
Statistical robustness report for saved cross-sectional OOS predictions.

The report computes daily IC/RankIC, long-short backtest returns, regime
slices, and moving-block bootstrap confidence intervals.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from cross_sectional_evaluate import (
    long_short_backtest,
    max_drawdown,
    pearson_corr,
    spearman_corr,
    t_stat,
)


DEFAULT_REGIMES = [
    ("2023_h1", "2023-01-01", "2023-06-30"),
    ("2023_h2", "2023-07-01", "2023-12-31"),
    ("2024_h2", "2024-07-01", "2024-12-31"),
]


def parse_model_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def prediction_path(root, horizon, prediction_file):
    return os.path.join(root, f"horizon_{horizon}d", prediction_file)


def load_predictions(path, models):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction file: {path}")
    predictions = pd.read_csv(path, parse_dates=["date", "decision_time", "target_time"])
    if models:
        predictions = predictions[predictions["model"].isin(models)].copy()
        if predictions.empty:
            raise ValueError(f"No requested models found in {path}")
    return predictions


def daily_model_stats(
    predictions,
    tx_cost_bps,
    portfolio_group_col=None,
    portfolio_mode="daily",
    selection_buffer=0.0,
):
    rows = []
    for model, group in predictions.groupby("model"):
        backtest = long_short_backtest(
            group,
            tx_cost_bps=tx_cost_bps,
            group_col=portfolio_group_col,
            portfolio_mode=portfolio_mode,
            selection_buffer=selection_buffer,
        ).set_index("target_time")

        for target_time, day in group.groupby("target_time"):
            bt = backtest.loc[target_time]
            rows.append({
                "model": model,
                "target_time": target_time,
                "n_names": int(day["ticker"].nunique()),
                "ic": pearson_corr(day["target_residual_next"], day["score"]),
                "rankic": spearman_corr(day["target_residual_next"], day["score"]),
                "directional_accuracy": float(
                    np.mean(np.sign(day["score"]) == np.sign(day["target_residual_next"]))
                ),
                "gross_return": float(bt["gross_return"]),
                "net_return": float(bt["net_return"]),
                "turnover": float(bt["turnover"]),
                "gross_exposure": float(bt["gross_exposure"]),
                "net_exposure": float(bt["net_exposure"]),
                "max_group_net_exposure": float(bt["max_group_net_exposure"])
                if "max_group_net_exposure" in bt and pd.notna(bt["max_group_net_exposure"])
                else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["model", "target_time"])


def summarize_daily_stats(daily, periods_per_year):
    rows = []
    for model, group in daily.groupby("model"):
        net = group["net_return"].to_numpy(float)
        gross = group["gross_return"].to_numpy(float)
        rankic = group["rankic"].dropna()
        ic = group["ic"].dropna()
        rows.append({
            "model": model,
            "n_days": int(group["target_time"].nunique()),
            "mean_ic": float(ic.mean()) if len(ic) else np.nan,
            "mean_rankic": float(rankic.mean()) if len(rankic) else np.nan,
            "ic_t_stat": t_stat(ic),
            "rankic_t_stat": t_stat(rankic),
            "directional_accuracy": float(group["directional_accuracy"].mean()),
            "gross_pnl": float(np.sum(gross)),
            "net_pnl": float(np.sum(net)),
            "annualized_return": float(np.mean(net) * periods_per_year),
            "annualized_volatility": float(np.std(net) * math.sqrt(periods_per_year)),
            "sharpe": float(np.mean(net) / np.std(net) * math.sqrt(periods_per_year))
            if np.std(net) > 0 else np.nan,
            "max_drawdown": max_drawdown(net),
            "mean_turnover": float(group["turnover"].mean()),
            "max_abs_net_exposure": float(group["net_exposure"].abs().max()),
            "max_group_net_exposure": float(group["max_group_net_exposure"].max(skipna=True))
            if group["max_group_net_exposure"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["mean_rankic", "net_pnl"], ascending=[False, False])


def moving_block_indices(n, block_size, rng):
    if n <= 0:
        return np.array([], dtype=int)
    block_size = max(1, min(block_size, n))
    starts = rng.integers(0, n - block_size + 1, size=math.ceil(n / block_size))
    indices = np.concatenate([np.arange(start, start + block_size) for start in starts])
    return indices[:n]


def sample_metric(values, periods_per_year):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "mean": np.nan,
            "sum": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    std = np.std(values)
    return {
        "mean": float(np.mean(values)),
        "sum": float(np.sum(values)),
        "sharpe": float(np.mean(values) / std * math.sqrt(periods_per_year)) if std > 0 else np.nan,
        "max_drawdown": max_drawdown(values),
    }


def bootstrap_model_stats(group, n_boot, block_size, seed, periods_per_year):
    rng = np.random.default_rng(seed)
    group = group.sort_values("target_time").reset_index(drop=True)
    n = len(group)
    observed = {
        "mean_rankic": float(group["rankic"].mean(skipna=True)),
        "mean_ic": float(group["ic"].mean(skipna=True)),
        "net_pnl": float(group["net_return"].sum()),
        "sharpe": sample_metric(group["net_return"], periods_per_year)["sharpe"],
        "mean_turnover": float(group["turnover"].mean()),
    }
    samples = {key: [] for key in observed}
    rankic = group["rankic"].to_numpy(float)
    ic = group["ic"].to_numpy(float)
    net = group["net_return"].to_numpy(float)
    turnover = group["turnover"].to_numpy(float)

    for _ in range(n_boot):
        idx = moving_block_indices(n, block_size, rng)
        samples["mean_rankic"].append(float(np.nanmean(rankic[idx])))
        samples["mean_ic"].append(float(np.nanmean(ic[idx])))
        samples["net_pnl"].append(float(np.nansum(net[idx])))
        samples["sharpe"].append(sample_metric(net[idx], periods_per_year)["sharpe"])
        samples["mean_turnover"].append(float(np.nanmean(turnover[idx])))

    rows = []
    for metric, values in samples.items():
        arr = np.asarray(values, dtype=float)
        rows.append({
            "metric": metric,
            "observed": observed[metric],
            "ci_low": float(np.nanpercentile(arr, 2.5)),
            "ci_high": float(np.nanpercentile(arr, 97.5)),
            "bootstrap_mean": float(np.nanmean(arr)),
            "p_le_zero": float(np.nanmean(arr <= 0.0)),
            "n_boot": int(n_boot),
            "block_size": int(block_size),
        })
    return pd.DataFrame(rows)


def bootstrap_report(daily, n_boot, block_size, seed, periods_per_year):
    rows = []
    for offset, (model, group) in enumerate(daily.groupby("model")):
        report = bootstrap_model_stats(group, n_boot, block_size, seed + offset, periods_per_year)
        report.insert(0, "model", model)
        rows.append(report)
    return pd.concat(rows, ignore_index=True)


def regime_filter(daily, start, end):
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return daily[(daily["target_time"] >= start_ts) & (daily["target_time"] <= end_ts)].copy()


def regime_report(daily, periods_per_year):
    rows = []
    regimes = [("all_oos", None, None)] + DEFAULT_REGIMES
    for name, start, end in regimes:
        regime_daily = daily if start is None else regime_filter(daily, start, end)
        if regime_daily.empty:
            continue
        summary = summarize_daily_stats(regime_daily, periods_per_year)
        summary.insert(0, "regime", name)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run(args):
    group_col = None if args.portfolio_group == "none" else f"{args.portfolio_group}_group"
    models = parse_model_list(args.models)
    prediction_file = prediction_path(args.predictions_root, args.horizon, args.prediction_file)
    predictions = load_predictions(prediction_file, models)
    if group_col and group_col not in predictions.columns:
        raise ValueError(
            f"Prediction file {prediction_file} does not contain required column {group_col}."
        )
    periods_per_year = 252.0 / args.horizon

    daily = daily_model_stats(
        predictions,
        tx_cost_bps=args.tx_cost_bps,
        portfolio_group_col=group_col,
        portfolio_mode=args.portfolio_mode,
        selection_buffer=args.selection_buffer,
    )
    summary = summarize_daily_stats(daily, periods_per_year)
    regimes = regime_report(daily, periods_per_year)
    bootstrap = bootstrap_report(
        daily,
        n_boot=args.n_boot,
        block_size=args.block_size,
        seed=args.seed,
        periods_per_year=periods_per_year,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    daily_path = os.path.join(args.output_dir, "daily_stats.csv")
    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    regime_path = os.path.join(args.output_dir, "regime_metrics.csv")
    bootstrap_path = os.path.join(args.output_dir, "bootstrap_ci.csv")
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    regimes.to_csv(regime_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    report = {
        "prediction_file": prediction_file,
        "horizon_days": args.horizon,
        "models": models,
        "tx_cost_bps": args.tx_cost_bps,
        "portfolio_group": args.portfolio_group,
        "portfolio_mode": args.portfolio_mode,
        "selection_buffer": args.selection_buffer,
        "periods_per_year": periods_per_year,
        "n_boot": args.n_boot,
        "block_size": args.block_size,
        "seed": args.seed,
        "outputs": {
            "daily_stats": daily_path,
            "summary_metrics": summary_path,
            "regime_metrics": regime_path,
            "bootstrap_ci": bootstrap_path,
        },
    }
    with open(os.path.join(args.output_dir, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(summary.to_string(index=False))
    print("\nBootstrap CI:")
    print(bootstrap.to_string(index=False))
    return summary, regimes, bootstrap


def main():
    parser = argparse.ArgumentParser(description="Bootstrap/regime report for saved OOS predictions")
    parser.add_argument("--predictions_root", required=True)
    parser.add_argument("--prediction_file", default="unique_oos_predictions.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--models", required=True)
    parser.add_argument("--tx_cost_bps", type=float, default=10.0)
    parser.add_argument("--portfolio_group", default="none", choices=["none", "country", "subsector"])
    parser.add_argument("--portfolio_mode", default="buffered", choices=["daily", "buffered"])
    parser.add_argument("--selection_buffer", type=float, default=0.15)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.n_boot <= 0:
        raise ValueError("--n_boot must be positive")
    if args.block_size <= 0:
        raise ValueError("--block_size must be positive")
    run(args)


if __name__ == "__main__":
    main()
