import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import __version__ as sklearn_version

from .data import daily_country_prices, daily_forecast_features


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NODES = ROOT / "data" / "processed_nodes.csv"
DEFAULT_PRICES = ROOT / "data" / "price_top100_2019-01-01_2024-12-31.csv"
DEFAULT_FORECASTS = ROOT / "data" / "forecast_revisions" / "country_hourly.csv"
DEFAULT_OUTPUT = ROOT / "quant_output" / "electricity_forecast"
RANDOM_SEED = 11

FOLDS = [
    {
        "name": "train_mar_may_test_jul_aug",
        "train_end": "2024-05-31",
        "val_start": "2024-06-01",
        "val_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2024-08-31",
    },
    {
        "name": "train_mar_jul_test_sep_oct",
        "train_end": "2024-07-31",
        "val_start": "2024-08-01",
        "val_end": "2024-08-31",
        "test_start": "2024-09-01",
        "test_end": "2024-10-31",
    },
    {
        "name": "train_mar_sep_test_nov_dec",
        "train_end": "2024-09-30",
        "val_start": "2024-10-01",
        "val_end": "2024-10-31",
        "test_start": "2024-11-01",
        "test_end": "2024-12-31",
    },
]


@dataclass
class PreparedPanel:
    panel: pd.DataFrame
    feature_sets: dict


class TrainOnlyTransformer:
    def __init__(self, columns, lower=0.01, upper=0.99):
        self.columns = list(columns)
        self.lower = lower
        self.upper = upper
        self.median = None
        self.lo = None
        self.hi = None
        self.scaler = StandardScaler()

    def fit(self, frame):
        values = frame[self.columns].astype(float).replace([np.inf, -np.inf], np.nan)
        self.median = values.median()
        values = values.fillna(self.median).fillna(0.0)
        self.lo = values.quantile(self.lower)
        self.hi = values.quantile(self.upper)
        values = values.clip(lower=self.lo, upper=self.hi, axis=1)
        self.scaler.fit(values)
        return self

    def transform(self, frame):
        values = frame[self.columns].astype(float).replace([np.inf, -np.inf], np.nan)
        values = values.fillna(self.median).fillna(0.0)
        values = values.clip(lower=self.lo, upper=self.hi, axis=1)
        return self.scaler.transform(values)


def build_panel(nodes_path=DEFAULT_NODES, price_path=DEFAULT_PRICES, forecast_path=DEFAULT_FORECASTS):
    prices = daily_country_prices(nodes_path, price_path)
    forecasts = daily_forecast_features(forecast_path)
    panel = forecasts.merge(prices, on=["country", "target_date"], how="inner")
    panel = panel.sort_values(["country", "target_date"]).reset_index(drop=True)

    grouped = panel.groupby("country", group_keys=False)["target_price"]
    panel["price_lag1"] = grouped.shift(1)
    panel["price_lag2"] = grouped.shift(2)
    panel["price_lag7"] = grouped.shift(7)
    panel["price_change_lag1"] = panel["price_lag1"] - panel["price_lag2"]
    panel["price_roll7"] = grouped.transform(lambda series: series.shift(1).rolling(7, min_periods=4).mean())
    panel["price_roll30"] = grouped.transform(lambda series: series.shift(1).rolling(30, min_periods=14).mean())
    panel["price_vol7"] = grouped.transform(lambda series: series.shift(1).rolling(7, min_periods=4).std())

    day_of_week = panel["target_date"].dt.dayofweek
    day_of_year = panel["target_date"].dt.dayofyear
    panel["sin_dow"] = np.sin(2 * np.pi * day_of_week / 7)
    panel["cos_dow"] = np.cos(2 * np.pi * day_of_week / 7)
    panel["sin_year"] = np.sin(2 * np.pi * day_of_year / 366)
    panel["cos_year"] = np.cos(2 * np.pi * day_of_year / 366)
    panel["decision_time"] = panel["target_date"] - pd.Timedelta(days=1) + pd.Timedelta(hours=9)

    country_dummies = pd.get_dummies(panel["country"], prefix="country", dtype=float)
    panel = pd.concat([panel, country_dummies], axis=1)
    panel = panel.dropna(subset=["target_price", "price_lag1", "price_roll7"]).copy()
    if not (panel["forecast_available_time"] < panel["decision_time"]).all():
        raise AssertionError("Forecast feature was not available before the decision time")
    if not (panel["decision_time"] < panel["target_date"]).all():
        raise AssertionError("Decision time must precede electricity delivery")

    price_columns = [
        "price_lag1", "price_lag2", "price_lag7", "price_change_lag1",
        "price_roll7", "price_roll30", "price_vol7",
    ]
    calendar_columns = ["sin_dow", "cos_dow", "sin_year", "cos_year"]
    country_columns = country_dummies.columns.tolist()
    forecast_level_columns = [
        column for column in panel.columns
        if "_previous_day2_" in column
    ]
    revision_columns = [column for column in panel.columns if "_revision_" in column]
    feature_sets = {
        "price_only": price_columns + calendar_columns + country_columns,
        "forecast_level_only": forecast_level_columns + calendar_columns + country_columns,
        "revision_only": revision_columns + calendar_columns + country_columns,
        "price_plus_revision": price_columns + revision_columns + calendar_columns + country_columns,
        "full": price_columns + forecast_level_columns + revision_columns + calendar_columns + country_columns,
    }
    return PreparedPanel(panel.reset_index(drop=True), feature_sets)


def utc_day(value):
    return pd.Timestamp(value, tz="UTC")


def split_panel(panel, fold):
    train = panel[panel["target_date"] <= utc_day(fold["train_end"])].copy()
    validation = panel[
        panel["target_date"].between(utc_day(fold["val_start"]), utc_day(fold["val_end"]))
    ].copy()
    test = panel[
        panel["target_date"].between(utc_day(fold["test_start"]), utc_day(fold["test_end"]))
    ].copy()
    return train, validation, test


def ridge_predict(train, validation, test, columns, alphas=(0.1, 1.0, 10.0, 100.0)):
    transformer = TrainOnlyTransformer(columns).fit(train)
    x_train = transformer.transform(train)
    x_validation = transformer.transform(validation)
    best_alpha = None
    best_mae = np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train, train["target_price"])
        mae = mean_absolute_error(validation["target_price"], model.predict(x_validation))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    fit_frame = pd.concat([train, validation], ignore_index=True)
    final_transformer = TrainOnlyTransformer(columns).fit(fit_frame)
    model = Ridge(alpha=best_alpha)
    model.fit(final_transformer.transform(fit_frame), fit_frame["target_price"])
    prediction = model.predict(final_transformer.transform(test))
    return prediction, float(best_alpha), float(best_mae)


def hist_gradient_predict(train, validation, test, columns):
    transformer = TrainOnlyTransformer(columns).fit(train)
    candidates = [
        {"learning_rate": 0.03, "max_leaf_nodes": 7},
        {"learning_rate": 0.05, "max_leaf_nodes": 15},
    ]
    best = None
    best_mae = np.inf
    for params in candidates:
        model = HistGradientBoostingRegressor(
            max_iter=150,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
            **params,
        )
        model.fit(transformer.transform(train), train["target_price"])
        mae = mean_absolute_error(validation["target_price"], model.predict(transformer.transform(validation)))
        if mae < best_mae:
            best_mae = mae
            best = params

    fit_frame = pd.concat([train, validation], ignore_index=True)
    final_transformer = TrainOnlyTransformer(columns).fit(fit_frame)
    model = HistGradientBoostingRegressor(
        max_iter=150,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
        **best,
    )
    model.fit(final_transformer.transform(fit_frame), fit_frame["target_price"])
    prediction = model.predict(final_transformer.transform(test))
    return prediction, best, float(best_mae)


def add_predictions(base, model, prediction, fold_name, metadata=None):
    out = base.copy()
    out["model"] = model
    out["prediction"] = np.asarray(prediction, dtype=float)
    out["fold"] = fold_name
    out["model_metadata"] = json.dumps(metadata or {}, sort_keys=True)
    return out


def fit_predict_fold(prepared, fold):
    train, validation, test = split_panel(prepared.panel, fold)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(f"Fold {fold['name']} contains an empty split")
    base = test[["target_date", "decision_time", "country", "target_price"]].copy()
    predictions = [
        add_predictions(base, "persistence", test["price_lag1"], fold["name"]),
        add_predictions(base, "rolling_7d", test["price_roll7"], fold["name"]),
    ]
    for feature_name, columns in prepared.feature_sets.items():
        prediction, alpha, validation_mae = ridge_predict(train, validation, test, columns)
        predictions.append(add_predictions(
            base,
            f"ridge_{feature_name}",
            prediction,
            fold["name"],
            {"alpha": alpha, "validation_mae": validation_mae, "n_features": len(columns)},
        ))
    prediction, params, validation_mae = hist_gradient_predict(
        train,
        validation,
        test,
        prepared.feature_sets["full"],
    )
    predictions.append(add_predictions(
        base,
        "hist_gradient_full",
        prediction,
        fold["name"],
        {**params, "validation_mae": validation_mae, "n_features": len(prepared.feature_sets["full"])},
    ))
    return pd.concat(predictions, ignore_index=True), train, validation, test


def metric_table(predictions):
    baseline = predictions[predictions["model"] == "persistence"].copy()
    baseline_mae = mean_absolute_error(baseline["target_price"], baseline["prediction"])
    rows = []
    for model, group in predictions.groupby("model"):
        y = group["target_price"].to_numpy(float)
        pred = group["prediction"].to_numpy(float)
        mae = mean_absolute_error(y, pred)
        rows.append({
            "model": model,
            "n_obs": int(len(group)),
            "n_days": int(group["target_date"].nunique()),
            "mae": float(mae),
            "rmse": float(math.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "bias": float(np.mean(pred - y)),
            "pearson": float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 else np.nan,
            "mae_improvement_vs_persistence": float(baseline_mae - mae),
            "mae_improvement_pct": float((baseline_mae - mae) / baseline_mae * 100.0),
        })
    return pd.DataFrame(rows).sort_values("mae")


def moving_block_indices(n, block_size, rng):
    block_size = max(1, min(int(block_size), n))
    starts = rng.integers(0, n - block_size + 1, size=math.ceil(n / block_size))
    return np.concatenate([np.arange(start, start + block_size) for start in starts])[:n]


def newey_west_t_stat(values, max_lag=7):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return np.nan
    centered = values - values.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    if long_run_variance <= 0:
        return np.nan
    return float(values.mean() / math.sqrt(long_run_variance / n))


def bootstrap_mae_improvement(predictions, n_boot=1000, block_size=7, seed=123):
    baseline = predictions[predictions["model"] == "persistence"][
        ["target_date", "country", "target_price", "prediction"]
    ].rename(columns={"prediction": "baseline_prediction"})
    rng = np.random.default_rng(seed)
    rows = []
    for model, group in predictions.groupby("model"):
        merged = group.merge(baseline, on=["target_date", "country", "target_price"], how="inner")
        merged["loss_improvement"] = (
            (merged["target_price"] - merged["baseline_prediction"]).abs()
            - (merged["target_price"] - merged["prediction"]).abs()
        )
        daily = merged.groupby("target_date")["loss_improvement"].mean().sort_index()
        observed = float(daily.mean())
        samples = np.empty(n_boot, dtype=float)
        values = daily.to_numpy(float)
        for i in range(n_boot):
            samples[i] = values[moving_block_indices(len(values), block_size, rng)].mean()
        rows.append({
            "model": model,
            "observed_mae_improvement": observed,
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "p_le_zero": float(np.mean(samples <= 0.0)),
            "newey_west_t_stat": newey_west_t_stat(values, max_lag=block_size),
            "n_days": int(len(values)),
            "n_boot": int(n_boot),
            "block_size": int(block_size),
        })
    return pd.DataFrame(rows).sort_values("observed_mae_improvement", ascending=False)


def bootstrap_pairwise_mae_improvement(
    predictions,
    comparisons,
    n_boot=1000,
    block_size=7,
    seed=321,
):
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["target_date", "country", "target_price"]
    for challenger, reference in comparisons:
        challenger_frame = predictions[predictions["model"] == challenger][keys + ["prediction"]]
        reference_frame = predictions[predictions["model"] == reference][keys + ["prediction"]]
        merged = challenger_frame.merge(
            reference_frame,
            on=keys,
            suffixes=("_challenger", "_reference"),
            how="inner",
        )
        merged["loss_improvement"] = (
            (merged["target_price"] - merged["prediction_reference"]).abs()
            - (merged["target_price"] - merged["prediction_challenger"]).abs()
        )
        daily = merged.groupby("target_date")["loss_improvement"].mean().sort_index()
        values = daily.to_numpy(float)
        samples = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            samples[i] = values[moving_block_indices(len(values), block_size, rng)].mean()
        rows.append({
            "challenger": challenger,
            "reference": reference,
            "observed_mae_improvement": float(values.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "p_le_zero": float(np.mean(samples <= 0.0)),
            "newey_west_t_stat": newey_west_t_stat(values, max_lag=block_size),
            "n_days": int(len(values)),
            "n_boot": int(n_boot),
            "block_size": int(block_size),
        })
    return pd.DataFrame(rows).sort_values("observed_mae_improvement", ascending=False)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = build_panel(args.nodes, args.prices, args.forecasts)
    all_predictions = []
    fold_metrics = []

    for fold in FOLDS:
        predictions, train, validation, test = fit_predict_fold(prepared, fold)
        fold_dir = output_dir / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)
        metrics = metric_table(predictions)
        metrics["fold"] = fold["name"]
        predictions.to_csv(fold_dir / "predictions.csv", index=False)
        metrics.to_csv(fold_dir / "metrics.csv", index=False)
        all_predictions.append(predictions)
        fold_metrics.append(metrics)
        print(f"finished {fold['name']}: train={len(train)} val={len(validation)} test={len(test)}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    duplicate_count = predictions.duplicated(["model", "target_date", "country"]).sum()
    if duplicate_count:
        raise AssertionError(f"Overlapping OOS predictions found: {duplicate_count}")
    metrics = metric_table(predictions)
    bootstrap = bootstrap_mae_improvement(
        predictions,
        n_boot=args.n_boot,
        block_size=args.block_size,
        seed=args.seed,
    )
    incremental = bootstrap_pairwise_mae_improvement(
        predictions,
        comparisons=[
            ("ridge_price_plus_revision", "ridge_price_only"),
            ("ridge_full", "ridge_price_only"),
            ("ridge_full", "ridge_price_plus_revision"),
            ("hist_gradient_full", "ridge_full"),
            ("ridge_forecast_level_only", "persistence"),
        ],
        n_boot=args.n_boot,
        block_size=args.block_size,
        seed=args.seed + 1,
    )
    country_rows = []
    for country, group in predictions.groupby("country"):
        country_metric = metric_table(group)
        country_metric["country"] = country
        country_rows.append(country_metric)
    country_metrics = pd.concat(country_rows, ignore_index=True)

    predictions.to_csv(output_dir / "oos_predictions.csv", index=False)
    pd.concat(fold_metrics, ignore_index=True).to_csv(output_dir / "fold_metrics.csv", index=False)
    metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_mae_improvement.csv", index=False)
    incremental.to_csv(output_dir / "feature_increment_bootstrap.csv", index=False)
    country_metrics.to_csv(output_dir / "country_metrics.csv", index=False)
    report = {
        "research_target": "next-day country average day-ahead electricity price",
        "decision_time": "09:00 UTC one day before delivery",
        "forecast_revision": "day-2 minus day-3 forecast for the same delivery hour",
        "target_day_rule": "UTC calendar day with at least 24 observed hourly prices",
        "runtime": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
        },
        "feature_sets": {name: columns for name, columns in prepared.feature_sets.items()},
        "folds": FOLDS,
        "n_oos_days": int(predictions["target_date"].nunique()),
        "n_countries": int(predictions["country"].nunique()),
        "best_model_by_mae": str(metrics.iloc[0]["model"]),
        "best_mae": float(metrics.iloc[0]["mae"]),
        "persistence_mae": float(metrics.loc[metrics["model"] == "persistence", "mae"].iloc[0]),
        "revision_increment_vs_price_only": incremental[
            (incremental["challenger"] == "ridge_price_plus_revision")
            & (incremental["reference"] == "ridge_price_only")
        ].iloc[0].to_dict(),
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nOOS summary")
    print(metrics.to_string(index=False))
    print("\nBootstrap MAE improvement")
    print(bootstrap.to_string(index=False))
    print("\nFeature increment bootstrap")
    print(incremental.to_string(index=False))
    return predictions, metrics, bootstrap


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Forecast European day-ahead electricity prices")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES))
    parser.add_argument("--prices", default=str(DEFAULT_PRICES))
    parser.add_argument("--forecasts", default=str(DEFAULT_FORECASTS))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
