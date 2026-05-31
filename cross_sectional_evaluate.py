"""
Daily cross-sectional alpha / RankIC evaluation.

Primary target:
  next-day residual stock return = next-day stock log return
  minus next-day equal-weight universe return.

The backtest trades raw next-day stock returns using a dollar-neutral
top/bottom ranked portfolio.
"""

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from quant_evaluate import FOLDS, load_raw_data, rolling_hourly_features, utc_ts


DATA_DIR = "data"
UNIVERSE_PATH = os.path.join(DATA_DIR, "cross_sectional_universe.csv")
PRICE_PATH = os.path.join(DATA_DIR, "cross_sectional", "daily_adj_close.csv")
OUTPUT_DIR = os.path.join("quant_output", "cross_sectional_rankic")
TX_COST_BPS = 10.0
MIN_USABLE_STOCKS = 20
MIN_PRICE_COVERAGE = 0.65
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99
RANDOM_SEED = 11
warnings.filterwarnings("ignore", category=PerformanceWarning)


@dataclass
class PreparedData:
    panel: pd.DataFrame
    feature_columns: dict
    usable_tickers: list
    horizon_days: int
    target_mode: str


def load_prices(price_path=PRICE_PATH):
    if not os.path.exists(price_path):
        raise FileNotFoundError(
            f"Missing {price_path}. Run fetch_market_data.py before cross-sectional evaluation."
        )
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, utc=True).normalize()
    return prices.sort_index()


def clean_prices(prices, min_coverage=MIN_PRICE_COVERAGE):
    prices = prices.sort_index().replace([np.inf, -np.inf], np.nan)
    coverage = prices.notna().mean()
    usable = coverage[coverage >= min_coverage].index.tolist()
    cleaned = prices[usable].ffill(limit=3)
    return cleaned.dropna(how="all"), usable


def spearman_corr(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    xr = x[valid].rank(method="average")
    yr = y[valid].rank(method="average")
    if xr.std(ddof=0) == 0 or yr.std(ddof=0) == 0:
        return np.nan
    return float(xr.corr(yr))


def pearson_corr(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].std(ddof=0) == 0 or y[valid].std(ddof=0) == 0:
        return np.nan
    return float(x[valid].corr(y[valid]))


def t_stat(values):
    values = pd.Series(values).dropna()
    if len(values) < 3 or values.std(ddof=1) == 0:
        return np.nan
    return float(values.mean() / (values.std(ddof=1) / math.sqrt(len(values))))


def max_drawdown(returns):
    if len(returns) == 0:
        return np.nan
    equity = np.cumsum(np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity - peak))


def add_stock_features(returns):
    features = {}
    market = returns.mean(axis=1)
    features["ret_1"] = returns
    features["ret_2"] = returns.shift(1)
    features["ret_5_sum"] = returns.rolling(5, min_periods=3).sum()
    features["ret_21_sum"] = returns.rolling(21, min_periods=10).sum()
    features["ret_5_mean"] = returns.rolling(5, min_periods=3).mean()
    features["ret_21_mean"] = returns.rolling(21, min_periods=10).mean()
    features["vol_21"] = returns.rolling(21, min_periods=10).std()
    features["rel_ret_1"] = returns.sub(market, axis=0)
    features["rel_ret_5_sum"] = features["ret_5_sum"].sub(market.rolling(5, min_periods=3).sum(), axis=0)
    return features, market


def forward_log_return(prices, horizon_days):
    return np.log(prices.shift(-horizon_days) / prices)


def add_cross_sectional_features(panel, columns):
    created = []
    grouped = panel.groupby("date", group_keys=False)
    for col in columns:
        if col not in panel.columns:
            continue
        z_col = f"{col}_xz"
        rank_col = f"{col}_xrank"
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0.0, np.nan)
        panel[z_col] = ((panel[col] - mean) / std).fillna(0.0)
        panel[rank_col] = grouped[col].rank(pct=True).sub(0.5).fillna(0.0)
        created.extend([z_col, rank_col])
    return created


def add_residual_targets(panel):
    g = panel.groupby("target_time", group_keys=False)
    panel["target_residual_market_next"] = panel["target_return_next"] - g["target_return_next"].transform("mean")
    country_mean = panel.groupby(["target_time", "country"])["target_return_next"].transform("mean")
    sector_mean = panel.groupby(["target_time", "subsector"])["target_return_next"].transform("mean")
    market_mean = g["target_return_next"].transform("mean")
    panel["target_residual_country_sector_next"] = (
        panel["target_return_next"] - country_mean - sector_mean + market_mean
    )
    return panel


def daily_context_features():
    raw = load_raw_data()
    hourly_features = rolling_hourly_features(raw.weather_price)
    decision = hourly_features[hourly_features.index.hour == 23].copy()
    decision.index = decision.index.normalize()
    return decision


def build_panel(
    universe_path=UNIVERSE_PATH,
    price_path=PRICE_PATH,
    min_usable_stocks=MIN_USABLE_STOCKS,
    horizon_days=1,
    target_mode="country_sector",
):
    universe = pd.read_csv(universe_path)
    prices, usable = clean_prices(load_prices(price_path))
    universe = universe[universe["ticker"].isin(prices.columns)].copy()
    prices = prices[universe["ticker"].tolist()]
    if len(prices.columns) < min_usable_stocks:
        raise ValueError(
            f"Only {len(prices.columns)} usable stocks after coverage filters; need at least {min_usable_stocks}."
        )

    returns = np.log(prices / prices.shift(1))
    stock_features, market_return = add_stock_features(returns)
    raw_target = forward_log_return(prices, horizon_days)
    target_date = pd.Series(returns.index, index=returns.index).shift(-horizon_days)
    context = daily_context_features()

    rows = []
    for ticker in prices.columns:
        meta = universe[universe["ticker"] == ticker].iloc[0]
        df = pd.DataFrame({
            "date": returns.index,
            "ticker": ticker,
            "country": meta["country"],
            "subsector": meta["subsector"],
            "decision_time": returns.index + pd.Timedelta(hours=23),
            "target_time": target_date + pd.Timedelta(hours=23),
            "target_return_next": raw_target[ticker].values,
            "market_return_1": market_return.values,
        })
        for name, frame in stock_features.items():
            df[name] = frame[ticker].values
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    panel = add_residual_targets(panel)
    if target_mode == "market":
        panel["target_residual_next"] = panel["target_residual_market_next"]
    elif target_mode == "country_sector":
        panel["target_residual_next"] = panel["target_residual_country_sector_next"]
    else:
        raise ValueError("target_mode must be 'market' or 'country_sector'")
    panel = panel.merge(
        context.reset_index().rename(columns={"index": "date"}),
        on="date",
        how="left",
    )
    panel["country_group"] = panel["country"]
    panel["subsector_group"] = panel["subsector"]

    country_feature_specs = [
        ("stock_price_country", "price_country_{country}", ["win_mean", "win_std", "last"]),
        ("stock_wind_country_speed", "wind_agg_country_{country}_speed", ["win_mean", "win_std", "last"]),
        ("stock_wind_country_ramp", "wind_agg_country_{country}_ramp", ["win_mean", "win_std", "last"]),
    ]
    panel["has_power_country_features"] = 0.0
    for country in panel["country"].dropna().unique():
        mask = panel["country"] == country
        for dst_prefix, src_template, suffixes in country_feature_specs:
            src_prefix = src_template.format(country=country)
            available = False
            for suffix in suffixes:
                src = f"{src_prefix}_{suffix}"
                dst = f"{dst_prefix}_{suffix}"
                if dst not in panel.columns:
                    panel[dst] = np.nan
                if src in panel.columns:
                    panel.loc[mask, dst] = panel.loc[mask, src]
                    available = True
            if available:
                panel.loc[mask, "has_power_country_features"] = 1.0

    panel = pd.get_dummies(panel, columns=["country", "subsector"], prefix=["country", "subsector"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["target_time", "target_return_next", "target_residual_next"])
    panel = panel[panel["decision_time"] < panel["target_time"]].copy()
    target_counts = panel.groupby("target_time")["ticker"].transform("nunique")
    panel = panel[target_counts >= min_usable_stocks].copy()

    stock_cols = [
        "ret_1", "ret_2", "ret_5_sum", "ret_21_sum", "ret_5_mean",
        "ret_21_mean", "vol_21", "rel_ret_1", "rel_ret_5_sum",
        "market_return_1",
    ]
    price_cols = [c for c in panel.columns if c.startswith("stock_price_country_")]
    weather_cols = [c for c in panel.columns if c.startswith((
        "ssr_", "t2m_", "u100_", "v100_", "wind_speed_",
        "wind_speed_ramp_", "wind_power_proxy_", "wind_agg_",
        "stock_wind_country_", "has_power_country_features",
    ))]
    group_label_cols = {"country_group", "subsector_group"}
    country_cols = [
        c for c in panel.columns
        if c.startswith(("country_", "subsector_")) and c not in group_label_cols
    ]
    stock_wind_cols = [c for c in weather_cols if c.startswith("stock_wind_country_")]
    stock_xs_cols = add_cross_sectional_features(panel, stock_cols)
    price_xs_cols = add_cross_sectional_features(panel, price_cols)
    stock_wind_xs_cols = add_cross_sectional_features(panel, stock_wind_cols)
    feature_columns = {
        "finance_only": stock_cols + stock_xs_cols,
        "price_only": price_cols + price_xs_cols,
        "wind_weather_only": weather_cols + stock_wind_xs_cols,
        "country_features_only": country_cols,
        "full": (
            stock_cols + price_cols + weather_cols
            + stock_xs_cols + price_xs_cols + stock_wind_xs_cols
            + country_cols
        ),
    }
    return PreparedData(panel, feature_columns, prices.columns.tolist(), horizon_days, target_mode)


def split_panel(panel, fold):
    train_end = utc_ts(fold["train_end"])
    val_start = utc_ts(fold["val_start"])
    val_end = utc_ts(fold["val_end"])
    test_start = utc_ts(fold["test_start"])
    test_end = utc_ts(fold["test_end"])
    train = panel[panel["target_time"] <= train_end].copy()
    val = panel[(panel["target_time"] >= val_start) & (panel["target_time"] <= val_end)].copy()
    test = panel[(panel["target_time"] >= test_start) & (panel["target_time"] <= test_end)].copy()
    return train, val, test


def transform_xy(train, test, columns):
    x_train = train[columns].copy().astype(float)
    x_test = test[columns].copy().astype(float)
    med = x_train.median()
    x_train = x_train.fillna(med).fillna(0.0)
    x_test = x_test.fillna(med).fillna(0.0)
    lo = x_train.quantile(WINSOR_Q_LOW)
    hi = x_train.quantile(WINSOR_Q_HIGH)
    x_train = x_train.clip(lower=lo, upper=hi, axis=1)
    x_test = x_test.clip(lower=lo, upper=hi, axis=1)
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def add_prediction(base, model_name, pred):
    out = base.copy()
    out["model"] = model_name
    out["score"] = np.asarray(pred, dtype=float)
    return out


def smooth_scores(predictions, span):
    if span <= 1:
        predictions["raw_score"] = predictions["score"]
        predictions["score_smoothing_span"] = 1
        return predictions
    out = predictions.sort_values(["model", "ticker", "target_time"]).copy()
    out["raw_score"] = out["score"]
    out["score"] = (
        out.groupby(["model", "ticker"], group_keys=False)["score"]
        .apply(lambda s: s.ewm(span=span, adjust=False).mean())
    )
    out["score_smoothing_span"] = int(span)
    return out.sort_index()


def fit_group_mean_baseline(train, test):
    group_cols = [
        c for c in train.columns
        if c.startswith(("country_", "subsector_")) and c not in {"country_group", "subsector_group"}
    ]
    if not group_cols:
        return np.zeros(len(test))
    x_train, x_test = transform_xy(train, test, group_cols)
    model = Ridge(alpha=1.0)
    model.fit(x_train, train["target_residual_next"])
    return model.predict(x_test)


def fit_predict_fold(prepared, fold, include_slow_models=True):
    train, _, test = split_panel(prepared.panel, fold)
    if train.empty or test.empty:
        raise ValueError(f"Empty train/test split for {fold['name']}")

    counts = test.groupby("target_time")["ticker"].nunique()
    if counts.min() < MIN_USABLE_STOCKS:
        raise ValueError(f"Fold {fold['name']} has a test day with only {counts.min()} stocks")

    base = test[[
        "date", "decision_time", "target_time", "ticker",
        "country_group", "subsector_group",
        "target_return_next", "target_residual_next",
    ]].copy()
    predictions = [
        add_prediction(base, "zero_score", np.zeros(len(test))),
        add_prediction(base, "lagged_return_reversal", -test["ret_1"].fillna(0.0)),
        add_prediction(base, "rolling_momentum_21d", test["ret_21_sum"].fillna(0.0)),
        add_prediction(base, "rolling_mean_5d", test["ret_5_mean"].fillna(0.0)),
        add_prediction(base, "country_sector_dummy", fit_group_mean_baseline(train, test)),
    ]

    model_specs = [
        ("ridge_finance_only", "finance_only", Ridge(alpha=1.0)),
        ("ridge_price_only", "price_only", Ridge(alpha=1.0)),
        ("ridge_wind_weather_only", "wind_weather_only", Ridge(alpha=1.0)),
        ("ridge_country_features_only", "country_features_only", Ridge(alpha=1.0)),
        ("ridge_full", "full", Ridge(alpha=1.0)),
        ("elasticnet_full", "full", ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=10000, random_state=RANDOM_SEED)),
    ]
    if include_slow_models:
        model_specs.append((
            "hist_gradient_boosting_full",
            "full",
            HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.04,
                l2_regularization=0.01,
                max_leaf_nodes=15,
                random_state=RANDOM_SEED,
            ),
        ))
    for model_name, feature_set, model in model_specs:
        cols = prepared.feature_columns[feature_set]
        x_train, x_test = transform_xy(train, test, cols)
        model.fit(x_train, train["target_residual_next"])
        predictions.append(add_prediction(base, model_name, model.predict(x_test)))

    return pd.concat(predictions, ignore_index=True), train, test


def select_side_tickers(group, side, entry_q, exit_q, prev_weights, portfolio_mode):
    ranked = group.sort_values("score")
    k = max(1, int(math.floor(len(group) * entry_q)))
    exit_k = max(k, int(math.ceil(len(group) * exit_q)))

    if side == "long":
        ordered = ranked["ticker"].tolist()[::-1]
        entry = ordered[:k]
        exit_set = set(ordered[:exit_k])
        prev_sign = 1
    else:
        ordered = ranked["ticker"].tolist()
        entry = ordered[:k]
        exit_set = set(ordered[:exit_k])
        prev_sign = -1

    if portfolio_mode != "buffered" or not prev_weights:
        return entry

    retained = [
        ticker for ticker in ordered
        if ticker in exit_set and np.sign(prev_weights.get(ticker, 0.0)) == prev_sign
    ][:k]
    selected = retained[:]
    selected_set = set(selected)
    for ticker in entry:
        if ticker not in selected_set:
            selected.append(ticker)
            selected_set.add(ticker)
        if len(selected) >= k:
            break
    return selected[:k]


def assign_group_weights(weights, group, gross_share, prev_weights, portfolio_mode, selection_buffer):
    q = 0.2 if len(group) >= 10 else 1 / 3
    exit_q = min(0.5, q + selection_buffer)
    longs = select_side_tickers(group, "long", q, exit_q, prev_weights, portfolio_mode)
    shorts = select_side_tickers(group, "short", q, exit_q, prev_weights, portfolio_mode)
    if not longs or not shorts:
        return
    for ticker in longs:
        weights[ticker] = 0.5 * gross_share / len(longs)
    for ticker in shorts:
        weights[ticker] = -0.5 * gross_share / len(shorts)


def day_long_short_weights(
    day,
    group_col=None,
    prev_weights=None,
    portfolio_mode="daily",
    selection_buffer=0.0,
):
    prev_weights = prev_weights or {}
    weights = {ticker: 0.0 for ticker in day["ticker"]}
    if len(day) < 3 or day["score"].std(ddof=0) <= 0:
        return weights

    if not group_col or group_col not in day.columns:
        q = 0.2 if len(day) >= 25 else 1 / 3
        exit_q = min(0.5, q + selection_buffer)
        longs = select_side_tickers(day, "long", q, exit_q, prev_weights, portfolio_mode)
        shorts = select_side_tickers(day, "short", q, exit_q, prev_weights, portfolio_mode)
        for ticker in longs:
            weights[ticker] = 0.5 / len(longs)
        for ticker in shorts:
            weights[ticker] = -0.5 / len(shorts)
        return weights

    eligible = []
    for _, group in day.groupby(group_col):
        if len(group) >= 3 and group["score"].std(ddof=0) > 0:
            eligible.append(group)
    total_names = sum(len(group) for group in eligible)
    if total_names == 0:
        return weights

    for group in eligible:
        gross_share = len(group) / total_names
        assign_group_weights(
            weights,
            group,
            gross_share,
            prev_weights,
            portfolio_mode,
            selection_buffer,
        )
    return weights


def long_short_backtest(
    predictions,
    tx_cost_bps=TX_COST_BPS,
    group_col=None,
    portfolio_mode="daily",
    selection_buffer=0.0,
):
    rows = []
    prev_weights = {}
    cost_rate = tx_cost_bps / 10000.0
    for target_time, day in predictions.sort_values("target_time").groupby("target_time"):
        day = day.dropna(subset=["score", "target_return_next"])
        weights = day_long_short_weights(
            day,
            group_col,
            prev_weights,
            portfolio_mode,
            selection_buffer,
        )
        all_tickers = set(prev_weights) | set(weights)
        turnover = sum(abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
        ret_map = day.set_index("ticker")["target_return_next"].to_dict()
        gross = sum(weights.get(t, 0.0) * ret_map.get(t, 0.0) for t in weights)
        net = gross - cost_rate * turnover
        max_group_net = np.nan
        if group_col and group_col in day.columns:
            group_map = day.set_index("ticker")[group_col].to_dict()
            group_nets = {}
            for ticker, weight in weights.items():
                group = group_map.get(ticker)
                group_nets[group] = group_nets.get(group, 0.0) + weight
            if group_nets:
                max_group_net = max(abs(value) for value in group_nets.values())
        rows.append({
            "target_time": target_time,
            "n_names": int(len(day)),
            "gross_return": float(gross),
            "net_return": float(net),
            "turnover": float(turnover),
            "gross_exposure": float(sum(abs(w) for w in weights.values())),
            "net_exposure": float(sum(weights.values())),
            "max_group_net_exposure": float(max_group_net) if pd.notna(max_group_net) else np.nan,
            "long_count": int(sum(1 for w in weights.values() if w > 0)),
            "short_count": int(sum(1 for w in weights.values() if w < 0)),
            "portfolio_mode": portfolio_mode,
            "selection_buffer": float(selection_buffer),
        })
        prev_weights = weights
    return pd.DataFrame(rows)


def metrics_for_predictions(
    predictions,
    tx_cost_bps=TX_COST_BPS,
    periods_per_year=252.0,
    portfolio_group_col=None,
    portfolio_mode="daily",
    selection_buffer=0.0,
):
    rows = []
    for model, group in predictions.groupby("model"):
        daily = []
        for _, day in group.groupby("target_time"):
            daily.append({
                "ic": pearson_corr(day["target_residual_next"], day["score"]),
                "rankic": spearman_corr(day["target_residual_next"], day["score"]),
            })
        daily = pd.DataFrame(daily)
        backtest = long_short_backtest(
            group,
            tx_cost_bps,
            portfolio_group_col,
            portfolio_mode,
            selection_buffer,
        )
        y = group["target_residual_next"].to_numpy(float)
        pred = group["score"].to_numpy(float)
        err = pred - y
        net = backtest["net_return"].to_numpy(float)
        gross = backtest["gross_return"].to_numpy(float)
        rows.append({
            "model": model,
            "n_obs": int(len(group)),
            "n_days": int(group["target_time"].nunique()),
            "mse": float(np.mean(err ** 2)),
            "mae": float(np.mean(np.abs(err))),
            "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(y))),
            "mean_ic": float(daily["ic"].mean(skipna=True)),
            "mean_rankic": float(daily["rankic"].mean(skipna=True)),
            "ic_t_stat": t_stat(daily["ic"]),
            "rankic_t_stat": t_stat(daily["rankic"]),
            "icir": float(daily["rankic"].mean(skipna=True) / daily["rankic"].std(ddof=1) * math.sqrt(periods_per_year)) if daily["rankic"].std(ddof=1) > 0 else np.nan,
            "annualized_return": float(np.mean(net) * periods_per_year),
            "annualized_volatility": float(np.std(net) * math.sqrt(periods_per_year)),
            "sharpe": float(np.mean(net) / np.std(net) * math.sqrt(periods_per_year)) if np.std(net) > 0 else np.nan,
            "max_drawdown": max_drawdown(net),
            "gross_pnl": float(np.sum(gross)),
            "net_pnl": float(np.sum(net)),
            "mean_turnover": float(backtest["turnover"].mean()),
            "max_group_net_exposure": float(backtest["max_group_net_exposure"].max(skipna=True)) if "max_group_net_exposure" in backtest else np.nan,
            "tx_cost_bps": float(tx_cost_bps),
            "periods_per_year": float(periods_per_year),
            "portfolio_mode": portfolio_mode,
            "selection_buffer": float(selection_buffer),
        })
    return pd.DataFrame(rows).sort_values(["mean_rankic", "net_pnl"], ascending=[False, False])


def dedupe_oos_predictions(predictions):
    if "fold_order" not in predictions.columns:
        return predictions.drop_duplicates(["model", "target_time", "ticker"], keep="last")
    return (
        predictions.sort_values(["model", "target_time", "ticker", "fold_order"])
        .drop_duplicates(["model", "target_time", "ticker"], keep="last")
        .sort_values(["model", "target_time", "ticker"])
    )


def run_one_horizon(args, horizon_days, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    prepared = build_panel(
        args.universe,
        args.prices,
        args.min_usable_stocks,
        horizon_days=horizon_days,
        target_mode=args.target_mode,
    )
    if args.write_panels:
        prepared.panel.to_csv(os.path.join(output_dir, "daily_panel.csv"), index=False)

    all_metrics = []
    all_predictions = []
    all_backtests = []
    periods_per_year = 252.0 / horizon_days
    portfolio_group_col = None if args.portfolio_group == "none" else f"{args.portfolio_group}_group"
    for fold_idx, fold in enumerate(FOLDS):
        fold_dir = os.path.join(output_dir, fold["name"])
        os.makedirs(fold_dir, exist_ok=True)
        predictions, train, test = fit_predict_fold(
            prepared,
            fold,
            include_slow_models=not args.skip_slow_models,
        )
        predictions = smooth_scores(predictions, args.score_smoothing_span)
        predictions["fold"] = fold["name"]
        predictions["fold_order"] = fold_idx
        predictions.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        if args.write_panels:
            train.to_csv(os.path.join(fold_dir, "train_panel.csv"), index=False)
            test.to_csv(os.path.join(fold_dir, "test_panel.csv"), index=False)

        metrics = metrics_for_predictions(
            predictions,
            args.tx_cost_bps,
            periods_per_year,
            portfolio_group_col,
            args.portfolio_mode,
            args.selection_buffer,
        )
        metrics["fold"] = fold["name"]
        metrics.to_csv(os.path.join(fold_dir, "metrics.csv"), index=False)
        metrics.to_json(os.path.join(fold_dir, "metrics.json"), orient="records", indent=2)

        for model, group in predictions.groupby("model"):
            bt = long_short_backtest(
                group,
                args.tx_cost_bps,
                portfolio_group_col,
                args.portfolio_mode,
                args.selection_buffer,
            )
            bt["model"] = model
            bt["fold"] = fold["name"]
            all_backtests.append(bt)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        print(f"\n=== {fold['name']} ===")
        print(metrics[["model", "mean_rankic", "rankic_t_stat", "sharpe", "net_pnl", "mean_turnover"]].to_string(index=False))

    summary = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    backtests = pd.concat(all_backtests, ignore_index=True)
    unique_oos_predictions = dedupe_oos_predictions(predictions)
    unique_oos_metrics = metrics_for_predictions(
        unique_oos_predictions,
        args.tx_cost_bps,
        periods_per_year,
        portfolio_group_col,
        args.portfolio_mode,
        args.selection_buffer,
    )
    summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    predictions.to_csv(os.path.join(output_dir, "summary_predictions.csv"), index=False)
    backtests.to_csv(os.path.join(output_dir, "summary_backtest.csv"), index=False)
    unique_oos_predictions.to_csv(os.path.join(output_dir, "unique_oos_predictions.csv"), index=False)
    unique_oos_metrics.to_csv(os.path.join(output_dir, "unique_oos_metrics.csv"), index=False)

    aggregate = summary.groupby("model").agg(
        mean_rankic=("mean_rankic", "mean"),
        mean_ic=("mean_ic", "mean"),
        rankic_t_mean=("rankic_t_stat", "mean"),
        sharpe_mean=("sharpe", "mean"),
        net_pnl_sum=("net_pnl", "sum"),
        turnover_mean=("mean_turnover", "mean"),
    ).sort_values("mean_rankic", ascending=False)
    aggregate.to_csv(os.path.join(output_dir, "aggregate_model_summary.csv"))

    outputs = {
        "summary_metrics": os.path.join(output_dir, "summary_metrics.csv"),
        "summary_predictions": os.path.join(output_dir, "summary_predictions.csv"),
        "summary_backtest": os.path.join(output_dir, "summary_backtest.csv"),
        "aggregate_model_summary": os.path.join(output_dir, "aggregate_model_summary.csv"),
        "unique_oos_predictions": os.path.join(output_dir, "unique_oos_predictions.csv"),
        "unique_oos_metrics": os.path.join(output_dir, "unique_oos_metrics.csv"),
    }
    if args.write_panels:
        outputs["daily_panel"] = os.path.join(output_dir, "daily_panel.csv")

    report = {
        "universe_path": args.universe,
        "prices": args.prices,
        "usable_tickers": prepared.usable_tickers,
        "n_usable_tickers": len(prepared.usable_tickers),
        "horizon_days": horizon_days,
        "target": f"{horizon_days}d_forward_{args.target_mode}_neutral_residual_return",
        "backtest": f"daily dollar-neutral scores evaluated on {horizon_days}d forward returns",
        "tx_cost_bps": args.tx_cost_bps,
        "score_smoothing_span": args.score_smoothing_span,
        "periods_per_year": periods_per_year,
        "portfolio_group": args.portfolio_group,
        "portfolio_mode": args.portfolio_mode,
        "selection_buffer": args.selection_buffer,
        "wrote_panels": bool(args.write_panels),
        "slow_models_included": not args.skip_slow_models,
        "feature_counts": {name: len(cols) for name, cols in prepared.feature_columns.items()},
        "outputs": outputs,
    }
    with open(os.path.join(output_dir, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return aggregate, report


def parse_horizons(text):
    horizons = []
    for item in text.split(","):
        value = int(item.strip())
        if value <= 0:
            raise ValueError("Horizons must be positive integers")
        horizons.append(value)
    return horizons


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    reports = {}
    aggregates = []
    for horizon in horizons:
        out_dir = args.output_dir if len(horizons) == 1 else os.path.join(args.output_dir, f"horizon_{horizon}d")
        aggregate, report = run_one_horizon(args, horizon, out_dir)
        aggregate = aggregate.reset_index()
        aggregate["horizon_days"] = horizon
        aggregates.append(aggregate)
        reports[f"horizon_{horizon}d"] = report

    horizon_summary = pd.concat(aggregates, ignore_index=True)
    horizon_summary.to_csv(os.path.join(args.output_dir, "horizon_model_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "horizons": horizons,
            "target_mode": args.target_mode,
            "tx_cost_bps": args.tx_cost_bps,
            "score_smoothing_span": args.score_smoothing_span,
            "portfolio_group": args.portfolio_group,
            "portfolio_mode": args.portfolio_mode,
            "selection_buffer": args.selection_buffer,
            "horizon_reports": reports,
            "horizon_model_summary": os.path.join(args.output_dir, "horizon_model_summary.csv"),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Cross-sectional RankIC evaluation")
    parser.add_argument("--universe", default=UNIVERSE_PATH)
    parser.add_argument("--prices", default=PRICE_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--tx_cost_bps", type=float, default=TX_COST_BPS)
    parser.add_argument("--min_usable_stocks", type=int, default=MIN_USABLE_STOCKS)
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--target_mode", default="country_sector", choices=["market", "country_sector"])
    parser.add_argument("--write_panels", action="store_true")
    parser.add_argument("--skip_slow_models", action="store_true")
    parser.add_argument("--score_smoothing_span", type=int, default=1)
    parser.add_argument("--portfolio_group", default="none", choices=["none", "country", "subsector"])
    parser.add_argument("--portfolio_mode", default="daily", choices=["daily", "buffered"])
    parser.add_argument("--selection_buffer", type=float, default=0.0)
    args = parser.parse_args()
    if args.score_smoothing_span < 1:
        raise ValueError("--score_smoothing_span must be at least 1")
    if not 0.0 <= args.selection_buffer <= 0.3:
        raise ValueError("--selection_buffer must be between 0.0 and 0.3")
    run(args)


if __name__ == "__main__":
    main()
