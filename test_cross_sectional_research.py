"""
Tests for the cross-sectional RankIC research pipeline.
Run with: python test_cross_sectional_research.py
"""

import os

import numpy as np
import pandas as pd

from cross_sectional_evaluate import (
    MIN_USABLE_STOCKS,
    PRICE_PATH,
    build_panel,
    dedupe_oos_predictions,
    long_short_backtest,
    metrics_for_predictions,
    smooth_scores,
    spearman_corr,
)
from statistical_report import bootstrap_model_stats, moving_block_indices
from stability_selection import mark_selected, summarize_validation_metrics


def test_rankic_matches_spearman_on_toy_data():
    y = [1.0, 2.0, 3.0, 4.0]
    pred = [10.0, 20.0, 30.0, 40.0]
    assert abs(spearman_corr(y, pred) - 1.0) < 1e-12
    pred_reverse = [40.0, 30.0, 20.0, 10.0]
    assert abs(spearman_corr(y, pred_reverse) + 1.0) < 1e-12


def test_long_short_portfolio_is_dollar_neutral():
    rows = []
    date = pd.Timestamp("2024-01-02 23:00:00", tz="UTC")
    for i in range(30):
        rows.append({
            "target_time": date,
            "ticker": f"S{i:02d}",
            "score": float(i),
            "target_return_next": 0.001 * i,
            "target_residual_next": 0.001 * (i - 15),
        })
    bt = long_short_backtest(pd.DataFrame(rows), tx_cost_bps=10)
    assert abs(bt.iloc[0]["gross_exposure"] - 1.0) < 1e-12
    assert abs(bt.iloc[0]["net_exposure"]) < 1e-12
    assert bt.iloc[0]["long_count"] == 6
    assert bt.iloc[0]["short_count"] == 6


def test_group_neutral_portfolio_has_zero_group_net_exposure():
    rows = []
    date = pd.Timestamp("2024-01-02 23:00:00", tz="UTC")
    for group in ["A", "B"]:
        for i in range(6):
            rows.append({
                "target_time": date,
                "ticker": f"{group}{i}",
                "subsector_group": group,
                "score": float(i),
                "target_return_next": 0.001 * i,
                "target_residual_next": 0.001 * (i - 3),
            })
    bt = long_short_backtest(pd.DataFrame(rows), tx_cost_bps=10, group_col="subsector_group")
    assert abs(bt.iloc[0]["gross_exposure"] - 1.0) < 1e-12
    assert abs(bt.iloc[0]["net_exposure"]) < 1e-12
    assert abs(bt.iloc[0]["max_group_net_exposure"]) < 1e-12


def test_transaction_cost_reduces_pnl_and_unchanged_zero_scores_have_zero_turnover():
    rows = []
    for day in pd.date_range("2024-01-02", periods=2, tz="UTC"):
        for i in range(10):
            rows.append({
                "target_time": day + pd.Timedelta(hours=23),
                "ticker": f"S{i:02d}",
                "score": 0.0,
                "target_return_next": 0.01,
                "target_residual_next": 0.0,
                "model": "zero_score",
            })
    pred = pd.DataFrame(rows)
    bt = long_short_backtest(pred, tx_cost_bps=10)
    assert (bt["turnover"] == 0.0).all()
    metrics = metrics_for_predictions(pred, tx_cost_bps=10).iloc[0]
    assert metrics["net_pnl"] <= metrics["gross_pnl"]


def test_buffered_portfolio_reduces_small_rank_churn():
    rows = []
    day1 = pd.Timestamp("2024-01-02 23:00:00", tz="UTC")
    day2 = pd.Timestamp("2024-01-03 23:00:00", tz="UTC")
    for i in range(30):
        rows.append({
            "target_time": day1,
            "ticker": f"S{i:02d}",
            "score": float(i),
            "target_return_next": 0.0,
            "target_residual_next": 0.0,
        })
        score = float(i)
        if i == 23:
            score = 24.5
        if i == 24:
            score = 23.5
        rows.append({
            "target_time": day2,
            "ticker": f"S{i:02d}",
            "score": score,
            "target_return_next": 0.0,
            "target_residual_next": 0.0,
        })
    pred = pd.DataFrame(rows)
    daily = long_short_backtest(pred, tx_cost_bps=10, portfolio_mode="daily")
    buffered = long_short_backtest(
        pred,
        tx_cost_bps=10,
        portfolio_mode="buffered",
        selection_buffer=0.1,
    )
    assert buffered.iloc[1]["turnover"] < daily.iloc[1]["turnover"]
    assert buffered.iloc[1]["net_return"] >= daily.iloc[1]["net_return"]


def test_score_smoothing_uses_past_scores_per_ticker():
    rows = []
    for i, score in enumerate([1.0, 3.0, 5.0]):
        rows.append({
            "target_time": pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(days=i),
            "ticker": "A",
            "model": "m",
            "score": score,
        })
    smoothed = smooth_scores(pd.DataFrame(rows), span=3).sort_values("target_time")
    assert smoothed["raw_score"].tolist() == [1.0, 3.0, 5.0]
    assert smoothed["score"].tolist() == [1.0, 2.0, 3.5]


def test_oos_dedup_keeps_latest_fold_prediction():
    date = pd.Timestamp("2024-01-02 23:00:00", tz="UTC")
    pred = pd.DataFrame([
        {"model": "m", "target_time": date, "ticker": "A", "score": 1.0, "fold_order": 0},
        {"model": "m", "target_time": date, "ticker": "A", "score": 2.0, "fold_order": 1},
    ])
    deduped = dedupe_oos_predictions(pred)
    assert len(deduped) == 1
    assert deduped.iloc[0]["score"] == 2.0


def test_moving_block_bootstrap_indices_are_valid():
    rng = np.random.default_rng(7)
    idx = moving_block_indices(11, 4, rng)
    assert len(idx) == 11
    assert idx.min() >= 0
    assert idx.max() < 11


def test_bootstrap_model_stats_has_confidence_intervals():
    dates = pd.date_range("2024-01-02", periods=20, tz="UTC")
    daily = pd.DataFrame({
        "target_time": dates,
        "rankic": np.linspace(-0.01, 0.03, len(dates)),
        "ic": np.linspace(-0.02, 0.02, len(dates)),
        "net_return": np.linspace(-0.001, 0.002, len(dates)),
        "turnover": np.full(len(dates), 0.1),
    })
    report = bootstrap_model_stats(daily, n_boot=20, block_size=5, seed=3, periods_per_year=252)
    assert {"metric", "observed", "ci_low", "ci_high", "p_le_zero"}.issubset(report.columns)
    assert set(["mean_rankic", "net_pnl", "sharpe"]).issubset(set(report["metric"]))


def test_stability_selection_marks_only_stable_models():
    metrics = pd.DataFrame([
        {"horizon_days": 5, "model": "stable", "fold": "a", "mean_rankic": 0.01, "rankic_t_stat": 1.0, "net_pnl": 0.1, "sharpe": 0.5, "mean_turnover": 0.1},
        {"horizon_days": 5, "model": "stable", "fold": "b", "mean_rankic": 0.02, "rankic_t_stat": 1.2, "net_pnl": 0.1, "sharpe": 0.6, "mean_turnover": 0.1},
        {"horizon_days": 5, "model": "stable", "fold": "c", "mean_rankic": -0.001, "rankic_t_stat": -0.1, "net_pnl": -0.1, "sharpe": -0.2, "mean_turnover": 0.1},
        {"horizon_days": 5, "model": "unstable", "fold": "a", "mean_rankic": 0.02, "rankic_t_stat": 1.0, "net_pnl": 0.1, "sharpe": 0.5, "mean_turnover": 0.1},
        {"horizon_days": 5, "model": "unstable", "fold": "b", "mean_rankic": -0.02, "rankic_t_stat": -1.0, "net_pnl": -0.1, "sharpe": -0.5, "mean_turnover": 0.1},
        {"horizon_days": 5, "model": "unstable", "fold": "c", "mean_rankic": -0.01, "rankic_t_stat": -0.5, "net_pnl": -0.1, "sharpe": -0.5, "mean_turnover": 0.1},
    ])
    summary = summarize_validation_metrics(metrics)

    class Args:
        min_positive_rankic_folds = 2
        min_mean_rankic = 0.0
        min_mean_rankic_t = -999.0
        min_net_pnl_sum = -999.0
        min_positive_net_pnl_folds = 0

    selected = mark_selected(summary, Args)
    flags = dict(zip(selected["model"], selected["selected"]))
    assert bool(flags["stable"])
    assert not bool(flags["unstable"])


def test_real_panel_if_prices_are_available():
    if not os.path.exists(PRICE_PATH):
        print(f"Skipping real panel test because {PRICE_PATH} is not present")
        return
    prepared = build_panel(horizon_days=3, target_mode="country_sector")
    assert len(prepared.usable_tickers) >= MIN_USABLE_STOCKS
    assert prepared.panel.groupby("target_time")["ticker"].nunique().min() >= MIN_USABLE_STOCKS
    assert (prepared.panel["decision_time"] < prepared.panel["target_time"]).all()
    assert prepared.panel["ticker"].notna().all()
    assert np.isfinite(prepared.panel["target_residual_next"]).all()
    assert prepared.horizon_days == 3
    assert prepared.target_mode == "country_sector"
    assert any(c.endswith("_xz") for c in prepared.feature_columns["full"])
    assert any(c.endswith("_xrank") for c in prepared.feature_columns["full"])
    assert "country_group" not in prepared.feature_columns["full"]
    assert "subsector_group" not in prepared.feature_columns["full"]

    target_gap = prepared.panel["target_time"] - prepared.panel["decision_time"]
    assert (target_gap >= pd.Timedelta(days=1)).all()

    daily_residual_mean = prepared.panel.groupby("target_time")["target_residual_market_next"].mean()
    assert daily_residual_mean.abs().max() < 1e-12


if __name__ == "__main__":
    test_rankic_matches_spearman_on_toy_data()
    test_long_short_portfolio_is_dollar_neutral()
    test_group_neutral_portfolio_has_zero_group_net_exposure()
    test_transaction_cost_reduces_pnl_and_unchanged_zero_scores_have_zero_turnover()
    test_buffered_portfolio_reduces_small_rank_churn()
    test_score_smoothing_uses_past_scores_per_ticker()
    test_oos_dedup_keeps_latest_fold_prediction()
    test_moving_block_bootstrap_indices_are_valid()
    test_bootstrap_model_stats_has_confidence_intervals()
    test_stability_selection_marks_only_stable_models()
    test_real_panel_if_prices_are_available()
    print("cross-sectional research tests passed")
