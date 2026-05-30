"""
Assertion tests for the quant research hardening layer.
Run with: python test_quant_research.py
"""

import numpy as np
import pandas as pd

from quant_evaluate import (
    FOLDS,
    build_fold_samples,
    fit_predict_models,
    load_raw_data,
    metrics_for_predictions,
    utc_ts,
)


def test_fold_boundaries_do_not_overlap():
    for fold in FOLDS:
        train_end = utc_ts(fold["train_end"])
        val_start = utc_ts(fold["val_start"])
        val_end = utc_ts(fold["val_end"])
        test_start = utc_ts(fold["test_start"])
        assert train_end < val_start <= val_end < test_start


def test_cost_adjusted_pnl_not_above_gross_when_costs_positive():
    df = pd.DataFrame({
        "window_end": pd.date_range("2024-01-01", periods=4, tz="UTC"),
        "target_time": pd.date_range("2024-01-02", periods=4, tz="UTC"),
        "y": [1.0, -1.0, 1.0, -1.0],
        "basket_return_next": [0.01, 0.01, 0.01, 0.01],
        "model": ["toy"] * 4,
        "y_pred": [1.0, -1.0, 1.0, -1.0],
    })
    metrics = metrics_for_predictions(df, tx_cost_bps=10)
    assert metrics.iloc[0]["net_pnl"] <= metrics.iloc[0]["gross_pnl"]


def test_real_fold_timestamp_discipline_and_metrics_alignment():
    raw = load_raw_data()
    samples, _ = build_fold_samples(raw, FOLDS[-1])
    assert not samples.empty
    assert (samples["window_end"] < samples["target_time"]).all()
    known_finance_time = samples["finance_feature_time"].dropna()
    assert (known_finance_time < samples.loc[known_finance_time.index, "window_end"]).all()

    predictions = fit_predict_models(samples)
    counts = predictions.groupby("model").size().unique()
    assert len(counts) == 1, "All models must be evaluated on the same samples"

    metrics = metrics_for_predictions(predictions)
    assert np.isfinite(metrics["mse"]).all()
    assert (metrics["n"] == counts[0]).all()


if __name__ == "__main__":
    test_fold_boundaries_do_not_overlap()
    test_cost_adjusted_pnl_not_above_gross_when_costs_positive()
    test_real_fold_timestamp_discipline_and_metrics_alignment()
    print("quant research tests passed")
