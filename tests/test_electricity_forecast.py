import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from electricity_forecast.data import daily_country_prices, daily_forecast_features
from electricity_forecast.evaluate import (
    FOLDS,
    TrainOnlyTransformer,
    bootstrap_pairwise_mae_improvement,
    build_panel,
)


def test_revision_is_day2_minus_day3_and_available_before_decision():
    rows = []
    for hour in range(24):
        rows.append({
            "country": "AAA",
            "valid_time": pd.Timestamp("2024-07-01", tz="UTC") + pd.Timedelta(hours=hour),
            "wind_speed_100m_previous_day2": 10.0,
            "wind_speed_100m_previous_day3": 8.0,
            "shortwave_radiation_previous_day2": 100.0,
            "shortwave_radiation_previous_day3": 90.0,
            "temperature_2m_previous_day2": 20.0,
            "temperature_2m_previous_day3": 19.0,
        })
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "forecast.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        daily = daily_forecast_features(path)
    assert daily.iloc[0]["wind_speed_100m_revision_mean"] == 2.0
    decision = daily.iloc[0]["target_date"] - pd.Timedelta(days=1) + pd.Timedelta(hours=9)
    assert daily.iloc[0]["forecast_available_time"] < decision


def test_incomplete_price_day_is_excluded():
    timestamps = list(pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"))
    timestamps.append(pd.Timestamp("2024-01-03 00:00:00", tz="UTC"))
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        nodes_path = directory / "nodes.csv"
        prices_path = directory / "prices.csv"
        pd.DataFrame([{
            "name": "node-a",
            "country": "AAA",
            "latitude": 50.0,
            "longitude": 5.0,
            "capacity_mw": 100.0,
        }]).to_csv(nodes_path, index=False)
        pd.DataFrame({"node-a": np.arange(len(timestamps), dtype=float)}, index=timestamps).to_csv(prices_path)
        daily = daily_country_prices(nodes_path, prices_path)
    assert daily["target_date"].tolist() == [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-02", tz="UTC"),
    ]
    assert (daily["target_hour_count"] == 24).all()


def test_transformer_uses_training_statistics_only():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    transformer = TrainOnlyTransformer(["x"]).fit(train)
    original_median = transformer.median["x"]
    transformed = transformer.transform(pd.DataFrame({"x": [1_000_000.0]}))
    assert transformer.median["x"] == original_median
    assert np.isfinite(transformed).all()


def test_fold_boundaries_do_not_overlap_within_fold():
    for fold in FOLDS:
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["val_start"])
        assert pd.Timestamp(fold["val_end"]) < pd.Timestamp(fold["test_start"])


def test_pairwise_bootstrap_detects_better_model():
    rows = []
    for date in pd.date_range("2024-01-01", periods=40, tz="UTC"):
        for country in ["A", "B"]:
            target = 50.0
            rows.extend([
                {"target_date": date, "country": country, "target_price": target, "model": "reference", "prediction": 60.0},
                {"target_date": date, "country": country, "target_price": target, "model": "challenger", "prediction": 52.0},
            ])
    result = bootstrap_pairwise_mae_improvement(
        pd.DataFrame(rows),
        [("challenger", "reference")],
        n_boot=100,
        block_size=7,
        seed=3,
    ).iloc[0]
    assert result["observed_mae_improvement"] == 8.0
    assert result["ci_low"] > 0.0


def test_real_panel_is_timestamp_clean():
    forecast_path = ROOT / "data" / "forecast_revisions" / "country_hourly.csv"
    if not forecast_path.exists():
        return
    prepared = build_panel()
    assert len(prepared.panel) > 1000
    assert prepared.panel["country"].nunique() == 7
    assert (prepared.panel["forecast_available_time"] < prepared.panel["decision_time"]).all()
    assert (prepared.panel["decision_time"] < prepared.panel["target_date"]).all()
    assert any("revision" in column for column in prepared.feature_sets["full"])


if __name__ == "__main__":
    test_revision_is_day2_minus_day3_and_available_before_decision()
    test_incomplete_price_day_is_excluded()
    test_transformer_uses_training_statistics_only()
    test_fold_boundaries_do_not_overlap_within_fold()
    test_pairwise_bootstrap_detects_better_model()
    test_real_panel_is_timestamp_clean()
    print("electricity forecast tests passed")
