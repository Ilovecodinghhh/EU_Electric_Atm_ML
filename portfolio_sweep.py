"""
Sweep transaction costs, portfolio grouping, and buffered selection settings
using saved cross-sectional prediction files.
"""

import argparse
import os

import pandas as pd

from cross_sectional_evaluate import metrics_for_predictions, parse_horizons


def parse_float_list(text):
    values = []
    for item in text.split(","):
        value = float(item.strip())
        if value < 0:
            raise ValueError("Sweep values must be non-negative")
        values.append(value)
    return values


def parse_group_list(text):
    groups = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [group for group in groups if group not in {"none", "country", "subsector"}]
    if invalid:
        raise ValueError(f"Invalid portfolio groups: {invalid}")
    return groups


def parse_model_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def prediction_path(root, horizon, prediction_file):
    return os.path.join(root, f"horizon_{horizon}d", prediction_file)


def run_sweep(args):
    horizons = parse_horizons(args.horizons)
    costs = parse_float_list(args.costs_bps)
    buffers = parse_float_list(args.buffers)
    groups = parse_group_list(args.portfolio_groups)
    models = set(parse_model_list(args.models))
    rows = []

    for horizon in horizons:
        path = prediction_path(args.predictions_root, horizon, args.prediction_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prediction file: {path}")
        predictions = pd.read_csv(path, parse_dates=["date", "decision_time", "target_time"])
        if models:
            predictions = predictions[predictions["model"].isin(models)].copy()
            if predictions.empty:
                raise ValueError(f"No requested models found in {path}")
        periods_per_year = 252.0 / horizon

        for group in groups:
            group_col = None if group == "none" else f"{group}_group"
            if group_col and group_col not in predictions.columns:
                raise ValueError(
                    f"Prediction file {path} does not contain required column {group_col}. "
                    "Regenerate predictions with the current cross_sectional_evaluate.py."
                )
            for buffer_value in buffers:
                mode = "daily" if buffer_value == 0 else "buffered"
                for cost_bps in costs:
                    metrics = metrics_for_predictions(
                        predictions,
                        tx_cost_bps=cost_bps,
                        periods_per_year=periods_per_year,
                        portfolio_group_col=group_col,
                        portfolio_mode=mode,
                        selection_buffer=buffer_value,
                    )
                    metrics["horizon_days"] = horizon
                    metrics["portfolio_group"] = group
                    metrics["portfolio_mode"] = mode
                    metrics["selection_buffer"] = buffer_value
                    metrics["tx_cost_bps"] = cost_bps
                    rows.append(metrics)

    output = pd.concat(rows, ignore_index=True)
    output = output.sort_values(
        ["horizon_days", "portfolio_group", "tx_cost_bps", "net_pnl", "mean_rankic"],
        ascending=[True, True, True, False, False],
    )
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Sweep portfolio construction settings on saved predictions")
    parser.add_argument("--predictions_root", required=True)
    parser.add_argument("--prediction_file", default="unique_oos_predictions.csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--costs_bps", default="0,5,10,20")
    parser.add_argument("--buffers", default="0,0.05,0.10,0.15,0.20")
    parser.add_argument("--portfolio_groups", default="none,subsector")
    parser.add_argument("--models", default="")
    args = parser.parse_args()
    output = run_sweep(args)
    print(output.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
