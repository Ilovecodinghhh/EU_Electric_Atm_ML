"""
Validation-based feature family / signal stability selection.

This script fits the existing cross-sectional candidate models, evaluates them
on each fold's validation period, selects stable models without looking at test
metrics, and writes pruned OOS prediction files for downstream bootstrap
reports.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from cross_sectional_evaluate import (
    FOLDS,
    build_panel,
    dedupe_oos_predictions,
    fit_predict_fold,
    metrics_for_predictions,
    parse_horizons,
    smooth_scores,
)


def validation_as_test_fold(fold):
    return {
        "name": f"{fold['name']}_validation",
        "train_end": fold["train_end"],
        "val_start": fold["val_start"],
        "val_end": fold["val_end"],
        "test_start": fold["val_start"],
        "test_end": fold["val_end"],
    }


def portfolio_group_column(name):
    return None if name == "none" else f"{name}_group"


def summarize_validation_metrics(metrics):
    metrics = metrics.copy()
    metrics["rankic_positive"] = metrics["mean_rankic"] > 0
    metrics["net_pnl_positive"] = metrics["net_pnl"] > 0
    summary = metrics.groupby(["horizon_days", "model"], as_index=False).agg(
        n_validation_folds=("fold", "nunique"),
        mean_rankic=("mean_rankic", "mean"),
        min_rankic=("mean_rankic", "min"),
        positive_rankic_folds=("rankic_positive", "sum"),
        mean_rankic_t=("rankic_t_stat", "mean"),
        net_pnl_sum=("net_pnl", "sum"),
        positive_net_pnl_folds=("net_pnl_positive", "sum"),
        sharpe_mean=("sharpe", "mean"),
        turnover_mean=("mean_turnover", "mean"),
    )
    return summary.sort_values(["horizon_days", "mean_rankic", "net_pnl_sum"], ascending=[True, False, False])


def mark_selected(summary, args):
    out = summary.copy()
    rankic_ok = out["positive_rankic_folds"] >= args.min_positive_rankic_folds
    mean_ok = out["mean_rankic"] >= args.min_mean_rankic
    t_ok = out["mean_rankic_t"] >= args.min_mean_rankic_t
    pnl_ok = out["net_pnl_sum"] >= args.min_net_pnl_sum
    pnl_folds_ok = out["positive_net_pnl_folds"] >= args.min_positive_net_pnl_folds
    out["selected"] = rankic_ok & mean_ok & t_ok & pnl_ok & pnl_folds_ok
    return out


def fit_validation_and_test_predictions(prepared, fold, fold_idx, include_slow_models, score_smoothing_span):
    val_fold = validation_as_test_fold(fold)
    val_predictions, _, _ = fit_predict_fold(
        prepared,
        val_fold,
        include_slow_models=include_slow_models,
    )
    test_predictions, _, _ = fit_predict_fold(
        prepared,
        fold,
        include_slow_models=include_slow_models,
    )

    val_predictions = smooth_scores(val_predictions, score_smoothing_span)
    test_predictions = smooth_scores(test_predictions, score_smoothing_span)
    for predictions, split in [(val_predictions, "validation"), (test_predictions, "test")]:
        predictions["fold"] = fold["name"]
        predictions["fold_order"] = fold_idx
        predictions["split"] = split
    return val_predictions, test_predictions


def evaluate_horizon(args, horizon):
    output_dir = os.path.join(args.output_dir, f"horizon_{horizon}d")
    os.makedirs(output_dir, exist_ok=True)
    periods_per_year = 252.0 / horizon
    group_col = portfolio_group_column(args.portfolio_group)
    prepared = build_panel(
        args.universe,
        args.prices,
        args.min_usable_stocks,
        horizon_days=horizon,
        target_mode=args.target_mode,
    )

    validation_predictions = []
    validation_metrics = []
    test_predictions = []
    include_slow_models = not args.skip_slow_models

    for fold_idx, fold in enumerate(FOLDS):
        val_pred, test_pred = fit_validation_and_test_predictions(
            prepared,
            fold,
            fold_idx,
            include_slow_models,
            args.score_smoothing_span,
        )
        val_metrics = metrics_for_predictions(
            val_pred,
            tx_cost_bps=args.tx_cost_bps,
            periods_per_year=periods_per_year,
            portfolio_group_col=group_col,
            portfolio_mode=args.portfolio_mode,
            selection_buffer=args.selection_buffer,
        )
        val_metrics["fold"] = fold["name"]
        val_metrics["split"] = "validation"
        val_metrics["horizon_days"] = horizon

        validation_predictions.append(val_pred)
        validation_metrics.append(val_metrics)
        test_predictions.append(test_pred)
        print(f"finished {fold['name']} horizon={horizon}")

    validation_predictions = pd.concat(validation_predictions, ignore_index=True)
    validation_metrics = pd.concat(validation_metrics, ignore_index=True)
    stability = mark_selected(summarize_validation_metrics(validation_metrics), args)
    selected_models = stability.loc[stability["selected"], "model"].tolist()

    test_predictions = pd.concat(test_predictions, ignore_index=True)
    pruned_test_predictions = test_predictions[test_predictions["model"].isin(selected_models)].copy()
    pruned_test_predictions = dedupe_oos_predictions(pruned_test_predictions) if selected_models else pruned_test_predictions

    if selected_models:
        pruned_metrics = metrics_for_predictions(
            pruned_test_predictions,
            tx_cost_bps=args.tx_cost_bps,
            periods_per_year=periods_per_year,
            portfolio_group_col=group_col,
            portfolio_mode=args.portfolio_mode,
            selection_buffer=args.selection_buffer,
        )
    else:
        pruned_metrics = pd.DataFrame()

    validation_predictions.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)
    validation_metrics.to_csv(os.path.join(output_dir, "validation_metrics.csv"), index=False)
    stability.to_csv(os.path.join(output_dir, "stability_summary.csv"), index=False)
    pruned_test_predictions.to_csv(os.path.join(output_dir, "unique_oos_predictions.csv"), index=False)
    pruned_metrics.to_csv(os.path.join(output_dir, "pruned_oos_metrics.csv"), index=False)

    with open(os.path.join(output_dir, "selected_models.json"), "w", encoding="utf-8") as f:
        json.dump({
            "horizon_days": horizon,
            "selected_models": selected_models,
            "selection_rules": {
                "min_positive_rankic_folds": args.min_positive_rankic_folds,
                "min_mean_rankic": args.min_mean_rankic,
                "min_mean_rankic_t": args.min_mean_rankic_t,
                "min_net_pnl_sum": args.min_net_pnl_sum,
                "min_positive_net_pnl_folds": args.min_positive_net_pnl_folds,
            },
        }, f, indent=2)

    return stability, pruned_metrics, selected_models


def run(args):
    horizons = parse_horizons(args.horizons)
    os.makedirs(args.output_dir, exist_ok=True)
    all_stability = []
    all_metrics = []
    selected = {}
    for horizon in horizons:
        stability, metrics, selected_models = evaluate_horizon(args, horizon)
        all_stability.append(stability)
        if not metrics.empty:
            metrics = metrics.copy()
            metrics["horizon_days"] = horizon
            all_metrics.append(metrics)
        selected[f"horizon_{horizon}d"] = selected_models

    stability = pd.concat(all_stability, ignore_index=True)
    stability.to_csv(os.path.join(args.output_dir, "stability_summary.csv"), index=False)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "pruned_oos_metrics.csv"),
            index=False,
        )
    with open(os.path.join(args.output_dir, "selected_models.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    print("\nStability summary:")
    print(stability.to_string(index=False))
    print("\nSelected models:")
    print(json.dumps(selected, indent=2))
    return stability, selected


def main():
    parser = argparse.ArgumentParser(description="Validation stability selection for cross-sectional models")
    parser.add_argument("--universe", default="data/cross_sectional_universe.csv")
    parser.add_argument("--prices", default="data/cross_sectional/daily_adj_close.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizons", default="3,5")
    parser.add_argument("--target_mode", default="market", choices=["market", "country_sector"])
    parser.add_argument("--tx_cost_bps", type=float, default=10.0)
    parser.add_argument("--min_usable_stocks", type=int, default=20)
    parser.add_argument("--portfolio_group", default="none", choices=["none", "country", "subsector"])
    parser.add_argument("--portfolio_mode", default="buffered", choices=["daily", "buffered"])
    parser.add_argument("--selection_buffer", type=float, default=0.15)
    parser.add_argument("--score_smoothing_span", type=int, default=5)
    parser.add_argument("--skip_slow_models", action="store_true")
    parser.add_argument("--min_positive_rankic_folds", type=int, default=2)
    parser.add_argument("--min_mean_rankic", type=float, default=0.0)
    parser.add_argument("--min_mean_rankic_t", type=float, default=-999.0)
    parser.add_argument("--min_net_pnl_sum", type=float, default=-999.0)
    parser.add_argument("--min_positive_net_pnl_folds", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
