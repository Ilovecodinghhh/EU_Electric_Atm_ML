# Research History

## Why the Project Changed Direction

The repository started as an ST-GCN prototype linking European wind-farm weather, electricity prices, and a small renewable-equity basket. The work later expanded into a 71-stock cross-sectional alpha study. Those experiments were useful for learning about timestamp discipline, out-of-sample validation, RankIC, transaction costs, turnover, and feature stability, but neither version justified keeping a graph neural network as the main research claim.

The main weaknesses were:

- the original four-stock PC1 target was too small and indirect for a defensible alpha study;
- realized ERA5 weather did not represent information genuinely available at the trading decision time;
- electricity prices were duplicated across wind-farm nodes even though they were country-level observations;
- repeated intraday samples were paired with effectively daily targets;
- ST-GCN complexity did not produce a stable gain over simpler baselines;
- the economic path from weather to next-day equity residual returns was weak and noisy.

The active project therefore predicts day-ahead electricity prices directly. This target is closer to the physical weather-to-generation-to-power-price mechanism and supports cleaner forecast-availability rules.

## What Was Retained

The current pipeline keeps the parts of the earlier work that survived scrutiny:

- country and wind-farm metadata;
- country-level day-ahead electricity prices;
- chronological fold construction;
- train-only preprocessing;
- explicit timestamp assertions;
- baseline and ablation comparisons;
- moving-block bootstrap and autocorrelation-robust t-statistics.

## Archived Work

The old ST-GCN, tensor-building, stock-market, RankIC, feature-pruning, and portfolio-sweep files were removed from the active tree to make the repository focused and reproducible. They remain recoverable from Git history. Their removal does not mean that the negative result is hidden: the key conclusion is that model complexity was not supported by stable out-of-sample evidence.

## Current Research Claim

Archived day-2 weather forecasts materially improve next-day country-level electricity price prediction when combined with lagged prices. Day-2 minus day-3 forecast revisions provide a much smaller but statistically positive incremental improvement. The evidence is preliminary because it covers only 183 complete OOS days in 2024 and seven countries.
