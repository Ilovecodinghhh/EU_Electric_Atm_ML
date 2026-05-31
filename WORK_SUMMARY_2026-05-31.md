# 2026-05-31 工作总结

## 今日目标

今天的核心目标是把项目从“能跑出图和模型结果”的原型，进一步推进成更接近传统量化研究标准的 cross-sectional alpha / RankIC pipeline。重点不是继续堆模型复杂度，而是检查信号是否稳定、是否 OOS 有效、是否扣交易成本后仍然站得住。

具体任务包括：

- 做 feature family pruning / stability selection。
- 剔除 validation 上不稳定或交易后表现差的信号。
- 在 pruning 后的模型集合上重新跑相同的 bootstrap 报告。
- 保留 RankIC、IC、t-stat、OOS、交易成本、turnover、max drawdown、regime breakdown 等面试中会被追问的证据链。

## 今日完成内容

### 1. 新增稳定性筛选脚本

新增脚本：

- `stability_selection.py`

该脚本的作用是用 validation fold 做 feature family / signal stability selection，不使用 test 结果来挑模型，避免“看了 OOS 再选模型”的二次泄漏。

主要逻辑：

- 对每个 horizon 单独做 validation-as-test 评估。
- 对候选模型计算 validation RankIC、RankIC t-stat、net PnL、positive fold count。
- 支持 score smoothing、buffered portfolio selection、transaction cost。
- 输出 pruning 后的 OOS prediction 文件，供后续 bootstrap/report 使用。

输出目录：

- `quant_output/stability_selection_market_smooth5_buffer15_coststable/`

关键输出：

- `stability_summary.csv`
- `selected_models.json`
- `validation_metrics.csv`
- `validation_predictions.csv`
- `horizon_3d/unique_oos_predictions.csv`
- `horizon_5d/unique_oos_predictions.csv`

### 2. 使用更严格的 cost-stable selection rule

最终采用的筛选逻辑是：

- validation mean RankIC 需要为正且超过最低门槛。
- 至少 2 个 validation folds 的 RankIC 为正。
- validation aggregate net PnL 需要大于等于 0。
- 保留交易后仍有经济意义的信号，而不是只看 RankIC。

这一步剔除了多个“统计上看起来尚可，但交易后不稳定”的模型。

### 3. 最终保留的模型

`selected_models.json` 中保留的模型如下：

| Horizon | 保留模型 |
|---|---|
| 3D | `lagged_return_reversal`, `ridge_wind_weather_only` |
| 5D | `lagged_return_reversal`, `rolling_momentum_21d`, `ridge_wind_weather_only` |

被剔除的主要模型包括：

- `ridge_price_only`
- `ridge_full`
- `elasticnet_full`
- `rolling_mean_5d`
- `country_sector_dummy`
- `ridge_country_features_only`
- `zero_score`
- `ridge_finance_only`

其中 `ridge_finance_only` 的 RankIC 并不完全差，但 validation aggregate net PnL 为负，因此在 cost-stable rule 下被剔除。这是今天比较重要的研究结论之一：不能只看 RankIC，要同时看交易成本和组合实现。

## Validation Stability Selection 结果

### 3D horizon

| Model | Selected | Mean RankIC | Min RankIC | Positive RankIC Folds | Mean RankIC t | Validation Net PnL |
|---|---:|---:|---:|---:|---:|---:|
| `lagged_return_reversal` | Yes | 0.0325 | -0.0059 | 2 | 2.1809 | 0.4611 |
| `ridge_wind_weather_only` | Yes | 0.0162 | 0.0043 | 3 | 1.4492 | 0.3506 |
| `ridge_finance_only` | No | 0.0173 | -0.0021 | 2 | 1.3135 | -0.0882 |
| `rolling_momentum_21d` | No | 0.0048 | -0.0094 | 2 | 0.3226 | 0.3448 |
| `ridge_full` | No | -0.0205 | -0.0357 | 0 | -1.6253 | -0.6176 |

### 5D horizon

| Model | Selected | Mean RankIC | Min RankIC | Positive RankIC Folds | Mean RankIC t | Validation Net PnL |
|---|---:|---:|---:|---:|---:|---:|
| `lagged_return_reversal` | Yes | 0.0346 | 0.0052 | 3 | 2.3666 | 0.6532 |
| `rolling_momentum_21d` | Yes | 0.0137 | -0.0034 | 2 | 0.8870 | 0.5228 |
| `ridge_wind_weather_only` | Yes | 0.0120 | -0.0038 | 2 | 1.2249 | 0.3035 |
| `ridge_finance_only` | No | 0.0142 | -0.0053 | 2 | 1.0690 | -0.1656 |
| `ridge_full` | No | -0.0321 | -0.0647 | 0 | -2.4982 | -0.8916 |

## Pruned Bootstrap 报告

稳定性筛选后，只对被保留模型重新跑 bootstrap 报告。

报告目录：

- `quant_output/stat_report_coststable_pruned_3d_buffer15_10bps/`
- `quant_output/stat_report_coststable_pruned_5d_buffer15_10bps/`

设定：

- Transaction cost: 10 bps
- Portfolio mode: buffered
- Selection buffer: 0.15
- Score smoothing span: 5
- Bootstrap samples: 1000
- Block size: 20

## OOS 汇总结果

### 5D horizon

| Model | RankIC | RankIC t | Net PnL | Sharpe | Turnover | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| `rolling_momentum_21d` | 0.0112 | 1.1055 | 0.5130 | 0.7226 | 0.0896 | -0.2771 |
| `ridge_wind_weather_only` | 0.0055 | 0.7114 | 0.2152 | 0.4394 | 0.2631 | -0.1992 |
| `lagged_return_reversal` | 0.0055 | 0.5581 | -0.1440 | -0.2119 | 0.5226 | -0.6633 |

Bootstrap confidence intervals:

| Model | Metric | 95% CI | P(metric <= 0) |
|---|---|---:|---:|
| `rolling_momentum_21d` | Mean RankIC | [-0.0274, 0.0510] | 0.268 |
| `rolling_momentum_21d` | Net PnL | [-0.4071, 1.7164] | 0.124 |
| `rolling_momentum_21d` | Sharpe | [-0.6524, 2.3717] | 0.124 |
| `ridge_wind_weather_only` | Mean RankIC | [-0.0176, 0.0295] | 0.326 |
| `ridge_wind_weather_only` | Net PnL | [-0.3623, 0.8181] | 0.237 |
| `lagged_return_reversal` | Net PnL | [-1.0358, 0.5038] | 0.696 |

### 3D horizon

| Model | RankIC | RankIC t | Net PnL | Sharpe | Turnover | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| `lagged_return_reversal` | 0.0185 | 1.9337 | -0.0130 | -0.0322 | 0.5218 | -0.3854 |
| `ridge_wind_weather_only` | 0.0131 | 1.6662 | 0.0970 | 0.3236 | 0.3048 | -0.2553 |

Bootstrap confidence intervals:

| Model | Metric | 95% CI | P(metric <= 0) |
|---|---|---:|---:|
| `lagged_return_reversal` | Mean RankIC | [-0.0114, 0.0449] | 0.117 |
| `lagged_return_reversal` | Net PnL | [-0.5847, 0.4863] | 0.567 |
| `ridge_wind_weather_only` | Mean RankIC | [-0.0064, 0.0291] | 0.097 |
| `ridge_wind_weather_only` | Net PnL | [-0.3275, 0.4524] | 0.386 |

## Regime Breakdown 观察

### 5D

- `rolling_momentum_21d` 是当前最好的候选信号，all-OOS net PnL 为 0.5130，Sharpe 为 0.7226。
- 但它并不是每个 regime 都稳定：2023 H2 很强，2024 H2 RankIC 为负但 PnL 仍为正。
- `ridge_wind_weather_only` 表现更温和，drawdown 较低，但 RankIC 和 t-stat 都不够强。
- `lagged_return_reversal` validation 看起来不错，但 OOS 扣费后为负，说明 turnover 太高会侵蚀信号。

### 3D

- `lagged_return_reversal` 的 RankIC t-stat 接近 2，但 net PnL 扣费后接近 0 且略负。
- `ridge_wind_weather_only` 的 RankIC 和 net PnL 都为正，但 bootstrap CI 仍然跨 0。
- 3D 信号整体更像“排序有一点信息，但组合实现不够强”。

## 测试与验证

今天补充并通过了稳定性筛选相关测试：

- `test_stability_selection_marks_only_stable_models`

同时确认以下测试通过：

- Python compile check
- `python test_cross_sectional_research.py`

已有测试覆盖的方向包括：

- RankIC 与 Spearman correlation 的一致性。
- long-short portfolio dollar-neutral。
- transaction cost 后 PnL 不高于 gross PnL。
- buffered selection 下 turnover 行为。
- score smoothing 行为。
- OOS prediction 去重。
- bootstrap utility。
- real panel 基础数据检查。

## 今日结论

今天最大的进展不是“找到了一个确定赚钱的 alpha”，而是把项目推进到了更严谨的量化研究状态：

- 有 OOS walk-forward 评估。
- 有 RankIC、IC、t-stat。
- 有交易成本、turnover、max drawdown。
- 有 regime breakdown。
- 有 bootstrap confidence interval。
- 有 validation-only stability selection。
- 有 feature family pruning。

目前不能严肃声称 alpha 已经统计显著。最接近可继续研究的候选是：

- 5D `rolling_momentum_21d`
- 3D / 5D `ridge_wind_weather_only`

但 bootstrap confidence interval 仍然跨 0，因此更稳妥的项目表述应该是：

> Built a leakage-aware cross-sectional alpha research pipeline for EU energy-transition equities, evaluating RankIC, OOS long-short performance, transaction costs, turnover, drawdown, bootstrap confidence intervals, and validation-based stability selection. After pruning unstable feature families, the strongest 5-day momentum candidate achieved positive OOS net PnL and Sharpe, but statistical significance remained inconclusive.

## 下一步建议

优先级最高的后续方向：

1. 扩大股票池，提高 cross-sectional breadth。
2. 进一步降低 turnover，尤其是 reversal 类信号。
3. 做 sector/country-neutral portfolio 版本和非 neutral 版本的系统对比。
4. 对 wind/weather feature 做更细的经济解释和 lag structure 测试。
5. 将 5D momentum 与 wind/weather signal 做简单 ensemble，但只能用 validation 选择权重。
6. 增加 borrow/shortability、liquidity、volume、market cap 过滤，避免不可交易股票污染结果。
7. 把 bootstrap 结果和 regime breakdown 自动汇总成最终 research report。

