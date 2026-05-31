# Data Audit

Finance range: 2019-01-02 00:00:00+00:00 to 2024-12-30 00:00:00+00:00

Finance columns: ORSTED.CO, VWS.CO, IBE.MC, EDPR.LS

Nodes: 100 across 39 clusters

Nodes inside Europe bounds: 99

## Invalid Geo Nodes

- Outside the Northeast (NLD): lat=38.0211, lon=-87.5172

Universe assumption: processed_nodes.csv is an ex-post top-capacity universe. Historical commissioning/investability dates are not encoded, so survivorship and availability bias remain research risks.

## Hourly Files

- `price_top100_2019-01-01_2024-12-31.csv`: 52585 rows, 100 columns, 0.0000% missing
- `ssr_top100_2019-01-01_2024-12-31.csv`: 52585 rows, 100 columns, 0.0000% missing
- `t2m_top100_2019-01-01_2024-12-31.csv`: 52585 rows, 100 columns, 0.0000% missing
- `u100_top100_2019-01-01_2024-12-31.csv`: 52585 rows, 100 columns, 0.0000% missing
- `v100_top100_2019-01-01_2024-12-31.csv`: 52585 rows, 100 columns, 0.0000% missing

## Electricity Price Granularity

- BEL: 1 unique series across 5 nodes
- DEU: 1 unique series across 24 nodes
- DNK: 1 unique series across 6 nodes
- FRA: 1 unique series across 23 nodes
- IRL: 1 unique series across 23 nodes
- NLD: 1 unique series across 9 nodes
- NOR: 1 unique series across 10 nodes

## Timestamp And Robust Scaling Policy

- finance_calendar: Use observed finance trading dates as the exchange calendar proxy.
- finance_features: Must be lagged to previous available trading close before model input.
- close_time_rule: Daily finance observations are treated as available at 23:00 UTC.
- targets: Must have target_time strictly after window_end and fall on an observed trading date.
- scalers: Must be fit only on each fold's training period.
- winsorization: Finance returns and electricity prices should be clipped with train-only 1%/99% quantiles.
