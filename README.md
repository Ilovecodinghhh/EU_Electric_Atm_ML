# European Day-Ahead Electricity Price Forecasting

This project tests whether archived weather forecast levels and forecast revisions improve next-day European day-ahead electricity price forecasts beyond simple price-history baselines.

The current research target is electricity price, not stock return. Earlier ST-GCN and cross-sectional equity experiments are summarized in [`docs/research_history.md`](docs/research_history.md) and remain available in Git history.

## Research Design

- **Target:** next-delivery-day country-average day-ahead electricity price in EUR/MWh.
- **Universe:** Belgium, Germany, Denmark, France, Ireland, the Netherlands, and Norway.
- **Decision time:** 09:00 UTC on the calendar day before delivery.
- **Weather source:** archived Open-Meteo Previous Runs forecasts.
- **Forecast level:** the day-2 forecast for each delivery hour.
- **Forecast revision:** day-2 forecast minus day-3 forecast for the same delivery hour.
- **Weather variables:** 100 m wind speed, shortwave radiation, and 2 m temperature.
- **Leakage control:** every forecast has an explicit availability timestamp that must precede the decision timestamp; all imputers, winsorization thresholds, scalers, and model parameters are fit inside each training fold.
- **Validation:** three expanding-window folds covering July through December 2024, with a separate validation month before each two-month test window.
- **Uncertainty:** seven-day moving-block bootstrap and Newey-West t-statistics on daily loss differences.
- **Target completeness:** a country-day is evaluated only when the UTC day contains at least 24 observed hourly prices; the incomplete 2024-12-31 endpoint is excluded.

The day-2 versus day-3 construction is conservative. It avoids relying on a forecast update that may not have been published by the 09:00 UTC decision time.

## Main Conclusions

1. **Weather forecasts add useful information beyond price history.** Full Ridge reduces pooled OOS MAE by 4.83 EUR/MWh (21.32%) versus persistence. The seven-day block-bootstrap 95% interval is 3.09 to 7.04 EUR/MWh, and the Newey-West t-statistic is 5.25.
2. **Forecast levels matter much more than forecast revisions.** Adding all weather features to price-only Ridge improves MAE by 2.91 EUR/MWh (95% interval: 1.76 to 3.97). Adding revisions alone to price history improves MAE by only 0.14 EUR/MWh (95% interval: 0.02 to 0.26), while a revision-only model performs poorly.
3. **Model complexity is not decisively rewarded.** HistGradient has the best pooled MAE, beating Full Ridge by 0.52 EUR/MWh, but the direct bootstrap interval is -0.40 to 1.45 and includes zero. Ridge is therefore the primary parsimonious specification; HistGradient is the best point-estimate challenger, not a statistically proven winner.
4. **The result is promising but narrow.** Full Ridge beats persistence in every OOS fold, but not in every country, and the test period covers only July through December 2024. The evidence supports a forecasting result, not a production trading or causal claim.
5. **The largest remaining risks are data provenance and market definition.** The retained price file lacks a documented original provider and product-level definition. The target is a UTC country-day proxy, and each country is represented by one capacity-weighted weather centroid. These must be upgraded before making operational claims.

## OOS Results

The combined out-of-sample period contains 183 complete UTC days and 1,281 country-day observations.

| Model | MAE | RMSE | R2 | MAE improvement vs persistence |
|---|---:|---:|---:|---:|
| HistGradient, full features | 17.30 | 27.96 | 0.604 | 23.62% |
| Ridge, full features | 17.83 | 27.48 | 0.618 | 21.32% |
| Ridge, price + revisions | 20.60 | 30.93 | 0.516 | 9.10% |
| Ridge, price only | 20.74 | 31.00 | 0.514 | 8.47% |
| Ridge, forecast levels only | 22.40 | 33.48 | 0.433 | 1.14% |
| Persistence | 22.66 | 32.85 | 0.454 | 0.00% |
| Ridge, revisions only | 26.82 | 38.98 | 0.231 | -18.39% |

Detailed tables are stored in [`quant_output/electricity_forecast`](quant_output/electricity_forecast).
These figures were generated in `raincalib-gpu` with Python 3.12.13,
NumPy 2.4.4, pandas 3.0.3, and scikit-learn 1.9.0.

## Run

The verified local environment for all experiments is `raincalib-gpu`. The
`environment.yml` file remains the portable dependency specification for a
fresh machine.

To create a standalone environment elsewhere:

```bash
conda env create -f environment.yml
```

Fetch the archived forecasts:

```bash
conda run -n raincalib-gpu python scripts/fetch_forecast_revisions.py
```

Run the walk-forward evaluation and bootstrap report:

```bash
conda run -n raincalib-gpu python scripts/run_electricity_forecast.py
```

Run the tests:

```bash
conda run -n raincalib-gpu python -m pytest -q
```

## Project Structure

```text
EU_Electric_Atm_ML/
|-- README.md
|-- environment.yml
|-- data/
|   |-- processed_nodes.csv
|   |-- price_top100_2019-01-01_2024-12-31.csv
|   `-- forecast_revisions/
|       |-- country_hourly.csv
|       `-- metadata.json
|-- docs/
|   |-- interview_questions.md
|   |-- interview_reference_answers.md
|   `-- research_history.md
|-- scripts/
|   |-- fetch_forecast_revisions.py
|   `-- run_electricity_forecast.py
|-- src/
|   `-- electricity_forecast/
|       |-- __init__.py
|       |-- data.py
|       |-- evaluate.py
|       `-- forecast_api.py
|-- tests/
|   `-- test_electricity_forecast.py
`-- quant_output/
    `-- electricity_forecast/
        |-- summary_metrics.csv
        |-- fold_metrics.csv
        |-- country_metrics.csv
        |-- bootstrap_mae_improvement.csv
        |-- feature_increment_bootstrap.csv
        `-- final_report.json
```

## Data Sources

- Electricity price and node metadata are the retained inputs from the original repository dataset.
- Archived weather forecasts are downloaded through the [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api).
- Open-Meteo's [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) documents the broader archived forecast product.

## Limitations

- The archived forecast experiment covers only March through December 2024, so the model has not been tested across multiple energy-market regimes.
- Country-level weather is represented by capacity-weighted wind-farm coordinates, not a production-grade numerical weather grid or generation forecast.
- The target is an average daily price, not an hourly auction curve, spread, or executable trading PnL.
- The retained electricity-price file has no provider URL or product-level provenance in the repository history, so it is treated as an unverified country-level proxy rather than an official exchange index.
- UTC calendar-day aggregation does not yet reproduce each bidding zone's local delivery day or daylight-saving convention.
- Publication timestamps are represented by a conservative lead-day rule rather than provider issue-time records for every forecast run.
- Norway behaves differently from the continental markets and needs country-specific diagnostics or model segmentation.

This is a research pipeline, not investment advice.
