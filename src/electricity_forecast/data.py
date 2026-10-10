from pathlib import Path

import numpy as np
import pandas as pd


def build_country_locations(nodes_path):
    """Return capacity-weighted coordinates for each represented price country."""
    nodes = pd.read_csv(nodes_path)
    required = {"name", "country", "latitude", "longitude", "capacity_mw"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"Missing node columns: {sorted(missing)}")

    rows = []
    for country, group in nodes.groupby("country", sort=True):
        weights = group["capacity_mw"].astype(float).clip(lower=0.0)
        if weights.sum() <= 0:
            weights = pd.Series(np.ones(len(group)), index=group.index)
        rows.append({
            "country": country,
            "latitude": float(np.average(group["latitude"], weights=weights)),
            "longitude": float(np.average(group["longitude"], weights=weights)),
            "capacity_mw": float(group["capacity_mw"].sum()),
            "node_count": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def load_country_hourly_prices(nodes_path, price_path):
    """Collapse duplicated node columns into one robust price series per country."""
    nodes = pd.read_csv(nodes_path)
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, utc=True)
    prices = prices.sort_index().replace([np.inf, -np.inf], np.nan)

    out = {}
    for country, group in nodes.groupby("country", sort=True):
        columns = [name for name in group["name"] if name in prices.columns]
        if not columns:
            continue
        out[country] = prices[columns].median(axis=1, skipna=True)
    if not out:
        raise ValueError("No country electricity-price series could be constructed")
    return pd.DataFrame(out, index=prices.index).sort_index()


def daily_country_prices(nodes_path, price_path, min_hours=24):
    hourly = load_country_hourly_prices(nodes_path, price_path)
    daily = hourly.resample("D").agg(["mean", "std", "min", "max"])
    daily.columns = [f"price_{stat}_{country}" for country, stat in daily.columns]
    hourly_counts = hourly.resample("D").count()

    rows = []
    countries = sorted(hourly.columns)
    for country in countries:
        frame = pd.DataFrame({
            "target_date": daily.index,
            "country": country,
            "target_price": daily[f"price_mean_{country}"].values,
            "target_price_std": daily[f"price_std_{country}"].values,
            "target_price_min": daily[f"price_min_{country}"].values,
            "target_price_max": daily[f"price_max_{country}"].values,
            "target_hour_count": hourly_counts[country].values,
        })
        frame = frame[frame["target_hour_count"] >= min_hours].copy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def daily_forecast_features(hourly_path):
    hourly = pd.read_csv(hourly_path, parse_dates=["valid_time"])
    hourly["valid_time"] = pd.to_datetime(hourly["valid_time"], utc=True)
    hourly["target_date"] = hourly["valid_time"].dt.normalize()
    value_columns = [
        column for column in hourly.columns
        if column not in {"country", "valid_time", "target_date", "latitude", "longitude"}
    ]
    if not value_columns:
        raise ValueError("Forecast revision file has no weather variables")

    grouped = hourly.groupby(["country", "target_date"])[value_columns]
    daily = grouped.agg(["mean", "std", "min", "max"])
    daily.columns = [f"{column}_{stat}" for column, stat in daily.columns]
    daily = daily.reset_index()

    for variable in ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]:
        for stat in ["mean", "std", "min", "max"]:
            lead2 = f"{variable}_previous_day2_{stat}"
            lead3 = f"{variable}_previous_day3_{stat}"
            if lead2 in daily and lead3 in daily:
                daily[f"{variable}_revision_{stat}"] = daily[lead2] - daily[lead3]

    # Day-2 forecasts for every delivery hour are available no later than
    # 23:00 UTC two calendar days before delivery, safely before the auction proxy.
    daily["forecast_available_time"] = daily["target_date"] - pd.Timedelta(days=2) + pd.Timedelta(hours=23)
    return daily.sort_values(["target_date", "country"]).reset_index(drop=True)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
