"""
Data audit for quant research assumptions.

Writes quant_output/data_audit.json and quant_output/data_audit.md.
"""

import json
import os
import hashlib

import pandas as pd


DATA_DIR = "data"
OUTPUT_DIR = "quant_output"
EUROPE_LAT_RANGE = (35.0, 72.0)
EUROPE_LON_RANGE = (-15.0, 35.0)


def read_hourly(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    finance_path = os.path.join(DATA_DIR, "Finance20192024", "raw_finance.csv")
    finance = pd.read_csv(finance_path, index_col=0, parse_dates=True)
    finance.index = pd.to_datetime(finance.index, utc=True)

    nodes = pd.read_csv(os.path.join(DATA_DIR, "processed_nodes.csv"))
    valid_geo = (
        nodes["latitude"].between(*EUROPE_LAT_RANGE)
        & nodes["longitude"].between(*EUROPE_LON_RANGE)
    )
    invalid_geo_nodes = nodes.loc[
        ~valid_geo, ["name", "country", "latitude", "longitude"]
    ].to_dict(orient="records")
    hourly_files = [
        "price_top100_2019-01-01_2024-12-31.csv",
        "ssr_top100_2019-01-01_2024-12-31.csv",
        "t2m_top100_2019-01-01_2024-12-31.csv",
        "u100_top100_2019-01-01_2024-12-31.csv",
        "v100_top100_2019-01-01_2024-12-31.csv",
    ]

    hourly_summary = {}
    for filename in hourly_files:
        df = read_hourly(os.path.join(DATA_DIR, filename))
        hourly_summary[filename] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": float(df.isna().sum().sum() / max(df.size, 1)),
        }

    price = read_hourly(os.path.join(DATA_DIR, "price_top100_2019-01-01_2024-12-31.csv"))
    price_hashes = {}
    for column in price.columns:
        values_hash = pd.util.hash_pandas_object(price[column], index=False).values
        price_hashes[column] = hashlib.blake2b(values_hash.tobytes(), digest_size=12).hexdigest()
    duplicate_price_groups = {}
    for country in sorted(nodes["country"].dropna().unique()):
        country_nodes = nodes.loc[nodes["country"] == country, "name"].tolist()
        present = [name for name in country_nodes if name in price.columns]
        if not present:
            continue
        n_unique_series = len({price_hashes[name] for name in present})
        duplicate_price_groups[country] = {
            "node_count": len(present),
            "unique_price_series": n_unique_series,
            "is_country_level_series": n_unique_series == 1 and len(present) > 1,
        }

    report = {
        "finance": {
            "path": finance_path,
            "columns": list(finance.columns),
            "rows": int(len(finance)),
            "start": str(finance.index.min()),
            "end": str(finance.index.max()),
            "missing_by_column": finance.isna().sum().astype(int).to_dict(),
        },
        "nodes": {
            "rows": int(len(nodes)),
            "valid_geo_rows": int(valid_geo.sum()),
            "invalid_geo_nodes": invalid_geo_nodes,
            "countries": sorted(nodes["country"].dropna().unique().tolist()),
            "capacity_mw_min": float(nodes["capacity_mw"].min()),
            "capacity_mw_max": float(nodes["capacity_mw"].max()),
            "clusters": int(nodes["physical_cluster"].nunique()),
            "assumption": (
                "processed_nodes.csv is an ex-post top-capacity universe. "
                "Historical commissioning/investability dates are not encoded, "
                "so survivorship and availability bias remain research risks."
            ),
        },
        "hourly": hourly_summary,
        "electricity_price": {
            "country_duplicate_series": duplicate_price_groups,
            "assumption": (
                "The current price input is country-level, not node-level, "
                "because all nodes within the same country often share one "
                "identical hourly price series."
            ),
        },
        "timestamp_policy": {
            "finance_calendar": "Use observed finance trading dates as the exchange calendar proxy.",
            "finance_features": "Must be lagged to previous available trading close before model input.",
            "close_time_rule": "Daily finance observations are treated as available at 23:00 UTC.",
            "targets": "Must have target_time strictly after window_end and fall on an observed trading date.",
            "scalers": "Must be fit only on each fold's training period.",
            "winsorization": "Finance returns and electricity prices should be clipped with train-only 1%/99% quantiles.",
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "data_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(OUTPUT_DIR, "data_audit.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Data Audit\n\n")
        f.write(f"Finance range: {report['finance']['start']} to {report['finance']['end']}\n\n")
        f.write(f"Finance columns: {', '.join(report['finance']['columns'])}\n\n")
        f.write(f"Nodes: {report['nodes']['rows']} across {report['nodes']['clusters']} clusters\n\n")
        f.write(f"Nodes inside Europe bounds: {report['nodes']['valid_geo_rows']}\n\n")
        if invalid_geo_nodes:
            f.write("## Invalid Geo Nodes\n\n")
            for node in invalid_geo_nodes:
                f.write(
                    f"- {node['name']} ({node['country']}): "
                    f"lat={node['latitude']}, lon={node['longitude']}\n"
                )
            f.write("\n")
        f.write(f"Universe assumption: {report['nodes']['assumption']}\n\n")
        f.write("## Hourly Files\n\n")
        for filename, item in hourly_summary.items():
            f.write(
                f"- `{filename}`: {item['rows']} rows, {item['columns']} columns, "
                f"{item['missing_pct']:.4%} missing\n"
            )
        f.write("\n## Electricity Price Granularity\n\n")
        for country, item in duplicate_price_groups.items():
            f.write(
                f"- {country}: {item['unique_price_series']} unique series across "
                f"{item['node_count']} nodes\n"
            )
        f.write("\n## Timestamp And Robust Scaling Policy\n\n")
        for key, value in report["timestamp_policy"].items():
            f.write(f"- {key}: {value}\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    audit()
