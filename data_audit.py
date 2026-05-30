"""
Data audit for quant research assumptions.

Writes quant_output/data_audit.json and quant_output/data_audit.md.
"""

import json
import os

import pandas as pd


DATA_DIR = "data"
OUTPUT_DIR = "quant_output"


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
        "timestamp_policy": {
            "finance_features": "Must be lagged to previous trading close before model input.",
            "targets": "Must have target_time strictly after window_end.",
            "scalers": "Must be fit only on each fold's training period.",
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
        f.write(f"Universe assumption: {report['nodes']['assumption']}\n\n")
        f.write("## Hourly Files\n\n")
        for filename, item in hourly_summary.items():
            f.write(
                f"- `{filename}`: {item['rows']} rows, {item['columns']} columns, "
                f"{item['missing_pct']:.4%} missing\n"
            )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    audit()
