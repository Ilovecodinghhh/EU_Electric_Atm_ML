"""
Fetch daily adjusted-close prices for the cross-sectional RankIC universe.

The downloader uses Yahoo's public chart endpoint via the standard library so
the project does not need an extra yfinance dependency.
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd


DATA_DIR = "data"
UNIVERSE_PATH = os.path.join(DATA_DIR, "cross_sectional_universe.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "cross_sectional")
PRICE_PATH = os.path.join(OUTPUT_DIR, "daily_adj_close.csv")
META_PATH = os.path.join(OUTPUT_DIR, "download_metadata.json")


def utc_seconds(date_text):
    dt = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def yahoo_chart_url(ticker, start, end):
    params = {
        "period1": utc_seconds(start),
        "period2": utc_seconds(end),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?{urllib.parse.urlencode(params)}"


def fetch_ticker(ticker, start, end, timeout=30):
    request = urllib.request.Request(
        yahoo_chart_url(ticker, start, end),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"No Yahoo result for {ticker}: {error}")

    result = result[0]
    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})
    adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")
    close = indicators.get("quote", [{}])[0].get("close")
    values = adjclose if adjclose is not None else close
    if not timestamps or values is None:
        raise RuntimeError(f"No daily close series returned for {ticker}")

    index = pd.to_datetime(timestamps, unit="s", utc=True).normalize()
    series = pd.Series(values, index=index, name=ticker, dtype="float64")
    return series.dropna()


def fetch_universe(universe_path=UNIVERSE_PATH, output_dir=OUTPUT_DIR, start="2019-01-01", end="2025-01-01", pause=0.4):
    os.makedirs(output_dir, exist_ok=True)
    universe = pd.read_csv(universe_path)
    prices = []
    meta = {
        "start": start,
        "end_exclusive": end,
        "universe_path": universe_path,
        "tickers": {},
    }

    for ticker in universe["ticker"]:
        try:
            series = fetch_ticker(ticker, start, end)
            prices.append(series)
            meta["tickers"][ticker] = {
                "status": "ok",
                "observations": int(series.notna().sum()),
                "start": str(series.index.min()),
                "end": str(series.index.max()),
            }
            print(f"ok {ticker}: {series.notna().sum()} rows")
        except Exception as exc:
            meta["tickers"][ticker] = {"status": "error", "error": str(exc)}
            print(f"error {ticker}: {exc}")
        time.sleep(pause)

    if not prices:
        raise RuntimeError("No market data downloaded")

    price_df = pd.concat(prices, axis=1, sort=True).sort_index()
    coverage = price_df.notna().mean().sort_values(ascending=False)
    failed = [ticker for ticker, info in meta["tickers"].items() if info["status"] != "ok"]
    meta["summary"] = {
        "configured_tickers": int(len(universe)),
        "downloaded_tickers": int(len(price_df.columns)),
        "failed_tickers": failed,
        "columns_ge_80pct_coverage": int((coverage >= 0.80).sum()),
        "columns_ge_65pct_coverage": int((coverage >= 0.65).sum()),
        "coverage": {ticker: float(value) for ticker, value in coverage.items()},
    }
    price_path = os.path.join(output_dir, "daily_adj_close.csv")
    meta_path = os.path.join(output_dir, "download_metadata.json")
    price_df.to_csv(price_path, index_label="date")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved {price_path}")
    print(f"saved {meta_path}")
    print(f"downloaded columns: {len(price_df.columns)}")
    print(f"columns with >=80% coverage: {(coverage >= 0.80).sum()}")
    return price_df, meta


def main():
    parser = argparse.ArgumentParser(description="Fetch cross-sectional daily adjusted-close data")
    parser.add_argument("--universe", default=UNIVERSE_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--pause", type=float, default=0.4)
    args = parser.parse_args()
    fetch_universe(args.universe, args.output_dir, args.start, args.end, args.pause)


if __name__ == "__main__":
    main()
