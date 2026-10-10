import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from electricity_forecast.forecast_api import fetch_country_revisions


def main():
    parser = argparse.ArgumentParser(description="Fetch archived day-2/day-3 weather forecasts by country")
    parser.add_argument("--nodes", default=str(ROOT / "data" / "processed_nodes.csv"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "forecast_revisions" / "country_hourly.csv"),
    )
    parser.add_argument(
        "--metadata",
        default=str(ROOT / "data" / "forecast_revisions" / "metadata.json"),
    )
    parser.add_argument("--start", default="2024-03-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--pause", type=float, default=0.25)
    args = parser.parse_args()
    fetch_country_revisions(
        args.nodes,
        args.output,
        args.metadata,
        start_date=args.start,
        end_date=args.end,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
