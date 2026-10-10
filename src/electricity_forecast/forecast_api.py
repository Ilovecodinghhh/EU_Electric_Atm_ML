import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .data import build_country_locations, ensure_parent


API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
BASE_VARIABLES = ("wind_speed_100m", "shortwave_radiation", "temperature_2m")
LEAD_DAYS = (2, 3)


def requested_fields(base_variables=BASE_VARIABLES, lead_days=LEAD_DAYS):
    return [f"{variable}_previous_day{lead}" for variable in base_variables for lead in lead_days]


def build_url(latitude, longitude, start_date, end_date, fields=None):
    fields = fields or requested_fields()
    params = {
        "latitude": round(float(latitude), 5),
        "longitude": round(float(longitude), 5),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(fields),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def fetch_location(latitude, longitude, start_date, end_date, timeout=120, retries=3):
    url = build_url(latitude, longitude, start_date, end_date)
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "EU-electricity-forecast-research/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            hourly = pd.DataFrame(payload["hourly"])
            hourly["valid_time"] = pd.to_datetime(hourly.pop("time"), utc=True)
            return hourly, payload.get("hourly_units", {}), url
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Forecast API failed after {retries} attempts: {last_error}")


def fetch_country_revisions(
    nodes_path,
    output_path,
    metadata_path,
    start_date="2024-03-01",
    end_date="2024-12-31",
    pause=0.25,
):
    locations = build_country_locations(nodes_path)
    frames = []
    metadata = {
        "source": "Open-Meteo Previous Runs API",
        "endpoint": API_URL,
        "start_date": start_date,
        "end_date": end_date,
        "lead_days": list(LEAD_DAYS),
        "revision_definition": "previous_day2 minus previous_day3 for the same valid hour",
        "availability_rule": "latest day-2 valid-hour forecast is available by target_date-2 23:00 UTC",
        "countries": {},
    }
    units = None

    for row in locations.itertuples(index=False):
        hourly, current_units, url = fetch_location(
            row.latitude,
            row.longitude,
            start_date,
            end_date,
        )
        hourly.insert(0, "country", row.country)
        hourly["latitude"] = row.latitude
        hourly["longitude"] = row.longitude
        frames.append(hourly)
        units = current_units
        non_null = {column: int(hourly[column].notna().sum()) for column in requested_fields()}
        metadata["countries"][row.country] = {
            "latitude": row.latitude,
            "longitude": row.longitude,
            "capacity_mw": row.capacity_mw,
            "node_count": row.node_count,
            "non_null_hours": non_null,
            "request_url": url,
        }
        print(f"fetched {row.country}: {len(hourly)} hours")
        time.sleep(pause)

    result = pd.concat(frames, ignore_index=True)
    useful = result[requested_fields()].notna().any(axis=1)
    result = result.loc[useful].sort_values(["country", "valid_time"]).reset_index(drop=True)
    if result.empty:
        raise RuntimeError("The forecast archive returned no usable observations")

    metadata["hourly_units"] = units
    metadata["rows"] = int(len(result))
    metadata["usable_start"] = str(result["valid_time"].min())
    metadata["usable_end"] = str(result["valid_time"].max())
    ensure_parent(output_path)
    ensure_parent(metadata_path)
    result.to_csv(output_path, index=False)
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result, metadata
