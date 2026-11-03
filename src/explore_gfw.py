"""Day 2: test the Global Fishing Watch API with one small query.

Saves the raw answer to data/raw/ and prints its structure, so we can
write down exactly what GFW returns. Needs the GFW_TOKEN environment variable.
"""
import json
import os
import sys
from pathlib import Path

import requests

URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
RAW_FILE = Path("data/raw/gfw_sample.json")

# Small box off the Sindh coast, as [longitude, latitude] pairs (ring must close).
REGION = {
    "type": "Polygon",
    "coordinates": [[[65.0, 23.0], [67.5, 23.0], [67.5, 25.0], [65.0, 25.0], [65.0, 23.0]]],
}

PARAMS = {
    "spatial-resolution": "LOW",
    "temporal-resolution": "ENTIRE",
    "spatial-aggregation": "false",
    "datasets[0]": "public-global-fishing-effort:latest",
    "date-range": "2022-05-01,2022-12-01",
    "format": "JSON",
    "group-by": "FLAG",
}


def get_token():
    """Read the API token from the environment, never from the code."""
    token = os.environ.get("GFW_TOKEN")
    if not token:
        sys.exit("GFW_TOKEN is not set. Set it first (see the steps), then run again.")
    return token


def fetch_report(token):
    """Send one report request and return the parsed JSON answer."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(URL, params=PARAMS, headers=headers,
                             json={"geojson": REGION}, timeout=60)
    response.raise_for_status()
    return response.json()


def describe(obj, indent=0, max_depth=4):
    """Print the shape of the JSON: keys, types and sizes."""
    pad = "  " * indent
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:10]:
            size = f" (len {len(value)})" if isinstance(value, (list, dict)) else f" = {str(value)[:40]}"
            print(f"{pad}{key}: {type(value).__name__}{size}")
            if indent < max_depth:
                describe(value, indent + 1, max_depth)
    elif isinstance(obj, list) and obj:
        print(f"{pad}[list of {len(obj)}] first item:")
        describe(obj[0], indent + 1, max_depth)


def main():
    token = get_token()
    try:
        data = fetch_report(token)
    except requests.exceptions.HTTPError as err:
        print(f"GFW returned an error: {err}")
        print(err.response.text[:300])
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        sys.exit("Could not connect. Check your internet connection.")
    except requests.exceptions.Timeout:
        sys.exit("The request timed out. Try again.")
    except ValueError:
        sys.exit("The answer was not valid JSON.")

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(data, indent=2))
    print(f"Saved raw answer to {RAW_FILE} ({RAW_FILE.stat().st_size / 1024:.1f} KB)\n")
    describe(data)


if __name__ == "__main__":
    main()