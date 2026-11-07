"""Day 3: look inside a Danish Maritime Authority (DMA) AIS file.

Reads the big CSV in chunks, prints the columns and what the 'Ship type' and
'Navigational status' fields contain, and saves a small sample of tracks to
data/processed/dma_sample.csv.

Usage:
    python src/explore_dma.py data/raw/aisdk-YYYY-MM-DD.csv
    python src/explore_dma.py data/raw/aisdk-YYYY-MM-DD.csv --max-rows 5000000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

USE_COLS = ["# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude",
            "Navigational status", "ROT", "SOG", "COG", "Ship type"]
CHUNK_ROWS = 500_000
OUT_FILE = Path("data/processed/dma_sample.csv")
SAMPLE_COLS = ["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "Ship type"]
VESSELS_PER_CLASS = 60
MIN_ROWS_PER_VESSEL = 100     # about 15+ minutes of reports
MAX_ROWS_PER_VESSEL = 600     # after cleaning, about 100 minutes
MIN_MOVING_SHARE = 0.2        # skip boats that sit still (they teach nothing about movement)
STOPPED_KN = 0.5
SEED = 42


def read_chunks(path, max_rows):
    """Yield chunks of the CSV, stopping once max_rows rows have been read."""
    seen = 0
    for chunk in pd.read_csv(path, usecols=USE_COLS, chunksize=CHUNK_ROWS, low_memory=False):
        yield chunk
        seen += len(chunk)
        if seen >= max_rows:
            break


def check_header(path):
    """Print every column name and stop if a column we need is missing."""
    header = pd.read_csv(path, nrows=0)
    print("Columns in the file:")
    print("  " + ", ".join(header.columns))
    missing = [c for c in USE_COLS if c not in header.columns]
    if missing:
        sys.exit(f"\nThese columns were not found: {missing}. Tell us, the file format may have changed.")


def summarise(data, ship_counts, status_counts):
    """Print the facts we need to write down in docs/data-notes.md."""
    print(f"\nRows read (Class A and B only): {len(data):,}")
    print(f"Different vessels (MMSI): {data['MMSI'].nunique():,}")
    print("\nShip type values (all rows read):")
    print(ship_counts.sort_values(ascending=False).head(15).astype(int).to_string())
    print("\nNavigational status values (all rows read):")
    print(status_counts.sort_values(ascending=False).head(10).astype(int).to_string())
    print("\nShare of missing values per column:")
    print((data.isna().mean() * 100).round(1).to_string())
    print("\nSpeed over ground (SOG, knots):")
    print(data["SOG"].describe().round(2).to_string())


def save_sample(data):
    """Clean first, then save whole tracks for some moving fishing and non-fishing boats."""
    data = data[(data["Latitude"].abs() <= 90) & (data["Longitude"].abs() <= 180)]
    data = data.drop_duplicates(["MMSI", "# Timestamp", "Latitude", "Longitude", "SOG"])
    data = data[~data["MMSI"].astype(str).str.startswith("97")]

    per_boat = data.groupby("MMSI").agg(rows=("SOG", "size"), moving=("SOG", lambda s: (s >= STOPPED_KN).mean()))
    good = per_boat[(per_boat["rows"] >= MIN_ROWS_PER_VESSEL) & (per_boat["moving"] >= MIN_MOVING_SHARE)].index
    print(f"\nBoats after cleaning: {len(per_boat):,}; long enough and moving: {len(good):,}")

    ship = data["Ship type"].fillna("Undefined").str.lower()
    fishing = data.loc[ship.eq("fishing") & data["MMSI"].isin(good), "MMSI"].unique()
    other = data.loc[~ship.isin(["fishing", "undefined"]) & data["MMSI"].isin(good), "MMSI"].unique()
    print(f"Moving boats with ship type 'Fishing': {len(fishing)}")
    print(f"Moving boats with a known non-fishing ship type: {len(other)}")
    if len(fishing) == 0 or len(other) == 0:
        print("Could not build a labelled sample. Try a bigger --max-rows, or a different day.")
        return

    rng = np.random.default_rng(SEED)
    parts = []
    for ids, label in ((fishing, "fishing"), (other, "other")):
        chosen = rng.permutation(ids)[:VESSELS_PER_CLASS]
        sub = data[data["MMSI"].isin(chosen)].groupby("MMSI").head(MAX_ROWS_PER_VESSEL)[SAMPLE_COLS].copy()
        sub["label"] = label
        parts.append(sub)
    sample = pd.concat(parts)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(sample):,} rows from {sample['MMSI'].nunique()} boats to {OUT_FILE} "
          f"({OUT_FILE.stat().st_size / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Explore a DMA AIS CSV file.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    args = parser.parse_args()

    try:
        check_header(args.csv_path)
        ship_counts = pd.Series(dtype="float64")
        status_counts = pd.Series(dtype="float64")
        frames = []
        for chunk in read_chunks(args.csv_path, args.max_rows):
            ship_counts = ship_counts.add(chunk["Ship type"].fillna("(missing)").value_counts(), fill_value=0)
            status_counts = status_counts.add(chunk["Navigational status"].fillna("(missing)").value_counts(), fill_value=0)
            frames.append(chunk.loc[chunk["Type of mobile"].isin(["Class A", "Class B"]), ["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "Navigational status", "ROT", "COG", "Ship type"]])
    except FileNotFoundError:
        sys.exit(f"File not found: {args.csv_path}")
    except pd.errors.EmptyDataError:
        sys.exit("The file is empty.")
    except pd.errors.ParserError as err:
        sys.exit(f"Could not read the file as CSV: {err}")

    if not frames:
        sys.exit("No rows were read.")
    data = pd.concat(frames, ignore_index=True)
    summarise(data, ship_counts, status_counts)
    save_sample(data)


if __name__ == "__main__":
    main()