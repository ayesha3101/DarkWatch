"""Checks that the data setup and preprocessing are complete and consistent.

Run:  python src/verify_data.py
Prints PASS / WARN / FAIL for every check and exits with an error if anything FAILs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROCESSED = Path("data/processed")
FILES = ["gfw_cells.csv", "density.npy", "cost.npy", "grid_info.json", "dma_sample.csv", "tracks_clean.csv"]
TRACK_COLUMNS = ["vessel_id", "time", "lat", "lon", "sog", "speed_kn", "turn_deg", "label"]
MIN_POINTS = 15
results = []


def check(ok, message, warn=False):
    """Record and print one check. warn=True means a failure is only a warning."""
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append(status)
    print(f"[{status}] {message}")


def check_files():
    print("\nFiles")
    for name in FILES:
        check((PROCESSED / name).exists(), f"data/processed/{name} exists")


def check_grids():
    print("\nGrids")
    try:
        density = np.load(PROCESSED / "density.npy")
        cost = np.load(PROCESSED / "cost.npy")
        info = json.loads((PROCESSED / "grid_info.json").read_text())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as err:
        check(False, f"could not load the grids: {err}")
        return None
    land = np.isinf(cost)
    check(density.shape == cost.shape, f"density and cost have the same shape {density.shape}")
    check((info["n_rows"], info["n_cols"]) == density.shape, "grid_info.json matches the grid shape")
    check(np.isfinite(density).all() and density.min() >= 0 and density.max() <= 1.0 + 1e-9,
          "density is finite and between 0 and 1")
    check(abs(density.max() - 1.0) < 1e-9, "density maximum is 1 (scaled)")
    check(set(np.unique(cost)) <= {1.0, np.inf}, "cost holds only 1.0 (water) and infinity (land)")
    check(0 < land.sum() < 0.5 * land.size, f"land cells look reasonable ({int(land.sum())} of {land.size})", warn=True)
    check((density[land] == 0).all(), "no fishing activity on land")
    check((density[~land] > 0).mean() > 0.1, "more than 10% of water cells have fishing activity", warn=True)
    return density


def check_gfw_cells(density):
    print("\nGFW cells")
    try:
        cells = pd.read_csv(PROCESSED / "gfw_cells.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError) as err:
        check(False, f"could not load gfw_cells.csv: {err}")
        return
    check(cells[["lat", "lon", "hours"]].notna().all().all(), "no missing lat, lon or hours")
    check((cells["hours"] > 0).all(), "all hours are positive")
    check(not cells.duplicated(["lat", "lon"]).any(), "one row per cell")
    if density is not None:
        check(int((density > 0).sum()) == len(cells), "cells with fishing in the grid match the cell table", warn=True)


def check_tracks():
    print("\nCleaned Danish tracks")
    try:
        df = pd.read_csv(PROCESSED / "tracks_clean.csv", parse_dates=["time"])
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as err:
        check(False, f"could not load tracks_clean.csv: {err}")
        return
    missing = [c for c in TRACK_COLUMNS if c not in df.columns]
    check(not missing, f"all expected columns are present {missing if missing else ''}")
    if missing:
        return
    check(df[["time", "lat", "lon", "label"]].notna().all().all(), "no missing time, lat, lon or label")
    check(df["lat"].abs().max() <= 90 and df["lon"].abs().max() <= 180, "all positions are valid")
    check(not df.duplicated(["vessel_id", "time"]).any(), "no duplicate boat and time")
    check((df.groupby("vessel_id")["label"].nunique() == 1).all(), "every boat has exactly one label")
    check(not df["vessel_id"].astype(str).str.startswith("97").any(), "no 97-prefix IDs")
    check(df.groupby("vessel_id").size().min() >= MIN_POINTS, f"every boat has at least {MIN_POINTS} points")
    check(df.groupby("vessel_id")["time"].apply(lambda s: s.is_monotonic_increasing).all(), "points are in time order")
    check(df["speed_kn"].dropna().max() <= 60, "speeds are plausible (60 knots or less)")
    check(df["turn_deg"].dropna().abs().max() <= 180, "turns are between -180 and 180 degrees")
    per_label = df.groupby("label")["vessel_id"].nunique()
    check(set(per_label.index) == {"fishing", "other"} and per_label.min() >= 20,
          f"both classes have at least 20 boats ({per_label.to_dict()})", warn=True)
    print(f"      {df['vessel_id'].nunique()} boats, {len(df):,} points")


def check_loader():
    print("\nLoader")
    try:
        from data import load_data
        loaded = load_data()
    except (ImportError, OSError, ValueError) as err:
        check(False, f"load_data() failed: {err}")
        return
    check(loaded["source"] == "real", "load_data() uses the real grids")
    check(loaded["real_tracks"] is not None, "load_data() finds the real tracks")


def check_repo():
    print("\nRepository hygiene")
    big = [p.name for p in PROCESSED.glob("*") if p.is_file() and p.stat().st_size > 10 * 1024 * 1024]
    check(not big, f"no file in data/processed is over 10 MB {big if big else ''}", warn=True)
    try:
        reqs = Path("requirements.txt").read_text().lower().replace("_", "-")
        check("global-land-mask" in reqs, "requirements.txt lists global-land-mask", warn=True)
        check("pygame" not in reqs, "requirements.txt has no pygame", warn=True)
    except FileNotFoundError:
        check(False, "requirements.txt exists", warn=True)
    try:
        ignore = Path(".gitignore").read_text()
        check("data/raw" in ignore and ".env" in ignore, ".gitignore hides data/raw and .env", warn=True)
    except FileNotFoundError:
        check(False, ".gitignore exists", warn=True)


def main():
    check_files()
    density = check_grids()
    check_gfw_cells(density)
    check_tracks()
    check_loader()
    check_repo()
    counts = {s: results.count(s) for s in ("PASS", "WARN", "FAIL")}
    print(f"\nResult: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()