"""Week 2 (Lab 10): exploratory analysis of the Global Fishing Watch fishing-activity data.

Loads the raw GFW answer, cleans it, adds up hours per map cell, prints the
findings we need to write down, and saves three charts.

Run:  python src/analysis.py
Needs data/raw/gfw_sample.json (made by src/explore_gfw.py).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_FILE = Path("data/raw/gfw_sample.json")
CELLS_FILE = Path("data/processed/gfw_cells.csv")
FIG_DIR = Path("outputs/figures")
GREEN = "#2e8b57"
REQUIRED = ["lat", "lon", "hours"]


def load_gfw(path=RAW_FILE):
    """Read the GFW JSON answer into a table with one row per cell and flag."""
    raw = json.loads(Path(path).read_text())
    rows = []
    for entry in raw["entries"]:
        for cell_rows in entry.values():
            rows.extend(cell_rows)
    table = pd.DataFrame(rows)
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise KeyError(f"columns not found in the GFW answer: {missing}")
    return table


def clean_gfw(table):
    """Drop unusable rows and name the empty flags. Returns (clean table, log)."""
    log = {"rows at start": len(table)}
    table = table.dropna(subset=REQUIRED)
    log["after dropping missing lat/lon/hours"] = len(table)
    table = table[table["hours"] > 0].copy()
    log["after dropping zero or negative hours"] = len(table)
    if "flag" in table.columns:
        table["flag"] = table["flag"].fillna("").replace("", "unknown")
    else:
        table["flag"] = "unknown"
    if "vesselIDs" not in table.columns:
        table["vesselIDs"] = np.nan
    return table, log


def cell_totals(table):
    """Add up hours per map cell (a cell has one row per flag in the raw data)."""
    return table.groupby(["lat", "lon"], as_index=False).agg(
        hours=("hours", "sum"), vessels=("vesselIDs", "sum"), flags=("flag", "nunique"))


def print_findings(table, cells, log):
    """Print the numbers to copy into the written findings."""
    print("Cleaning log:")
    for step, rows in log.items():
        print(f"  {step}: {rows:,}")
    hours = cells["hours"]
    print(f"\nMap cells with fishing: {len(cells):,}")
    print(f"Area covered: lat {cells['lat'].min()} to {cells['lat'].max()}, "
          f"lon {cells['lon'].min()} to {cells['lon'].max()}")
    print(f"Total fishing hours: {hours.sum():,.0f}")
    print(f"Hours per cell: median {hours.median():.1f}, mean {hours.mean():.1f}, max {hours.max():.1f}")

    q1, q3 = hours.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    outliers = cells[hours > upper]
    print(f"\nOutlier cells (above {upper:.1f} hours): {len(outliers)} "
          f"({len(outliers) / len(cells) * 100:.1f}% of cells) hold "
          f"{outliers['hours'].sum() / hours.sum() * 100:.1f}% of all hours")
    top_n = max(1, int(len(cells) * 0.1))
    print(f"Busiest 10% of cells ({top_n}) hold "
          f"{hours.nlargest(top_n).sum() / hours.sum() * 100:.1f}% of all hours")
    print("\nFive busiest cells:")
    print(cells.nlargest(5, "hours")[["lat", "lon", "hours", "vessels"]].round(1).to_string(index=False))

    by_flag = table.groupby("flag")["hours"].sum().sort_values(ascending=False)
    print("\nHours by flag (top 8):")
    print((by_flag.head(8) / by_flag.sum() * 100).round(1).astype(str).add(" %").to_string())
    print(f"Share of hours with unknown flag: {by_flag.get('unknown', 0) / by_flag.sum() * 100:.1f}%")


def make_charts(table, cells):
    """Save three charts: hours per cell, hours by flag, and a cell map."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(cells["hours"]), bins=25, color=GREEN, edgecolor="black")
    ax.set_xlabel("log10 of fishing hours in a cell")
    ax.set_ylabel("number of cells")
    ax.set_title("A few cells have far more fishing than the rest")
    fig.savefig(FIG_DIR / "gfw_hours_per_cell.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    by_flag = table.groupby("flag")["hours"].sum().sort_values(ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_flag.index, by_flag.values, color=GREEN, edgecolor="black")
    ax.set_ylabel("fishing hours")
    ax.set_title("Fishing hours by flag (top 8)")
    fig.savefig(FIG_DIR / "gfw_hours_by_flag.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    points = ax.scatter(cells["lon"], cells["lat"], c=np.log10(cells["hours"]), cmap="Greens",
                        s=60, marker="s", edgecolor="black", linewidth=0.3)
    fig.colorbar(points, label="log10 of fishing hours")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Where fishing hours were recorded (one square per cell)")
    fig.savefig(FIG_DIR / "gfw_cell_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        table, log = clean_gfw(load_gfw())
    except FileNotFoundError:
        sys.exit(f"{RAW_FILE} not found. Run python src/explore_gfw.py first.")
    except (json.JSONDecodeError, KeyError) as err:
        sys.exit(f"Could not read the GFW file: {err}")
    cells = cell_totals(table)
    print_findings(table, cells, log)
    make_charts(table, cells)
    CELLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(CELLS_FILE, index=False)
    print(f"\nSaved {CELLS_FILE} and 3 charts in {FIG_DIR}/")


if __name__ == "__main__":
    main()