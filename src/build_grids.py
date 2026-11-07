"""Week 2: build the real density and cost grids for our area.

density[y, x]: fishing activity from 0 to 1 (hours, log-scaled so a few very busy
               cells do not hide all the others).
cost[y, x]:    1.0 = open water, np.inf = land (from the global-land-mask package).
Row 0 is the north edge, so y grows southwards. Both grids have the same shape.

Run:  python src/build_grids.py
Needs data/processed/gfw_cells.csv (made by src/analysis.py).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELLS_FILE = Path("data/processed/gfw_cells.csv")
OUT_DIR = Path("data/processed")

LAT_MAX, LON_MIN, STEP = 25.0, 65.0, 0.1   # north-west corner and cell size in degrees
N_ROWS, N_COLS = 21, 26                    # lat 25.0 to 23.0 and lon 65.0 to 67.5


def cell_centres():
    """Latitude and longitude of every grid cell, shape (N_ROWS, N_COLS)."""
    lats = LAT_MAX - STEP * np.arange(N_ROWS)
    lons = LON_MIN + STEP * np.arange(N_COLS)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid


def build_hours_grid(cells):
    """Put each cell's hours into the grid. Returns (grid, number of cells outside)."""
    rows = np.round((LAT_MAX - cells["lat"].to_numpy()) / STEP).astype(int)
    cols = np.round((cells["lon"].to_numpy() - LON_MIN) / STEP).astype(int)
    inside = (rows >= 0) & (rows < N_ROWS) & (cols >= 0) & (cols < N_COLS)
    grid = np.zeros((N_ROWS, N_COLS))
    np.add.at(grid, (rows[inside], cols[inside]), cells["hours"].to_numpy()[inside])
    return grid, int((~inside).sum())


def build_land_mask():
    """True where a grid cell is land."""
    from global_land_mask import globe
    lat_grid, lon_grid = cell_centres()
    return globe.is_land(lat_grid, lon_grid)


def save_preview(density, land, path="outputs/figures/real_map.png"):
    shown = np.ma.masked_where(land, density)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_facecolor("#444444")
    image = ax.imshow(shown, origin="upper", cmap="Greens",
                      extent=[LON_MIN - STEP / 2, LON_MIN + STEP * (N_COLS - 0.5),
                              LAT_MAX - STEP * (N_ROWS - 0.5), LAT_MAX + STEP / 2])
    fig.colorbar(image, label="fishing activity (0-1, log-scaled)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Data source: real (Global Fishing Watch), grey = land")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        cells = pd.read_csv(CELLS_FILE)
        land = build_land_mask()
    except FileNotFoundError:
        sys.exit(f"{CELLS_FILE} not found. Run python src/analysis.py first.")
    except ImportError as err:
        sys.exit(f"Could not import the land mask package: {err}\n"
                 "Install it into THIS environment with:  python -m pip install global-land-mask")

    hours, outside = build_hours_grid(cells)
    on_land = (hours > 0) & land
    print(f"Grid: {N_COLS} x {N_ROWS} cells, {int(land.sum())} land cells")
    print(f"GFW cells outside the grid: {outside}")
    print(f"GFW cells that fall on land (coast cells, the land mask is finer than 0.1 degree): {int(on_land.sum())}")

    density = np.log1p(hours)
    density[land] = 0.0
    density = density / density.max()
    cost = np.where(land, np.inf, 1.0)
    assert cost.shape == density.shape

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "density.npy", density)
    np.save(OUT_DIR / "cost.npy", cost)
    (OUT_DIR / "grid_info.json").write_text(json.dumps(
        {"lat_max": LAT_MAX, "lon_min": LON_MIN, "step_deg": STEP, "n_rows": N_ROWS, "n_cols": N_COLS}, indent=2))
    save_preview(density, land)
    print(f"Saved density.npy, cost.npy and grid_info.json in {OUT_DIR}/ and outputs/figures/real_map.png")


if __name__ == "__main__":
    main()