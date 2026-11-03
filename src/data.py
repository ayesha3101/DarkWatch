"""Data loading, and the seeded simulator we use when real data is missing.

Grid convention (agreed between both members):
    cost[y, x]    2D NumPy array. 1.0 = open water, higher = harder to cross,
                  np.inf = land (cannot be entered). Every finite cost is >= 1,
                  which is what makes the A* heuristic admissible.
    density[y, x] 2D NumPy array in 0..1: how much fishing happens in that cell.
Tracks are a Pandas table with the columns in TRACK_COLUMNS.
"""
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
TRACK_COLUMNS = ["vessel_id", "step", "x", "y", "speed", "turn", "label"]


def generate_ocean(seed=42, width=40, height=30):
    """Make a seeded grid ocean. Returns (cost, density)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    cost = np.ones((height, width))
    cost[(yy < 3 + 2 * np.sin(xx / 5.0))] = np.inf                       # coast along the top
    cost[(xx - 0.7 * width) ** 2 + (yy - 0.6 * height) ** 2 < 9] = np.inf  # an island
    cost[(xx >= 5) & (xx < 12) & (yy >= 15) & (yy < 22)] = 3.0             # slow, restricted zone

    density = np.zeros((height, width))
    for _ in range(4):
        cx, cy = rng.uniform(4, width - 4), rng.uniform(6, height - 4)
        spread = rng.uniform(2.5, 5.0)
        density += rng.uniform(0.5, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread ** 2))
    density[np.isinf(cost)] = 0.0
    return cost, density / density.max()


def generate_tracks(cost, density, seed=42, n_vessels=40, n_steps=120):
    """Make labelled tracks: 'fishing' boats go slowly and turn a lot, 'other' boats go fast and straight."""
    rng = random.Random(seed)
    height, width = cost.shape
    rows = []
    for vessel_id in range(n_vessels):
        label = "fishing" if vessel_id % 2 == 0 else "other"
        x, y = _start_position(rng, cost, density, label)
        heading = rng.uniform(0, 2 * math.pi)
        for step in range(n_steps):
            if label == "fishing":
                speed, turn = max(0.0, rng.gauss(0.3, 0.1)), rng.gauss(0, 0.9)
            else:
                speed, turn = max(0.0, rng.gauss(1.0, 0.15)), rng.gauss(0, 0.1)
            heading += turn
            nx, ny = x + speed * math.cos(heading), y + speed * math.sin(heading)
            blocked = not (0 <= nx < width and 0 <= ny < height) or np.isinf(cost[int(ny), int(nx)])
            if blocked:
                heading += math.pi
            else:
                x, y = nx, ny
            rows.append((vessel_id, step, x, y, speed, turn, label))
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _start_position(rng, cost, density, label, tries=1000):
    """Pick a water cell; fishing boats prefer busy fishing areas."""
    height, width = cost.shape
    for _ in range(tries):
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        if np.isinf(cost[int(y), int(x)]):
            continue
        if label == "other" or rng.random() < density[int(y), int(x)]:
            return x, y
    raise RuntimeError("Could not find a start position in water.")


def load_data(seed=42):
    """Load processed real data if all files exist, otherwise use the simulator.

    Returns a dict with cost, density, tracks and source ('real' or 'simulated').
    """
    try:
        cost = np.load(PROCESSED_DIR / "cost.npy")
        density = np.load(PROCESSED_DIR / "density.npy")
        tracks = pd.read_csv(PROCESSED_DIR / "tracks.csv")
        missing = set(TRACK_COLUMNS) - set(tracks.columns)
        if missing:
            raise KeyError(f"tracks.csv is missing columns: {sorted(missing)}")
        if cost.shape != density.shape:
            raise ValueError("cost and density grids have different shapes")
        return {"cost": cost, "density": density, "tracks": tracks, "source": "real"}
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError) as err:
        print(f"Real data not available ({err}). Using the simulator instead.")
        cost, density = generate_ocean(seed)
        return {"cost": cost, "density": density, "tracks": generate_tracks(cost, density, seed),
                "source": "simulated"}


def save_preview(data, path="outputs/figures/simulated_map.png"):
    """Save a picture of the density map, land and a few tracks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = np.ma.masked_where(np.isinf(data["cost"]), data["density"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#444444")
    image = ax.imshow(shown, origin="upper", cmap="Greens")
    fig.colorbar(image, label="fishing activity (0-1)")
    for vessel_id, track in list(data["tracks"].groupby("vessel_id"))[:6]:
        ax.plot(track["x"] - 0.5, track["y"] - 0.5, linewidth=1, label=f"boat {vessel_id} ({track['label'].iloc[0]})")
    ax.set_title(f"Data source: {data['source']} (grey = land)")
    ax.legend(fontsize=7, loc="lower right")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    data = load_data()
    cost, tracks = data["cost"], data["tracks"]
    print(f"Source: {data['source']}")
    print(f"Grid: {cost.shape[1]} x {cost.shape[0]} cells, {int(np.isinf(cost).sum())} land cells")
    print(f"Tracks: {len(tracks)} rows, {tracks['vessel_id'].nunique()} vessels")
    print(tracks["label"].value_counts().to_string())
    print(f"Preview saved to {save_preview(data)}")