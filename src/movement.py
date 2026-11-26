# kernels, residuals (predicted - actual),used in belief.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_AT_EQUATOR = 111.32
CELL_KM_DEMO = 11.1        # a 0.1 degree cell, the size of our real map cells
STEP_MINUTES_DEMO = 120    # one step of the demo (longer than we can measure, so the spread is stretched)
FIT_STEP_MINUTES = 30      # the step the real tracks let us measure
LOOKBACK_MIN = 5           # minutes used to measure the last known velocity
SEED = 42

TRACKS_FILE = Path("data/processed/tracks_clean.csv")
SPLIT_FILE = Path("data/processed/boat_split.csv")
TABLE_FILE = Path("outputs/results/movement_table.csv")


def to_km(lat, lon, lat0, lon0):# calc distance from long and lat and covert into kilometeres
    east = (np.asarray(lon) - lon0) * KM_PER_DEG_LON_AT_EQUATOR * np.cos(np.radians(lat0))
    north = (np.asarray(lat) - lat0) * KM_PER_DEG_LAT
    return east, north

def split_boats(tracks, seed=SEED, test_share=0.3):# train test split keep equal labels in bot groups
    rng = np.random.default_rng(seed)
    boats = tracks.groupby("vessel_id")["label"].first().sort_index()
    rows = []
    for label, group in boats.groupby(boats):
        ids = rng.permutation(group.index.to_numpy())
        n_test = max(1, int(round(len(ids) * test_share)))
        for i, vessel_id in enumerate(ids):
            rows.append({"vessel_id": vessel_id, "label": label, "group": "test" if i < n_test else "train"})
    return pd.DataFrame(rows).sort_values("vessel_id").reset_index(drop=True)

def load_or_make_split(tracks, path=SPLIT_FILE):
    try:
        saved = pd.read_csv(path)
        if set(saved["vessel_id"]) == set(tracks["vessel_id"]):
            return saved
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        pass
    split = split_boats(tracks)
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)
    return split

def last_velocity(points, lookback=LOOKBACK_MIN): #calc velocity using last known position
    indexed = points.set_index("time")[["lat", "lon"]]
    last_time = indexed.index[-1]
    earlier = last_time - pd.Timedelta(minutes=lookback)
    if earlier not in indexed.index:
        return 0.0, 0.0
    lat, lon = indexed.loc[last_time]
    east, north = to_km(indexed.loc[earlier, "lat"], indexed.loc[earlier, "lon"], lat, lon)
    return -float(east) / lookback, -float(north) / lookback


def residuals(tracks, step_minutes, lookback=LOOKBACK_MIN):
    parts = []
    for vessel_id, group in tracks.groupby("vessel_id"):
        g = group.drop_duplicates("time").set_index("time").sort_index()[["lat", "lon"]]
        now = g.index
        before = g.reindex(now - pd.Timedelta(minutes=lookback))
        after = g.reindex(now + pd.Timedelta(minutes=step_minutes))
        ok = before["lat"].notna().to_numpy() & after["lat"].notna().to_numpy()
        if not ok.any():
            continue
        lat0, lon0 = g["lat"].to_numpy()[ok], g["lon"].to_numpy()[ok]
        be, bn = to_km(before["lat"].to_numpy()[ok], before["lon"].to_numpy()[ok], lat0, lon0)
        ae, an = to_km(after["lat"].to_numpy()[ok], after["lon"].to_numpy()[ok], lat0, lon0)
        ve, vn = -be / lookback, -bn / lookback            # km per minute
        parts.append(pd.DataFrame({"vessel_id": vessel_id,
                                   "res_east": ae - ve * step_minutes,
                                   "res_north": an - vn * step_minutes}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["vessel_id", "res_east", "res_north"])


class MovementModel: #custom probailites model ( not regular models, based on probaility kernels)
    def __init__(self, step_minutes, cell_km, radius=5, fit_step_minutes=None):
        self.step_minutes = step_minutes
        self.fit_step_minutes = fit_step_minutes or step_minutes     # horizon measured on real tracks
        self.spread_scale = step_minutes / self.fit_step_minutes     # 1 = no stretching
        self.cell_km = cell_km
        self.radius = radius                  # the table covers offsets -radius..+radius
        self.kernel = None
        self.n_samples = 0
        self.boat_ids = []
        self.clipped_share = 0.0

    def fit(self, tracks):
        """Count residuals into the table. Use training boats only."""
        res = residuals(tracks, self.fit_step_minutes)
        if res.empty:
            raise ValueError("no usable pairs of points: tracks are too short or have too many gaps")
        dx = np.rint(res["res_east"].to_numpy() * self.spread_scale / self.cell_km).astype(int)
        dy = np.rint(-res["res_north"].to_numpy() * self.spread_scale / self.cell_km).astype(int)   # y grows southwards
        r = self.radius
        self.clipped_share = float(((np.abs(dx) > r) | (np.abs(dy) > r)).mean())
        counts = np.zeros((2 * r + 1, 2 * r + 1))
        for ox, oy in ((dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy), (dy, dx), (-dy, dx), (dy, -dx), (-dy, -dx)):
            np.add.at(counts, (np.clip(oy, -r, r) + r, np.clip(ox, -r, r) + r), 1)   # equal in every direction
        self.kernel = counts / counts.sum()
        self.n_samples = len(res)
        self.boat_ids = sorted(res["vessel_id"].unique().tolist())
        return self

    def stay_probability(self):# how likely is the boat in the same cell
        return float(self.kernel[self.radius, self.radius])

    def drift_cells(self, v_east, v_north):# how likely is the boat to drift to other cells
        return v_east * self.step_minutes / self.cell_km, -v_north * self.step_minutes / self.cell_km

    def table(self):
        """The move probabilities as a table, biggest first (zero rows left out)."""
        r = self.radius
        rows = [{"dx": x - r, "dy": y - r, "probability": self.kernel[y, x]}
                for y in range(2 * r + 1) for x in range(2 * r + 1) if self.kernel[y, x] > 0]
        return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def default_kernel(radius=5, p_stay=0.83, p_side=0.027, p_diag=0.009):#when real data is not available
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    c = radius
    kernel[c, c] = p_stay
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        kernel[c + dy, c + dx] = p_side
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        kernel[c + dy, c + dx] = p_diag
    ring = [(c + dy, c + dx) for dy in range(-2, 3) for dx in range(-2, 3) if max(abs(dx), abs(dy)) == 2]
    rest = 1.0 - kernel.sum()
    for y, x in ring:
        kernel[y, x] = rest / len(ring)
    return kernel

def load_demo_kernel():# return the kernel probs
    try:
        tracks = pd.read_csv(TRACKS_FILE, parse_dates=["time"])
        split = load_or_make_split(tracks)
        ids = split.loc[(split["group"] == "train") & (split["label"] == "fishing"), "vessel_id"]
        model = MovementModel(STEP_MINUTES_DEMO, CELL_KM_DEMO, fit_step_minutes=FIT_STEP_MINUTES)
        return model.fit(tracks[tracks["vessel_id"].isin(ids)]).kernel, "measured on real tracks"
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError) as err:
        print(f"Movement table not measured ({err}). Using the fallback table.")
        return default_kernel(), "fallback table"


def main():
    try:
        tracks = pd.read_csv(TRACKS_FILE, parse_dates=["time"])
    except FileNotFoundError:
        sys.exit(f"{TRACKS_FILE} not found. Run python src/clean_tracks.py first.")
    split = load_or_make_split(tracks)
    counts = split.groupby(["group", "label"]).size().unstack(fill_value=0)
    print("Boats per group (split by boat, seed 42):\n" + counts.to_string())

    train = tracks[tracks["vessel_id"].isin(split.loc[(split["group"] == "train") & (split["label"] == "fishing"), "vessel_id"])]
    try:
        model = MovementModel(STEP_MINUTES_DEMO, CELL_KM_DEMO, fit_step_minutes=FIT_STEP_MINUTES).fit(train)
    except ValueError as err:
        sys.exit(f"Could not fit the movement model: {err}")
    print(f"\nDemo model: step {STEP_MINUTES_DEMO} min, cell {CELL_KM_DEMO} km, trained on "
          f"{len(model.boat_ids)} training fishing boats, {model.n_samples:,} samples")
    print(f"Spread measured at {FIT_STEP_MINUTES} min and stretched x{model.spread_scale:g} to the demo step (an assumption)")
    print(f"Probability of staying in the same cell: {model.stay_probability():.3f}")
    print(f"Share of residuals outside the table (clipped to its edge): {model.clipped_share:.3%}")
    table = model.table()
    print("\nMost likely moves (dx cells east, dy cells south):\n" + table.head(8).round(4).to_string(index=False))
    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_FILE, index=False)
    print(f"\nSaved {TABLE_FILE} and {SPLIT_FILE}")

if __name__ == "__main__":
    main()