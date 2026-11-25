#test belief model on real data
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from belief import Belief
from movement import LOOKBACK_MIN, MovementModel, last_velocity, load_or_make_split, to_km

TRACKS_FILE = Path("data/processed/tracks_clean.csv")
FIG_DIR, RESULT_DIR = Path("outputs/figures"), Path("outputs/results")
CELL_KM, STEP_MIN, GRID, HORIZON_STEPS, START_EVERY = 1.0, 5, 61, 6, 5
CENTRE = GRID // 2
GREEN, BLACK, GREY = "#2e8b57", "#111111", "#888888"


def episodes(track):
    """Yield (last_seen_row_index, velocity, truth positions) for blackout starts that have all needed points."""
    g = track.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    times = set(g["time"])
    for i in range(0, len(g), START_EVERY):
        t0 = g.loc[i, "time"]
        need = [t0 - pd.Timedelta(minutes=LOOKBACK_MIN)] + [t0 + pd.Timedelta(minutes=STEP_MIN * k) for k in range(1, HORIZON_STEPS + 1)]
        if all(t in times for t in need):
            yield g, i, t0


def run_episode(model, g, i, t0):
    """Return rows of errors and coverage for one blackout."""
    now = g.loc[i]
    ve, vn = last_velocity(g[g["time"] <= t0], LOOKBACK_MIN)
    drift = model.drift_cells(ve, vn)
    with_drift = Belief((GRID, GRID), (CENTRE, CENTRE))
    no_drift = Belief((GRID, GRID), (CENTRE, CENTRE))
    rows = []
    for k in range(1, HORIZON_STEPS + 1):
        minutes = STEP_MIN * k
        with_drift.predict(model.kernel, drift)
        no_drift.predict(model.kernel)
        later = g[g["time"] == t0 + pd.Timedelta(minutes=minutes)].iloc[0]
        east, north = (float(v) for v in to_km(later["lat"], later["lon"], now["lat"], now["lon"]))
        mx, my = with_drift.mean_position()
        row = {"minutes": minutes,
               "last_seen_km": np.hypot(east, north),
               "constant_speed_km": np.hypot(east - ve * minutes, north - vn * minutes),
               "belief_km": np.hypot(east - (mx - CENTRE) * CELL_KM, north + (my - CENTRE) * CELL_KM)}
        tx, ty = CENTRE + int(round(east / CELL_KM)), CENTRE - int(round(north / CELL_KM))
        inside = 0 <= tx < GRID and 0 <= ty < GRID
        for name, b in (("drift", with_drift), ("nodrift", no_drift)):
            mask = b.top_mask(0.9)
            row[f"coverage90_{name}"] = float(inside and mask[ty, tx])
            row[f"area90_{name}_km2"] = float(mask.sum()) * CELL_KM ** 2
        rows.append(row)
    return rows


def make_chart(summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(summary["minutes"], summary["last_seen_km"], marker="o", color=BLACK, label="last seen position")
    ax.plot(summary["minutes"], summary["belief_km"], marker="o", color=GREEN, label="belief (average position)")
    ax.set_xlabel("minutes since AIS stopped")
    ax.set_ylabel("average error in km")
    ax.set_title("Belief vs last seen position on held-out real boats")
    ax.legend()
    fig.savefig(FIG_DIR / "belief_error_vs_minutes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        tracks = pd.read_csv(TRACKS_FILE, parse_dates=["time"])
    except FileNotFoundError:
        sys.exit(f"{TRACKS_FILE} not found. Run python src/clean_tracks.py first.")
    split = load_or_make_split(tracks)
    train_ids = split.loc[(split["group"] == "train") & (split["label"] == "fishing"), "vessel_id"]
    test_ids = split.loc[(split["group"] == "test") & (split["label"] == "fishing"), "vessel_id"]
    try:
        model = MovementModel(STEP_MIN, CELL_KM).fit(tracks[tracks["vessel_id"].isin(train_ids)])
    except ValueError as err:
        sys.exit(f"Could not fit the movement model: {err}")
    print(f"Movement table: step {STEP_MIN} min, cell {CELL_KM} km, {len(model.boat_ids)} training boats, "
          f"{model.n_samples:,} samples, chance of staying {model.stay_probability():.2f}")

    rows, n_episodes = [], 0
    for vessel_id in test_ids:
        track = tracks[tracks["vessel_id"] == vessel_id]
        for g, i, t0 in episodes(track):
            for row in run_episode(model, g, i, t0):
                rows.append({"vessel_id": vessel_id, **row})
            n_episodes += 1
    if not rows:
        sys.exit("No test episodes: the test tracks are too short or have gaps.")
    df = pd.DataFrame(rows)
    per_boat = df.groupby(["vessel_id", "minutes"]).mean().reset_index()      # every boat counts once
    summary = per_boat.groupby("minutes").mean(numeric_only=True).drop(columns="vessel_id").round(3).reset_index()
    print(f"Test boats: {df['vessel_id'].nunique()}, blackout episodes: {n_episodes}\n")
    print(summary.to_string(index=False))
    last = summary.iloc[-1]
    print(f"\nAfter {int(last['minutes'])} minutes the average error is {last['last_seen_km']:.2f} km for the last seen "
          f"position and {last['belief_km']:.2f} km for the belief.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / "belief_vs_last_seen.csv", index=False)
    make_chart(summary)
    print(f"Saved {RESULT_DIR / 'belief_vs_last_seen.csv'} and {FIG_DIR / 'belief_error_vs_minutes.png'}")


if __name__ == "__main__":
    main()