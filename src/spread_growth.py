"""How fast does the prediction error of a hidden boat grow with time? (Lab 9)

For the TRAINING fishing boats, we predict where each boat will be 5, 10, ... 30 minutes
after a moment (using its last velocity) and measure how far the real position is from
that prediction. Then we fit a straight line on a log-log plot: error ~ minutes ^ alpha.

Why: our tracks are only 1.5 to 2 hours long, so we can measure the error up to about
30 minutes. If the demo uses a longer step, the spread has to be EXTENDED beyond what
the data shows. This script prints the measured growth and what it would predict at
longer steps, so the extension is a stated assumption and not a guess.

Run:  python src/spread_growth.py
"""
import sys

import numpy as np
import pandas as pd

from movement import CELL_KM_DEMO, TRACKS_FILE, load_or_make_split, residuals

HORIZONS = [5, 10, 15, 20, 25, 30]
EXTEND_TO = [60, 120, 240]


def main():
    try:
        tracks = pd.read_csv(TRACKS_FILE, parse_dates=["time"])
    except FileNotFoundError:
        sys.exit(f"{TRACKS_FILE} not found. Run python src/clean_tracks.py first.")
    split = load_or_make_split(tracks)
    ids = split.loc[(split["group"] == "train") & (split["label"] == "fishing"), "vessel_id"]
    train = tracks[tracks["vessel_id"].isin(ids)]

    rows = []
    for minutes in HORIZONS:
        res = residuals(train, minutes)
        if res.empty:
            continue
        dist = np.hypot(res["res_east"], res["res_north"])
        rows.append({"minutes": minutes, "samples": len(res), "median_km": dist.median(),
                     "p90_km": dist.quantile(0.9), "rms_km": float(np.sqrt((dist ** 2).mean()))})
    if len(rows) < 3:
        sys.exit("Not enough horizons with data to fit a growth line.")
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))

    alpha, intercept = np.polyfit(np.log(table["minutes"]), np.log(table["rms_km"]), 1)
    print(f"\nFitted growth: error is about {np.exp(intercept):.3f} km x minutes^{alpha:.2f}")
    print("(alpha 1 = grows in a straight line, 0.5 = grows like a random walk)")
    print(f"\nCell size in the demo: {CELL_KM_DEMO} km")
    print(f"{'minutes':>8}{'error km':>10}{'in cells':>10}   note")
    for minutes in HORIZONS[-1:] + EXTEND_TO:
        km = float(np.exp(intercept) * minutes ** alpha)
        note = "measured range" if minutes <= HORIZONS[-1] else "EXTENDED beyond the data"
        print(f"{minutes:>8}{km:>10.2f}{km / CELL_KM_DEMO:>10.2f}   {note}")


if __name__ == "__main__":
    main()