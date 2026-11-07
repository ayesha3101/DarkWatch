"""Week 2 (Lab 10): clean the Danish AIS sample so it is ready for the movement model and classifier.

Steps: remove invalid positions, remove duplicates, remove 97-prefix IDs, label each
boat once from its real ship type, resample to one point per minute, work out speed
and turning from the positions, drop short tracks and boats that mostly sit still.

Run:  python src/clean_tracks.py
Needs data/processed/dma_sample.csv (made by src/explore_dma.py).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_FILE = Path("data/processed/dma_sample.csv")
OUT_FILE = Path("data/processed/tracks_clean.csv")
FIG_DIR = Path("outputs/figures")
MIN_POINTS = 15          # at least this many one-minute points per boat
MIN_MOVING_SHARE = 0.2   # boats that move less than this share of the time are dropped
STOPPED_KN = 0.5         # below this speed a boat counts as stopped
MAX_GAP_MIN = 2          # do not compute speed across gaps longer than this
NEEDED = ["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "Ship type"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two points on the Earth."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((np.radians(lon2) - np.radians(lon1)) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Direction from point 1 to point 2, in degrees from north (0 to 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2) - np.radians(lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def read_sample(path):
    df = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    df["time"] = pd.to_datetime(df["# Timestamp"], format="%d/%m/%Y %H:%M:%S")
    return df


def label_per_vessel(df):
    """One label per boat from its real ship types (ignoring 'Undefined'). Conflicting boats are dropped."""
    known = df[df["Ship type"].notna() & (df["Ship type"] != "Undefined")]
    labels = {}
    for mmsi, kinds in known.groupby("MMSI")["Ship type"].agg(lambda s: {k.lower() for k in s}).items():
        if len(kinds) == 1:
            labels[mmsi] = "fishing" if "fishing" in kinds else "other"
    df = df[df["MMSI"].isin(labels)].copy()
    df["label"] = df["MMSI"].map(labels)
    return df


def resample_minutes(df):
    """Keep the last report in every minute for each boat (no invented points)."""
    parts = []
    for mmsi, group in df.sort_values("time").groupby("MMSI"):
        minute = (group.set_index("time")[["Latitude", "Longitude", "SOG", "label"]]
                  .resample("1min").last().dropna(subset=["Latitude"]))
        minute["vessel_id"] = mmsi
        parts.append(minute.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={"Latitude": "lat", "Longitude": "lon", "SOG": "sog"})


def add_motion(df):
    """Work out speed (knots) and turning (degrees) from the positions."""
    df = df.sort_values(["vessel_id", "time"]).reset_index(drop=True)
    g = df.groupby("vessel_id")
    prev_lat, prev_lon = g["lat"].shift(), g["lon"].shift()
    minutes = g["time"].diff().dt.total_seconds() / 60
    speed = haversine_km(prev_lat, prev_lon, df["lat"], df["lon"]) / (minutes / 60) / 1.852
    speed[minutes > MAX_GAP_MIN] = np.nan
    df["speed_kn"] = speed
    bearing = bearing_deg(prev_lat, prev_lon, df["lat"], df["lon"])
    bearing[(speed < STOPPED_KN) | speed.isna()] = np.nan   # direction is noise when stopped
    prev_bearing = bearing.groupby(df["vessel_id"]).shift()
    df["turn_deg"] = (bearing - prev_bearing + 540) % 360 - 180
    return df


def make_charts(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    moving = df[df["speed_kn"] >= STOPPED_KN]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, colour in (("fishing", "#2e8b57"), ("other", "#111111")):
        ax.hist(moving.loc[moving["label"] == label, "speed_kn"].clip(upper=30), bins=30,
                alpha=0.6, color=colour, label=label, density=True)
    ax.set_xlabel("speed in knots (moving points only, capped at 30)")
    ax.set_ylabel("share of points")
    ax.set_title("Fishing boats and other boats move at different speeds")
    ax.legend()
    fig.savefig(FIG_DIR / "dma_speed_by_label.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    turns = df.dropna(subset=["turn_deg"])
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [turns.loc[turns["label"] == lab, "turn_deg"].abs() for lab in ("fishing", "other")]
    ax.boxplot(data, patch_artist=True, boxprops={"facecolor": "#2e8b57"}, medianprops={"color": "black"})
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["fishing", "other"])
    ax.set_ylabel("size of turn between minutes (degrees)")
    ax.set_title("Turning by boat type")
    fig.savefig(FIG_DIR / "dma_turning_by_label.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        df = read_sample(IN_FILE)
    except FileNotFoundError:
        sys.exit(f"{IN_FILE} not found. Run python src/explore_dma.py first.")
    except (KeyError, ValueError) as err:
        sys.exit(f"Could not read the sample: {err}")

    steps = [("start", len(df), df["MMSI"].nunique())]

    def record(name, frame, id_col="MMSI"):
        steps.append((name, len(frame), frame[id_col].nunique()))

    df = df[(df["Latitude"].abs() <= 90) & (df["Longitude"].abs() <= 180)]
    record("remove invalid positions", df)
    df = df.drop_duplicates(["MMSI", "time", "Latitude", "Longitude", "SOG"])
    record("remove exact duplicates", df)
    df = df[~df["MMSI"].astype(str).str.startswith("97")]
    record("remove 97-prefix IDs", df)
    df = label_per_vessel(df)
    record("keep boats with one clear label", df)
    df = add_motion(resample_minutes(df))
    record("resample to one point per minute", df, "vessel_id")

    per_boat = df.groupby("vessel_id").agg(
        points=("time", "size"),
        moving=("speed_kn", lambda s: (s.dropna() >= STOPPED_KN).mean() if s.notna().any() else 0.0))
    keep = per_boat[(per_boat["points"] >= MIN_POINTS) & (per_boat["moving"] >= MIN_MOVING_SHARE)].index
    short = int((per_boat["points"] < MIN_POINTS).sum())
    still = int(((per_boat["points"] >= MIN_POINTS) & (per_boat["moving"] < MIN_MOVING_SHARE)).sum())
    df = df[df["vessel_id"].isin(keep)]
    record(f"drop short tracks ({short}) and mostly-stopped boats ({still})", df, "vessel_id")

    print(f"{'step':<62}{'rows':>9}{'boats':>7}")
    for name, rows, boats in steps:
        print(f"{name:<62}{rows:>9,}{boats:>7}")
    if df.empty:
        sys.exit("No tracks left. Lower MIN_POINTS or MIN_MOVING_SHARE, or take a bigger sample.")

    print("\nBoats per label:\n" + df.groupby("label")["vessel_id"].nunique().to_string())
    print("\nOne-minute points per boat:\n" + df.groupby("vessel_id").size().describe().round(1).to_string())
    print("\nMedian speed (knots) and median size of turn (degrees) by label:")
    print(df.assign(abs_turn=df["turn_deg"].abs()).groupby("label")[["speed_kn", "abs_turn"]].median().round(2).to_string())

    out = df[["vessel_id", "time", "lat", "lon", "sog", "speed_kn", "turn_deg", "label"]]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    make_charts(out)
    print(f"\nSaved {OUT_FILE} ({OUT_FILE.stat().st_size / 1024:.0f} KB) and 2 charts in {FIG_DIR}/")


if __name__ == "__main__":
    main()