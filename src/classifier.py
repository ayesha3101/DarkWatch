"""Tell fishing boats from other boats by HOW they move (Lab 11).

Rows are 15-minute windows of a track. For every window we work out a few numbers about
speed and turning (the features). The label (fishing or other) comes from the boat's real
ship type, but the ship type itself is NEVER a feature.

The split is by BOAT: all windows of a boat are either in train or in test (the same split
as the movement model, data/processed/boat_split.csv). Splitting by row would leak the answer,
because windows from one boat look alike.

Models compared on the same split:
  Naive Bayes    GaussianNB(): assumes each feature is bell-shaped for each class and the
                 features are independent. var_smoothing=1e-9 (default) only avoids dividing by zero.
  Decision tree  DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42): max_depth and
                 min_samples_leaf stop it from memorising the training boats.
  Majority       DummyClassifier("most_frequent"): always answers the most common class. The baseline.

Run:  python src/classifier.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from movement import TRACKS_FILE, load_or_make_split

FIG_DIR, RESULT_DIR = Path("outputs/figures"), Path("outputs/results")
WINDOW_MIN = 15
MIN_POINTS = 10          # a window needs at least this many points with a speed
STOPPED_KN = 0.5
SEED = 42
LABELS = ["fishing", "other"]
FEATURES = ["mean_speed", "std_speed", "median_speed", "max_speed",
            "share_stopped", "mean_abs_turn", "std_turn", "mean_speed_change"]


def make_windows(tracks):
    """One row per 15-minute window: vessel_id, label and the feature numbers."""
    rows = []
    for vessel_id, track in tracks.groupby("vessel_id"):
        track = track.sort_values("time")
        window = ((track["time"] - track["time"].iloc[0]).dt.total_seconds() // (60 * WINDOW_MIN)).astype(int)
        for _, part in track.groupby(window):
            speeds = part["speed_kn"].dropna()
            if len(speeds) < MIN_POINTS:
                continue
            turns = part["turn_deg"].dropna()
            rows.append({"vessel_id": vessel_id, "label": part["label"].iloc[0],
                         "mean_speed": speeds.mean(), "std_speed": speeds.std(), "median_speed": speeds.median(),
                         "max_speed": speeds.max(), "share_stopped": float((speeds < STOPPED_KN).mean()),
                         "mean_abs_turn": turns.abs().mean() if len(turns) else 0.0,
                         "std_turn": turns.std() if len(turns) > 1 else 0.0,
                         "mean_speed_change": speeds.diff().abs().mean()})
    return pd.DataFrame(rows).fillna(0.0)


def split_windows(windows, split):
    """Train and test windows, taken from the by-boat split."""
    group = windows["vessel_id"].map(split.set_index("vessel_id")["group"])
    return windows[group == "train"], windows[group == "test"]


def models():
    return {"Naive Bayes": GaussianNB(),
            "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED),
            "Always the majority class": DummyClassifier(strategy="most_frequent")}


def evaluate(train, test):
    """Fit every model on train and score it on test. Returns (table, confusion matrices, fitted models)."""
    rows, matrices, fitted = [], {}, {}
    for name, model in models().items():
        model.fit(train[FEATURES], train["label"])
        predicted = model.predict(test[FEATURES])
        by_boat = pd.Series(predicted == "fishing", index=test.index).groupby(test["vessel_id"]).mean() > 0.5
        truth = test.groupby("vessel_id")["label"].first() == "fishing"
        rows.append({"model": name,
                     "window_accuracy": accuracy_score(test["label"], predicted),
                     "window_f1_fishing": f1_score(test["label"], predicted, pos_label="fishing", zero_division=0),
                     "boat_accuracy": float((by_boat == truth.reindex(by_boat.index)).mean())})
        matrices[name] = confusion_matrix(test["label"], predicted, labels=LABELS)
        fitted[name] = model
    return pd.DataFrame(rows).round(3), matrices, fitted


def make_chart(matrices):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 4))
    for ax, (name, m) in zip(np.atleast_1d(axes), matrices.items()):
        ax.imshow(m, cmap="Greens")
        for (i, j), value in np.ndenumerate(m):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
        ax.set_xticks([0, 1], LABELS)
        ax.set_yticks([0, 1], LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("real")
        ax.set_title(name, fontsize=10)
    fig.savefig(FIG_DIR / "classifier_confusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        tracks = pd.read_csv(TRACKS_FILE, parse_dates=["time"])
    except FileNotFoundError:
        sys.exit(f"{TRACKS_FILE} not found. Run python src/clean_tracks.py first.")
    windows = make_windows(tracks)
    train, test = split_windows(windows, load_or_make_split(tracks))
    if train.empty or test.empty or train["label"].nunique() < 2:
        sys.exit("Not enough windows in train or test to compare the models.")
    print(f"Windows: {len(train)} train from {train['vessel_id'].nunique()} boats, "
          f"{len(test)} test from {test['vessel_id'].nunique()} boats. Features: {', '.join(FEATURES)}")
    print(f"Fishing share of test windows: {(test['label'] == 'fishing').mean():.2f}\n")

    table, matrices, fitted = evaluate(train, test)
    print(table.to_string(index=False))
    for name, m in matrices.items():
        print(f"\n{name} (rows = real, columns = predicted; order {LABELS}):\n{m}")
    tree = fitted["Decision tree"]
    print("\nWhat the decision tree used most:")
    print(pd.Series(tree.feature_importances_, index=FEATURES).sort_values(ascending=False).head(4).round(3).to_string())

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULT_DIR / "classifier_results.csv", index=False)
    make_chart(matrices)
    print(f"\nSaved {RESULT_DIR / 'classifier_results.csv'} and {FIG_DIR / 'classifier_confusion.png'}")


if __name__ == "__main__":
    main()