"""Tests for the classifier. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classifier import FEATURES, evaluate, make_windows, split_windows   # noqa: E402
from movement import split_boats                                          # noqa: E402


def fake_tracks(n_per_label=20, minutes=60, seed=0):
    """Fishing boats slow with more turning, other boats fast and straight."""
    rng = np.random.default_rng(seed)
    rows, start, vid = [], pd.Timestamp("2026-10-31 00:00:00"), 0
    for label, speed, turn in (("fishing", 3.0, 8.0), ("other", 11.0, 1.0)):
        for _ in range(n_per_label):
            vid += 1
            for m in range(minutes):
                rows.append({"vessel_id": vid, "time": start + pd.Timedelta(minutes=m), "label": label,
                             "speed_kn": max(0.0, rng.normal(speed, 1.5)), "turn_deg": rng.normal(0, turn)})
    return pd.DataFrame(rows)


def test_windows_use_only_movement_features():
    windows = make_windows(fake_tracks())
    assert set(windows.columns) == {"vessel_id", "label", *FEATURES}
    assert not any("type" in c.lower() for c in windows.columns)


def test_every_boat_is_in_only_one_group():
    tracks = fake_tracks()
    train, test = split_windows(make_windows(tracks), split_boats(tracks))
    assert not set(train["vessel_id"]) & set(test["vessel_id"])
    assert len(train) > 0 and len(test) > 0


def test_models_beat_the_majority_baseline_on_separable_data():
    tracks = fake_tracks()
    train, test = split_windows(make_windows(tracks), split_boats(tracks))
    table, matrices, _ = evaluate(train, test)
    scores = table.set_index("model")["window_accuracy"]
    assert scores["Naive Bayes"] > scores["Always the majority class"]
    assert scores["Decision tree"] > scores["Always the majority class"]
    assert all(m.sum() == len(test) for m in matrices.values())