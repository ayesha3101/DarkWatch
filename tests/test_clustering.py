"""Tests for clustering and the MCTS sweep. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classifier import FEATURES                          # noqa: E402
from clustering import choose_k, profile, scale          # noqa: E402
from mcts_sweep import summarise, sweep                  # noqa: E402


def two_mode_windows(n=60, seed=0):
    """Slow wiggly windows and fast straight windows, clearly separated."""
    rng = np.random.default_rng(seed)
    rows = []
    for mode, (speed, turn, stopped) in enumerate(((3.0, 8.0, 0.3), (11.0, 1.0, 0.0))):
        for i in range(n):
            row = {f: rng.normal(0, 0.1) + 1.0 for f in FEATURES}
            row.update({"mean_speed": rng.normal(speed, 0.5), "median_speed": rng.normal(speed, 0.5),
                        "mean_abs_turn": rng.normal(turn, 0.5), "share_stopped": stopped + abs(rng.normal(0, 0.02)),
                        "vessel_id": mode * 100 + i % 10, "label": "fishing" if mode == 0 else "other"})
            rows.append(row)
    return pd.DataFrame(rows)


def test_two_clear_modes_are_found_with_k_equal_two():
    windows = two_mode_windows()
    table, best_k = choose_k(scale(windows))
    assert best_k == 2
    scores = table.set_index("k")["silhouette"]
    assert scores[2] > scores[3] + 0.05          # two groups fit clearly better than three


def test_clustering_is_repeatable():
    x = scale(two_mode_windows())
    assert choose_k(x)[0].equals(choose_k(x)[0])


def test_the_scaled_features_have_no_label_information():
    windows = two_mode_windows()
    x = scale(windows)
    assert x.shape == (len(windows), len(FEATURES))
    assert np.allclose(x.mean(axis=0), 0, atol=1e-9)


def test_profile_describes_each_cluster():
    windows = two_mode_windows()
    clusters = (windows["mean_speed"] > 7).astype(int).to_numpy()
    prof = profile(windows, clusters)
    assert list(prof.index) == [0, 1]
    assert prof.loc[0, "mean_speed"] < prof.loc[1, "mean_speed"]
    assert prof.loc[0, "share_fishing"] == 1.0 and prof.loc[1, "share_fishing"] == 0.0


def test_mcts_sweep_runs_and_reports_every_row():
    df = sweep(episodes=1, seed=3, sims=(5, 10))
    table = summarise(df)
    assert {"Last known position", "Belief + A*", "MCTS-5", "MCTS-10"} <= set(table.index)
    assert (table["seconds_per_episode"] > 0).all()