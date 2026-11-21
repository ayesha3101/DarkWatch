"""Tests for the belief filter. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from belief import Belief, shift   # noqa: E402

KERNEL = np.array([[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]])
DELTA = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def test_belief_sums_to_one_after_every_step():
    b = Belief((9, 9), (4, 4))
    assert abs(b.p.sum() - 1) < 1e-12
    for _ in range(5):
        b.predict(KERNEL, drift=(0.3, -0.2))
        assert abs(b.p.sum() - 1) < 1e-9
        b.update((2, 2), detected=False)
        assert abs(b.p.sum() - 1) < 1e-9


def test_predict_spreads_the_belief():
    b = Belief((9, 9), (4, 4))
    before = b.entropy()
    b.predict(KERNEL)
    assert b.entropy() > before


def test_searching_an_empty_cell_lowers_it_and_raises_the_others():
    b = Belief((5, 5), (2, 2))
    b.predict(KERNEL)
    before = b.p.copy()
    b.update((1, 2), detected=False, radius=0, p_detect=0.9)   # search the cell left of the middle, find nothing
    assert b.p[2, 1] < before[2, 1]
    assert b.p[2, 2] > before[2, 2]


def test_miss_update_matches_the_hand_calculation():
    b = Belief((3, 3), (1, 1))
    b.predict(KERNEL)
    b.update((0, 1), detected=False, radius=0, p_detect=0.9)
    assert abs(b.p[1, 0] - 0.01 / 0.91) < 1e-12
    assert abs(b.p[1, 1] - 0.6 / 0.91) < 1e-12


def test_detection_collapses_to_the_boat_cell():
    b = Belief((5, 5), (2, 2))
    b.predict(KERNEL)
    b.update((1, 1), detected=True, boat_cell=(2, 1))
    assert b.p[1, 2] == 1.0 and abs(b.p.sum() - 1) < 1e-12


def test_drift_moves_the_average_position():
    b = Belief((21, 21), (5, 10))
    b.predict(DELTA, drift=(3.0, 0.0))
    assert abs(b.mean_position()[0] - 8.0) < 1e-9
    b.predict(DELTA, drift=(0.5, 0.5))
    x, y = b.mean_position()
    assert abs(x - 8.5) < 1e-9 and abs(y - 10.5) < 1e-9


def test_fractional_shift_keeps_the_total():
    p = np.zeros((7, 7))
    p[3, 3] = 1.0
    assert abs(shift(p, 0.37, -1.62).sum() - 1.0) < 1e-12


def test_no_belief_on_land():
    water = np.ones((5, 5), dtype=bool)
    water[:, 3:] = False
    b = Belief((5, 5), (2, 2), water=water)
    for _ in range(3):
        b.predict(KERNEL, drift=(1.0, 0.0))
    assert b.p[~water].sum() == 0 and abs(b.p.sum() - 1) < 1e-9


def test_contradicting_evidence_resets_to_uniform_over_water():
    b = Belief((5, 5), (2, 2))
    b.update((2, 2), detected=False, radius=1, p_detect=1.0)   # certain to see it, saw nothing, but all mass is there
    assert b.lost
    assert np.allclose(b.p, 1 / 25)


def test_top_mask_holds_the_requested_mass():
    b = Belief((9, 9), (4, 4))
    for _ in range(3):
        b.predict(KERNEL)
    mask = b.top_mask(0.9)
    assert b.p[mask].sum() >= 0.9 - 1e-9
    assert mask.sum() < mask.size