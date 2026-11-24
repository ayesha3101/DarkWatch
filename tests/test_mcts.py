"""Tests for the generic Monte Carlo tree search. Run: python -m pytest tests -q"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcts import MCTS   # noqa: E402


class Bandit:
    """One decision, three actions with noisy rewards (average 0.2, 0.8 and 0.5)."""
    means = {"a": 0.2, "b": 0.8, "c": 0.5}

    def actions(self, state):
        return list(self.means) if state == "start" else []

    def is_terminal(self, state):
        return state != "start"

    def sample_world(self, rng):
        return None

    def step(self, state, action, world, rng):
        return "end", self.means[action] + rng.uniform(-0.1, 0.1)

    def rollout(self, state, world, rng):
        return 0.0


class DelayedReward:
    """Action A pays nothing now but 1.0 next step; action B pays 0.5 now and nothing after."""

    def actions(self, state):
        return ["A", "B"] if state == "s0" else ["go"] if state in ("sA", "sB") else []

    def is_terminal(self, state):
        return state == "end"

    def sample_world(self, rng):
        return None

    def step(self, state, action, world, rng):
        table = {("s0", "A"): ("sA", 0.0), ("s0", "B"): ("sB", 0.5), ("sA", "go"): ("end", 1.0), ("sB", "go"): ("end", 0.0)}
        return table[(state, action)]

    def rollout(self, state, world, rng):
        return 0.0


def test_picks_the_best_arm():
    best, stats = MCTS(Bandit(), simulations=300, rng=random.Random(1)).search("start")
    assert best == "b"
    assert stats["b"][0] > stats["a"][0]


def test_visits_add_up_to_the_number_of_simulations():
    _, stats = MCTS(Bandit(), simulations=120, rng=random.Random(2)).search("start")
    assert sum(v for v, _ in stats.values()) == 120


def test_looks_ahead_past_a_small_immediate_reward():
    best, _ = MCTS(DelayedReward(), simulations=200, max_depth=2, rng=random.Random(3)).search("s0")
    assert best == "A"          # the return from A is 1.0, from B only 0.5


def test_depth_one_only_sees_the_immediate_reward():
    best, _ = MCTS(DelayedReward(), simulations=200, max_depth=1, rng=random.Random(3)).search("s0")
    assert best == "B"


def test_same_seed_gives_the_same_answer():
    a = MCTS(Bandit(), simulations=60, rng=random.Random(5)).search("start")
    b = MCTS(Bandit(), simulations=60, rng=random.Random(5)).search("start")
    assert a == b


def test_no_actions_returns_none():
    assert MCTS(Bandit(), simulations=10).search("end") == (None, {})


def test_a_single_simulation_still_works_and_zero_is_refused():
    best, _ = MCTS(Bandit(), simulations=1, rng=random.Random(0)).search("start")
    assert best in Bandit.means
    with pytest.raises(ValueError):
        MCTS(Bandit(), simulations=0)