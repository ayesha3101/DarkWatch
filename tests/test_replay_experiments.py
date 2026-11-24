"""Tests for the replay file and the experiment runner. Run: python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import BeliefAStarPatrol, MCTSPatrol, RandomPatrol   # noqa: E402
from data import generate_ocean                                   # noqa: E402
from experiments import build_steps, run_step, run_steps         # noqa: E402
from movement import default_kernel                               # noqa: E402
from replay import find_agent, record_episode, save              # noqa: E402

COST, DENSITY = generate_ocean(7)
DATA = {"cost": COST, "density": DENSITY, "source": "simulated"}
KERNEL = default_kernel()


def test_replay_has_the_agreed_structure():
    replay = record_episode(BeliefAStarPatrol, DATA, KERNEL, seed=2)
    assert replay["format"] == 1 and replay["agent"] == "Belief + A*"
    m = replay["map"]
    assert (len(m["cost"]), len(m["cost"][0])) == (m["height"], m["width"]) == COST.shape
    assert any(v is None for row in m["cost"] for v in row)             # land is null in the file
    assert len(replay["frames"]) == replay["summary"]["steps"] + 1
    frame = replay["frames"][0]
    assert {"step", "patrol", "fuel", "boats", "belief", "target", "decision", "found_total"} <= set(frame)


def test_the_belief_in_every_frame_adds_up_to_the_number_of_tracked_dark_boats():
    replay = record_episode(BeliefAStarPatrol, DATA, KERNEL, seed=5)
    for frame in replay["frames"]:
        dark = sum(b["state"] == "dark" for b in frame["boats"])
        total = 0.0 if frame["belief"] is None else float(np.sum(frame["belief"]))
        assert abs(total - dark) < 0.1                                   # rounding to 4 decimals
    assert replay["frames"][-1]["fuel"] >= 0


def test_replay_is_json_and_repeatable(tmp_path):
    a = record_episode(RandomPatrol, DATA, KERNEL, seed=3)
    b = record_episode(RandomPatrol, DATA, KERNEL, seed=3)
    assert a == b
    path = save(a, tmp_path / "x.json")
    assert json.loads(path.read_text())["seed"] == 3


def test_agents_without_a_belief_have_no_belief_map():
    assert record_episode(RandomPatrol, DATA, KERNEL, seed=1)["frames"][1]["belief"] is None


def test_decisions_are_recorded_for_the_planning_agents():
    replay = record_episode(BeliefAStarPatrol, DATA, KERNEL, seed=5)
    decisions = [f["decision"] for f in replay["frames"] if f["decision"]]
    assert decisions and all({"x", "y", "utility"} <= set(c) for d in decisions for c in d["candidates"])
    mcts = record_episode(MCTSPatrol, DATA, KERNEL, seed=5)
    assert any(f["decision"] and "visits" in f["decision"]["candidates"][0] for f in mcts["frames"] if f["decision"])


def test_unknown_agent_name_gives_a_helpful_error():
    assert find_agent("belief + a*") is BeliefAStarPatrol
    with pytest.raises(ValueError, match="Choose one of"):
        find_agent("nonsense")


def test_experiment_runner_records_ok_and_failed_steps(tmp_path):
    steps = {"fine": [sys.executable, "-c", "print('hello')"], "broken": [sys.executable, "-c", "raise SystemExit(3)"],
             "missing": ["definitely-not-a-program-xyz"]}
    table = run_steps(steps, log_dir=tmp_path)
    assert dict(zip(table["step"], table["status"])) == {"fine": "ok", "broken": "FAILED", "missing": "FAILED"}
    assert (tmp_path / "fine.txt").read_text().strip() == "hello"


def test_experiment_runner_can_choose_steps_and_the_quick_set_is_smaller():
    steps = build_steps(quick=False, include_slow=True)
    assert {"movement", "search", "belief", "classifier", "clustering", "patrols", "sensitivity", "mcts-sweep"} == set(steps)
    assert "--skip-slow" in build_steps(quick=True)["patrols"]
    assert "100" in build_steps()["patrols"]