"""Bad input, missing files and impossible situations must give clear errors, not crashes. Run: python -m pytest tests -q"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data                                                 # noqa: E402
from belief import Belief                                   # noqa: E402
from movement import default_kernel                         # noqa: E402
from run_patrols import at_least_two                        # noqa: E402
from search import AStar                                    # noqa: E402
from world import World, WorldConfig                        # noqa: E402

COST, DENSITY = data.generate_ocean(7)
KERNEL = default_kernel()


@pytest.mark.parametrize("bad", [{"n_boats": 0}, {"fuel": 0}, {"patrol_speed": -1}, {"idle_fuel": -0.1},
                                  {"p_dark": 1.5}, {"p_detect": -0.2}, {"radius": -1}, {"max_steps": 0},
                                  {"boat_knots": (5.0, 1.0)}])
def test_invalid_settings_are_refused_with_a_message(bad):
    with pytest.raises(ValueError):
        WorldConfig(**bad)


def test_a_map_with_no_water_is_refused():
    with pytest.raises(ValueError, match="water"):
        World(np.full((5, 5), np.inf), np.zeros((5, 5)), KERNEL)


def test_mismatched_density_is_refused():
    with pytest.raises(ValueError, match="same shape"):
        World(COST, np.zeros((3, 3)), KERNEL)


@pytest.mark.parametrize("kernel", [np.zeros((3, 3)), np.ones((4, 4)) / 16, -np.eye(3), np.ones((3, 5)) / 15])
def test_a_bad_movement_table_is_refused(kernel):
    with pytest.raises(ValueError, match="kernel"):
        World(COST, DENSITY, kernel)


def test_belief_cell_outside_the_grid_is_refused():
    with pytest.raises(ValueError, match="outside"):
        Belief((5, 5), (7, 2))


def test_targets_on_land_or_outside_the_map_make_the_patrol_wait_not_crash():
    world = World(COST, DENSITY, KERNEL, seed=1)
    land = tuple(int(v) for v in np.argwhere(np.isinf(COST))[0][::-1])
    for target in (land, (-3, 2), (999, 999), None):
        before = world.patrol
        world.step(target)
        assert world.patrol == before
    assert world.fuel_used > 0


def test_a_corrupted_grid_file_falls_back_to_the_simulator(tmp_path, monkeypatch):
    (tmp_path / "cost.npy").write_bytes(b"this is not a numpy file")
    (tmp_path / "density.npy").write_bytes(b"")
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path)
    loaded = data.load_data()
    assert loaded["source"] == "simulated" and np.isfinite(loaded["cost"]).any()


def test_mismatched_real_grids_fall_back_to_the_simulator(tmp_path, monkeypatch):
    np.save(tmp_path / "cost.npy", np.ones((4, 4)))
    np.save(tmp_path / "density.npy", np.ones((5, 5)))
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path)
    assert data.load_data()["source"] == "simulated"


def test_search_on_an_enclosed_goal_reports_no_path():
    grid = np.ones((5, 5))
    grid[1:4, 1:4] = np.inf
    grid[2, 2] = 1.0                       # a water cell completely walled in
    result = AStar(grid).search((0, 0), (2, 2))
    assert not result.found and result.reason == "no path"


def test_episodes_option_must_be_a_number_of_at_least_two():
    assert at_least_two("5") == 5
    for bad in ("0", "1", "-4", "abc", "2.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            at_least_two(bad)