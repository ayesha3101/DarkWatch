"""Tests for the MCTS patrol agent and its helpers. Run: python -m pytest tests -q"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import MCTSPatrol, make_mcts                       # noqa: E402
from belief import Belief                                       # noqa: E402
from data import generate_ocean                                 # noqa: E402
from mcts_patrol import PathCache, spaced_candidates            # noqa: E402
from movement import default_kernel                             # noqa: E402
from search import AStar                                        # noqa: E402
from world import Observation, World                            # noqa: E402

COST, DENSITY = generate_ocean(7)
KERNEL = default_kernel()


def water_cells(n=8, seed=0):
    ys, xs = np.where(np.isfinite(COST))
    rng = random.Random(seed)
    picks = rng.sample(range(len(xs)), n)
    return [(int(xs[i]), int(ys[i])) for i in picks]


def test_path_cache_move_stays_within_the_budget_and_follows_the_cheapest_route():
    cache = PathCache(COST)
    for start, goal in zip(water_cells(8, 1), water_cells(8, 2)):
        cell, spent = cache.move(start, goal, 4.0)
        assert spent <= 4.0 + 1e-9
        route = AStar(COST).search(start, cell)
        assert route.found and abs(route.cost - spent) < 1e-9          # what it spent is the cheapest cost to where it got
        full, _ = cache.move(start, goal, 1e9)
        assert full == goal and abs(cache.cost(start, goal) - AStar(COST).search(start, goal).cost) < 1e-9


def test_path_cache_unreachable_goal_stays_put():
    wall = np.ones((5, 5))
    wall[:, 2] = np.inf
    cache = PathCache(wall)
    assert cache.move((0, 0), (4, 4), 10.0) == ((0, 0), 0.0)


def test_spaced_candidates_are_ordered_and_spread_out():
    mass = np.zeros((10, 10))
    mass[2, 2], mass[2, 3], mass[7, 7], mass[5, 1] = 0.5, 0.4, 0.3, 0.1
    cells = spaced_candidates(mass, k=3, spacing=3)
    assert cells[0] == (2, 2) and (3, 2) not in cells                  # the neighbour of the best cell is skipped
    assert cells == [(2, 2), (7, 7), (1, 5)]


def test_mcts_goes_for_the_cell_where_the_boat_almost_certainly_is():
    world = World(COST, DENSITY, KERNEL, seed=1)
    agent = MCTSPatrol(world, rng=random.Random(4), simulations=150)
    start = world.patrol
    target = next(c for c in water_cells(40, 3) if 3 <= max(abs(c[0] - start[0]), abs(c[1] - start[1])) <= 6)
    belief = Belief(COST.shape, target, world.water)
    agent.beliefs[0], agent.velocities[0] = belief, (0.0, 0.0)
    obs = Observation(1, start, 100.0)
    chosen = agent.choose_target(obs)
    assert chosen is not None and max(abs(chosen[0] - target[0]), abs(chosen[1] - target[1])) <= 1


def test_mcts_returns_none_when_there_is_nothing_to_look_for():
    world = World(COST, DENSITY, KERNEL, seed=1)
    agent = MCTSPatrol(world, rng=random.Random(0), simulations=20)
    assert agent.choose_target(Observation(0, world.patrol, 100.0)) is None


def test_make_mcts_sets_name_and_simulations():
    cls = make_mcts(50)
    assert cls.name == "MCTS-50" and cls.SIMULATIONS == 50 and issubclass(cls, MCTSPatrol)