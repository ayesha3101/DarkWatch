"""Tests for BFS, UCS and A*. Run with:  python -m pytest tests -q"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search import AStar, BFS, UCS            # noqa: E402
from search_experiment import make_map, pick_endpoints   # noqa: E402

INF = np.inf
# The 5 x 5 example used for the hand trace (see src/search_demo.py). Start (0, 0), goal (4, 4).
SMALL = np.array([[1, INF, 1, 1, 1],
                  [1, 1, 1, 2, 1],
                  [1, 2, 2, 1, 1],
                  [1, INF, 1, INF, 1],
                  [INF, 1, 1, 1, 1]], dtype=float)
BEST_PATH = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 2), (4, 2), (4, 3), (4, 4)]


def reference_cost(planner, start, goal):
    """Slow, simple check: relax every move over and over until nothing improves."""
    best = {start: 0.0}
    changed = True
    while changed:
        changed = False
        for state, g in list(best.items()):
            for nxt, step in planner.neighbors(state):
                if g + step < best.get(nxt, math.inf) - 1e-12:
                    best[nxt] = g + step
                    changed = True
    return best.get(goal, math.inf)


def path_cost(planner, path):
    """Add up the real cost of walking a path, move by move."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        steps = dict(planner.neighbors(a))
        assert b in steps, f"illegal move {a} -> {b}"
        total += steps[b]
    return total


def test_ucs_and_astar_match_reference_on_random_maps():
    for seed in range(15):
        cost = make_map(30, 20, seed)
        ends = pick_endpoints(cost, np.random.default_rng(seed), 10)
        if ends is None:
            continue
        ref = reference_cost(UCS(cost), *ends)
        for cls in (UCS, AStar):
            result = cls(cost).search(*ends)
            assert result.found == math.isfinite(ref)
            if result.found:
                assert abs(result.cost - ref) < 1e-9
                assert abs(path_cost(cls(cost), result.path) - result.cost) < 1e-9


def test_astar_expands_fewer_nodes_than_ucs_overall():
    ucs_total = astar_total = 0
    for seed in range(15):
        cost = make_map(30, 20, seed, obstacle_share=0.15)
        ends = pick_endpoints(cost, np.random.default_rng(seed), 10)
        if ends is None:
            continue
        ucs_total += UCS(cost).search(*ends).nodes_expanded
        astar_total += AStar(cost).search(*ends).nodes_expanded
    assert astar_total < ucs_total


def test_bfs_has_fewest_moves_but_not_always_lowest_cost():
    for seed in range(15):
        cost = make_map(30, 20, seed)
        ends = pick_endpoints(cost, np.random.default_rng(seed), 10)
        if ends is None:
            continue
        bfs, ucs = BFS(cost).search(*ends), UCS(cost).search(*ends)
        assert bfs.found == ucs.found
        if bfs.found:
            assert len(bfs.path) <= len(ucs.path)
            assert bfs.cost >= ucs.cost - 1e-9


def test_small_map_known_answer():
    """Numbers worked out by hand and checked against the reference."""
    ucs, astar, bfs = (cls(SMALL).search((0, 0), (4, 4)) for cls in (UCS, AStar, BFS))
    assert ucs.path == astar.path == BEST_PATH
    assert abs(ucs.cost - reference_cost(UCS(SMALL), (0, 0), (4, 4))) < 1e-9
    assert abs(ucs.cost - astar.cost) < 1e-9
    assert (bfs.nodes_expanded, ucs.nodes_expanded, astar.nodes_expanded) == (18, 18, 9)
    assert bfs.cost > ucs.cost + 0.9          # BFS has the same number of moves but costs more fuel


def test_no_path_does_not_crash():
    wall = np.ones((5, 5))
    wall[:, 2] = INF
    for cls in (BFS, UCS, AStar):
        result = cls(wall).search((0, 0), (4, 4))
        assert not result.found and result.reason == "no path"


def test_start_equals_goal():
    result = AStar(SMALL).search((0, 0), (0, 0))
    assert result.found and result.path == [(0, 0)] and result.cost == 0


def test_bad_endpoints_are_reported_not_raised():
    planner = AStar(SMALL)
    assert planner.search((1, 0), (4, 4)).reason == "start is on land"
    assert planner.search((0, 0), (1, 3)).reason == "goal is on land"
    assert planner.search((0, 0), (9, 9)).reason == "goal is outside the grid"


def test_fuel_limit():
    result = AStar(SMALL).search((0, 0), (4, 4), max_cost=3.0)
    assert not result.found and "fuel" in result.reason


def test_no_corner_cutting():
    grid = np.ones((2, 2))
    grid[0, 1] = INF   # land at x=1, y=0: moving (0,0) -> (1,1) diagonally would clip its corner
    moves = [state for state, _ in AStar(grid).neighbors((0, 0))]
    assert (1, 1) not in moves


def test_distances_from_matches_ucs_on_random_maps():
    for seed in range(10):
        cost = make_map(30, 20, seed)
        ends = pick_endpoints(cost, np.random.default_rng(seed), 10)
        if ends is None:
            continue
        start, goal = ends
        dist, parent = UCS(cost).distances_from(start)
        ucs = UCS(cost).search(start, goal)
        assert (goal in dist) == ucs.found
        if ucs.found:
            assert abs(dist[goal] - ucs.cost) < 1e-9
            path = UCS.rebuild(parent, goal)
            assert path[0] == start and abs(path_cost(UCS(cost), path) - dist[goal]) < 1e-9