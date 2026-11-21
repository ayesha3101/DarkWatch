"""Tests for the environment and the patrol agents. Run: python -m pytest tests -q"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import ALL_AGENTS, BeliefAStarPatrol, RandomPatrol, SweepPatrol   # noqa: E402
from data import generate_ocean                                              # noqa: E402
from movement import default_kernel                                          # noqa: E402
from run_patrols import paired_table, run_episode, summarise                 # noqa: E402
import pandas as pd                                                         # noqa: E402
from world import World, WorldConfig                                         # noqa: E402

COST, DENSITY = generate_ocean(7)
KERNEL = default_kernel()
DATA = {"cost": COST, "density": DENSITY}


def run(agent_class, seed, config=None):
    return run_episode(agent_class, DATA, KERNEL, seed, config or WorldConfig())


def test_default_kernel_sums_to_one_and_is_symmetric():
    assert abs(KERNEL.sum() - 1.0) < 1e-12
    assert np.allclose(KERNEL, KERNEL[::-1, :]) and np.allclose(KERNEL, KERNEL[:, ::-1])


def test_same_seed_gives_identical_results_for_every_agent():
    for agent_class in ALL_AGENTS:
        assert run(agent_class, 5) == run(agent_class, 5), agent_class.name


def test_different_seeds_give_different_episodes():
    results = {tuple(run(RandomPatrol, seed).values()) for seed in range(6)}
    assert len(results) > 1


def test_boats_move_the_same_whatever_the_patrol_does():
    a, b = World(COST, DENSITY, KERNEL, seed=3), World(COST, DENSITY, KERNEL, seed=3)
    rng = random.Random(0)
    for _ in range(15):
        a.step(None)
        b.step((rng.randrange(COST.shape[1]), rng.randrange(COST.shape[0])))
    assert [(x.x, x.y, x.dark) for x in a.boats] == [(x.x, x.y, x.dark) for x in b.boats]


def test_patrol_stays_in_water_and_fuel_never_goes_negative():
    world = World(COST, DENSITY, KERNEL, seed=4)
    rng = random.Random(1)
    while not world.done():
        world.step((rng.randrange(COST.shape[1]), rng.randrange(COST.shape[0])))
        assert world.passable(world.patrol)
        assert world.fuel >= -1e-9
    assert world.fuel_used <= world.config.fuel + 1e-9


def test_patrol_burns_at_most_patrol_speed_when_moving():
    world = World(COST, DENSITY, KERNEL, seed=4)
    far = (world.patrol[0], min(COST.shape[0] - 1, world.patrol[1] + 15))
    world.step(far)
    assert 0 < world.fuel_used <= world.config.patrol_speed + 1e-9


def test_standing_still_costs_idle_fuel():
    world = World(COST, DENSITY, KERNEL, seed=4)
    world.step(None)
    assert abs(world.fuel_used - world.config.idle_fuel) < 1e-12


def test_a_boat_is_found_only_when_dark_and_close():
    cfg = WorldConfig(p_dark=1.0, p_detect=1.0, radius=1)     # every boat goes dark at once and is always seen if close
    world = World(COST, DENSITY, KERNEL, seed=2, config=cfg)
    obs = world.step(None)
    for boat_id, cell in obs.found:
        boat = world.boats[boat_id]
        assert boat.dark and max(abs(cell[0] - world.patrol[0]), abs(cell[1] - world.patrol[1])) <= 1


def test_observation_never_lists_dark_or_found_boats_as_visible():
    world = World(COST, DENSITY, KERNEL, seed=6, config=WorldConfig(p_dark=0.5))
    for _ in range(10):
        obs = world.step(None)
        hidden = {b.id for b in world.boats if b.dark or b.found}
        assert not hidden & {i for i, _, _ in obs.visible}


def test_beliefs_always_sum_to_one_and_stay_off_land():
    world = World(COST, DENSITY, KERNEL, seed=8, config=WorldConfig(p_dark=0.4))
    agent = BeliefAStarPatrol(world, rng=random.Random(0))
    obs = world.observation()
    for _ in range(12):
        agent.observe(obs)
        for belief in agent.beliefs.values():
            assert abs(belief.p.sum() - 1.0) < 1e-9
            assert belief.p[~world.water].sum() == 0
        obs = world.step(agent.choose_target(obs))


def test_sweep_waypoints_are_all_water():
    agent = SweepPatrol(World(COST, DENSITY, KERNEL, seed=1))
    assert agent.waypoints and all(COST[y, x] == 1.0 or np.isfinite(COST[y, x]) for x, y in agent.waypoints)


def test_summary_numbers_are_consistent():
    s = run(BeliefAStarPatrol, 9)
    assert 0 <= s["found"] <= s["went_dark"] <= WorldConfig().n_boats
    assert s["found_per_fuel"] == s["found"] / s["fuel_used"]


def test_belief_agent_beats_random_on_average():
    seeds = range(100, 116)
    belief = np.mean([run(BeliefAStarPatrol, s)["found_per_fuel"] for s in seeds])
    random_ = np.mean([run(RandomPatrol, s)["found_per_fuel"] for s in seeds])
    assert belief > random_


def test_found_by_budget_never_decreases_and_ends_at_the_total():
    world = World(COST, DENSITY, KERNEL, seed=11, config=WorldConfig(p_dark=0.5, p_detect=1.0))
    rng = random.Random(2)
    while not world.done():
        world.step((rng.randrange(COST.shape[1]), rng.randrange(COST.shape[0])))
    counts = [world.found_by(b) for b in range(0, 140, 5)]
    assert counts == sorted(counts)
    assert world.found_by(10_000) == world.summary()["found"]


def test_summary_has_the_budget_columns():
    s = run(RandomPatrol, 3)
    assert all(f"found_at_{b}" in s for b in (25, 50, 75, 100))
    assert s["found_at_25"] <= s["found_at_50"] <= s["found_at_75"] <= s["found_at_100"] <= s["found"]


def test_paired_table_compares_on_the_same_episodes():
    rows = []
    for episode, (a, b) in enumerate([(1, 2), (2, 2), (0, 3), (1, 1)]):
        for agent, found in (("Last known position", a), ("Other", b)):
            rows.append({"agent": agent, "episode": episode, "found": found, "found_at_50": found,
                         "fuel_used": 100.0, "found_per_fuel": found / 100.0, "went_dark": 4,
                         **{f"found_at_{x}": found for x in (25, 75, 100)}})
    paired = paired_table(pd.DataFrame(rows))
    boats = paired[paired["metric"] == "boats found"].iloc[0]
    assert boats["mean_difference"] == 1.0
    assert (boats["episodes_better"], boats["episodes_worse"], boats["episodes_tied"]) == (2, 0, 2)
    assert "Other" in set(summarise(pd.DataFrame(rows)).index)