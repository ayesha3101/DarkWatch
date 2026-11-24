"""Record an episode as a JSON file that the web page plays back (format: docs/replay-format.md).

Run:  python src/replay.py --agent "Belief + A*" --seed 42
      python src/replay.py --all --seed 42          (one file for every strategy, same boats)

The page only DRAWS this file. All the thinking (belief, routes, choices) was done here in Python.
Frame t shows the world at time t, the belief after looking at what the patrol saw at t,
and the target the agent chose for the next step. The true positions of dark boats are in
the file so the page can offer a "show hidden boats" switch; it should not show them by default.
"""
import argparse
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agents import ALL_AGENTS
from data import load_data
from movement import CELL_KM_DEMO, STEP_MINUTES_DEMO, load_demo_kernel
from world import World, WorldConfig

OUT_DIR = Path("outputs/replay")
GRID_INFO = Path("data/processed/grid_info.json")
FORMAT_VERSION = 1


def find_agent(name):
    """The agent class with this name (ignoring case). Raises ValueError with the valid names."""
    for agent_class in ALL_AGENTS:
        if agent_class.name.lower() == name.lower():
            return agent_class
    raise ValueError(f"unknown agent '{name}'. Choose one of: {', '.join(a.name for a in ALL_AGENTS)}")


def belief_map(agent):
    """All tracked dark boats' beliefs added up, rounded (or None for agents without a belief)."""
    beliefs = getattr(agent, "beliefs", None)
    if beliefs is None:
        return None
    total = sum((b.p for b in beliefs.values()), np.zeros_like(next(iter(beliefs.values())).p)) if beliefs else None
    return None if total is None else np.round(total, 4).tolist()


def make_frame(world, agent, target):
    """Everything the page needs to draw one moment."""
    return {
        "step": world.step_count,
        "patrol": {"x": world.patrol[0], "y": world.patrol[1]},
        "fuel": round(world.fuel, 3),
        "fuel_used": round(world.fuel_used, 3),
        "found_total": sum(b.found for b in world.boats),
        "target": list(target) if target is not None else None,
        "boats": [{"id": b.id, "x": round(b.x, 3), "y": round(b.y, 3),
                   "state": "found" if b.found else "dark" if b.dark else "visible"} for b in world.boats],
        "belief": belief_map(agent),
        "decision": agent.last_decision,
    }


def record_episode(agent_class, data, kernel, seed, config=None):
    """Play one episode and return the replay as a dictionary."""
    config = config or WorldConfig()
    world = World(data["cost"], data["density"], kernel, seed=seed, config=config)
    agent = agent_class(world, rng=random.Random(seed * 7919 + 17))
    frames, obs = [], world.observation()
    while True:
        agent.observe(obs)
        target = None if world.done() else agent.choose_target(obs)
        frames.append(make_frame(world, agent, target))
        if world.done():
            break
        obs = world.step(target)
    return {
        "format": FORMAT_VERSION,
        "agent": agent_class.name,
        "seed": seed,
        "map": {
            "source": data["source"],
            "width": int(world.width), "height": int(world.height),
            "cost": [[None if math.isinf(v) else float(v) for v in row] for row in world.cost],
            "density": np.round(world.density, 3).tolist(),
            "port": list(world.port),
            "grid_info": json.loads(GRID_INFO.read_text()) if GRID_INFO.exists() else None,
        },
        "config": asdict(config) | {"step_minutes": STEP_MINUTES_DEMO, "cell_km": CELL_KM_DEMO},
        "summary": world.summary(),
        "frames": frames,
    }


def default_path(agent_name, seed):
    """Where a replay of this agent and seed is saved by default."""
    slug = re.sub(r"[^a-z0-9]+", "_", agent_name.lower()).strip("_")
    return OUT_DIR / f"{slug}_seed{seed}.json"


def save(replay, path=None):
    """Write the replay as JSON and return the path."""
    path = Path(path) if path else default_path(replay["agent"], replay["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replay, allow_nan=False, separators=(",", ":")))
    return path


def main():
    parser = argparse.ArgumentParser(description="Record an episode for the web page.")
    parser.add_argument("--agent", default="Belief + A*")
    parser.add_argument("--all", action="store_true", help="record every strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="file name (one agent only)")
    args = parser.parse_args()

    try:
        classes = ALL_AGENTS if args.all else [find_agent(args.agent)]
    except ValueError as err:
        raise SystemExit(str(err))
    data, (kernel, _) = load_data(args.seed), load_demo_kernel()
    for agent_class in classes:
        replay = record_episode(agent_class, data, kernel, args.seed)
        path = save(replay, None if args.all else args.out)
        s = replay["summary"]
        print(f"{agent_class.name}: {len(replay['frames'])} frames, found {s['found']} of {s['went_dark']} dark boats, "
              f"fuel used {s['fuel_used']} -> {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()