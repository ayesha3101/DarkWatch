"""Compare BFS, UCS and A* on many seeded maps (Labs 04-05).

Run:  python src/search_experiment.py
      python src/search_experiment.py --seed 42 --maps 30

Prints a results table and saves two charts and a per-instance CSV in outputs/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from search import AStar, BFS, UCS

PLANNERS = [BFS, UCS, AStar]
FIG_DIR, RESULT_DIR = Path("outputs/figures"), Path("outputs/results")
GREEN, BLACK = "#2e8b57", "#111111"


def make_map(width, height, seed, obstacle_share=0.25, slow_share=0.15):
    """Seeded map: land cells (inf), slow cells (cost 3) and open water (cost 1)."""
    rng = np.random.default_rng(seed)
    cost = np.ones((height, width))
    cost[rng.random((height, width)) < slow_share] = 3.0
    cost[rng.random((height, width)) < obstacle_share] = np.inf
    return cost


def pick_endpoints(cost, rng, min_distance):
    """Pick a start and goal in water that are at least min_distance apart."""
    ys, xs = np.where(np.isfinite(cost))
    cells = list(zip(xs.tolist(), ys.tolist()))
    for _ in range(500):
        a, b = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= min_distance:
            return a, b
    return None


def run_instance(group, index, cost, start, goal):
    """Run all three planners on one map. Returns one row per planner."""
    rows = []
    for planner_class in PLANNERS:
        result = planner_class(cost).search(start, goal)
        rows.append({"group": group, "instance": index, "planner": planner_class.name,
                     "found": result.found, "nodes_expanded": result.nodes_expanded,
                     "cost": result.cost if result.found else np.nan,
                     "moves": len(result.path) - 1 if result.found else np.nan,
                     "frontier_peak": result.frontier_peak})
    return rows


def real_map_cost():
    """The real cost grid, or None if the real data is not available."""
    from data import load_data
    data = load_data()
    return data["cost"] if data["source"] == "real" else None


def summarise(df):
    """Average over instances that all three planners solved."""
    solved = df[df["found"]]
    solved = solved[solved.groupby(["group", "instance"])["planner"].transform("count") == len(PLANNERS)]
    table = solved.groupby(["group", "planner"], sort=False)[["nodes_expanded", "cost", "moves", "frontier_peak"]].mean()
    table["maps"] = solved.groupby(["group", "planner"], sort=False).size()
    return table.round(2), solved


def make_charts(solved):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    group = solved[solved["group"] == "random maps"]
    means = group.groupby("planner", sort=False)["nodes_expanded"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(means.index, means.values, color=[BLACK, "#777777", GREEN], edgecolor="black")
    ax.set_ylabel("average nodes expanded")
    ax.set_title("A* looks at far fewer cells than BFS and UCS")
    fig.savefig(FIG_DIR / "search_nodes_expanded.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    wide = group.pivot(index="instance", columns="planner", values="nodes_expanded")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(wide["UCS"], wide["A*"], color=GREEN, edgecolor="black")
    top = float(max(wide["UCS"].max(), wide["A*"].max()))
    ax.plot([0, top], [0, top], color=BLACK, linestyle="--", label="same as UCS")
    ax.set_xlabel("UCS nodes expanded")
    ax.set_ylabel("A* nodes expanded")
    ax.set_title("One dot per map (below the line = A* is better)")
    ax.legend()
    fig.savefig(FIG_DIR / "search_ucs_vs_astar.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare BFS, UCS and A*.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maps", type=int, default=30)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--height", type=int, default=30)
    args = parser.parse_args()

    rows, skipped = [], 0
    for i in range(args.maps):
        cost = make_map(args.width, args.height, args.seed + i)
        ends = pick_endpoints(cost, np.random.default_rng(args.seed + i), min(args.width, args.height) // 2)
        if ends is None:
            skipped += 1
            continue
        rows += run_instance("random maps", i, cost, *ends)

    real = real_map_cost()
    if real is not None:
        rng = np.random.default_rng(args.seed)
        for i in range(10):
            ends = pick_endpoints(real, rng, 10)
            if ends:
                rows += run_instance("real map", i, real, *ends)

    df = pd.DataFrame(rows)
    table, solved = summarise(df)
    unsolved = df.groupby(["group", "instance"])["found"].all().eq(False).sum()
    print(f"Seed {args.seed}. Maps with no path (left out of the averages): {int(unsolved)}. "
          f"Maps where no start/goal pair was found: {skipped}\n")
    print(table.to_string())

    wide = solved.pivot_table(index=["group", "instance"], columns="planner", values="cost")
    print(f"\nA* cost differs from UCS on {int((abs(wide['A*'] - wide['UCS']) > 1e-9).sum())} of {len(wide)} maps (should be 0).")
    print(f"BFS path is more expensive than UCS on {int((wide['BFS'] > wide['UCS'] + 1e-9).sum())} of {len(wide)} maps.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / "search_comparison.csv", index=False)
    make_charts(solved)
    print(f"\nSaved {RESULT_DIR / 'search_comparison.csv'} and 2 charts in {FIG_DIR}/")


if __name__ == "__main__":
    main()