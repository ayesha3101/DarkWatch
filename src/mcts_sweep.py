"""How many simulations does MCTS need? (Lab 8)

Run:  python src/mcts_sweep.py
      python src/mcts_sweep.py --episodes 30 --seed 42 --sims 10 50 200 500

MCTS with each number of simulations is run on the SAME seeded episodes, next to
"Last known position" and "Belief + A*" as reference lines. It reports boats found by
fuel 50, boats found per 100 fuel and the seconds one episode takes, so the cost of more
simulations is visible. More simulations should give a steadier estimate of which
candidate is best, but each decision gets slower.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import BeliefAStarPatrol, LastKnownPatrol, make_mcts
from data import load_data
from movement import load_demo_kernel
from run_patrols import run_episode
from world import WorldConfig

FIG_DIR, RESULT_DIR = Path("outputs/figures"), Path("outputs/results")
GREEN, BLACK, GREY = "#2e8b57", "#111111", "#999999"
DEFAULT_SIMS = (10, 50, 200, 500)


def sweep(episodes, seed, sims=DEFAULT_SIMS, config=None):
    """One row per (strategy, episode), with the seconds the episode took."""
    config, (kernel, _), data = config or WorldConfig(), load_demo_kernel(), load_data(seed)
    classes = [LastKnownPatrol, BeliefAStarPatrol] + [make_mcts(n) for n in sims]
    rows = []
    for agent_class in classes:
        for i in range(episodes):
            start = time.perf_counter()
            result = run_episode(agent_class, data, kernel, seed + i, config)
            rows.append({"agent": agent_class.name, "episode": i, "seconds": time.perf_counter() - start, **result})
    return pd.DataFrame(rows)


def summarise(df):
    g = df.groupby("agent", sort=False)
    table = pd.DataFrame({"found_by_50": g["found_at_50"].mean(),
                          "found_by_50_se": g["found_at_50"].std() / np.sqrt(g.size()),
                          "found_per_100_fuel": 100 * g["found_per_fuel"].mean(),
                          "found_share": (df["found"] / df["went_dark"].clip(lower=1)).groupby(df["agent"], sort=False).mean(),
                          "seconds_per_episode": g["seconds"].mean()})
    return table.round(3)


def make_chart(table, sims):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = table.loc[[f"MCTS-{n}" for n in sims]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(sims, rows["found_by_50"], yerr=rows["found_by_50_se"], marker="o", color=GREEN, capsize=4, label="MCTS")
    for name, colour in (("Belief + A*", GREY), ("Last known position", BLACK)):
        ax.axhline(table.loc[name, "found_by_50"], color=colour, linestyle="--", label=name)
    ax.set_xscale("log")
    ax.set_xticks(sims, [str(n) for n in sims])
    ax.set_xlabel("simulations per decision")
    ax.set_ylabel("boats found by fuel 50 (mean, error bar = standard error)")
    ax.set_title("Effect of the number of MCTS simulations")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "mcts_simulations_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MCTS simulation-count sweep.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sims", type=int, nargs="+", default=list(DEFAULT_SIMS))
    args = parser.parse_args()

    df = sweep(args.episodes, args.seed, tuple(args.sims))
    table = summarise(df)
    print(f"\n{args.episodes} episodes per row, seed {args.seed}, same boats for every row.\n")
    print(table.to_string())
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULT_DIR / "mcts_sweep.csv")
    make_chart(table, tuple(args.sims))
    print(f"\nSaved {RESULT_DIR / 'mcts_sweep.csv'} and {FIG_DIR / 'mcts_simulations_sweep.png'}")


if __name__ == "__main__":
    main()