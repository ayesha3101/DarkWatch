#k check kerha k jo humari patrol starategies wou tab bhi reliable hain if ( k humari boat ki speeds change hojaaajeyin, tou accurate result degi ya)
# and check on 2 diff ais switch probs 0.08, 0.25
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import BeliefAStarPatrol, GreedyPatrol, LastKnownPatrol, MCTSPatrol, RandomPatrol
from data import load_data
from movement import load_demo_kernel
from run_patrols import BASELINE, run_episode
from world import WorldConfig

FIG_DIR, RESULT_DIR = Path("outputs/figures"), Path("outputs/results")
SPEEDS = [("slow boats (1.5-5 kn)", (1.5, 5.0)), ("medium boats (4-10 kn)", (4.0, 10.0)), ("fast boats (8-16 kn)", (8.0, 16.0))]
DARK_CHANCES = (0.08, 0.25)
GREEN, BLACK, GREY = "#2e8b57", "#111111", "#999999"


def settings():
    return [(f"{label}, AIS off {p:.2f}", WorldConfig(boat_knots=knots, p_dark=p)) for label, knots in SPEEDS for p in DARK_CHANCES]

def run_setting(name, config, agents, data, kernel, episodes, seed):
    rows = []
    for agent_class in agents:
        for i in range(episodes):
            rows.append({"setting": name, "agent": agent_class.name, "episode": i,
                         **run_episode(agent_class, data, kernel, seed + i, config)})
    return pd.DataFrame(rows)


def paired(df, strategy="Belief + A*", baseline=BASELINE):
    """Strategy minus baseline on the same episodes, per setting, for two measures."""
    rows = []
    for setting, part in df.groupby("setting", sort=False):
        for column, label, scale in (("found_at_50", "boats found by fuel 50", 1), ("found_per_fuel", "boats found per 100 fuel", 100)):
            wide = part.pivot(index="episode", columns="agent", values=column) * scale
            if strategy not in wide or baseline not in wide:
                continue
            diff = wide[strategy] - wide[baseline]
            se = diff.std() / np.sqrt(len(diff))
            rows.append({"setting": setting, "measure": label, "mean_difference": diff.mean(),
                         "ci95_low": diff.mean() - 1.96 * se, "ci95_high": diff.mean() + 1.96 * se})
    return pd.DataFrame(rows).round(3)


def make_chart(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = df.groupby(["setting", "agent"], sort=False)["found_at_50"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / len(table.columns)
    for k, agent in enumerate(table.columns):
        colour = GREEN if agent == "Belief + A*" else BLACK if agent == BASELINE else GREY
        ax.bar(np.arange(len(table)) + k * width, table[agent], width, label=agent, color=colour, edgecolor="white")
    ax.set_xticks(np.arange(len(table)) + width * (len(table.columns) - 1) / 2, table.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("boats found by fuel 50 (mean)")
    ax.set_title("Strategies as boats get faster and AIS goes off sooner")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "sensitivity_found_by_50.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of the strategies to boat speed and AIS-off chance.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-mcts", action="store_true")
    args = parser.parse_args()

    data, (kernel, kernel_source) = load_data(args.seed), load_demo_kernel()
    agents = [RandomPatrol, LastKnownPatrol, GreedyPatrol, BeliefAStarPatrol] + ([MCTSPatrol] if args.with_mcts else [])
    df = pd.concat([run_setting(n, c, agents, data, kernel, args.episodes, args.seed) for n, c in settings()], ignore_index=True)

    summary = df.groupby(["setting", "agent"], sort=False).agg(found=("found", "mean"), found_by_50=("found_at_50", "mean"),
                                                                found_per_fuel=("found_per_fuel", "mean")).round(3)
    summary["found_per_fuel"] = (summary["found_per_fuel"] * 100).round(3)
    summary = summary.rename(columns={"found_per_fuel": "found_per_100_fuel"})
    diff = paired(df)
    print(f"\nMap: {data['source']}. Movement table: {kernel_source}. {args.episodes} episodes per cell, seed {args.seed}.\n")
    print(summary.to_string())
    print(f"\nBelief + A* minus '{BASELINE}' on the same episodes (95% interval; if it contains 0 the gap may be luck):\n")
    print(diff.to_string(index=False))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / "sensitivity.csv")
    diff.to_csv(RESULT_DIR / "sensitivity_paired.csv", index=False)
    make_chart(df)
    print(f"\nSaved tables in {RESULT_DIR}/ and {FIG_DIR / 'sensitivity_found_by_50.png'}")


if __name__ == "__main__":
    main()