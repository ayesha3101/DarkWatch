"""Run every patrol strategy on the same seeded episodes and compare them (no window needed).

Run:  python src/run_patrols.py
      python src/run_patrols.py --episodes 100 --seed 42
      python src/run_patrols.py --skip-slow        (leaves out the slow MCTS agent)

Episode i uses seed + i for every strategy, so they all face the same boats and the same luck.
Besides boats found per fuel used, it reports boats found within the SAME fuel budget
(25, 50, 75 and 100 units) and a paired comparison against "Last known position".
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import ALL_AGENTS
from data import load_data
from movement import load_demo_kernel
from world import FUEL_BUDGETS, World, WorldConfig

FIG_DIR, RESULT_DIR = Path("outputs/figures"), Path("outputs/results")
GREEN, BLACK, GREY = "#2e8b57", "#111111", "#999999"
BASELINE = "Last known position"


def run_episode(agent_class, data, kernel, seed, config):
    """One episode. Returns the summary dict."""
    world = World(data["cost"], data["density"], kernel, seed=seed, config=config)
    agent = agent_class(world, rng=random.Random(seed * 7919 + 17))
    obs = world.observation()
    while not world.done():
        agent.observe(obs)
        obs = world.step(agent.choose_target(obs))
    return world.summary()


def run_all(episodes, seed, config=None, agents=None, skip_slow=False):
    """Run every agent on `episodes` seeds. Returns (rows as a table, map source, movement table source)."""
    config = config or WorldConfig()
    data = load_data(seed)
    kernel, kernel_source = load_demo_kernel()
    classes = [a for a in (agents or ALL_AGENTS) if not (skip_slow and getattr(a, "slow", False))]
    rows = []
    for agent_class in classes:
        for i in range(episodes):
            rows.append({"agent": agent_class.name, "episode": i, **run_episode(agent_class, data, kernel, seed + i, config)})
    return pd.DataFrame(rows), data["source"], kernel_source


def summarise(df):
    """Average per strategy, with the boats found within each fuel budget."""
    df = df.assign(found_per_100_fuel=100 * df["found_per_fuel"], found_share=df["found"] / df["went_dark"].clip(lower=1))
    g = df.groupby("agent", sort=False)
    table = pd.DataFrame({
        "found": g["found"].mean(), "found_share": g["found_share"].mean(), "fuel_used": g["fuel_used"].mean(),
        "found_per_100_fuel": g["found_per_100_fuel"].mean(),
        "std_error": g["found_per_100_fuel"].std() / np.sqrt(g.size())})
    for budget in FUEL_BUDGETS:
        table[f"found_by_{budget}"] = g[f"found_at_{budget}"].mean()
    return table.round(3)


def paired_table(df, baseline=BASELINE):
    """Each strategy minus the baseline on the SAME episodes: mean difference, 95% interval and win counts."""
    metrics = (("found_per_fuel", "boats found per 100 fuel", 100), ("found", "boats found", 1),
               ("found_at_50", "boats found by fuel 50", 1))
    rows = []
    for column, label, scale in metrics:
        wide = df.pivot(index="episode", columns="agent", values=column) * scale
        if baseline not in wide.columns:
            continue
        for agent in wide.columns:
            if agent == baseline:
                continue
            diff = wide[agent] - wide[baseline]
            se = diff.std() / np.sqrt(len(diff))
            rows.append({"metric": label, "strategy": agent, "mean_difference": diff.mean(),
                         "ci95_low": diff.mean() - 1.96 * se, "ci95_high": diff.mean() + 1.96 * se,
                         "episodes_better": int((diff > 0).sum()), "episodes_worse": int((diff < 0).sum()),
                         "episodes_tied": int((diff == 0).sum())})
    return pd.DataFrame(rows).round(3)


def make_charts(table):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colours = [GREY] * len(table)
    if "Belief + A*" in table.index:
        colours[list(table.index).index("Belief + A*")] = GREEN
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(table.index, table["found_per_100_fuel"], yerr=table["std_error"], color=colours, edgecolor=BLACK, capsize=4)
    ax.set_ylabel("boats found per 100 fuel used (error bar = standard error)")
    ax.set_title("Patrol strategies on the same seeded episodes")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.savefig(FIG_DIR / "patrol_found_per_fuel.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(table.index, table["found_share"] * 100, color=colours, edgecolor=BLACK)
    ax.set_ylabel("share of dark boats found (%)")
    ax.set_title("How many of the boats that went dark were found")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.savefig(FIG_DIR / "patrol_found_share.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, row in table.iterrows():
        colour = GREEN if name == "Belief + A*" else BLACK if name == BASELINE else GREY
        ax.plot(FUEL_BUDGETS, [row[f"found_by_{b}"] for b in FUEL_BUDGETS], marker="o", color=colour,
                linewidth=2.5 if colour != GREY else 1.2, label=name)
    ax.set_xlabel("fuel budget (units used)")
    ax.set_ylabel("boats found by then (mean)")
    ax.set_title("Boats found within the same fuel budget")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "patrol_found_by_budget.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def at_least_two(text):
    """argparse type: a whole number of episodes, at least 2 (the paired comparison needs two)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number")
    if value < 2:
        raise argparse.ArgumentTypeError("episodes must be at least 2")
    return value


def main():
    parser = argparse.ArgumentParser(description="Compare patrol strategies.")
    parser.add_argument("--episodes", type=at_least_two, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-slow", action="store_true", help="leave out slow agents such as MCTS")
    args = parser.parse_args()

    df, source, kernel_source = run_all(args.episodes, args.seed, skip_slow=args.skip_slow)
    table, paired = summarise(df), paired_table(df)
    print(f"\nMap: {source}. Movement table: {kernel_source}. {args.episodes} episodes per strategy, seed {args.seed}.\n")
    print(table.to_string())
    print(f"\nEach strategy minus '{BASELINE}' on the same episodes (95% interval; if it contains 0 the gap may be luck):\n")
    print(paired.to_string(index=False))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / "patrol_episodes.csv", index=False)
    table.to_csv(RESULT_DIR / "patrol_comparison.csv")
    paired.to_csv(RESULT_DIR / "patrol_paired.csv", index=False)
    make_charts(table)
    print(f"\nSaved 3 tables in {RESULT_DIR}/ and 3 charts in {FIG_DIR}/")


if __name__ == "__main__":
    main()