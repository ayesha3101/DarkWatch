import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
LOG_DIR, RESULT_DIR, FIG_DIR = Path("outputs/logs"), Path("outputs/results"), Path("outputs/figures")
SEED = 42


def build_steps(seed=SEED, quick=False, include_slow=False):
    """The steps in order, as {name: command}. Movement comes first because it makes the train/test split."""
    py = sys.executable
    steps = {
        "movement": [py, str(SRC / "movement.py")],
        "search": [py, str(SRC / "search_experiment.py"), "--seed", str(seed), "--maps", "5" if quick else "30"],
        "belief": [py, str(SRC / "belief_experiment.py")],
        "classifier": [py, str(SRC / "classifier.py")],
        "clustering": [py, str(SRC / "clustering.py")],
        "patrols": [py, str(SRC / "run_patrols.py"), "--seed", str(seed), "--episodes", "5" if quick else "100"]
                   + (["--skip-slow"] if quick else []),
    }
    if include_slow:
        steps["sensitivity"] = [py, str(SRC / "sensitivity.py"), "--seed", str(seed), "--episodes", "3" if quick else "30"]
        steps["mcts-sweep"] = [py, str(SRC / "mcts_sweep.py"), "--seed", str(seed), "--episodes", "2" if quick else "30"] \
                              + (["--sims", "10"] if quick else [])
    return steps


def run_step(name, command, log_dir=LOG_DIR):
    """Run one step, save its printout, and return a result row. Never raises for a failing step."""
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        done = subprocess.run(command, capture_output=True, text=True)
        status, output = ("ok" if done.returncode == 0 else "FAILED"), done.stdout + done.stderr
    except OSError as err:
        status, output = "FAILED", f"could not start: {err}"
    log = log_dir / f"{name}.txt"
    log.write_text(output)
    return {"step": name, "status": status, "seconds": round(time.perf_counter() - start, 1), "log": str(log)}


def run_steps(steps, log_dir=LOG_DIR, only=None):
    """Run the chosen steps in order. Returns a table of results."""
    rows = []
    for name, command in steps.items():
        if only and name not in only:
            continue
        print(f"running {name} ...", flush=True)
        row = run_step(name, command, log_dir)
        print(f"  {row['status']} in {row['seconds']} s (log: {row['log']})", flush=True)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run all experiments.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    parser.add_argument("--all", action="store_true", help="include the slow sensitivity test and MCTS sweep")
    parser.add_argument("--only", nargs="+", help="run only these steps")
    args = parser.parse_args()

    steps = build_steps(args.seed, args.quick, args.all)
    unknown = [n for n in (args.only or []) if n not in steps]
    if unknown:
        sys.exit(f"unknown step(s): {', '.join(unknown)}. Choose from: {', '.join(steps)}")
    table = run_steps(steps, only=args.only)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULT_DIR / "experiments_summary.csv", index=False)
    print("\n" + table.to_string(index=False))
    charts = sorted(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    print(f"\n{len(charts)} charts in {FIG_DIR}/ and tables in {RESULT_DIR}/")
    failed = table[table["status"] != "ok"]
    if len(failed):
        print(f"{len(failed)} step(s) failed. Read their logs in {LOG_DIR}/.")
        sys.exit(1)


if __name__ == "__main__":
    main()