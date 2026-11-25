"""Check that the repository works from a fresh clone, using only what is committed.

Run:  python src/fresh_clone_check.py --repo https://github.com/<you>/<repo>.git
      python src/fresh_clone_check.py --repo .              (a local folder that is a git repository)
      python src/fresh_clone_check.py --repo . --skip-tests (skips the slow test run)
      python src/fresh_clone_check.py --repo . --skip-docs  (until the proposal and report are in docs/)

It clones into a temporary folder, makes a new virtual environment, installs requirements.txt,
runs the tests, the data check, a quick experiment run and a replay export, and checks the
repository itself: required files, files that must not be committed, secrets, big files and
who made the commits. It prints PASS / WARN / FAIL for each check and exits with an error if
anything FAILs. Do this on a different computer too, following only the README.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REQUIRED_DOCS = ["docs/proposal.pdf", "docs/proposal.md"]
REQUIRED = (["README.md", "requirements.txt", ".gitignore"]
            + [f"src/{n}.py" for n in (
                "agents", "analysis", "belief", "belief_demo", "belief_experiment", "build_grids", "build_web", "classifier",
                "clean_tracks", "clustering", "data", "experiments", "explore_dma", "explore_gfw", "fresh_clone_check",
                "mcts", "mcts_patrol", "mcts_sweep", "movement", "replay", "run_patrols", "search", "search_demo",
                "search_experiment", "sensitivity", "server", "verify_data", "world")]
            + [f"src/web/{n}" for n in ("index.html", "style.css", "render.js", "app.js")]
            + [f"tests/test_{n}.py" for n in (
                "belief", "classifier", "clustering", "fresh_clone_check", "mcts", "mcts_patrol", "movement",
                "replay_experiments", "robustness", "search", "server_web", "world_agents")])
NOTES = [f"docs/{n}.md" for n in ("data-notes", "eda-findings", "search-notes", "belief-notes", "environment-notes",
                                  "mcts-notes", "ml-notes", "replay-format")]
REQUIRED_DATA = ["data/processed/cost.npy", "data/processed/density.npy", "data/processed/tracks_clean.csv",
                 "data/processed/boat_split.csv"]
REPORT_FILES = ["docs/report.pdf", "docs/report.md"]
FORBIDDEN_PARTS = {".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
SECRET_PATTERNS = [r"eyJ[A-Za-z0-9_-]{20,}",
                   r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"][A-Za-z0-9_.\-]{16,}['\"]"]
HISTORY_PATTERNS = [r"eyJ[A-Za-z0-9_-]{20,}",     # the same two ideas in the plain regex style git understands (searched with -i)
                    r"(api[_-]?key|token|secret|password)[[:space:]]*[:=][[:space:]]*['\"][A-Za-z0-9_.-]{16,}['\"]"]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".html", ".css", ".js", ".env"}
BIG_FILE_MB = 10
results = []


def check(status, message):
    """Record and print one result. status is PASS, WARN or FAIL."""
    results.append(status)
    print(f"[{status}] {message}", flush=True)


def resolve_repo(text):
    """A local folder becomes an absolute path (git clone runs somewhere else); a URL is left alone."""
    local = Path(text).expanduser()
    return str(local.resolve()) if local.exists() else text


def run(command, cwd, timeout=3600):
    """Run a command. Returns (ok, seconds, last lines of the output)."""
    start = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        ok, text = done.returncode == 0, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as err:
        ok, text = False, str(err)
    return ok, time.perf_counter() - start, "\n".join("    | " + line for line in text.splitlines()[-6:])


def git(args, cwd):
    """Output of a git command, or None if it failed."""
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else None


def tracked_files(clone):
    return [line for line in (git(["ls-files"], clone) or "").splitlines() if line]


def forbidden_files(files):
    """Committed files that should not be in the repository."""
    bad = []
    for name in files:
        parts = Path(name).parts
        if set(parts) & FORBIDDEN_PARTS or name.endswith(".pyc") or Path(name).name == ".env":
            bad.append(name)
        elif parts[:2] == ("data", "raw") and Path(name).name != ".gitkeep":
            bad.append(name)
    return bad


def big_files(clone, files):
    return [name for name in files if (Path(clone) / name).is_file() and (Path(clone) / name).stat().st_size > BIG_FILE_MB * 1024 * 1024]


def secret_files(clone, files):
    """Names of committed text files that look like they contain a key or token (the match itself is never printed)."""
    found = []
    for name in files:
        path = Path(clone) / name
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file() and path.stat().st_size < 1_000_000:
            text = path.read_text(errors="ignore")
            if any(re.search(p, text) for p in SECRET_PATTERNS):
                found.append(name)
    return found


def secret_commits(clone):
    """Commits (anywhere in the history) whose changes look like they add a key or token."""
    hits = set()
    for pattern in HISTORY_PATTERNS:
        out = git(["log", "--all", "--oneline", "-i", "-G", pattern], clone)
        if out is None:
            raise RuntimeError("git could not search the history")      # never pass silently
        hits.update(line.split()[0] for line in out.splitlines() if line)
    return sorted(hits)


def commit_authors(clone):
    """Counter of commits per author name."""
    out = git(["log", "--all", "--format=%an"], clone) or ""
    return Counter(line for line in out.splitlines() if line)


def venv_python(folder):
    return folder / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")


def check_repository(clone, allow_simulated=False, skip_docs=False):
    """Checks on the committed files, which need no installation."""
    files = tracked_files(clone)
    missing = [f for f in REQUIRED if f not in files]
    check("PASS" if not missing else "FAIL", f"all {len(REQUIRED)} code, test and document files are committed"
          + (f" ({len(missing)} missing, never committed or pushed: {', '.join(missing)})" if missing else ""))
    if skip_docs:
        check("WARN", "docs/ skipped for now: add the proposal and the report (PDF and Markdown) before you submit")
    else:
        missing_docs = [f for f in REQUIRED_DOCS if f not in files]
        check("PASS" if not missing_docs else "FAIL", "the proposal is in docs/ as PDF and Markdown" + (f" (missing: {', '.join(missing_docs)})" if missing_docs else ""))
        missing_notes = [f for f in NOTES if f not in files]
        check("PASS" if not missing_notes else "WARN", "the notes in docs/ are committed" + (f" (missing: {', '.join(missing_notes)})" if missing_notes else ""))
    missing_data = [f for f in REQUIRED_DATA if f not in files]
    if missing_data:
        check("WARN" if allow_simulated else "FAIL",
              "processed data is committed, so the numbers can be reproduced (missing: " + ", ".join(missing_data) + ")")
    else:
        check("PASS", "processed data is committed")
    if not skip_docs:
        missing_report = [f for f in REPORT_FILES if f not in files]
        check("PASS" if not missing_report else "WARN", "report is in docs/ as PDF and Markdown" + (f" (missing: {', '.join(missing_report)})" if missing_report else ""))
    bad = forbidden_files(files)
    check("PASS" if not bad else "FAIL", "no virtual environment, cache, .env or raw data committed" + (f" (found: {', '.join(bad[:5])})" if bad else ""))
    big = big_files(clone, files)
    check("PASS" if not big else "FAIL", f"no committed file over {BIG_FILE_MB} MB" + (f" (found: {', '.join(big)})" if big else ""))
    secrets = secret_files(clone, files)
    check("PASS" if not secrets else "FAIL", "no key or token in the committed files" + (f" (look at: {', '.join(secrets)})" if secrets else ""))
    try:
        commits = secret_commits(clone)
        check("PASS" if not commits else "WARN", "no key or token added anywhere in the history" + (f" (commits to look at: {', '.join(commits[:5])})" if commits else ""))
    except RuntimeError as err:
        check("WARN", f"could not scan the history for keys ({err})")
    authors = commit_authors(clone)
    summary = ", ".join(f"{name}: {n}" for name, n in authors.most_common())
    check("PASS" if len(authors) >= 2 else "WARN",
          f"commits come from at least two authors ({summary or 'no commits found'})")


def main():
    parser = argparse.ArgumentParser(description="Check the repository from a fresh clone.")
    parser.add_argument("--repo", required=True, help="repository URL or a local folder")
    parser.add_argument("--skip-tests", action="store_true", help="skip the (slow) test run")
    parser.add_argument("--skip-docs", action="store_true", help="do not check docs/ yet (use until the proposal and report are added)")
    parser.add_argument("--allow-simulated", action="store_true", help="do not fail if the processed data is not committed")
    parser.add_argument("--keep", action="store_true", help="keep the temporary clone")
    args = parser.parse_args()

    if shutil.which("git") is None:
        sys.exit("git was not found. Install Git first.")
    work = Path(tempfile.mkdtemp(prefix="darkwatch_clone_"))
    clone = work / "repo"
    print(f"Working in {work}\n", flush=True)
    try:
        ok, secs, tail = run(["git", "clone", "--quiet", resolve_repo(args.repo), str(clone)], work)
        check("PASS" if ok else "FAIL", f"clone ({secs:.0f} s)" + ("" if ok else f"\n{tail}"))
        if not ok:
            return
        check_repository(clone, args.allow_simulated, args.skip_docs)

        ok, secs, tail = run([sys.executable, "-m", "venv", ".venv"], clone)
        check("PASS" if ok else "FAIL", f"make a virtual environment ({secs:.0f} s)" + ("" if ok else f"\n{tail}"))
        if not ok:
            return
        py = str(venv_python(clone))
        ok, secs, tail = run([py, "-m", "pip", "install", "-q", "-r", "requirements.txt"], clone)
        check("PASS" if ok else "FAIL", f"pip install -r requirements.txt ({secs:.0f} s)" + ("" if ok else f"\n{tail}"))
        if not ok:
            return

        steps = [("data checks (verify_data.py)", [py, "src/verify_data.py"], "WARN" if args.allow_simulated else "FAIL"),
                 ("quick experiments (experiments.py --quick)", [py, "src/experiments.py", "--quick"], "WARN" if args.allow_simulated else "FAIL"),
                 ("replay export (replay.py)", [py, "src/replay.py", "--agent", "Belief + A*", "--seed", "42"], "FAIL")]
        if not args.skip_tests:
            steps.insert(0, ("tests (pytest)", [py, "-m", "pytest", "-q"], "FAIL"))
        for label, command, bad_status in steps:
            ok, secs, tail = run(command, clone)
            check("PASS" if ok else bad_status, f"{label} ({secs:.0f} s)" + ("" if ok else f"\n{tail}"))
    finally:
        if args.keep:
            print(f"\nKept the clone in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
        counts = {s: results.count(s) for s in ("PASS", "WARN", "FAIL")}
        print(f"\nResult: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed")
    if "FAIL" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()