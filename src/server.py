"""Local web server for the DarkWatch page. It only listens on this computer (127.0.0.1).

Run:  python src/server.py          then open  http://127.0.0.1:5000
      python src/server.py --port 8000

The page asks this server to run an episode in Python (agent + seed) and draws the replay it gets back.
All the thinking happens here; the page only draws. Nothing is put on the internet.
"""
import argparse
import threading
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from agents import ALL_AGENTS
from data import load_data
from movement import load_demo_kernel
from replay import find_agent, record_episode

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_SEED = 9_999_999


def create_app(data=None, kernel=None):
    """Make the Flask app. `data` and `kernel` can be given (tests do); otherwise they are loaded once."""
    if data is None:
        data = load_data()
    if kernel is None:
        kernel, _ = load_demo_kernel()
    app = Flask(__name__, static_folder=None)
    lock = threading.Lock()                       # one episode at a time keeps the shared route memory safe

    @lru_cache(maxsize=64)
    def episode(agent_name, seed):
        with lock:
            return record_episode(find_agent(agent_name), data, kernel, seed)

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/agents")
    def agents():
        return jsonify([{"name": a.name, "slow": bool(getattr(a, "slow", False))} for a in ALL_AGENTS])

    @app.get("/api/replay")
    def replay():
        name = request.args.get("agent", "Belief + A*")
        try:
            seed = int(request.args.get("seed", "42"))
        except ValueError:
            return jsonify(error="The seed must be a whole number."), 400
        if not 0 <= seed <= MAX_SEED:
            return jsonify(error=f"The seed must be between 0 and {MAX_SEED}."), 400
        try:
            find_agent(name)
        except ValueError as err:
            return jsonify(error=str(err)), 400
        return jsonify(episode(name, seed))

    @app.get("/<path:name>")
    def static_file(name):
        return send_from_directory(WEB_DIR, name)       # refuses paths that leave the web folder

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve the DarkWatch page on this computer.")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app = create_app()
    print(f"DarkWatch is running at http://127.0.0.1:{args.port}  (press Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()