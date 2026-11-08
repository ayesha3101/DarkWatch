"""Route search on the cost grid: BFS, UCS and A*, written from scratch (Labs 04-05).

Grid convention (same as data.py): cost[y, x] is 1.0 for open water, higher for
harder cells, and np.inf for land. A state is a cell written as (x, y).

Moves: 8 directions. A step costs 1 (straight) or sqrt(2) (diagonal), multiplied by
the cost of the cell we enter. A diagonal move is not allowed to cut across the
corner of a land cell. The path cost is the fuel used.
"""
import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

SQRT2 = math.sqrt(2.0)
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass
class SearchResult:
    """What a search returns. It never raises for 'no path': check .found and .reason."""

    found: bool
    path: list
    cost: float
    nodes_expanded: int
    frontier_peak: int
    reason: str
    order: list = field(default_factory=list)   # (state, g, priority) in the order nodes were expanded


class Planner:
    """Base class: holds the grid and the move rules. Subclasses write search()."""

    name = "Planner"

    def __init__(self, cost):
        self.cost = np.asarray(cost, dtype=float)
        if self.cost.ndim != 2:
            raise ValueError("cost must be a 2D grid")
        self.height, self.width = self.cost.shape
        water = self.cost[np.isfinite(self.cost)]
        self.min_cost = float(water.min()) if water.size else 1.0

    def passable(self, x, y):
        """True if (x, y) is inside the grid and not land."""
        return 0 <= x < self.width and 0 <= y < self.height and np.isfinite(self.cost[y, x])

    def neighbors(self, state):
        """Yield (next_state, step_cost) for every legal move from state."""
        x, y = state
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not self.passable(nx, ny):
                continue
            diagonal = dx != 0 and dy != 0
            if diagonal and not (self.passable(x + dx, y) and self.passable(x, y + dy)):
                continue   # no cutting across a land corner
            yield (nx, ny), (SQRT2 if diagonal else 1.0) * self.cost[ny, nx]

    def check_endpoints(self, start, goal):
        """Return a reason string if start or goal is unusable, otherwise None."""
        for label, (x, y) in (("start", start), ("goal", goal)):
            if not (0 <= x < self.width and 0 <= y < self.height):
                return f"{label} is outside the grid"
            if not np.isfinite(self.cost[y, x]):
                return f"{label} is on land"
        return None

    @staticmethod
    def rebuild(parent, goal):
        """Follow parent links back from the goal to the start."""
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]

    def search(self, start, goal, max_cost=None):
        raise NotImplementedError


class BFS(Planner):
    """Breadth-first search: expands cells by number of moves, using a FIFO queue.

    It finds the path with the fewest moves, which is NOT always the cheapest path
    here, because cells have different costs. max_cost (the fuel limit) is only used
    to skip cells that are already too expensive.
    """

    name = "BFS"

    def search(self, start, goal, max_cost=None):
        problem = self.check_endpoints(start, goal)
        if problem:
            return SearchResult(False, [], math.inf, 0, 0, problem)
        if start == goal:
            return SearchResult(True, [start], 0.0, 0, 1, "ok")

        frontier = deque([start])          # FIFO queue
        parent = {start: None}             # also acts as the explored/visited set
        fuel = {start: 0.0}
        expanded, peak, order = 0, 1, []
        while frontier:
            state = frontier.popleft()
            expanded += 1
            order.append((state, fuel[state], None))
            for nxt, step in self.neighbors(state):
                if nxt in parent:
                    continue
                used = fuel[state] + step
                if max_cost is not None and used > max_cost:
                    continue
                parent[nxt], fuel[nxt] = state, used
                if nxt == goal:
                    return SearchResult(True, self.rebuild(parent, goal), used, expanded, peak, "ok", order)
                frontier.append(nxt)
            peak = max(peak, len(frontier))
        reason = "no path" if max_cost is None else "no path within the fuel limit"
        return SearchResult(False, [], math.inf, expanded, peak, reason, order)


class UCS(Planner):
    """Uniform-cost search: always expands the cheapest cell so far (priority queue on g)."""

    name = "UCS"

    def priority(self, g, state, goal):
        """What the priority queue sorts by. UCS uses only the cost so far."""
        return g

    def search(self, start, goal, max_cost=None):
        problem = self.check_endpoints(start, goal)
        if problem:
            return SearchResult(False, [], math.inf, 0, 0, problem)

        tie = itertools.count()            # keeps the order stable when priorities are equal
        frontier = [(self.priority(0.0, start, goal), next(tie), 0.0, start)]   # heap
        best_g = {start: 0.0}
        parent = {start: None}
        explored = set()
        expanded, peak, order = 0, 1, []
        while frontier:
            prio, _, g, state = heapq.heappop(frontier)
            if state in explored:
                continue                   # an older, worse entry for a cell we already expanded
            if state == goal:
                return SearchResult(True, self.rebuild(parent, goal), g, expanded, peak, "ok", order)
            explored.add(state)
            expanded += 1
            order.append((state, g, prio))
            for nxt, step in self.neighbors(state):
                new_g = g + step
                if max_cost is not None and new_g > max_cost:
                    continue
                if new_g < best_g.get(nxt, math.inf):
                    best_g[nxt], parent[nxt] = new_g, state
                    heapq.heappush(frontier, (self.priority(new_g, nxt, goal), next(tie), new_g, nxt))
            peak = max(peak, len(frontier))
        reason = "no path" if max_cost is None else "no path within the fuel limit"
        return SearchResult(False, [], math.inf, expanded, peak, reason, order)


class AStar(UCS):
    """A* search: like UCS, but sorts by f = g + h, where h estimates the cost still to go.

    Heuristic h = (octile distance to the goal) x (cheapest cell cost in the grid).
    Octile distance is the number of moves needed on an empty 8-connected grid:
    max(dx, dy) + (sqrt(2) - 1) * min(dx, dy). Why it is admissible (never too high):
    every real move costs at least min_cost times 1 (straight) or sqrt(2) (diagonal),
    and no path can reach the goal with fewer moves than the octile distance.
    It is also consistent, so A* never needs to re-open a cell.
    """

    name = "A*"

    def heuristic(self, state, goal):
        dx, dy = abs(state[0] - goal[0]), abs(state[1] - goal[1])
        return (max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)) * self.min_cost

    def priority(self, g, state, goal):
        return g + self.heuristic(state, goal)