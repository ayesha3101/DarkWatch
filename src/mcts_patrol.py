"""The patrol problem that the generic MCTS in mcts.py plans for (Lab 8).

Public state    (patrol cell, fuel left). It does not depend on where the boats really are.
Action          head for one of a few candidate cells (the cells with the most belief).
Hidden world    for every simulation we SAMPLE a position for each dark boat from its belief
                (a "particle"), then let those particles move with the same movement model
                the belief uses. Many simulations with different samples cover the uncertainty.
Reward          boats found in the step, minus a small price for the fuel burnt
                (PRICE_OF_FUEL = 0.01, so one boat is worth 100 fuel units).
Evaluation      after the tree depth is used up, a rollout takes ROLLOUT_STEPS more steps,
                each time heading for a random candidate. Its total reward is the leaf value.
Limits          at most max_depth levels of the tree (3) and `simulations` simulations
                (tested with a sweep in src/mcts_sweep.py).
"""
import bisect
import math

import numpy as np

from search import AStar, Planner

_CACHES = {}


class PathCache:
    """Cheapest routes from any cell. Each start cell is worked out once (full Dijkstra) and remembered."""

    def __init__(self, cost):
        self.planner = AStar(cost)
        self.trees = {}
        self.moves = {}

    @classmethod
    def for_grid(cls, cost):
        """One shared cache per map, so it also survives from one episode to the next."""
        key = hash(np.asarray(cost, dtype=float).tobytes())
        if key not in _CACHES:
            _CACHES[key] = cls(cost)
        return _CACHES[key]

    def _tree(self, start):
        if start not in self.trees:
            self.trees[start] = self.planner.distances_from(start)
        return self.trees[start]

    def cost(self, start, goal):
        """Cheapest fuel cost from start to goal (infinity if there is no route)."""
        return self._tree(start)[0].get(goal, math.inf)

    def move(self, start, goal, budget):
        """Walk from start toward goal along the cheapest route, using at most `budget` fuel.
        Returns (cell reached, fuel spent). Stays put if the goal is unreachable."""
        key = (start, goal, round(budget, 3))
        if key not in self.moves:
            dist, parent = self._tree(start)
            reached = start
            if goal != start and goal in dist:
                for cell in Planner.rebuild(parent, goal)[1:]:
                    if dist[cell] > budget + 1e-9:
                        break
                    reached = cell
            self.moves[key] = (reached, dist.get(reached, 0.0))
        return self.moves[key]


def spaced_candidates(mass, k=6, spacing=3):
    """The k cells with the most belief, no two closer than `spacing` cells (so they are different places)."""
    chosen = []
    for flat in np.argsort(mass, axis=None)[::-1]:
        y, x = divmod(int(flat), mass.shape[1])
        if mass[y, x] <= 1e-9:
            break
        if all(max(abs(x - cx), abs(y - cy)) >= spacing for cx, cy in chosen):
            chosen.append((x, y))
            if len(chosen) == k:
                break
    return chosen


class PatrolProblem:
    """Fits the interface that mcts.MCTS expects (see the top of mcts.py)."""

    PRICE_OF_FUEL = 0.01
    ROLLOUT_STEPS = 3

    def __init__(self, cost, kernel, config, paths, beliefs, velocities, candidates):
        self.config, self.paths, self.candidates = config, paths, candidates
        self.water = np.isfinite(cost)
        self.height, self.width = cost.shape
        self.kernel_cum = np.cumsum(kernel.ravel()).tolist()
        self.kernel_width = kernel.shape[1]
        self.kernel_radius = kernel.shape[0] // 2
        self.sources = [(np.cumsum(b.p.ravel()).tolist(), velocities[bid]) for bid, b in beliefs.items()]

    # ----- what MCTS asks for -----
    def actions(self, state):
        return list(self.candidates) if state[1] > 1e-9 else []

    def is_terminal(self, state):
        return state[1] <= 1e-9

    def sample_world(self, rng):
        """One guess of where every dark boat is, drawn from its belief: [x, y, vx, vy] each."""
        particles = []
        for cum, (vx, vy) in self.sources:
            index = min(bisect.bisect_left(cum, rng.random() * cum[-1]), len(cum) - 1)
            y, x = divmod(index, self.width)
            particles.append([float(x), float(y), vx, vy])
        return particles

    def step(self, state, action, particles, rng):
        cell, fuel = state
        new_cell, spent = self.paths.move(cell, action, min(self.config.patrol_speed, fuel))
        if spent == 0.0:
            spent = min(self.config.idle_fuel, fuel)
        found = self._advance(particles, new_cell, rng)
        return (new_cell, fuel - spent), found - self.PRICE_OF_FUEL * spent

    def rollout(self, state, particles, rng):
        total = 0.0
        for _ in range(self.ROLLOUT_STEPS):
            if self.is_terminal(state):
                break
            state, reward = self.step(state, rng.choice(self.candidates), particles, rng)
            total += reward
        return total

    # ----- the simulated boats -----
    def _offset(self, u):
        index = min(bisect.bisect_left(self.kernel_cum, u * self.kernel_cum[-1]), len(self.kernel_cum) - 1)
        row, col = divmod(index, self.kernel_width)
        return col - self.kernel_radius, row - self.kernel_radius

    def _passable(self, x, y):
        cx, cy = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
        return 0 <= cx < self.width and 0 <= cy < self.height and bool(self.water[cy, cx])

    def _advance(self, particles, patrol_cell, rng):
        """Move every particle one step, let the sensor look, remove the found ones. Returns how many were found."""
        found, alive = 0, []
        cfg = self.config
        for p in particles:
            ox, oy = self._offset(rng.random())
            nx, ny = p[0] + p[2] + ox, p[1] + p[3] + oy
            if self._passable(nx, ny):
                p[0], p[1] = nx, ny
            else:
                p[2], p[3] = -p[2], -p[3]
            bx, by = int(math.floor(p[0] + 0.5)), int(math.floor(p[1] + 0.5))
            if max(abs(bx - patrol_cell[0]), abs(by - patrol_cell[1])) <= cfg.radius and rng.random() < cfg.p_detect:
                found += 1
            else:
                alive.append(p)
        particles[:] = alive
        return found