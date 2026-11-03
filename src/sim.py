"""Simulation core for the DarkWatch starter (synthetic data, no pygame here).

Keeping game logic separate from drawing lets the experiments run headless.
Replace GreedyPatrol with your A*/MCTS agent and Belief.diffuse with your HMM.
"""
import math
import random

import numpy as np

W, H = 40, 30
YY, XX = np.mgrid[0:H, 0:W]


class Ocean:
    """Grid ocean with a synthetic fishing-density map (swap for real data)."""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        d = np.zeros((H, W))
        for _ in range(5):
            cx, cy = rng.uniform(4, W - 4), rng.uniform(4, H - 4)
            s = rng.uniform(2.5, 5.0)
            d += rng.uniform(0.5, 1.0) * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * s * s))
        self.density = d / d.max()


class Belief:
    """Probability map of where one vessel might be (placeholder for an HMM)."""

    def __init__(self):
        self.p = np.zeros((H, W))

    def collapse(self, x, y):
        """Observation: we know exactly where the vessel is."""
        self.p[:] = 0.0
        self.p[int(y), int(x)] = 1.0

    def diffuse(self, rate=0.5):
        """Transition model: mass spreads to the 4 neighbours each step."""
        q = np.pad(self.p, 1, mode="edge")
        nb = (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]) / 4.0
        self.p = (1 - rate) * self.p + rate * nb
        self.p /= self.p.sum()


class Vessel:
    """A fishing vessel doing a correlated random walk; AIS may go dark."""

    def __init__(self, rng):
        self.x, self.y = rng.uniform(3, W - 3), rng.uniform(3, H - 3)
        self.px, self.py = self.x, self.y
        self.heading = rng.uniform(0, 2 * math.pi)
        self.visible = True
        self.last_known = (self.x, self.y)
        self.trail = []
        self.belief = Belief()
        self.belief.collapse(self.x, self.y)

    def step(self, rng):
        self.px, self.py = self.x, self.y
        self.heading += rng.gauss(0, 0.35)
        nx = self.x + 0.3 * math.cos(self.heading)
        ny = self.y + 0.3 * math.sin(self.heading)
        if not 1 <= nx < W - 1:
            self.heading, nx = math.pi - self.heading, self.x
        if not 1 <= ny < H - 1:
            self.heading, ny = -self.heading, self.y
        self.x, self.y = nx, ny
        self.trail = (self.trail + [(nx, ny)])[-40:]


class PatrolAgent:
    """Base agent. Subclasses override choose_target (your A*/MCTS goes here)."""

    name = "Base"
    speed = 0.6
    fuel_cost = 0.15

    def __init__(self):
        self.x, self.y = W / 2, H / 2
        self.px, self.py = self.x, self.y
        self.heading = 0.0
        self.fuel = 100.0
        self.target = None

    def choose_target(self, world):
        raise NotImplementedError

    def move(self, target):
        self.px, self.py = self.x, self.y
        self.target = target
        if target is None:
            return
        dx, dy = target[0] - self.x, target[1] - self.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        step = min(self.speed, d)
        self.x += dx / d * step
        self.y += dy / d * step
        self.heading = math.atan2(dy, dx)
        self.fuel = max(0.0, self.fuel - self.fuel_cost)


class GreedyPatrol(PatrolAgent):
    """Baseline: head for the cell with best (dark-vessel belief / distance)."""

    name = "Greedy belief"

    def choose_target(self, world):
        dark = [v.belief.p for v in world.vessels if not v.visible]
        if not dark:
            return None
        score = sum(dark) / (np.hypot(XX - self.x, YY - self.y) + 3.0)
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        return ix + 0.5, iy + 0.5


class World:
    """Ties ocean, vessels and patrol together; one step() = one time tick."""

    def __init__(self, seed=7, n_vessels=6, agent_cls=GreedyPatrol):
        self.rng = random.Random(seed)
        self.ocean = Ocean(seed)
        self.vessels = [Vessel(self.rng) for _ in range(n_vessels)]
        self.patrol = agent_cls()
        self.step_count = 0
        self.detections = 0
        self.history = []

    def step(self):
        if self.patrol.fuel <= 0:
            return
        for v in self.vessels:
            v.step(self.rng)
            if v.visible:
                v.belief.collapse(v.x, v.y)
                v.last_known = (v.x, v.y)
                if self.rng.random() < 0.015:
                    v.visible = False
            else:
                v.belief.diffuse()
                if self.rng.random() < 0.004:
                    v.visible = True
                    v.belief.collapse(v.x, v.y)
        self.patrol.move(self.patrol.choose_target(self))
        for v in self.vessels:
            if not v.visible and math.hypot(self.patrol.x - v.x, self.patrol.y - v.y) < 2.5:
                v.visible = True
                v.belief.collapse(v.x, v.y)
                self.detections += 1
        self.step_count += 1
        self.history.append(self.detections)