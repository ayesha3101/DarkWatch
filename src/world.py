"""The patrol environment: boats that move and go dark, and a patrol with fuel and a sensor.

PEAS (Lab 3)
  Performance  boats found per unit of fuel
  Environment  grid ocean (cost grid), fishing boats that move, AIS that switches off
  Actuators    the patrol moves toward a target cell along an A* route (limited per step)
  Sensors      AIS of boats that are still transmitting; a detection square around the patrol

One step = STEP_MINUTES_DEMO minutes. Everything random comes from ONE seeded stream,
and the same number of random draws is used every step whatever the patrol does, so two
patrols on the same seed face exactly the same boats and the same luck.

Agents only get an `Observation` (never the hidden boat positions).
"""
import bisect
import math
import random
from dataclasses import dataclass, field

import numpy as np

from movement import CELL_KM_DEMO, STEP_MINUTES_DEMO
from search import AStar

FUEL_BUDGETS = (25, 50, 75, 100)     # fuel budgets used to compare strategies fairly
CELLS_PER_KNOT = 1.852 * (STEP_MINUTES_DEMO / 60.0) / CELL_KM_DEMO   # cells a boat moves per step per knot


@dataclass
class WorldConfig:
    """All the numbers that define an episode (assumptions, stated in the report)."""
    n_boats: int = 6
    fuel: float = 100.0            # fuel units, 1 unit = one straight step over open water
    patrol_speed: float = 4.0      # most fuel units the patrol can burn moving in one step
    idle_fuel: float = 0.25        # fuel burnt in a step when the patrol does not move
    p_dark: float = 0.08           # chance per step that a boat with AIS on switches it off
    p_detect: float = 0.9          # chance the patrol sees a dark boat inside its square
    radius: int = 1                # the square is (2 * radius + 1) cells wide
    max_steps: int = 80
    boat_knots: tuple = (1.5, 5.0)

    def __post_init__(self):
        """Refuse settings that make no sense, with a message that says which one."""
        if self.n_boats < 1:
            raise ValueError("n_boats must be at least 1")
        if self.fuel <= 0 or self.patrol_speed <= 0:
            raise ValueError("fuel and patrol_speed must be positive")
        if self.idle_fuel < 0:
            raise ValueError("idle_fuel cannot be negative")
        for name in ("p_dark", "p_detect"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.radius < 0 or self.max_steps < 1:
            raise ValueError("radius cannot be negative and max_steps must be at least 1")
        low, high = self.boat_knots
        if low < 0 or high < low:
            raise ValueError("boat_knots must be (low, high) with 0 <= low <= high")


@dataclass
class Boat:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    dx: float = 0.0                # displacement during the last step (what AIS would show)
    dy: float = 0.0
    dark: bool = False
    found: bool = False

    @property
    def cell(self):
        return int(math.floor(self.x + 0.5)), int(math.floor(self.y + 0.5))


@dataclass
class Observation:
    """What the patrol knows at the start of a step."""
    step: int
    patrol_cell: tuple
    fuel: float
    visible: list = field(default_factory=list)     # (id, cell, (dx, dy)) for boats with AIS on
    went_dark: list = field(default_factory=list)   # (id, last_seen_cell, last_velocity) this step
    found: list = field(default_factory=list)       # (id, cell) found this step
    searched: tuple = None                          # (patrol_cell, radius) used by the sensor this step


class World:
    """One episode. Create a new World (with a seed) for every episode."""

    def __init__(self, cost, density, kernel, seed=42, config=None, port=None):
        self.cost = np.asarray(cost, dtype=float)
        self.density = np.asarray(density, dtype=float)
        self.kernel = np.asarray(kernel, dtype=float)
        if self.cost.ndim != 2 or not np.isfinite(self.cost).any():
            raise ValueError("the map must be a 2D grid with at least one water cell")
        if self.density.shape != self.cost.shape:
            raise ValueError("density and cost must have the same shape")
        k = self.kernel
        if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2 == 0 or (k < 0).any() or abs(k.sum() - 1.0) > 1e-6:
            raise ValueError("kernel must be an odd-sized square table of non-negative numbers that add up to 1")
        self.config = config or WorldConfig()
        self.seed = seed
        self.height, self.width = self.cost.shape
        self.water = np.isfinite(self.cost)
        self.router = AStar(self.cost)
        self.rng = random.Random(seed)
        self._cum = np.cumsum(self.kernel.ravel()).tolist()
        self.port = port or self._default_port()
        self.patrol = self.port
        self.fuel = self.config.fuel
        self.fuel_used = 0.0
        self.step_count = 0
        self.went_dark_total = 0
        self.trace = [(0.0, 0)]            # (fuel used so far, boats found so far) after every step
        self.boats = self._spawn_boats()
        self._last = Observation(0, self.patrol, self.fuel)

    # ----- set-up -----
    def _default_port(self):
        """Water cell nearest to the north-east part of the map (a harbour on the coast)."""
        ys, xs = np.where(self.water)
        target = (0.75 * (self.width - 1), 0.1 * (self.height - 1))
        i = int(np.argmin((xs - target[0]) ** 2 + (ys - target[1]) ** 2))
        return int(xs[i]), int(ys[i])

    def _spawn_boats(self):
        """Boats start in water, more of them where fishing activity is high."""
        ys, xs = np.where(self.water)
        weights = (self.density[ys, xs] + 1e-3).tolist()
        cells = list(zip(xs.tolist(), ys.tolist()))
        boats = []
        for i in range(self.config.n_boats):
            cx, cy = self.rng.choices(cells, weights=weights)[0]
            heading = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(*self.config.boat_knots) * CELLS_PER_KNOT
            vx, vy = speed * math.cos(heading), speed * math.sin(heading)
            boats.append(Boat(i, cx + self.rng.uniform(-0.4, 0.4), cy + self.rng.uniform(-0.4, 0.4), vx, vy, vx, vy))
        return boats

    # ----- helpers -----
    def passable(self, cell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and bool(self.water[y, x])

    def _sample_offset(self, u):
        """Pick a random (dx, dy) move from the movement table using one uniform number."""
        index = min(bisect.bisect_left(self._cum, u * self._cum[-1]), len(self._cum) - 1)
        row, col = divmod(index, self.kernel.shape[1])
        return col - self.kernel.shape[1] // 2, row - self.kernel.shape[0] // 2

    def _edge_cost(self, a, b):
        return dict(self.router.neighbors(a))[b]

    def _move_patrol(self, target):
        """Walk toward the target along an A* route, burning at most patrol_speed fuel."""
        route = None
        if target is not None and target != self.patrol and self.passable(target):
            route = self.router.search(self.patrol, target, max_cost=self.fuel)
        spent, here = 0.0, self.patrol
        if route is not None and route.found:
            budget = min(self.config.patrol_speed, self.fuel)
            for nxt in route.path[1:]:
                step_cost = self._edge_cost(here, nxt)
                if spent + step_cost > budget + 1e-9:
                    break
                spent, here = spent + step_cost, nxt
        if spent == 0.0:
            spent = min(self.config.idle_fuel, self.fuel)
        self.patrol = here
        self.fuel -= spent
        self.fuel_used += spent

    # ----- the episode -----
    def observation(self):
        """What the patrol sees now."""
        visible = [(b.id, b.cell, (b.dx, b.dy)) for b in self.boats if not b.dark and not b.found]
        last = self._last
        return Observation(self.step_count, self.patrol, self.fuel, visible, last.went_dark, last.found, last.searched)

    def done(self):
        return (self.fuel <= 1e-9 or self.step_count >= self.config.max_steps
                or all(b.found for b in self.boats))

    def step(self, target):
        """Patrol moves toward `target` (or stays if None), boats move, boats may go dark, the sensor looks."""
        cfg = self.config
        self.step_count += 1
        self._move_patrol(target)
        went_dark, found = [], []
        for boat in self.boats:
            u_move, u_dark, u_detect = self.rng.random(), self.rng.random(), self.rng.random()   # always drawn
            before_cell, before_velocity = boat.cell, (boat.dx, boat.dy)
            ox, oy = self._sample_offset(u_move)
            nx, ny = boat.x + boat.vx + ox, boat.y + boat.vy + oy
            if self.passable((int(math.floor(nx + 0.5)), int(math.floor(ny + 0.5)))):
                boat.dx, boat.dy, boat.x, boat.y = nx - boat.x, ny - boat.y, nx, ny
            else:
                boat.vx, boat.vy = -boat.vx, -boat.vy      # bounce off land or the edge
                boat.dx = boat.dy = 0.0
            if not boat.dark and not boat.found and u_dark < cfg.p_dark:
                boat.dark = True
                self.went_dark_total += 1
                went_dark.append((boat.id, before_cell, before_velocity))
            near = max(abs(boat.cell[0] - self.patrol[0]), abs(boat.cell[1] - self.patrol[1])) <= cfg.radius
            if boat.dark and not boat.found and near and u_detect < cfg.p_detect:
                boat.found = True
                found.append((boat.id, boat.cell))
        self._last = Observation(self.step_count, self.patrol, self.fuel, [], went_dark, found, (self.patrol, cfg.radius))
        self.trace.append((self.fuel_used, sum(b.found for b in self.boats)))
        return self.observation()

    def found_by(self, budget):
        """Boats found by the time `budget` fuel units had been used. After the episode ends the count stays flat,
        so a strategy that stops early is not punished (and not rewarded) for the fuel it did not use."""
        count = 0
        for used, found in self.trace:
            if used > budget + 1e-9:
                break
            count = found
        return count

    def summary(self):
        """The numbers we compare strategies with."""
        found = sum(b.found for b in self.boats)
        row = {"found": found, "went_dark": self.went_dark_total, "fuel_used": round(self.fuel_used, 3),
               "steps": self.step_count,
               "found_per_fuel": found / self.fuel_used if self.fuel_used > 0 else 0.0}
        row.update({f"found_at_{b}": self.found_by(b) for b in FUEL_BUDGETS})
        return row