#one base class other greedy , random ,fixed sweep , a star + 
import random
import numpy as np
from belief import Belief
from mcts import MCTS
from mcts_patrol import PatrolProblem, PathCache, spaced_candidates
from search import AStar

def chebyshev(a, b): #calculates idtsnace btw 2 grids and picks mx
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def square_mass(mass, radius): #adds probabilities of neighbouring cells includes centre cells
    h, w = mass.shape
    padded = np.pad(mass, radius)
    out = np.zeros_like(mass)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out += padded[dy:dy + h, dx:dx + w]
    return out

class PatrolAgent:
   #base class only 
    name = "PatrolAgent"

    def __init__(self, world, rng=None):
        self.cost = world.cost
        self.water = world.water
        self.kernel = world.kernel
        self.config = world.config
        self.rng = rng or random.Random(0)
        self.last_decision = None      # what the agent weighed last time (the replay file shows it)

    def observe(self, obs):
        """Look at the new observation (update memory). Default: nothing to remember."""

    def choose_target(self, obs):
        #whcih cell to move
        raise NotImplementedError


class RandomPatrol(PatrolAgent):
    name = "Random"
    RANGE = 6

    def __init__(self, world, rng=None):
        super().__init__(world, rng)
        self.target = None

    def observe(self, obs): #ager target current grid hai so turn target to null and calc again
        if self.target == obs.patrol_cell:
            self.target = None

    def choose_target(self, obs):  #random targetssss
        if self.target is None:
            x, y = obs.patrol_cell
            h, w = self.water.shape
            ys, xs = np.where(self.water[max(0, y - self.RANGE):y + self.RANGE + 1, max(0, x - self.RANGE):x + self.RANGE + 1])
            if len(xs):
                k = self.rng.randrange(len(xs))
                self.target = (int(xs[k]) + max(0, x - self.RANGE), int(ys[k]) + max(0, y - self.RANGE))
        return self.target


class SweepPatrol(PatrolAgent):
    name = "Fixed sweep"  #name of the patrol type

    def __init__(self, world, rng=None):
        super().__init__(world, rng)
        step = 2 * self.config.radius + 1
        h, w = self.water.shape
        self.waypoints = []
        for i, y in enumerate(range(self.config.radius, h, step)):
            xs = [x for x in range(self.config.radius, w, step) if self.water[y, x]]
            self.waypoints += [(x, y) for x in (xs if i % 2 == 0 else xs[::-1])]
        self.index, self.stuck, self.last_cell = 0, 0, None

    def observe(self, obs): #if no way to move
        if not self.waypoints:
            return
        self.stuck = self.stuck + 1 if obs.patrol_cell == self.last_cell else 0
        self.last_cell = obs.patrol_cell
        if obs.patrol_cell == self.waypoints[self.index] or self.stuck >= 3:
            self.index, self.stuck = (self.index + 1) % len(self.waypoints), 0

    def choose_target(self, obs):
        return self.waypoints[self.index] if self.waypoints else None


class LastKnownPatrol(PatrolAgent):
    name = "Last known position"

    def __init__(self, world, rng=None):
        super().__init__(world, rng)
        self.targets = {}

    def observe(self, obs):
        found_ids = {i for i, _ in obs.found}
        for boat_id in found_ids:
            self.targets.pop(boat_id, None)
        for boat_id, last_seen, _ in obs.went_dark:
            if boat_id not in found_ids:
                self.targets[boat_id] = last_seen
        for boat_id, cell in list(self.targets.items()):
            if chebyshev(obs.patrol_cell, cell) <= self.config.radius and obs.searched is not None:
                del self.targets[boat_id]

    def choose_target(self, obs):
        if not self.targets:
            return None
        px, py = obs.patrol_cell
        return min(self.targets.values(), key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)

class BeliefAgent(PatrolAgent):
    name = "BeliefAgent"

    def __init__(self, world, rng=None):
        super().__init__(world, rng)
        self.beliefs, self.velocities = {}, {}

    def observe(self, obs):
        found_ids = {i for i, _ in obs.found}
        for boat_id in list(self.beliefs):
            if boat_id in found_ids:
                del self.beliefs[boat_id], self.velocities[boat_id]
            else:
                self.beliefs[boat_id].predict(self.kernel, self.velocities[boat_id])
        for boat_id, last_seen, velocity in obs.went_dark:
            if boat_id in found_ids:
                continue
            belief = Belief(self.cost.shape, last_seen, self.water)
            belief.predict(self.kernel, velocity)
            self.beliefs[boat_id], self.velocities[boat_id] = belief, velocity
        if obs.searched is not None:
            cell, radius = obs.searched
            for belief in self.beliefs.values():
                belief.update(cell, detected=False, radius=radius, p_detect=self.config.p_detect)

    def square_mass(self):
        """Belief of all dark boats added up inside the sensor square around every cell (land gets 0)."""
        total = np.zeros(self.cost.shape)
        for belief in self.beliefs.values():
            total += belief.p
        mass = square_mass(total, self.config.radius)
        mass[~self.water] = 0.0
        return mass


class GreedyPatrol(BeliefAgent):
    """Head for the cell with the most belief, whatever it costs to get there."""

    name = "Greedy belief"

    def choose_target(self, obs):
        mass = self.square_mass()
        if mass.max() <= 1e-9:
            return None
        ys, xs = np.where(mass >= mass.max() - 1e-12)
        px, py = obs.patrol_cell
        k = int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))
        self.last_decision = {"candidates": [{"x": int(xs[k]), "y": int(ys[k]), "mass": round(float(mass.max()), 4)}]}
        return int(xs[k]), int(ys[k])


class BeliefAStarPatrol(BeliefAgent):
    """Our agent. Among the cells with the most belief, pick the one with the best expected
    detections per unit of fuel: p_detect x belief in the square / (A* route cost + 1)."""

    name = "Belief + A*"
    CANDIDATES = 12

    def __init__(self, world, rng=None):
        super().__init__(world, rng)
        self.router = AStar(self.cost)

    def choose_target(self, obs):
        mass = self.square_mass()
        if mass.max() <= 1e-9:
            return None
        best, best_utility, weighed = None, 0.0, []
        for flat in np.argsort(mass, axis=None)[::-1][:self.CANDIDATES]:
            y, x = divmod(int(flat), mass.shape[1])
            if mass[y, x] <= 1e-9:
                break
            route = self.router.search(obs.patrol_cell, (x, y), max_cost=obs.fuel)
            if not route.found:
                continue
            utility = self.config.p_detect * mass[y, x] / (route.cost + 1.0)
            weighed.append({"x": x, "y": y, "mass": round(float(mass[y, x]), 4),
                            "route_cost": round(float(route.cost), 3), "utility": round(float(utility), 5)})
            if utility > best_utility:
                best, best_utility = (x, y), utility
        self.last_decision = {"candidates": weighed}
        return best


class MCTSPatrol(BeliefAgent):
    """Chooses the target with Monte Carlo tree search (src/mcts.py) over a few candidate cells.

    Each simulation samples a position for every dark boat from its belief, so the tree learns
    which candidate does best on average over the uncertainty. More simulations = a better
    estimate but a slower decision."""

    name = "MCTS"
    slow = True                 # run_patrols.py can skip it with --skip-slow
    SIMULATIONS = 200
    CANDIDATES = 6
    MAX_DEPTH = 3

    def __init__(self, world, rng=None, simulations=None):
        super().__init__(world, rng)
        self.simulations = simulations or self.SIMULATIONS
        self.paths = PathCache.for_grid(self.cost)

    def choose_target(self, obs):
        candidates = spaced_candidates(self.square_mass(), self.CANDIDATES)
        if not candidates:
            return None
        problem = PatrolProblem(self.cost, self.kernel, self.config, self.paths, self.beliefs, self.velocities, candidates)
        best, stats = MCTS(problem, self.simulations, max_depth=self.MAX_DEPTH, rng=self.rng).search((obs.patrol_cell, obs.fuel))
        self.last_decision = {"candidates": [{"x": a[0], "y": a[1], "visits": v, "mean_return": round(float(m), 4)}
                                             for a, (v, m) in stats.items()]}
        return best


def make_mcts(simulations):
    """A copy of MCTSPatrol with a fixed number of simulations and its own name (used by the sweep)."""
    return type(f"MCTS{simulations}", (MCTSPatrol,), {"name": f"MCTS-{simulations}", "SIMULATIONS": simulations})


ALL_AGENTS = [RandomPatrol, SweepPatrol, LastKnownPatrol, GreedyPatrol, BeliefAStarPatrol, MCTSPatrol]