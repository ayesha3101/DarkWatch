"""Bayesian belief filter for a hidden boat (Lab 09).

The belief is a probability map p[y, x] over grid cells (it always sums to 1).
Two steps repeat over time:
  predict  the transition model: the boat may have moved (drift + spread table),
  update   the sensor model and Bayes' rule: the patrol looked and either found the
           boat or found nothing. Finding nothing is evidence too.

Sensor model: if the patrol is at cell P, it detects a boat inside the square of cells
within `radius` of P with probability `p_detect` (and never outside it).
Bayes' rule on a miss: new belief of a cell = old belief x P(miss | boat there),
then divide by the total. P(miss | boat in the square) = 1 - p_detect, and
P(miss | boat outside) = 1.
"""
import numpy as np

DETECT_RADIUS = 1      # cells
DETECT_PROB = 0.9


def _shift_int(p, sx, sy):
    """Move the map sx cells right and sy cells down. Mass pushed off the edge is dropped."""
    h, w = p.shape
    out = np.zeros_like(p)
    out[slice(max(0, sy), min(h, h + sy)), slice(max(0, sx), min(w, w + sx))] = \
        p[slice(max(0, -sy), min(h, h - sy)), slice(max(0, -sx), min(w, w - sx))]
    return out


def shift(p, dx, dy):
    """Move the map by a fractional number of cells, splitting mass between neighbouring cells."""
    ix, iy = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = dx - ix, dy - iy
    out = np.zeros_like(p)
    for sx, wx in ((ix, 1 - fx), (ix + 1, fx)):
        for sy, wy in ((iy, 1 - fy), (iy + 1, fy)):
            if wx * wy > 0:
                out += wx * wy * _shift_int(p, sx, sy)
    return out


def convolve(p, kernel):
    """Spread the map with a (2R+1) x (2R+1) table of move probabilities."""
    r = kernel.shape[0] // 2
    out = np.zeros_like(p)
    for j in range(kernel.shape[0]):
        for i in range(kernel.shape[1]):
            if kernel[j, i] > 0:
                out += kernel[j, i] * _shift_int(p, i - r, j - r)
    return out


class Belief:
    """Probability map of one hidden boat. Cells are (x, y); the map is indexed p[y, x]."""

    def __init__(self, shape, cell, water=None):
        self.water = np.ones(shape, dtype=bool) if water is None else np.asarray(water, dtype=bool)
        self.p = np.zeros(shape)
        self.lost = False                  # True if the evidence contradicted the belief
        self.collapse(cell)

    def collapse(self, cell):
        """We know exactly where the boat is."""
        if not (0 <= cell[0] < self.p.shape[1] and 0 <= cell[1] < self.p.shape[0]):
            raise ValueError(f"cell {cell} is outside the {self.p.shape[1]} x {self.p.shape[0]} grid")
        self.p[:] = 0.0
        self.p[cell[1], cell[0]] = 1.0
        self.lost = False

    def _normalise(self):
        total = self.p.sum()
        if total <= 1e-12:                 # nothing left: start again, equal on all water cells
            self.p = self.water / self.water.sum()
            self.lost = True
        else:
            self.p = self.p / total

    def predict(self, kernel, drift=(0.0, 0.0)):
        """Transition step: move the map by the drift (cells), then spread it with the kernel."""
        self.p = convolve(shift(self.p, *drift), kernel) * self.water
        self._normalise()

    def update(self, patrol_cell, detected=False, boat_cell=None, radius=DETECT_RADIUS, p_detect=DETECT_PROB):
        """Sensor step. If detected, the boat's cell is known. If not, lower the belief in the searched square."""
        if detected:
            if boat_cell is None:
                raise ValueError("boat_cell is needed when the boat was detected")
            self.collapse(boat_cell)
            return
        px, py = patrol_cell
        h, w = self.p.shape
        self.p[max(0, py - radius):min(h, py + radius + 1), max(0, px - radius):min(w, px + radius + 1)] *= (1.0 - p_detect)
        self._normalise()

    def mean_position(self):
        """Average position (x, y) in cells, weighted by the belief."""
        ys, xs = np.indices(self.p.shape)
        return float((self.p * xs).sum()), float((self.p * ys).sum())

    def entropy(self):
        """How spread out the belief is, in bits (0 = certain)."""
        q = self.p[self.p > 0]
        return float(-(q * np.log2(q)).sum())

    def top_mask(self, mass=0.9):
        """True on the smallest set of cells that holds at least `mass` of the belief."""
        order = np.argsort(self.p, axis=None)[::-1]
        keep = np.searchsorted(np.cumsum(self.p.ravel()[order]), mass - 1e-12) + 1
        mask = np.zeros(self.p.size, dtype=bool)
        mask[order[:keep]] = True
        return mask.reshape(self.p.shape)