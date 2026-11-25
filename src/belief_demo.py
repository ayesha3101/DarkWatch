#just a demo run to see if our algos work
import numpy as np

from belief import Belief

KERNEL = np.array([[0.0, 0.1, 0.0],
                   [0.1, 0.6, 0.1],
                   [0.0, 0.1, 0.0]])


def show(title, grid):
    print(f"\n{title}")
    for row in grid:
        print("   " + "  ".join(f"{v:7.4f}" for v in row))


def main():
    belief = Belief((3, 3), cell=(1, 1))
    show("Start: boat last seen in the middle (x=1, y=1)", belief.p)

    belief.predict(KERNEL)
    show("After predict: spread with the move table (sums to 1)", belief.p)

    before = belief.p.copy()
    belief.p[1, 0] *= 1 - 0.9            # patrol at (x=0, y=1): multiply that cell by P(miss | boat there) = 0.1
    show("Bayes numerator: searched cell x 0.1, other cells x 1", belief.p)
    total = belief.p.sum()
    print(f"\nTotal of the numerators = {total:.4f}  (this is P(miss))")
    belief.p = belief.p / total
    show("After update: divide by the total (sums to 1)", belief.p)
    print(f"\nSearched cell went from {before[1, 0]:.4f} to {belief.p[1, 0]:.4f}; "
          f"the middle went from {before[1, 1]:.4f} to {belief.p[1, 1]:.4f}")


if __name__ == "__main__":
    main()