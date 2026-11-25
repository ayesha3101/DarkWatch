#search demo (bfs, ucs and A star)
import numpy as np

from search import AStar, BFS, UCS

INF = np.inf
GRID = np.array([[1, INF, 1, 1, 1],
                 [1, 1, 1, 2, 1],
                 [1, 2, 2, 1, 1],
                 [1, INF, 1, INF, 1],
                 [INF, 1, 1, 1, 1]], dtype=float)
START, GOAL = (0, 0), (4, 4)
def show_grid():
    print("Cost grid (# = land, S = start, G = goal). x goes right, y goes down.")
    for y in range(GRID.shape[0]):
        cells = []
        for x in range(GRID.shape[1]):
            mark = "S" if (x, y) == START else "G" if (x, y) == GOAL else None
            cells.append(mark or ("#" if np.isinf(GRID[y, x]) else str(int(GRID[y, x]))))
        print("  " + " ".join(cells))


def main():
    show_grid()
    print(f"\n{'planner':<6}{'cost':>8}{'moves':>7}{'expanded':>10}   path")
    for planner_class in (BFS, UCS, AStar):
        r = planner_class(GRID).search(START, GOAL)
        print(f"{planner_class.name:<6}{r.cost:>8.3f}{len(r.path) - 1:>7}{r.nodes_expanded:>10}   {r.path}")

    r = AStar(GRID).search(START, GOAL)
    print("\nA* expansion order: g = cost so far, f = g + h (h = estimate to go)")
    print(f"{'#':>3}  {'cell':<8}{'g':>7}{'h':>8}{'f':>8}")
    for i, (state, g, f) in enumerate(r.order, start=1):
        print(f"{i:>3}  {str(state):<8}{g:>7.3f}{f - g:>8.3f}{f:>8.3f}")


if __name__ == "__main__":
    main()