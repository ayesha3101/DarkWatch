"""Monte Carlo tree search, written from scratch (Lab 8).

This file is the general algorithm. It knows nothing about boats: it only talks to a
`problem` object (see src/mcts_patrol.py for ours). The four steps of every simulation:

  1 Selection      walk down the tree, always taking the child with the best UCT score
  2 Expansion      at the first node with an untried action, try it and add the new node
  3 Simulation     from that new node, play on with a cheap default policy (the rollout)
  4 Backpropagation  add the total reward to every node on the way back up

UCT score of a child = (average return) / scale + c * sqrt(ln(parent visits) / child visits)
  - the first part favours actions that did well (exploitation),
  - the second favours actions tried few times (exploration),
  - scale is the biggest return seen so far, so returns of any size give a score near 0 to 1,
  - c is how much we explore. We use c = 1.0.
Depth limit: the tree is grown max_depth levels deep, then the rollout takes over.
The best action is the most visited child at the root (more robust than the best average).

What a `problem` must provide:
  actions(state)                      the actions available in a public state (a list)
  is_terminal(state)                  True when nothing more can happen
  sample_world(rng)                   hidden information for ONE simulation (may be None)
  step(state, action, world, rng)     returns (new_state, reward). The public part of the new
                                      state must not depend on the hidden world.
  rollout(state, world, rng)          total reward of the default policy from state
"""
import math
import random


class Node:
    """One public state in the tree, with the statistics of how good it has been."""

    __slots__ = ("state", "untried", "children", "visits", "total")

    def __init__(self, state, actions):
        self.state = state
        self.untried = list(actions)      # actions not tried yet
        self.children = {}                # action -> Node
        self.visits = 0
        self.total = 0.0                  # sum of returns that passed through this node

    @property
    def mean(self):
        return self.total / self.visits if self.visits else 0.0


class MCTS:
    """Plan one decision with `simulations` simulations."""

    def __init__(self, problem, simulations=200, c=1.0, max_depth=3, rng=None):
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        self.problem, self.simulations = problem, simulations
        self.c, self.max_depth = c, max_depth
        self.rng = rng or random.Random(0)
        self.scale = 1e-9                 # biggest |return| seen so far

    def search(self, state):
        """Return (best action, {action: (visits, average return)}). The action is None if there are none."""
        root = Node(state, self.problem.actions(state))
        if not root.untried:
            return None, {}
        self.rng.shuffle(root.untried)
        for _ in range(self.simulations):
            self._simulate(root)
        stats = {a: (child.visits, child.mean) for a, child in root.children.items()}
        best = max(root.children, key=lambda a: (root.children[a].visits, root.children[a].mean))
        return best, stats

    def _simulate(self, root):
        problem, rng = self.problem, self.rng
        world = problem.sample_world(rng)
        node, path, depth = root, [], 0               # path holds (node, action, reward) for every step taken

        while depth < self.max_depth and not problem.is_terminal(node.state):
            if node.untried:                          # 2. expansion: try a new action, add one node, stop descending
                action = node.untried.pop()
                new_state, reward = problem.step(node.state, action, world, rng)
                child = Node(new_state, problem.actions(new_state))
                rng.shuffle(child.untried)
                node.children[action] = child
                path.append((node, action, reward))
                node, depth = child, depth + 1
                break
            action = self._best_child(node)           # 1. selection
            new_state, reward = problem.step(node.state, action, world, rng)
            path.append((node, action, reward))
            node, depth = node.children[action], depth + 1

        tail = 0.0 if problem.is_terminal(node.state) else problem.rollout(node.state, world, rng)   # 3. simulation

        total = tail                                  # 4. backpropagation: each node gets the return from there on
        for parent, action, reward in reversed(path):
            total += reward
            child = parent.children[action]
            child.visits += 1
            child.total += total
            self.scale = max(self.scale, abs(total))
        root.visits += 1

    def _best_child(self, node):
        """The child with the highest UCT score."""
        log_n = math.log(max(1, node.visits))

        def uct(action):
            child = node.children[action]
            return child.mean / self.scale + self.c * math.sqrt(log_n / child.visits)

        return max(node.children, key=uct)