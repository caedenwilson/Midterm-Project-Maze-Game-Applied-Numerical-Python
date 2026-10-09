import random
from collections import deque

import numpy as np

# Directions as (d_row, d_col): North, East, South, West
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class Maze:
    """A multi-floor maze. Nodes are (floor, row, col) cells.

    Each floor is a bool grid twice the cell size plus one (True = floor,
    False = wall), so passages between cells are tiles of their own.
    Generation: randomized depth-first search (recursive backtracker) for a
    perfect maze, then a "braid" pass that knocks out extra walls to create loops.
    """

    def __init__(self, size=8, floors=3, loop_chance=0.08, ghosts_per_floor=2):
        if size < 3 or floors < 1:
            raise ValueError("size must be >= 3 and floors >= 1")
        self.size = size
        self.floors = floors
        self.shape = (2 * size + 1,) * 2
        self.grid = np.stack([self._carve_floor(size, loop_chance) for _ in range(floors)])
        self.stairs = {}  # (f, r, c) -> destination floor (same r, c)
        self.goal = self.map_pos = (-1, -1, -1)  # placeholders until placed
        self._place_stairs()
        self._place_objects(ghosts_per_floor)

    # ---------- generation ----------
    @staticmethod
    def _carve_floor(n, loop_chance):
        grid = np.zeros((2 * n + 1, 2 * n + 1), dtype=bool)
        visited = np.zeros((n, n), dtype=bool)
        start = (random.randrange(n), random.randrange(n))
        visited[start] = True
        grid[2 * start[0] + 1, 2 * start[1] + 1] = True
        stack = [start]  # iterative DFS, so no recursion limit
        while stack:
            r, c = stack[-1]
            options = [(r + dr, c + dc, dr, dc) for dr, dc in DIRS
                       if 0 <= r + dr < n and 0 <= c + dc < n and not visited[r + dr, c + dc]]
            if not options:
                stack.pop()
                continue
            nr, nc, dr, dc = random.choice(options)
            grid[2 * r + 1 + dr, 2 * c + 1 + dc] = True  # knock out the wall between
            grid[2 * nr + 1, 2 * nc + 1] = True
            visited[nr, nc] = True
            stack.append((nr, nc))

        # Braid: remove some remaining walls so the maze contains loops
        for r in range(n):
            for c in range(n):
                for dr, dc in ((0, 1), (1, 0)):
                    wr, wc = 2 * r + 1 + dr, 2 * c + 1 + dc
                    if r + dr < n and c + dc < n and not grid[wr, wc] \
                            and random.random() < loop_chance:
                        grid[wr, wc] = True
        return grid

    def _place_stairs(self):
        n = self.size
        used = [set() for _ in range(self.floors)]  # one staircase per cell
        for f in range(self.floors - 1):
            free = [(r, c) for r in range(n) for c in range(n)
                    if (r, c) not in used[f] and (r, c) not in used[f + 1]]
            for r, c in random.sample(free, 2):
                self.stairs[(f, r, c)] = f + 1
                self.stairs[(f + 1, r, c)] = f
                used[f].add((r, c))
                used[f + 1].add((r, c))

    def _place_objects(self, ghosts_per_floor):
        n = self.size
        nodes = [(f, r, c) for f in range(self.floors) for r in range(n) for c in range(n)]
        plain = [x for x in nodes if x not in self.stairs]

        self.player_start = random.choice(plain)
        dist = self.distances(self.player_start)
        far = max(dist.values())
        visible = self.observe(*self.player_start)["cells"]
        candidates = [x for x in plain if dist[x] >= 0.6 * far
                      and not (x[0] == self.player_start[0] and (x[1], x[2]) in visible)]
        self.goal = random.choice(candidates or [max(plain, key=dist.get)])

        self.map_pos = random.choice([x for x in plain
                                      if x not in (self.player_start, self.goal)])
        # The map shows a route from a random location (not the map's own) to the goal
        self.map_route_start = random.choice([x for x in nodes if x != self.goal])
        self.map_path = self.shortest_path(self.map_route_start, self.goal)

        self.ghost_starts = []
        for f in range(self.floors):
            spots = [x for x in nodes if x[0] == f and x != self.goal
                     and (f != self.player_start[0]
                          or abs(x[1] - self.player_start[1]) + abs(x[2] - self.player_start[2]) >= 4)]
            self.ghost_starts += random.sample(spots, min(ghosts_per_floor, len(spots)))

    # ---------- queries ----------
    def is_open(self, f, r, c, d):
        dr, dc = DIRS[d]
        return bool(self.grid[f, 2 * r + 1 + dr, 2 * c + 1 + dc])

    def open_dirs(self, f, r, c):
        return frozenset(d for d in range(4) if self.is_open(f, r, c, d))

    def neighbors(self, node):
        f, r, c = node
        for d in self.open_dirs(f, r, c):
            yield (f, r + DIRS[d][0], c + DIRS[d][1])
        if node in self.stairs:
            yield (self.stairs[node], r, c)

    def distances(self, start):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in self.neighbors(cur):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist

    def shortest_path(self, start, goal):
        prev = {start: None}
        queue = deque([start])
        while queue and goal not in prev:
            cur = queue.popleft()
            for nxt in self.neighbors(cur):
                if nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
        path, node = [], goal
        while node is not None:
            path.append(node)
            node = prev[node]
        return path[::-1]

    def observe(self, f, r, c, ghosts=()):
        """Everything visible from (f, r, c): straight down each corridor, never through walls."""
        cells = {(r, c): self.open_dirs(f, r, c)}
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r, c
            while self.is_open(f, rr, cc, d):
                rr, cc = rr + dr, cc + dc
                cells[(rr, cc)] = self.open_dirs(f, rr, cc)
        return {
            "floor": f,
            "cells": cells,  # (r, c) -> set of open directions
            "stairs": {(sr, sc): dest for (sf, sr, sc), dest in self.stairs.items()
                       if sf == f and (sr, sc) in cells},
            "goal": self.goal if self.goal[0] == f and self.goal[1:] in cells else None,
            "map": self.map_pos if self.map_pos[0] == f and self.map_pos[1:] in cells else None,
            "ghosts": [g for g in ghosts if g[0] == f and g[1:] in cells],
        }


if __name__ == "__main__":
    m = Maze()
    print(f"{m.floors} floors, {len(m.stairs) // 2} staircases, "
          f"start {m.player_start}, goal {m.goal}, map {m.map_pos}")
