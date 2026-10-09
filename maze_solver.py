import heapq

from maze import DIRS

STAIR_COST = 4  # prefer finishing a floor before hopping to another


class MazeSolver:
    """Explores a multi-floor maze using only what the player can see.

    It remembers every cell it has seen (and each cell's open sides). With
    that memory it runs Dijkstra over the known graph and heads for the
    nearest unexplored cell, or for the goal as soon as the goal has been seen.
    Because the search is over a graph, loops and staircases need no special
    handling. Visible ghosts (and the tiles next to them) are avoided.
    """

    def __init__(self):
        self.known = {}   # (f, r, c) -> set of open directions
        self.stairs = {}  # (f, r, c) -> destination floor
        self.goal = None  # (f, r, c) once seen

    def observe(self, obs):
        f = obs["floor"]
        for (r, c), dirs in obs["cells"].items():
            self.known[(f, r, c)] = dirs
        for (r, c), dest in obs["stairs"].items():
            self.stairs[(f, r, c)] = dest
        if obs["goal"]:
            self.goal = obs["goal"]

    def _edges(self, node):
        f, r, c = node
        for d in self.known.get(node, ()):
            yield (f, r + DIRS[d][0], c + DIRS[d][1]), 1
        if node in self.stairs:
            yield (self.stairs[node], r, c), STAIR_COST

    def _plan(self, start, blocked):
        """Cheapest path to the goal (if seen) or else the nearest unexplored cell."""
        dist = {start: 0}
        prev = {}
        heap = [(0, start)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            if node != start and (node == self.goal if self.goal else node not in self.known):
                path = [node]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            for nxt, cost in self._edges(node):
                if nxt in blocked:
                    continue
                if d + cost < dist.get(nxt, float("inf")):
                    dist[nxt] = d + cost
                    prev[nxt] = node
                    heapq.heappush(heap, (d + cost, nxt))
        return None

    def next_action(self, pos, obs):
        """Returns ('move', direction), ('stairs',) or ('wait',)."""
        self.observe(obs)
        ghosts = set(obs["ghosts"])
        danger = set(ghosts)
        for f, r, c in ghosts:
            danger.update((f, r + dr, c + dc) for dr, dc in DIRS)

        path = self._plan(pos, danger) or self._plan(pos, ghosts)
        if not path:
            return ("wait",)
        nxt = path[1]
        if nxt[0] != pos[0]:
            return ("stairs",)
        return ("move", DIRS.index((nxt[1] - pos[1], nxt[2] - pos[2])))
