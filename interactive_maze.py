import random
import sys

import matplotlib.pyplot as plt
import numpy as np

from maze import Maze, DIRS
from maze_solver import MazeSolver

VIEW_W, VIEW_H = 200, 120  # raycaster resolution
MINI_R = 7                 # minimap shows a fixed window of +/- this many tiles around the player
MAX_LIVES = 3
GHOST_SPEED = 0.6          # chance a ghost drifts each turn
GHOSTS_PER_FLOOR = 1
WALL_COLORS = [(0.85, 0.3, 0.25), (0.3, 0.5, 0.85), (0.3, 0.75, 0.4),
               (0.8, 0.7, 0.25), (0.7, 0.4, 0.8)]  # one color per floor

INSTRUCTIONS = ("W/UP forward   S/DOWN back   A/LEFT D/RIGHT turn   E stairs   "
                "M read map (when standing on it)   SPACE wait   X autopilot   Q quit")


# ---------- sprite textures (RGBA, drawn procedurally) ----------
def _make_sprites(T=24):
    y, x = np.mgrid[0:T, 0:T]
    nx = (x - (T - 1) / 2) / (T / 2)
    ny = (y - (T - 1) / 2) / (T / 2)

    def rgba(mask, color):
        t = np.zeros((T, T, 4))
        t[mask, :3] = color
        t[mask, 3] = 1
        return t

    # Ghost: dome + wavy skirt, with eyes and a mouth
    dome = (ny <= -0.2) & (nx ** 2 + (ny + 0.2) ** 2 <= 0.85)
    skirt = (ny > -0.2) & (ny < 0.7 + 0.15 * np.sin(x * 1.8)) & (np.abs(nx) <= 0.92)
    body = dome | skirt
    eyes = ((nx + 0.35) ** 2 + (ny + 0.25) ** 2 < 0.04) | ((nx - 0.35) ** 2 + (ny + 0.25) ** 2 < 0.04)
    mouth = (nx / 0.25) ** 2 + ((ny - 0.2) / 0.2) ** 2 < 1
    ghost = rgba(body, (0.85, 0.92, 1.0))
    ghost[(eyes | mouth) & body, :3] = (0.1, 0.1, 0.25)

    # Exit door: glowing green arch
    arch = ((np.abs(nx) <= 0.7) & (ny >= -0.3)) | (nx ** 2 + (ny + 0.3) ** 2 <= 0.49)
    inner = ((np.abs(nx) <= 0.4) & (ny >= -0.3)) | (nx ** 2 + (ny + 0.3) ** 2 <= 0.16)
    goal = rgba(arch, (0.15, 0.8, 0.35))
    goal[inner & arch, :3] = (0.75, 1.0, 0.75)

    # Map: little scroll
    paper = (np.abs(nx) <= 0.8) & (np.abs(ny) <= 0.5)
    rolls = (np.abs(nx) <= 0.85) & (np.abs(ny) > 0.5) & (np.abs(ny) <= 0.7)
    scroll = rgba(paper | rolls, (0.6, 0.4, 0.2))
    scroll[paper, :3] = (0.95, 0.85, 0.5)
    scroll[paper & (y % 4 == 0), :3] = (0.5, 0.3, 0.15)

    # Stairs: rising / falling steps
    k = x // (T // 4)
    up = rgba(ny >= 1 - 2 * (k + 1) / 4, (0.3, 0.85, 1.0))
    down = rgba(ny >= 1 - 2 * (4 - k) / 4, (0.65, 0.4, 1.0))

    # name -> (texture, size relative to wall height, lift off the floor)
    return {"ghost": (ghost, 0.6, 0.15), "goal": (goal, 0.85, 0.0), "map": (scroll, 0.3, 0.0),
            "stairs_up": (up, 0.4, 0.0), "stairs_down": (down, 0.4, 0.0)}


SPRITES = _make_sprites()


class Game:
    def __init__(self, size=8, floors=3):
        self.size, self.floors = size, floors
        for key in plt.rcParams:  # disable matplotlib's default key bindings
            if key.startswith("keymap."):
                plt.rcParams[key] = []

        self.fig = plt.figure(figsize=(10.5, 7))
        self.fig.canvas.manager.set_window_title("Ghost Maze")
        self.ax_view = self.fig.add_axes([0.02, 0.13, 0.66, 0.85])
        self.ax_map = self.fig.add_axes([0.71, 0.33, 0.27, 0.45])
        self.hud = self.fig.text(0.845, 0.96, "", ha="center", va="top", fontsize=13)
        self.status = self.fig.text(0.02, 0.085, "", fontsize=11, color="darkred")
        self.fig.text(0.02, 0.025, INSTRUCTIONS, fontsize=8.5)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: self.timer.stop())
        self.timer = self.fig.canvas.new_timer(interval=150)
        self.timer.add_callback(self.auto_step)
        self.new_game()

    def new_game(self):
        self.timer.stop()
        self.maze = Maze(self.size, self.floors, ghosts_per_floor=GHOSTS_PER_FLOOR)
        self.pos = self.maze.player_start
        self.ghosts = list(self.maze.ghost_starts)
        self.facing = next(d for d in range(4) if self.maze.is_open(*self.pos, d))
        self.lives = MAX_LIVES
        self.won = self.lost = self.viewing_map = self.auto = self.flash = False
        self.msg = ""
        self.seen = np.zeros(self.maze.grid.shape, dtype=bool)  # tiles the player has laid eyes on
        self.solver = MazeSolver()
        self.look()
        self.draw()

    def close(self):
        self.timer.stop()
        plt.close(self.fig)

    # ---------- game logic ----------
    def look(self):
        """Update what the player (and the solver) can see from here."""
        self.obs = self.maze.observe(*self.pos, self.ghosts)
        f = self.pos[0]
        for r, c in self.obs["cells"]:
            self.seen[f, 2 * r:2 * r + 3, 2 * c:2 * c + 3] = True
        self.solver.observe(self.obs)

    def on_key(self, event):
        key = event.key
        if self.won or self.lost:
            if key == "y":
                self.new_game()
            elif key in ("n", "q"):
                self.close()
            return
        if key == "q":
            self.close()
        elif key == "x":
            self.auto = not self.auto
            self.timer.start() if self.auto else self.timer.stop()
            self.draw()
        elif self.auto:
            return  # manual input is ignored while the autopilot drives
        elif key in ("up", "w"):
            self.do("move", self.facing)
        elif key in ("down", "s"):
            self.do("move", (self.facing + 2) % 4)
        elif key in ("left", "a"):
            self.do("turn", -1)
        elif key in ("right", "d"):
            self.do("turn", 1)
        elif key in ("e", "enter"):
            self.do("stairs")
        elif key in (" ", "space"):
            self.do("wait")
        elif key == "m":
            if self.pos == self.maze.map_pos:
                self.viewing_map = not self.viewing_map
                self.msg = ""
            else:
                self.msg = "There is no map here."
            self.draw()

    def do(self, action, arg=None):
        self.msg, self.viewing_map = "", False
        acted = False
        if action == "turn":
            self.facing = (self.facing + arg) % 4
        elif action == "move":
            if self.maze.is_open(*self.pos, arg):
                f, r, c = self.pos
                self.pos = (f, r + DIRS[arg][0], c + DIRS[arg][1])
                acted = True
            else:
                self.msg = "A wall blocks the way."
        elif action == "stairs":
            if self.pos in self.maze.stairs:
                self.pos = (self.maze.stairs[self.pos],) + self.pos[1:]
                acted = True
            else:
                self.msg = "There are no stairs here."
        elif action == "wait":
            acted = True
        if acted:
            self.world_step()
        self.look()
        if self.won:
            self.timer.stop()
        elif self.lost:
            self.timer.stop()
        elif not self.msg:
            if self.pos in self.maze.stairs:
                up = self.maze.stairs[self.pos] > self.pos[0]
                self.msg = f"Stairs {'up' if up else 'down'} - press E"
            elif self.pos == self.maze.map_pos:
                self.msg = "You found a map! Press M to read it."
        self.draw()

    def world_step(self):
        if self.pos == self.maze.goal:
            self.won = True
            return
        self.collide()
        if self.lost:
            return
        for i, (f, r, c) in enumerate(self.ghosts):  # ghosts drift around their floor
            options = [(f, r + DIRS[d][0], c + DIRS[d][1]) for d in self.maze.open_dirs(f, r, c)]
            if options and random.random() < GHOST_SPEED:
                self.ghosts[i] = random.choice(options)
        self.collide()

    def collide(self):
        for i, ghost in enumerate(self.ghosts):
            if ghost == self.pos:
                self.lives -= 1
                self.flash = True
                self.msg = "BOO! A ghost got you!"
                f = ghost[0]
                n = self.size
                spots = [(f, r, c) for r in range(n) for c in range(n)
                         if f != self.pos[0] or abs(r - self.pos[1]) + abs(c - self.pos[2]) >= 4]
                self.ghosts[i] = random.choice(spots)
                if self.lives <= 0:
                    self.lost = True
                    return

    def auto_step(self):
        if self.won or self.lost:
            self.timer.stop()
            return
        action = self.solver.next_action(self.pos, self.obs)
        if action[0] == "move":
            self.facing = action[1]
        self.do(*action)

    # ---------- rendering ----------
    def render_view(self):
        """First-person raycast. Only what a ray can reach is drawn, so nothing leaks."""
        H, W = VIEW_H, VIEW_W
        f, r, c = self.pos
        grid = self.maze.grid[f]
        rows, cols = grid.shape
        tile_r, tile_c = 2 * r + 1, 2 * c + 1
        py, px = tile_r + 0.5, tile_c + 0.5
        dr, dc = DIRS[self.facing]
        plr, plc = dc * 0.66, -dr * 0.66  # camera plane, to the player's right
        det = dr * plc - plr * dc

        img = np.empty((H, W, 3))
        img[:H // 2] = (0.1, 0.1, 0.16)  # ceiling
        img[H // 2:] = (0.2, 0.16, 0.12)  # floor
        zbuf = np.full(W, np.inf)
        wall = np.array(WALL_COLORS[f % len(WALL_COLORS)])

        for x in range(W):
            cam = 2 * x / W - 1
            ray_r, ray_c = dr + plr * cam, dc + plc * cam
            map_r, map_c = tile_r, tile_c
            delta_r = abs(1 / ray_r) if ray_r else np.inf
            delta_c = abs(1 / ray_c) if ray_c else np.inf
            step_r = -1 if ray_r < 0 else 1
            step_c = -1 if ray_c < 0 else 1
            side_r = np.inf if not ray_r else ((py - map_r) if ray_r < 0 else (map_r + 1 - py)) * delta_r
            side_c = np.inf if not ray_c else ((px - map_c) if ray_c < 0 else (map_c + 1 - px)) * delta_c
            while True:
                if side_r < side_c:
                    side_r += delta_r
                    map_r += step_r
                    side = 0
                else:
                    side_c += delta_c
                    map_c += step_c
                    side = 1
                if not (0 <= map_r < rows and 0 <= map_c < cols) or not grid[map_r, map_c]:
                    break
            if side == 0:
                dist = (map_r - py + (1 - step_r) / 2) / ray_r
            else:
                dist = (map_c - px + (1 - step_c) / 2) / ray_c
            dist = max(dist, 1e-3)
            zbuf[x] = dist
            h = min(H, int(H / dist))
            top = (H - h) // 2
            shade = max(0.15, 1 / (1 + 0.2 * dist)) * (1 if side == 0 else 0.7)
            img[top:top + h, x] = wall * shade

        # Sprites (stairs, goal, map, ghosts), far to near, hidden by walls via the z-buffer
        things = []
        for (sf, sr, sc), dest in self.maze.stairs.items():
            if sf == f:
                things.append((sr, sc, "stairs_up" if dest > f else "stairs_down"))
        for pos, key in ((self.maze.goal, "goal"), (self.maze.map_pos, "map")):
            if pos[0] == f:
                things.append((pos[1], pos[2], key))
        things += [(g[1], g[2], "ghost") for g in self.ghosts if g[0] == f]

        projected = []
        for sr, sc, key in things:
            rel_r, rel_c = 2 * sr + 1.5 - py, 2 * sc + 1.5 - px
            a = (rel_r * plc - plr * rel_c) / det  # depth
            b = (dr * rel_c - dc * rel_r) / det    # sideways offset
            if a > 0.2:
                projected.append((a, b, key))
        for a, b, key in sorted(projected, reverse=True):
            tex, scale, lift = SPRITES[key]
            size = int(scale * H / a)
            if size < 2:
                continue
            x0 = int(W / 2 * (1 + b / a)) - size // 2
            bottom = int(H / 2 + 0.5 * H / a - lift * H / a)
            top = bottom - size
            ys = np.arange(max(top, 0), min(bottom, H))
            if len(ys) == 0:
                continue
            ty = np.clip((ys - top) * tex.shape[0] // size, 0, tex.shape[0] - 1)
            shade = max(0.3, 1 / (1 + 0.15 * a))
            for x in range(max(x0, 0), min(x0 + size, W)):
                if zbuf[x] <= a:
                    continue
                col = tex[ty, min(tex.shape[1] - 1, (x - x0) * tex.shape[1] // size)]
                keep = col[:, 3] > 0.5
                img[ys[keep], x] = col[keep, :3] * shade

        if self.flash:
            img = img * 0.5 + np.array([0.5, 0.0, 0.0])
            self.flash = False
        return np.clip(img, 0, 1)

    def draw_minimap(self):
        """Only previously seen tiles, in a window centered on the player (no edge-of-maze clues)."""
        f, r, c = self.pos
        g, s = self.maze.grid[f], self.seen[f]
        img = np.zeros(g.shape + (3,))
        img[s & g] = (0.8, 0.8, 0.75)
        img[s & ~g] = (0.25, 0.25, 0.4)
        marks = [(pos[1:], col) for pos, col in
                 ((self.maze.goal, (0.1, 0.9, 0.3)), (self.maze.map_pos, (1.0, 0.85, 0.2)))
                 if pos[0] == f]
        for (sf, sr, sc), dest in self.maze.stairs.items():
            if sf == f:
                marks.append(((sr, sc), (0.3, 0.85, 1.0) if dest > f else (0.65, 0.4, 1.0)))
        for (mr, mc), col in marks:
            if s[2 * mr + 1, 2 * mc + 1]:
                img[2 * mr + 1, 2 * mc + 1] = col
        R = MINI_R
        img = np.pad(img, ((R, R), (R, R), (0, 0)))
        cy, cx = 2 * r + 1 + R, 2 * c + 1 + R
        ax = self.ax_map
        ax.clear()
        ax.imshow(img[cy - R:cy + R + 1, cx - R:cx + R + 1], interpolation="nearest")
        dr, dc = DIRS[self.facing]
        ax.arrow(R - dc * 0.6, R - dr * 0.6, dc * 1.6, dr * 1.6, head_width=1.3,
                 length_includes_head=True, color="red")
        ax.set_xlim(-0.5, 2 * R + 0.5)
        ax.set_ylim(2 * R + 0.5, -0.5)
        ax.set_title("Explored so far\ncyan/purple=stairs up/down  green=exit  yellow=map",
                     fontsize=7)
        ax.axis("off")

    def draw_map_view(self):
        """The paper map: all floors, with a route to the goal and every staircase marked."""
        m, ax = self.maze, self.ax_view
        n, gap = m.shape[0], 2
        img = np.zeros((n, m.floors * n + (m.floors - 1) * gap, 3))
        offs = [f * (n + gap) for f in range(m.floors)]
        for f, off in enumerate(offs):
            sub = img[:, off:off + n]
            sub[m.grid[f]] = (0.9, 0.85, 0.7)
            sub[~m.grid[f]] = (0.25, 0.2, 0.15)
        ax.imshow(img, interpolation="nearest")
        for f, off in enumerate(offs):
            ax.text(off + n / 2 - 0.5, -1.0, f"Floor {f + 1}", ha="center", fontsize=10)
        for (f, r, c), dest in m.stairs.items():
            ax.text(offs[f] + 2 * c + 1, 2 * r + 1, "▲" if dest > f else "▼", ha="center",
                    va="center", fontsize=8, color="navy")
        gf, gr, gc = m.goal
        ax.text(offs[gf] + 2 * gc + 1, 2 * gr + 1, "★", ha="center", va="center",
                fontsize=14, color="green")
        path = m.map_path
        i = 0
        while i < len(path):  # draw the route one floor at a time
            j = i
            while j + 1 < len(path) and path[j + 1][0] == path[i][0]:
                j += 1
            f = path[i][0]
            ax.plot([offs[f] + 2 * p[2] + 1 for p in path[i:j + 1]],
                    [2 * p[1] + 1 for p in path[i:j + 1]], color="orangered", lw=2)
            if j + 1 < len(path):  # staircase on the route
                nf = path[j + 1][0]
                tag = "UP" if nf > f else "DOWN"
                ax.text(offs[f] + 2 * path[j][2] + 1, 2 * path[j][1] + 1, f"{'▲' if nf > f else '▼'} {tag}",
                        ha="center", va="center", fontsize=8, weight="bold",
                        bbox=dict(facecolor="yellow", edgecolor="black", pad=1))
            i = j + 1
        sf, sr, sc = path[0]
        ax.text(offs[sf] + 2 * sc + 1, 2 * sr + 1, "S", ha="center", va="center", fontsize=9,
                weight="bold", color="white", bbox=dict(facecolor="orangered", pad=1))
        ax.set_ylim(n - 0.5, -2.5)
        ax.set_title("MAP - route from S to the exit (★). Press M to put it away.", fontsize=10)
        ax.axis("off")

    def draw(self):
        ax = self.ax_view
        ax.clear()
        if self.viewing_map:
            self.draw_map_view()
        else:
            ax.imshow(self.render_view(), interpolation="nearest", aspect="auto")
            ax.axis("off")
            if self.pos in self.maze.stairs and not (self.won or self.lost):
                up = self.maze.stairs[self.pos] > self.pos[0]
                ax.text(0.5, 0.1, f"{'▲ Climb up' if up else '▼ Go down'}: press E",
                        transform=ax.transAxes, ha="center", va="center", color="black",
                        fontsize=16, weight="bold",
                        bbox=dict(facecolor="cyan" if up else "violet", alpha=0.9))
            if self.won:
                ax.text(0.5, 0.5, "YOU ESCAPED THE MAZE!\n\nPlay again? (Y / N)",
                        transform=ax.transAxes, ha="center", va="center", color="gold",
                        fontsize=22, weight="bold", bbox=dict(facecolor="black", alpha=0.85))
            elif self.lost:
                ax.text(0.5, 0.5, "GAME OVER\nThe ghosts got you!\n\nPlay again? (Y / N)",
                        transform=ax.transAxes, ha="center", va="center", color="red",
                        fontsize=22, weight="bold", bbox=dict(facecolor="black", alpha=0.85))
        self.draw_minimap()
        hearts = "♥" * self.lives + "♡" * (MAX_LIVES - self.lives)
        self.hud.set_text(f"Floor {self.pos[0] + 1} of {self.floors}\nLives: {hearts}"
                          + ("\nAUTOPILOT" if self.auto else ""))
        self.status.set_text(self.msg)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    size, floors = 8, 3
    try:
        if len(sys.argv) > 1:
            size = int(sys.argv[1])
        if len(sys.argv) > 2:
            floors = int(sys.argv[2])
        if not (3 <= size <= 20 and 1 <= floors <= 5):
            raise ValueError
    except ValueError:
        sys.exit("Usage: python interactive_maze.py [size 3-20] [floors 1-5]")

    print(INSTRUCTIONS)
    game = Game(size, floors)
    plt.show()
