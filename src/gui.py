"""DarkWatch console GUI (Pygame). Reads the World, never changes game rules.

Keys: SPACE pause | S step | UP/DOWN speed | R reset | H heatmap | G god view | ESC quit
Optional: put a TTF at assets/font.ttf (e.g. Inter, cite it in the report).
"""
import math
import os
import sys

import numpy as np
import pygame

from sim import H, W, World

BG, PANEL, LINE = (10, 18, 32), (17, 28, 46), (38, 56, 84)
TEXT, MUTED = (214, 226, 240), (120, 140, 166)
CYAN, AMBER, RED, WHITE = (0, 200, 230), (255, 176, 32), (255, 84, 84), (235, 242, 250)
CELL, MAP_X, MAP_Y = 22, 20, 70
MAP_W, MAP_H = W * CELL, H * CELL
WIN_W, WIN_H = 1280, 760
PANEL_X, PANEL_W = 920, 340
FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "font.ttf")


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def to_screen(x, y):
    return int(MAP_X + x * CELL), int(MAP_Y + y * CELL)


def rgba_surface(rgb, alpha, size):
    """Turn (H,W,3) colours + (H,W) alpha into a smoothly scaled surface."""
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(s)[:] = rgb.transpose(1, 0, 2).astype(np.uint8)
    pygame.surfarray.pixels_alpha(s)[:] = np.clip(alpha, 0, 255).T.astype(np.uint8)
    return pygame.transform.smoothscale(s, size)


def heat_surface(density):
    stops = np.array([[12, 44, 92], [18, 150, 165], [250, 214, 90]], float)
    t = np.clip(density, 0, 1) * 2
    i = np.minimum(t.astype(int), 1)
    f = (t - i)[..., None]
    return rgba_surface(stops[i] * (1 - f) + stops[i + 1] * f, 40 + 90 * density, (MAP_W, MAP_H))


def draw_ship(surf, cx, cy, heading, color, size=11):
    pts = [(cx + math.cos(heading + a) * size * k, cy + math.sin(heading + a) * size * k)
           for a, k in ((0, 1.0), (2.5, 0.8), (-2.5, 0.8))]
    pygame.draw.polygon(surf, color, pts)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("DarkWatch")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.fonts = {n: self.load_font(n) for n in (13, 15, 18, 28)}
        self.seed, self.speed = 7, 1.0
        self.show_heat, self.god = True, False
        self.reset()

    @staticmethod
    def load_font(size):
        if os.path.exists(FONT_PATH):
            try:
                return pygame.font.Font(FONT_PATH, size)
            except (OSError, pygame.error):
                pass
        return pygame.font.SysFont("segoeui,helveticaneue,helvetica,arial", size)

    def reset(self):
        self.world = World(seed=self.seed)
        self.heat = heat_surface(self.world.ocean.density)
        self.disp = np.zeros((H, W))
        self.acc, self.paused = 0.0, False

    def step_dt(self):
        return 1.0 / (6 * self.speed)

    def text(self, s, size, color, pos):
        self.screen.blit(self.fonts[size].render(s, True, color), pos)

    def belief_target(self):
        t = np.zeros((H, W))
        for v in self.world.vessels:
            if not v.visible:
                t = np.maximum(t, v.belief.p / v.belief.p.max())
        return t

    def handle(self, e):
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()
        if e.type != pygame.KEYDOWN:
            return
        if e.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif e.key == pygame.K_s and self.paused:
            self.world.step()
            self.acc = self.step_dt()
        elif e.key == pygame.K_UP:
            self.speed = min(4.0, self.speed * 1.5)
        elif e.key == pygame.K_DOWN:
            self.speed = max(0.25, self.speed / 1.5)
        elif e.key == pygame.K_r:
            self.seed += 1
            self.reset()
        elif e.key == pygame.K_h:
            self.show_heat = not self.show_heat
        elif e.key == pygame.K_g:
            self.god = not self.god

    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000
            for e in pygame.event.get():
                self.handle(e)
            if not self.paused:
                self.acc += dt
                while self.acc >= self.step_dt():
                    self.world.step()
                    self.acc -= self.step_dt()
            self.draw()
            pygame.display.flip()

    # ---------- drawing ----------
    def draw(self):
        w, s = self.world, self.screen
        a = 1.0 if self.paused else min(1.0, self.acc / self.step_dt())
        s.fill(BG)
        self.text("DARKWATCH", 28, CYAN, (20, 12))
        self.text("Patrol planning under uncertainty  |  synthetic demo data", 13, MUTED, (22, 48))
        self.draw_map(a)
        self.draw_panel()
        self.draw_tooltip()

    def draw_map(self, a):
        w, s = self.world, self.screen
        pygame.draw.rect(s, (8, 14, 26), (MAP_X, MAP_Y, MAP_W, MAP_H))
        if self.show_heat:
            s.blit(self.heat, (MAP_X, MAP_Y))
        for i in range(0, W + 1, 5):
            x = MAP_X + i * CELL
            pygame.draw.line(s, LINE, (x, MAP_Y), (x, MAP_Y + MAP_H), 1)
            self.text(str(i), 13, MUTED, (x - 6, MAP_Y + MAP_H + 4))
        for j in range(0, H + 1, 5):
            pygame.draw.line(s, LINE, (MAP_X, MAP_Y + j * CELL), (MAP_X + MAP_W, MAP_Y + j * CELL), 1)
        pygame.draw.rect(s, LINE, (MAP_X, MAP_Y, MAP_W, MAP_H), 2)

        self.disp += (self.belief_target() - self.disp) * 0.2
        glow = np.empty((H, W, 3))
        glow[:] = AMBER
        s.blit(rgba_surface(glow, (self.disp ** 0.6) * 200, (MAP_W, MAP_H)), (MAP_X, MAP_Y))

        for v in w.vessels:
            n = len(v.trail)
            if v.visible:
                for i, (tx, ty) in enumerate(v.trail):
                    pygame.draw.circle(s, mix((8, 14, 26), WHITE, 0.6 * i / max(1, n)), to_screen(tx, ty), 2)
                pygame.draw.circle(s, WHITE, to_screen(lerp(v.px, v.x, a), lerp(v.py, v.y, a)), 4)
            else:
                pygame.draw.circle(s, AMBER, to_screen(*v.last_known), 7, 2)
                if self.god:
                    pygame.draw.circle(s, RED, to_screen(v.x, v.y), 3)

        p = w.patrol
        px, py = to_screen(lerp(p.px, p.x, a), lerp(p.py, p.y, a))
        if p.target:
            tx, ty = to_screen(*p.target)
            d = math.hypot(tx - px, ty - py)
            for k in range(0, int(d), 14):
                f1, f2 = k / d, min(d, k + 7) / d
                pygame.draw.line(s, CYAN, (lerp(px, tx, f1), lerp(py, ty, f1)),
                                 (lerp(px, tx, f2), lerp(py, ty, f2)), 2)
            pygame.draw.circle(s, CYAN, (tx, ty), 5, 1)
        frac = p.fuel / 100.0
        if frac > 0.01:
            pygame.draw.arc(s, CYAN if frac > 0.3 else RED, (px - 18, py - 18, 36, 36),
                            math.pi / 2, math.pi / 2 + 2 * math.pi * frac, 3)
        draw_ship(s, px, py, p.heading, CYAN)

    def draw_panel(self):
        w, s, p = self.world, self.screen, self.world.patrol
        pygame.draw.rect(s, PANEL, (PANEL_X, MAP_Y, PANEL_W, MAP_H), border_radius=12)
        x, y = PANEL_X + 20, MAP_Y + 18
        dark = sum(1 for v in w.vessels if not v.visible)
        rows = [("STRATEGY", p.name, CYAN), ("STEP", str(w.step_count), TEXT),
                ("DETECTIONS", str(w.detections), TEXT),
                ("VESSELS DARK", f"{dark} / {len(w.vessels)}", AMBER if dark else TEXT),
                ("SPEED", f"{self.speed:.2f}x" + ("  (paused)" if self.paused else ""), TEXT)]
        for label, val, col in rows:
            self.text(label, 13, MUTED, (x, y))
            self.text(val, 18, col, (x, y + 16))
            y += 52
        self.text("FUEL" + ("  - EMPTY (press R)" if p.fuel <= 0 else ""), 13, MUTED, (x, y))
        pygame.draw.rect(s, LINE, (x, y + 20, 300, 12), border_radius=6)
        pygame.draw.rect(s, CYAN if p.fuel > 30 else RED,
                         (x, y + 20, int(3 * p.fuel), 12), border_radius=6)
        y += 52
        self.text("CUMULATIVE DETECTIONS", 13, MUTED, (x, y))
        r = pygame.Rect(x, y + 22, 300, 100)
        pygame.draw.rect(s, (8, 14, 26), r, border_radius=6)
        h = w.history[-300:]
        if len(h) > 1:
            top = max(5, max(h))
            pts = [(r.x + i * 300 / (len(h) - 1), r.bottom - 4 - (v / top) * (r.h - 8))
                   for i, v in enumerate(h)]
            pygame.draw.lines(s, CYAN, False, pts, 2)
        ly = MAP_Y + MAP_H - 98
        for i, (txt, col) in enumerate([("white dot: vessel with AIS", WHITE),
                                        ("amber ring: last known position", AMBER),
                                        ("amber glow: belief (where it may be)", AMBER),
                                        ("cyan ship: patrol, ring = fuel", CYAN)]):
            pygame.draw.circle(s, col, (x + 5, ly + i * 20 + 8), 4)
            self.text(txt, 13, MUTED, (x + 18, ly + i * 20))
        self.text("SPACE pause  S step  UP/DOWN speed  R reset  H heat  G god", 13, MUTED,
                  (MAP_X, WIN_H - 22))

    def draw_tooltip(self):
        mx, my = pygame.mouse.get_pos()
        cx, cy = (mx - MAP_X) // CELL, (my - MAP_Y) // CELL
        if not (0 <= cx < W and 0 <= cy < H):
            return
        mass = sum(v.belief.p[cy, cx] for v in self.world.vessels if not v.visible)
        lines = [f"cell ({cx},{cy})", f"fishing density {self.world.ocean.density[cy, cx]:.2f}",
                 f"belief mass {mass:.3f}"]
        box = pygame.Rect(min(mx + 14, WIN_W - 190), min(my + 14, WIN_H - 70), 176, 58)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=8)
        pygame.draw.rect(self.screen, LINE, box, 1, border_radius=8)
        for i, t in enumerate(lines):
            self.text(t, 13, TEXT, (box.x + 10, box.y + 8 + i * 16))


if __name__ == "__main__":
    App().run()