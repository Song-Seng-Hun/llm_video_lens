#!/usr/bin/env python3
"""Draw the animated pixel-art banner at the top of the READMEs.

    python3 tools/build_banner.py

writes assets/banner-dark.svg and assets/banner-light.svg: a film strip of eight frames with a
cut, a lens that reads each frame (pose keypoints, a tracked point going from blue to red), the
brief Claude writes from it and the trajectory panel. The pixel helpers, palettes, font and the
Claude sprite come from the profile map (Song-Seng-Hun/Song-Seng-Hun tools/build_worldmap.py), so
the banners look alike. 160x90 grid, 4 px per cell, SMIL animation (GitHub plays it in README
images). Standard library only.
"""

import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"

P = 4                 # px per grid cell
W, H = 160, 90        # grid size

# ---------------------------------------------------------------- palettes (from build_worldmap.py)

COMMON = {
    "skin": "#F1C9A5", "hair": "#3B2A22", "glass": "#1A1A2A",
    "coat": "#4A6FA5", "coat_d": "#38598A", "pants": "#2E3A55", "scarf": "#E8C468",
    "orange": "#D97757", "yellow": "#E3B341", "teal": "#4FA3A5",
    "green": "#6DBE73", "blue": "#5E8FD6", "red": "#E5566B",
    "white": "#F7F5EE", "paper": "#FFFDF6", "ink": "#4A5070",
    "flower1": "#E8A0B4", "flower2": "#E8D27A",
}

PALETTES = {
    "dark": {
        **COMMON,
        "K": "#0B0E1C", "frame": "#B89B5E", "frame_d": "#5E4C2C",
        "sky0": "#0C1026", "sky1": "#111735", "sky2": "#171F45", "sky3": "#1D2855", "sky4": "#253366",
        "star": "#F4EBD9", "star_d": "#7F88B5", "cloud": "#2A3566", "cloud_d": "#222C58",
        "mtn_far": "#1A2350", "g0": "#1F3B35", "g1": "#264840", "g2": "#19312C", "tuft": "#2F5A4D",
        "path": "#5C4B3F", "path_e": "#3F332B", "path_h": "#6E5A4B",
        "wall": "#8B8FA8", "wall_m": "#6C7192", "wall_d": "#4C5172",
        "roof": "#7B4A57", "roof_d": "#5A3442", "wood": "#6B4E3A", "wood_d": "#4A3527",
        "metal": "#9AA3B5", "metal_d": "#5F6880", "win": "#262E4D",
        "lit": "#FFD27A", "glow": ("#FFB85C", 0.32), "smoke": ("#9AA3B5", 0.7),
        "text": "#F4EBD9", "sub": "#8E96C2", "shadow": "#0B0E1C",
    },
    "light": {
        **COMMON,
        "K": "#2B2F45", "frame": "#C9A86A", "frame_d": "#8A6E3E",
        "sky0": "#B9D8EA", "sky1": "#C6E0EE", "sky2": "#D2E7F0", "sky3": "#DDEDF1", "sky4": "#E8F2EF",
        "star": "#FFFFFF", "star_d": "#FFFFFF", "cloud": "#FFFFFF", "cloud_d": "#E3ECF2",
        "mtn_far": "#A9C2D4", "g0": "#B4CE98", "g1": "#C1D8A6", "g2": "#A2BE87", "tuft": "#8CB072",
        "path": "#E2D2AE", "path_e": "#C9B48C", "path_h": "#EEE2C6",
        "wall": "#F2EEE6", "wall_m": "#DAD3C6", "wall_d": "#B9B0A1",
        "roof": "#C97E6C", "roof_d": "#A8655A", "wood": "#A47C5B", "wood_d": "#86624A",
        "metal": "#C6CDD6", "metal_d": "#8E97A6", "win": "#7F95AA",
        "lit": "#FFE08A", "glow": ("#FFC94A", 0.38), "smoke": ("#8E97A6", 0.55),
        "text": "#2B2F45", "sub": "#6A7192", "shadow": "#F7F3E8",
    },
}

# sprite characters -> palette keys
CHARS = {
    "K": "K", "h": "hair", "f": "skin", "G": "glass", "c": "coat", "C": "coat_d",
    "n": "pants", "s": "scarf", "o": "orange", "w": "white", "p": "paper", "i": "ink",
    "y": "yellow", "m": "metal", "M": "metal_d", "l": "lit", "r": "red", "g": "green",
    "W": "wood_d", "v": "wood", "R": "roof", "x": "text", "u": "sub",
}

# ---------------------------------------------------------------- canvas (from build_worldmap.py)


class Canvas:
    def __init__(self):
        self.px = {}

    def set(self, x, y, col):
        self.px[(x, y)] = col

    def rect(self, x, y, w, h, col):
        for j in range(y, y + h):
            for i in range(x, x + w):
                self.px[(i, j)] = col

    def box(self, x, y, w, h, fill, edge="K"):
        self.rect(x, y, w, h, edge)
        if w > 2 and h > 2:
            self.rect(x + 1, y + 1, w - 2, h - 2, fill)

    def sprite(self, rows, x, y):
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch != ".":
                    self.px[(x + i, y + j)] = CHARS[ch]

    def merge(self, other):
        self.px.update(other.px)


def outlined(c, col="K"):
    out = Canvas()
    for (x, y) in c.px:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, y + dy) not in c.px:
                out.px[(x + dx, y + dy)] = col
    out.px.update(c.px)
    return out


def rects(cells):
    """Cover a set of cells with few rectangles (row runs merged downwards)."""
    rows = {}
    for x, y in cells:
        rows.setdefault(y, []).append(x)
    out, open_ = [], {}
    for y in sorted(rows):
        xs = sorted(rows[y])
        runs, start, prev = [], xs[0], xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                runs.append((start, prev - start + 1))
                start = x
            prev = x
        runs.append((start, prev - start + 1))
        nxt = {}
        for key in runs:
            if key in open_ and sum(open_[key]) == y:
                y0, h = open_.pop(key)
                nxt[key] = (y0, h + 1)
            else:
                nxt[key] = (y, 1)
        out += [(k[0], y0, k[1], h) for k, (y0, h) in open_.items()]
        open_ = nxt
    out += [(k[0], y0, k[1], h) for k, (y0, h) in open_.items()]
    return out


def svg_paths(c, pal, unit=P, ox=0, oy=0):
    by_col = {}
    for xy, col in c.px.items():
        by_col.setdefault(col, set()).add(xy)
    out = []
    for col in sorted(by_col):
        d = "".join(f"M{x*unit+ox} {y*unit+oy}h{w*unit}v{h*unit}h{-w*unit}z"
                    for x, y, w, h in sorted(rects(by_col[col]), key=lambda r: (r[1], r[0])))
        fill = pal[col]
        if isinstance(fill, tuple):
            out.append(f'<path fill="{fill[0]}" fill-opacity="{fill[1]}" d="{d}"/>')
        else:
            out.append(f'<path fill="{fill}" d="{d}"/>')
    return "".join(out)

# ---------------------------------------------------------------- animation helpers (from build_worldmap.py)

T = 12.0              # loop length in seconds
DT = 1 / 16           # timeline step in seconds
TICKS = [k * DT for k in range(int(T / DT))]
FADE = T - 1.0        # lights go out and everything resets from here


def _kt(t):
    return f"{t / T:.4f}".rstrip("0").rstrip(".")


def _steps(changes):
    ts, vs = [], []
    for t, v in changes:
        if vs and vs[-1] == v:
            continue
        if ts and abs(t - ts[-1]) < 1e-9:
            vs[-1] = v
            continue
        ts.append(t)
        vs.append(v)
    ts[0] = 0.0
    return ";".join(_kt(t) or "0" for t in ts), ";".join(str(v) for v in vs)


def timeline(attr, changes):
    """Discrete animation on the shared T-second loop."""
    k, v = _steps(changes)
    if attr == "translate":
        return (f'<animateTransform attributeName="transform" type="translate" dur="{T}s" '
                f'repeatCount="indefinite" calcMode="discrete" keyTimes="{k}" values="{v}"/>')
    return (f'<animate attributeName="{attr}" dur="{T}s" repeatCount="indefinite" '
            f'calcMode="discrete" keyTimes="{k}" values="{v}"/>')


def loop(attr, dur, values, keytimes=None, begin=0.0, discrete=True, translate=False):
    """Free-running ambient animation with its own period."""
    kt = f' keyTimes="{keytimes}"' if keytimes else ""
    mode = ' calcMode="discrete"' if discrete else ""
    beg = f' begin="{begin:.2f}s"' if begin else ""
    if translate:
        return (f'<animateTransform attributeName="transform" type="translate" dur="{dur}s"'
                f'{beg} repeatCount="indefinite"{mode}{kt} values="{values}"/>')
    return f'<animate attributeName="{attr}" dur="{dur}s"{beg} repeatCount="indefinite"{mode}{kt} values="{values}"/>'


def group(inner, anim="", opacity=None, transform=None):
    attrs = ""
    if opacity is not None:
        attrs += f' opacity="{opacity}"'
    if transform:
        attrs += f' transform="{transform}"'
    return f"<g{attrs}>{inner}{anim}</g>"


def shown(inner, t_on, t_off):
    """Visible from t_on to t_off on the loop."""
    return group(inner, timeline("opacity", [(0, 0), (t_on, 1), (t_off, 0)]), opacity=0)


def sparkle(pal, x, y, t_on):
    small, big = Canvas(), Canvas()
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        small.set(x + dx, y + dy, "lit")
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
        big.set(x + dx, y + dy, "lit" if abs(dx) + abs(dy) < 2 else "white")
    return (group(svg_paths(small, pal), timeline("opacity", [(0, 0), (t_on, 1), (t_on + 0.2, 0),
                                                            (t_on + 0.45, 1), (t_on + 0.65, 0)]), opacity=0)
            + group(svg_paths(big, pal), timeline("opacity", [(0, 0), (t_on + 0.2, 1), (t_on + 0.45, 0)]),
                    opacity=0))

# ---------------------------------------------------------------- font 5x7 (from build_worldmap.py)


FONT_SRC = """
A .###. #...# #...# ##### #...# #...# #...#
B ####. #...# #...# ####. #...# #...# ####.
C .###. #...# #.... #.... #.... #...# .###.
D ####. #...# #...# #...# #...# #...# ####.
E ##### #.... #.... ####. #.... #.... #####
F ##### #.... #.... ####. #.... #.... #....
G .###. #...# #.... #.### #...# #...# .####
H #...# #...# #...# ##### #...# #...# #...#
I ### .#. .#. .#. .#. .#. ###
J ..### ...#. ...#. ...#. ...#. #..#. .##..
K #...# #..#. #.#.. ##... #.#.. #..#. #...#
L #.... #.... #.... #.... #.... #.... #####
M #...# ##.## #.#.# #.#.# #...# #...# #...#
N #...# #...# ##..# #.#.# #..## #...# #...#
O .###. #...# #...# #...# #...# #...# .###.
P ####. #...# #...# ####. #.... #.... #....
Q .###. #...# #...# #...# #.#.# #..#. .##.#
R ####. #...# #...# ####. #.#.. #..#. #...#
S .#### #.... #.... .###. ....# ....# ####.
T ##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..
U #...# #...# #...# #...# #...# #...# .###.
V #...# #...# #...# #...# #...# .#.#. ..#..
W #...# #...# #...# #.#.# #.#.# #.#.# .#.#.
X #...# #...# .#.#. ..#.. .#.#. #...# #...#
Y #...# #...# .#.#. ..#.. ..#.. ..#.. ..#..
Z ##### ....# ...#. ..#.. .#... #.... #####
0 .###. #...# #..## #.#.# ##..# #...# .###.
1 .#. ##. .#. .#. .#. .#. ###
2 .###. #...# ....# ...#. ..#.. .#... #####
3 ####. ....# ....# .###. ....# ....# ####.
4 ...#. ..##. .#.#. #..#. ##### ...#. ...#.
5 ##### #.... ####. ....# ....# #...# .###.
6 .###. #.... #.... ####. #...# #...# .###.
. . . . . . . #
· . . . # . . .
- ... ... ... ### ... ... ...
"""

FONT = {" ": ["..."] * 7}
for line in FONT_SRC.strip().splitlines():
    ch, *rows = line.split()
    FONT[ch] = rows


def text_width(s):
    return sum(len(FONT[ch][0]) + 1 for ch in s) - 1


def draw_text(c, s, x, y, col, shadow="shadow"):
    for pass_col, dy in ((shadow, 1), (col, 0)):
        if pass_col is None:
            continue
        cx = x
        for ch in s:
            for j, row in enumerate(FONT[ch]):
                for i, v in enumerate(row):
                    if v == "#":
                        c.set(cx + i, y + j + dy, pass_col)
            cx += len(FONT[ch][0]) + 1
    return x + text_width(s)

# ---------------------------------------------------------------- sprites (traveler and Claude from build_worldmap.py)


# Claude: body = cols 1-11 x rows 0-7 from the body's top-left, arms beside it, legs on rows 8-9.
CLAUDE_EYES = {
    "happy": [(2, 2), (3, 3), (2, 4), (10, 2), (9, 3), (10, 4)],   # > <
    "dot": [(3, 2), (3, 3), (9, 2), (9, 3)],
    "blink": [(2, 3), (3, 3), (9, 3), (10, 3)],
    "right": [(4, 2), (4, 3), (10, 2), (10, 3)],
}
_ARM_L, _ARM_R = [(0, 3), (0, 4), (0, 5)], [(12, 3), (12, 4), (12, 5)]
_UP_L, _UP_R = [(0, -2), (0, -1), (0, 0)], [(12, -2), (12, -1), (12, 0)]
CLAUDE_ARMS = {
    "down": _ARM_L + _ARM_R,
    "up": _UP_L + _UP_R,
    "wave1": _UP_L + _ARM_R,
    "wave2": [(0, 1), (-1, 0), (-1, -1)] + _ARM_R,
}
CLAUDE_LEGS = {
    "stand": [(x, y) for x in (2, 4, 8, 10) for y in (8, 9)],
    "stepA": [(2, 8), (2, 9), (8, 8), (8, 9), (4, 8), (10, 8)],
    "stepB": [(4, 8), (4, 9), (10, 8), (10, 9), (2, 8), (8, 8)],
}


def claude_frame(eyes, arms, legs, bob):
    c = Canvas()
    ox, oy = 1, 3 - bob          # two spare rows on top for raised arms
    c.rect(ox + 1, oy, 11, 8, "orange")
    for x, y in CLAUDE_ARMS[arms] + CLAUDE_LEGS[legs]:
        c.set(ox + x, oy + y, "orange")
    for x, y in CLAUDE_EYES[eyes]:
        c.set(ox + x, oy + y, "K")
    return outlined(c)


# ---------------------------------------------------------------- scene

THEME = {
    "dark": {"film": "#141826", "hole": "#3A4266", "shot_a": "#2F6F73", "shot_b": "#5A4A8A",
             "figure": "#F7F5EE", "board": "#F2EEE6", "panel": "#101528", "tint": ("#FFFFFF", 0.14)},
    "light": {"film": "#2B2F45", "hole": "#8E97B5", "shot_a": "#8FCFCB", "shot_b": "#B7A6E0",
              "figure": "#2B2F45", "board": "#FFFDF6", "panel": "#E8EEF5", "tint": ("#FFFFFF", 0.3)},
}
# the tracked point goes from blue (early) to red (late), as in trajectory.png
TRACK = ["#4C8FE0", "#5F83D6", "#7377CB", "#866BC0", "#9A5FB4", "#AE54A0", "#C9508A", "#E5566B"]
for k, col in enumerate(TRACK):
    COMMON[f"t{k}"] = col

FW, FH, GAP = 16, 14, 2                  # frame size and gap, in cells
FX0 = (W - (8 * FW + 7 * GAP)) // 2      # left edge of the first frame
FY0 = 18                                 # top of the frames
CUT_AFTER = 3                            # the cut sits between frame 4 and frame 5
FLOOR = 80                               # floor row; Claude stands on it
CLAUDE_AT = (30, FLOOR - 1)
BOARD = (50, 42, 46, 37)                 # the brief: x, y, w, h
PANEL = (104, 42, 48, 37)                # the trajectory panel: x, y, w, h


def frame_x(k):
    return FX0 + k * (FW + GAP)


# joints per frame, relative to the frame: head, neck, hip, left hand, right hand (tracked), feet
# shot A: someone raises their right hand; shot B: someone walks to the right.
POSES = []
for k, (rh, lh) in enumerate((((12, 8), (4, 8)), ((12, 6), (4, 8)), ((11, 4), (4, 8)), ((10, 2), (4, 8)))):
    POSES.append({"head": (8, 3), "neck": (8, 5), "hip": (8, 9), "lhand": lh, "rhand": rh,
                  "lfoot": (6, 12), "rfoot": (10, 12)})
for k, dx in enumerate((-3, -1, 1, 3)):
    swing = 1 if k % 2 else -1
    POSES.append({"head": (8 + dx, 3), "neck": (8 + dx, 5), "hip": (8 + dx, 9),
                  "lhand": (6 + dx - swing, 8), "rhand": (10 + dx + swing, 8),
                  "lfoot": (6 + dx + swing, 12), "rfoot": (10 + dx - swing, 12)})
BONES = [("head", "neck"), ("neck", "hip"), ("neck", "lhand"), ("neck", "rhand"),
         ("hip", "lfoot"), ("hip", "rfoot")]


def line(c, a, b, col):
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
    for i in range(n + 1):
        c.set(round(a[0] + (b[0] - a[0]) * i / n), round(a[1] + (b[1] - a[1]) * i / n), col)


def draw_room(c):
    bands = ["sky0", "sky1", "sky2", "sky3", "sky4"]
    for y in range(2, FLOOR):
        k = min(4, (y - 2) * 5 // (FLOOR - 2))
        for x in range(2, W - 2):
            c.set(x, y, bands[k])
    for y in range(FLOOR, H - 2):
        for x in range(2, W - 2):
            c.set(x, y, "wood_d" if (y - FLOOR) % 3 == 0 or (x + 7 * ((y - FLOOR) // 3)) % 23 == 0 else "wood")


def draw_strip(c):
    top, bottom = FY0 - 4, FY0 + FH + 3
    c.rect(2, top, W - 4, bottom - top + 1, "film")
    for x in range(4, W - 4, 4):
        c.rect(x, top + 1, 2, 2, "hole")
        c.rect(x, bottom - 2, 2, 2, "hole")
    for k, pose in enumerate(POSES):
        x0 = frame_x(k)
        c.rect(x0, FY0, FW, FH, "shot_a" if k <= CUT_AFTER else "shot_b")
        p = {name: (x0 + jx, FY0 + jy) for name, (jx, jy) in pose.items()}
        for a, b in BONES:
            line(c, p[a], p[b], "figure")
        hx, hy = p["head"]
        c.rect(hx - 1, hy - 1, 3, 3, "figure")
    gx = frame_x(CUT_AFTER) + FW
    c.rect(gx, top, GAP, bottom - top + 1, "red")             # the cut
    tl = bottom + 3                                           # timeline under the strip
    c.rect(FX0, tl, 8 * FW + 7 * GAP, 1, "metal_d")
    c.rect(gx, tl - 1, GAP, 3, "red")


def draw_board(c):
    x, y, w, h = BOARD
    c.box(x, y, w, h, "board")
    c.rect(x + 6, y + h, 2, FLOOR - y - h, "wood_d")          # easel legs
    c.rect(x + w - 8, y + h, 2, FLOOR - y - h, "wood_d")
    c.box(x + w // 2 - 4, y - 2, 9, 3, "metal_d")             # clip


def draw_panel(c):
    x, y, w, h = PANEL
    c.box(x, y, w, h, "panel")
    c.rect(x + 6, y + h, 2, FLOOR - y - h, "wood_d")
    c.rect(x + w - 8, y + h, 2, FLOOR - y - h, "wood_d")


def lens_canvas():
    """A magnifying glass centred on (0, 0): tinted glass, a metal rim and a wooden handle."""
    glass, rim = Canvas(), Canvas()
    for dy in range(-11, 12):
        for dx in range(-11, 12):
            r = math.hypot(dx, dy)
            if r < 9.2:
                glass.set(dx, dy, "tint")
            elif r < 10.6:
                rim.set(dx, dy, "metal" if dx + dy < 0 else "metal_d")
    for i in range(6):
        rim.rect(7 + i, 7 + i, 2, 2, "wood_d" if i < 1 else "wood")
    glass.set(-5, -5, "white")
    glass.set(-4, -6, "white")
    glass.set(-6, -4, "white")
    return glass, outlined(rim)

# ---------------------------------------------------------------- story
# The lens visits the eight frames in turn; each one gets its keypoints and a tracked point, a line
# on the brief and a step on the trajectory. Crossing the cut flashes it. Claude cheers at the end.

DWELL, STEP = 0.6, 0.4
START = 0.5
ARRIVE = [START + k * (DWELL + STEP) for k in range(8)]       # lens settles on frame k
T_CUT = ARRIVE[CUT_AFTER] + DWELL + STEP / 2
DONE = ARRIVE[7] + DWELL


def snap(x, y):
    return f"{round(x) * P} {round(y) * P}"


def lens_x(t):
    centres = [frame_x(k) + FW / 2 for k in range(8)]
    if t <= ARRIVE[0]:
        return centres[0]
    for k in range(7):
        leave = ARRIVE[k] + DWELL
        if t < leave:
            return centres[k]
        if t < ARRIVE[k + 1]:
            return centres[k] + (centres[k + 1] - centres[k]) * (t - leave) / STEP
    return centres[7]


def lens_layer(pal):
    glass, rim = lens_canvas()
    pos = [(t, snap(lens_x(t), FY0 + FH / 2)) for t in TICKS]
    vis = [(0, 0), (START - 0.25, 1), (DONE + 0.5, 0)]
    body = svg_paths(glass, pal) + svg_paths(rim, pal)
    return group(group(body, timeline("opacity", vis), opacity=0), timeline("translate", pos))


def frame_overlays(pal):
    out = ""
    for k, pose in enumerate(POSES):
        x0 = frame_x(k)
        glow = Canvas()                                       # frame border while the lens is on it
        for i in range(-1, FW + 1):
            glow.set(x0 + i, FY0 - 1, "lit")
            glow.set(x0 + i, FY0 + FH, "lit")
        for j in range(FY0 - 1, FY0 + FH + 1):
            glow.set(x0 - 1, j, "lit")
            glow.set(x0 + FW, j, "lit")
        out += shown(svg_paths(glow, pal), ARRIVE[k], ARRIVE[k] + DWELL)
        keys = Canvas()                                       # pose keypoints stay until the reset
        for name, (jx, jy) in pose.items():
            keys.set(x0 + jx, FY0 + jy, f"t{k}" if name == "rhand" else "lit")
        hx, hy = x0 + pose["rhand"][0], FY0 + pose["rhand"][1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            keys.set(hx + dx, hy + dy, f"t{k}")
        tick = Canvas()                                       # its step on the timeline
        tick.rect(x0 + FW // 2 - 1, FY0 + FH + 5, 3, 3, f"t{k}")
        out += shown(svg_paths(keys, pal) + svg_paths(tick, pal), ARRIVE[k] + 0.125, FADE)
    flash = Canvas()
    gx = frame_x(CUT_AFTER) + FW
    flash.rect(gx - 1, FY0 - 4, GAP + 2, FH + 8, "lit")
    out += group(svg_paths(flash, pal), timeline("opacity", [(0, 0), (T_CUT, 1), (T_CUT + 0.15, 0),
                                                           (T_CUT + 0.3, 1), (T_CUT + 0.45, 0)]), opacity=0)
    label = Canvas()                                          # unit 2 px
    draw_text(label, "CUT", (gx + 1) * 2 - text_width("CUT") // 2, (FY0 - 9) * 2, "red")
    out += shown(svg_paths(label, pal, unit=2), T_CUT, FADE)
    return out


def brief_layer(pal):
    x, y, w, h = BOARD
    title = Canvas()                                          # unit 2 px
    draw_text(title, "BRIEF", (x + w // 2) * 2 - text_width("BRIEF") // 2, (y + 3) * 2, "ink", shadow=None)
    out = svg_paths(title, pal, unit=2)
    rows = list(range(8))
    rows.insert(CUT_AFTER + 1, "cut")
    rng = random.Random(5)
    for i, row in enumerate(rows):
        ly = y + 9 + 3 * i
        c = Canvas()
        if row == "cut":
            c.rect(x + 4, ly, w - 8, 1, "red")
            t_on = T_CUT + 0.25
        else:
            c.rect(x + 4, ly - 1, 2, 2, f"t{row}")
            c.rect(x + 8, ly, rng.randrange(18, w - 12), 1, "ink")
            t_on = ARRIVE[row] + 0.375
        out += shown(svg_paths(c, pal), t_on, FADE)
    return out


def trajectory_layer(pal):
    """The tracked point's path, blue to red, broken at the cut (tracks never cross a cut)."""
    x, y, w, h = PANEL
    title = Canvas()                                          # unit 2 px
    draw_text(title, "TRACK", (x + w // 2) * 2 - text_width("TRACK") // 2, (y + 3) * 2, "sub", shadow=None)
    out = svg_paths(title, pal, unit=2)
    pts = []
    for k, pose in enumerate(POSES):
        gx, gy = frame_x(k) + pose["rhand"][0], FY0 + pose["rhand"][1]
        px = x + 3 + (gx - FX0) * (w - 7) / (8 * FW + 7 * GAP)
        py = y + 4 + (gy - FY0) * (h - 9) / FH
        pts.append((round(px), round(py) + 4))
    for k, (px, py) in enumerate(pts):
        c = Canvas()
        if k and k != CUT_AFTER + 1:
            line(c, pts[k - 1], (px, py), f"t{k}")
        c.rect(px - 1, py - 1, 3, 3, f"t{k}")
        out += shown(svg_paths(c, pal), ARRIVE[k] + 0.25, FADE)
    return out


def claude_pose(t, k):
    blink = any(a <= t < a + 0.125 for a in (2.2, 6.4, 10.2))
    if DONE <= t < FADE - 0.5:
        return ("happy", "wave1" if (k // 4) % 2 else "wave2", "stand")
    return ("blink" if blink else "right", "down", "stand")


def claude_layer(pal):
    seq = [(t, claude_pose(t, k)) for k, t in enumerate(TICKS)]
    frames = {pose: claude_frame(*pose, 0) for pose in dict.fromkeys(p for _, p in seq)}
    ox, oy = (CLAUDE_AT[0] - 7) * P, (CLAUDE_AT[1] - 13) * P
    return "".join(group(svg_paths(fc, pal, ox=ox, oy=oy),
                         timeline("opacity", [(t, 1 if p == pose else 0) for t, p in seq]),
                         opacity=1 if pose == seq[0][1] else 0)
                   for pose, fc in frames.items())


def frame_border():
    f = Canvas()
    f.rect(0, 0, W, H, "frame_d")
    f.rect(1, 1, W - 2, H - 2, "frame")
    for y in range(2, H - 2):
        for x in range(2, W - 2):
            f.px.pop((x, y), None)
    for x, y in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        f.px.pop((x, y), None)
    for x, y in ((1, 1), (W - 2, 1), (1, H - 2), (W - 2, H - 2)):
        f.set(x, y, "frame_d")
    return f


def build(theme):
    pal = {**PALETTES[theme], **THEME[theme], **{f"t{k}": c for k, c in enumerate(TRACK)}}
    bg = Canvas()
    draw_room(bg)
    draw_strip(bg)
    draw_board(bg)
    draw_panel(bg)
    title = Canvas()                                          # unit 3 px
    draw_text(title, "LLM VIDEO LENS", W * P // 6 - text_width("LLM VIDEO LENS") // 2, 3, "text")
    desc = ("Pixel-art banner: a magnifying lens moves along a film strip of eight frames with a cut in "
            "the middle; each frame it reads gets pose keypoints and a tracked point that turns from blue "
            "to red, Claude writes a line of the brief for it, and a panel draws the point's path, broken "
            "at the cut.")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W * P} {H * P}" width="{W * P}" '
        f'height="{H * P}" shape-rendering="crispEdges" role="img" aria-labelledby="t d">'
        f'<title id="t">LLM Video Lens</title><desc id="d">{desc}</desc>'
        + svg_paths(bg, pal) + frame_overlays(pal) + brief_layer(pal) + trajectory_layer(pal)
        + claude_layer(pal) + lens_layer(pal)
        + svg_paths(title, pal, unit=3) + svg_paths(frame_border(), pal)
        + "</svg>\n"
    )


def main():
    OUT.mkdir(exist_ok=True)
    for theme in PALETTES:
        path = OUT / f"banner-{theme}.svg"
        path.write_text(build(theme), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
