"""Отрисовка кадра через Pygame (работает и без дисплея: SDL_VIDEODRIVER=dummy)."""
from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

SCALE = 6


def render_frame(env) -> np.ndarray:
    if not pygame.get_init():
        pygame.init()
    s = int(env.scenario.size * SCALE)
    surf = pygame.Surface((s, s))
    surf.fill((235, 240, 228))
    for o in env.scenario.obstacles:
        pygame.draw.circle(surf, (90, 120, 80), (int(o.x * SCALE), int(o.y * SCALE)), int(o.r * SCALE))
    if len(env.trajectory) > 1:
        pygame.draw.lines(surf, (120, 150, 220), False, [(int(x * SCALE), int(y * SCALE)) for x, y in env.trajectory], 2)
    t = env.target
    tx, ty = int(t.x * SCALE), int(t.y * SCALE)
    if t.hp > 0:
        pygame.draw.rect(surf, (200, 60, 50), (tx - 9, ty - 9, 18, 18))
    else:  # цель уничтожена
        pygame.draw.line(surf, (200, 60, 50), (tx - 11, ty - 11), (tx + 11, ty + 11), 4)
        pygame.draw.line(surf, (200, 60, 50), (tx - 11, ty + 11), (tx + 11, ty - 11), 4)
    d = env.drone
    cx, cy = int(d.x * SCALE), int(d.y * SCALE)
    # сектор поражения
    for sgn in (-1, 1):
        a = d.heading + sgn * math.radians(15)
        pygame.draw.line(surf, (240, 180, 60), (cx, cy), (cx + int(25 * SCALE * math.cos(a)), cy + int(25 * SCALE * math.sin(a))), 1)
    pygame.draw.circle(surf, (30, 60, 160), (cx, cy), 9)
    pygame.draw.line(surf, (255, 255, 255), (cx, cy), (cx + int(12 * math.cos(d.heading)), cy + int(12 * math.sin(d.heading))), 3)
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
