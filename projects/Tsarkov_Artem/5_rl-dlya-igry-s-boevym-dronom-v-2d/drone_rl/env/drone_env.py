"""DroneCombatEnv — 2D-симулятор боевого дрона (Gymnasium API).

Задача агента: долететь до наземной цели, не врезавшись в препятствия и
стены, и поразить её, выбрав момент атаки (цель в зоне поражения, в секторе
наведения и в прямой видимости). Боезапас ограничен, после выстрела —
перезарядка (см. weapon.py, паттерн State).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..telemetry.events import TelemetryEvent, TelemetrySubject
from .commands import ACTION_TO_COMMAND
from .scenarios import Obstacle, Scenario, ScenarioFactory
from .weapon import Weapon

N_RAYS = 9
RAY_FOV = math.pi  # 180 градусов перед дроном
RAY_RANGE = 30.0
FIRE_RANGE = 25.0
FIRE_CONE = math.radians(15)
DRONE_RADIUS = 1.5
TARGET_HP = 2


@dataclass
class Drone:
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    max_speed: float = 2.0
    accel: float = 0.4
    turn_rate: float = math.radians(12)
    weapon: Weapon = field(default_factory=Weapon)
    fired_this_step: bool = False


@dataclass
class Target:
    x: float
    y: float
    hp: int = TARGET_HP
    vx: float = 0.0
    vy: float = 0.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def ray_circle(ox: float, oy: float, dx: float, dy: float, c: Obstacle) -> float | None:
    fx, fy = ox - c.x, oy - c.y
    b = fx * dx + fy * dy
    cc = fx * fx + fy * fy - c.r * c.r
    disc = b * b - cc
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    return t if t >= 0 else None


class DroneCombatEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, scenario: str = "mixed", max_steps: int = 400, render_mode: str | None = None,
                 telemetry: TelemetrySubject | None = None, moving_target: bool = True) -> None:
        super().__init__()
        self.builder = ScenarioFactory.create(scenario)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.telemetry = telemetry or TelemetrySubject()
        self.moving_target = moving_target
        self.action_space = spaces.Discrete(len(ACTION_TO_COMMAND))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(8 + N_RAYS,), dtype=np.float32)
        self._renderer = None
        self.scenario: Scenario | None = None

    # ------------------------------------------------------------------ helpers
    def _ray(self, ang: float) -> float:
        d = self.drone
        dx, dy = math.cos(ang), math.sin(ang)
        best = RAY_RANGE
        for o in self.scenario.obstacles:
            t = ray_circle(d.x, d.y, dx, dy, o)
            if t is not None and t < best:
                best = t
        s = self.scenario.size
        for t in ((-d.x / dx) if dx < 0 else (s - d.x) / dx if dx > 0 else math.inf,
                  (-d.y / dy) if dy < 0 else (s - d.y) / dy if dy > 0 else math.inf):
            if 0 <= t < best:
                best = t
        return best

    def _line_of_sight(self) -> bool:
        d, t = self.drone, self.target
        dist = math.hypot(t.x - d.x, t.y - d.y)
        if dist < 1e-6:
            return True
        dx, dy = (t.x - d.x) / dist, (t.y - d.y) / dist
        return all((h := ray_circle(d.x, d.y, dx, dy, o)) is None or h > dist for o in self.scenario.obstacles)

    def _rel(self) -> tuple[float, float]:
        d, t = self.drone, self.target
        dist = math.hypot(t.x - d.x, t.y - d.y)
        ang = _wrap(math.atan2(t.y - d.y, t.x - d.x) - d.heading)
        return dist, ang

    def _collided(self) -> bool:
        d, s = self.drone, self.scenario.size
        if not (DRONE_RADIUS <= d.x <= s - DRONE_RADIUS and DRONE_RADIUS <= d.y <= s - DRONE_RADIUS):
            return True
        return any(math.hypot(d.x - o.x, d.y - o.y) < o.r + DRONE_RADIUS for o in self.scenario.obstacles)

    def _obs(self) -> np.ndarray:
        d = self.drone
        dist, ang = self._rel()
        rays = [self._ray(d.heading - RAY_FOV / 2 + i * RAY_FOV / (N_RAYS - 1)) / RAY_RANGE for i in range(N_RAYS)]
        w = d.weapon
        obs = [
            d.speed / d.max_speed,
            min(dist / self.scenario.size, 1.0) * 2 - 1,
            math.sin(ang),
            math.cos(ang),
            1.0 if w.ready else -1.0,
            w.ammo / w.max_ammo,
            self.target.hp / TARGET_HP,
            1.0 if self._line_of_sight() else -1.0,
            *[r * 2 - 1 for r in rays],
        ]
        return np.asarray(obs, dtype=np.float32)

    def can_hit(self) -> bool:
        dist, ang = self._rel()
        return dist <= FIRE_RANGE and abs(ang) <= FIRE_CONE and self._line_of_sight()

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.scenario = self.builder.build(self.np_random)
        dx, dy = self.scenario.drone_start
        tx, ty = self.scenario.target_start
        self.drone = Drone(dx, dy, heading=float(self.np_random.uniform(0, 2 * math.pi)))
        self.target = Target(tx, ty)
        self.steps = 0
        self.episode_id = f"{int(self.np_random.integers(0, 2**48)):012x}"
        self.ep_return = 0.0
        self.hits = 0
        self.shots = 0
        self.trajectory: list[tuple[float, float]] = [(dx, dy)]
        self._prev_dist = self._rel()[0]
        return self._obs(), {"episode_id": self.episode_id, "scenario": self.scenario.name}

    def _move_target(self) -> None:
        if not self.moving_target:
            return
        t = self.target
        if self.steps % 40 == 0:
            a = self.np_random.uniform(0, 2 * math.pi)
            t.vx, t.vy = 0.3 * math.cos(a), 0.3 * math.sin(a)
        nx, ny = t.x + t.vx, t.y + t.vy
        s = self.scenario.size
        blocked = not (5 < nx < s - 5 and 5 < ny < s - 5) or any(
            math.hypot(nx - o.x, ny - o.y) < o.r + 2 for o in self.scenario.obstacles)
        if blocked:
            t.vx, t.vy = -t.vx, -t.vy
        else:
            t.x, t.y = nx, ny

    def step(self, action: int):
        action = int(action)
        d = self.drone
        d.fired_this_step = False
        could_hit = self.can_hit()
        cmd = ACTION_TO_COMMAND[action]
        cmd.execute(d)
        d.weapon.tick() if not d.fired_this_step else None

        d.x += d.speed * math.cos(d.heading)
        d.y += d.speed * math.sin(d.heading)
        self.trajectory.append((d.x, d.y))
        self._move_target()
        self.steps += 1

        reward = -0.005
        dist, ang = self._rel()
        reward += 0.05 * (self._prev_dist - dist)
        self._prev_dist = dist
        terminated, outcome = False, None

        if d.fired_this_step:
            self.shots += 1
            if could_hit:
                self.target.hp -= 1
                self.hits += 1
                reward += 3.0
                if self.target.hp <= 0:
                    reward += 10.0
                    terminated, outcome = True, "target_destroyed"
            else:
                reward -= 0.3
        elif action == 4:  # пытался стрелять во время перезарядки
            reward -= 0.02

        if not terminated and self._collided():
            reward -= 5.0
            terminated, outcome = True, "collision"

        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            outcome = "timeout"
        if not terminated and not truncated and d.weapon.ammo == 0 and self.target.hp > 0 and d.weapon.state.name == "empty":
            terminated, outcome = True, "out_of_ammo"
            reward -= 2.0

        self.ep_return += reward
        self.telemetry.emit(TelemetryEvent("step", self.episode_id, self.steps, {
            "action": cmd.name, "x": round(d.x, 2), "y": round(d.y, 2), "heading": round(d.heading, 3),
            "dist": round(dist, 2), "reward": round(reward, 4), "weapon": d.weapon.state.name,
        }))
        info = {"episode_id": self.episode_id, "outcome": outcome, "hits": self.hits, "shots": self.shots}
        if terminated or truncated:
            summary = {"scenario": self.scenario.name, "outcome": outcome, "steps": self.steps,
                       "return": round(self.ep_return, 3), "hits": self.hits, "shots": self.shots,
                       "success": outcome == "target_destroyed"}
            info["summary"] = summary
            self.telemetry.emit(TelemetryEvent("episode_end", self.episode_id, self.steps, summary))
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        from .render import render_frame
        return render_frame(self)
