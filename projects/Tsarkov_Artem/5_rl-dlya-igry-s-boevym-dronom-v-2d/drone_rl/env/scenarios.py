"""Генерация сценариев (карт) — паттерн Factory.

ScenarioFactory по имени создаёт генератор карты: «open» — без препятствий,
«forest» — случайные круглые препятствия, «corridor» — стена с проходом.
Тренер обучает на смеси сценариев (curriculum), инференс может выбрать любой.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    x: float
    y: float
    r: float


@dataclass
class Scenario:
    name: str
    size: float
    obstacles: list[Obstacle] = field(default_factory=list)
    drone_start: tuple[float, float] = (10.0, 10.0)
    target_start: tuple[float, float] = (90.0, 90.0)


class ScenarioBuilder:
    name = "base"

    def __init__(self, size: float = 100.0) -> None:
        self.size = size

    def _free_point(self, rng: np.random.Generator, obstacles: list[Obstacle], margin: float) -> tuple[float, float]:
        for _ in range(200):
            p = rng.uniform(margin, self.size - margin, size=2)
            if all(np.hypot(p[0] - o.x, p[1] - o.y) > o.r + margin for o in obstacles):
                return float(p[0]), float(p[1])
        return self.size / 2, self.size / 2

    def obstacles(self, rng: np.random.Generator) -> list[Obstacle]:
        return []

    def build(self, rng: np.random.Generator) -> Scenario:
        obs = self.obstacles(rng)
        drone = self._free_point(rng, obs, 6.0)
        for _ in range(100):
            target = self._free_point(rng, obs, 6.0)
            if np.hypot(target[0] - drone[0], target[1] - drone[1]) > 40:
                break
        return Scenario(self.name, self.size, obs, drone, target)


class OpenScenario(ScenarioBuilder):
    name = "open"


class ForestScenario(ScenarioBuilder):
    name = "forest"

    def __init__(self, size: float = 100.0, n_obstacles: int = 7) -> None:
        super().__init__(size)
        self.n = n_obstacles

    def obstacles(self, rng: np.random.Generator) -> list[Obstacle]:
        return [Obstacle(float(rng.uniform(15, 85)), float(rng.uniform(15, 85)), float(rng.uniform(4, 9))) for _ in range(self.n)]


class CorridorScenario(ScenarioBuilder):
    name = "corridor"

    def obstacles(self, rng: np.random.Generator) -> list[Obstacle]:
        gap = rng.uniform(25, 75)
        return [Obstacle(50.0, float(y), 4.0) for y in np.arange(4, 100, 7) if abs(y - gap) > 12]

    def build(self, rng: np.random.Generator) -> Scenario:
        obs = self.obstacles(rng)
        return Scenario(self.name, self.size, obs, (float(rng.uniform(8, 30)), float(rng.uniform(10, 90))),
                        (float(rng.uniform(70, 92)), float(rng.uniform(10, 90))))


class MixedScenario(ScenarioBuilder):
    """Curriculum: случайно выбирает один из сценариев."""

    name = "mixed"

    def __init__(self, size: float = 100.0) -> None:
        super().__init__(size)
        self.builders = [OpenScenario(size), ForestScenario(size), CorridorScenario(size)]
        self.p = [0.2, 0.5, 0.3]

    def build(self, rng: np.random.Generator) -> Scenario:
        b = self.builders[rng.choice(len(self.builders), p=self.p)]
        return b.build(rng)


class ScenarioFactory:
    _registry: dict[str, type[ScenarioBuilder]] = {
        "open": OpenScenario,
        "forest": ForestScenario,
        "corridor": CorridorScenario,
        "mixed": MixedScenario,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> ScenarioBuilder:
        try:
            return cls._registry[name](**kwargs)
        except KeyError as e:
            raise ValueError(f"Unknown scenario '{name}'. Available: {sorted(cls._registry)}") from e

    @classmethod
    def register(cls, name: str, builder: type[ScenarioBuilder]) -> None:
        cls._registry[name] = builder

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)
