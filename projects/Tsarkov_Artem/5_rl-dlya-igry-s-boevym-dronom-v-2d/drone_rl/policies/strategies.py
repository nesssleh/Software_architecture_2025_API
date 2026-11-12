"""Политики управления — паттерн Strategy.

Все политики реализуют один интерфейс act(obs, env) -> action, поэтому
симулятор, inference-сервис и оценка (evaluate) работают с любой из них:
случайной (baseline), эвристической (baseline) и обученной PPO.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class PolicyStrategy(ABC):
    name = "abstract"

    @abstractmethod
    def act(self, obs: np.ndarray, env=None) -> int: ...


class RandomPolicy(PolicyStrategy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env=None) -> int:
        return int(self.rng.integers(0, 5))


class HeuristicPolicy(PolicyStrategy):
    """«Наводись и лети»: поворот на цель, полёт вперёд, выстрел в секторе.

    Использует только наблюдение (obs) — тот же объём информации, что у агента.
    Есть простое уклонение по центральному лучу, но без планирования пути.
    """

    name = "heuristic"

    def act(self, obs, env=None) -> int:
        sin_a, cos_a = obs[2], obs[3]
        dist = (obs[1] + 1) / 2 * 100
        ready, los = obs[4] > 0, obs[7] > 0
        rays = (obs[8:] + 1) / 2 * 30
        ang = np.arctan2(sin_a, cos_a)
        if ready and los and dist <= 25 and abs(ang) <= np.radians(15):
            return 4
        center = rays[len(rays) // 2 - 1: len(rays) // 2 + 2].min()
        if center < 6:  # препятствие прямо по курсу — отворачиваем в сторону свободных лучей
            left, right = rays[len(rays) // 2 + 1:].mean(), rays[: len(rays) // 2].mean()
            return 2 if left > right else 3
        if abs(ang) > np.radians(10):
            return 2 if ang > 0 else 3
        if dist < 12:
            return 0
        return 1


class PPOPolicy(PolicyStrategy):
    name = "ppo"

    def __init__(self, path: str | Path, deterministic: bool = True) -> None:
        from stable_baselines3 import PPO

        self.model = PPO.load(str(path), device="cpu")
        self.deterministic = deterministic

    def act(self, obs, env=None) -> int:
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)


def make_policy(name: str, model_path: str | None = None) -> PolicyStrategy:
    if name == "random":
        return RandomPolicy()
    if name == "heuristic":
        return HeuristicPolicy()
    if name == "ppo":
        if not model_path:
            raise ValueError("model_path is required for ppo policy")
        return PPOPolicy(model_path)
    raise ValueError(f"unknown policy {name}")
