"""Оценка политики на N эпизодах с фиксированными seed (одинаковые карты для всех)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .env.drone_env import DroneCombatEnv
from .policies.strategies import PolicyStrategy


def evaluate(policy: PolicyStrategy, episodes: int = 100, scenario: str = "mixed", seed0: int = 10_000) -> dict:
    env = DroneCombatEnv(scenario=scenario)
    returns, lengths, outcomes, acc = [], [], Counter(), []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy.act(obs, env))
            done = term or trunc
        s = info["summary"]
        returns.append(s["return"])
        lengths.append(s["steps"])
        outcomes[s["outcome"]] += 1
        if s["shots"]:
            acc.append(s["hits"] / s["shots"])
    return {
        "policy": policy.name,
        "scenario": scenario,
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": outcomes["target_destroyed"] / episodes,
        "collision_rate": outcomes["collision"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "mean_length": float(np.mean(lengths)),
        "hit_accuracy": float(np.mean(acc)) if acc else 0.0,
        "outcomes": dict(outcomes),
    }
