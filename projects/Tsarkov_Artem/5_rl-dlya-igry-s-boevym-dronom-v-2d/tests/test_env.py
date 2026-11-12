import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from drone_rl.env.commands import ACTION_TO_COMMAND, FireCommand
from drone_rl.env.drone_env import DroneCombatEnv
from drone_rl.env.scenarios import ScenarioFactory
from drone_rl.env.weapon import EmptyState, ReadyState, ReloadingState, Weapon
from drone_rl.evaluate import evaluate
from drone_rl.policies.strategies import HeuristicPolicy, RandomPolicy
from drone_rl.telemetry.events import MemoryObserver, TelemetrySubject


def test_gymnasium_api_compliance():
    check_env(DroneCombatEnv(), skip_render_check=True)


def test_observation_in_bounds():
    env = DroneCombatEnv()
    obs, _ = env.reset(seed=1)
    for _ in range(100):
        obs, *_rest, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_weapon_state_machine():
    w = Weapon(ammo=2, cooldown_steps=2)
    assert isinstance(w.state, ReadyState)
    assert w.fire() and isinstance(w.state, ReloadingState)
    assert not w.fire()  # во время перезарядки выстрел невозможен
    w.tick(); w.tick()
    assert isinstance(w.state, ReadyState)
    assert w.fire() and isinstance(w.state, EmptyState) and w.ammo == 0


def test_scenario_factory():
    assert set(ScenarioFactory.available()) >= {"open", "forest", "corridor", "mixed"}
    with pytest.raises(ValueError):
        ScenarioFactory.create("moon")
    sc = ScenarioFactory.create("open").build(np.random.default_rng(0))
    assert sc.obstacles == []


def test_hit_when_aligned():
    env = DroneCombatEnv(scenario="open", moving_target=False)
    env.reset(seed=0)
    env.drone.x, env.drone.y = 50, 50
    env.target.x, env.target.y = 60, 50
    env.drone.heading = 0.0
    env._prev_dist = 10
    assert env.can_hit()
    _, r, term, _, info = env.step(4)
    assert info["hits"] == 1 and r > 2


def test_collision_terminates():
    env = DroneCombatEnv(scenario="open")
    env.reset(seed=0)
    env.drone.x, env.drone.y, env.drone.heading, env.drone.speed = 1.6, 50, math.pi, 2.0
    _, r, term, _, info = env.step(1)
    assert term and info["outcome"] == "collision" and r < -4


def test_command_mapping():
    assert len(ACTION_TO_COMMAND) == 5 and isinstance(ACTION_TO_COMMAND[4], FireCommand)


def test_telemetry_observer_receives_events():
    subj = TelemetrySubject()
    mem = MemoryObserver()
    subj.attach(mem)
    env = DroneCombatEnv(telemetry=subj)
    env.reset(seed=3)
    done = False
    while not done:
        *_, term, trunc, _ = env.step(0)
        done = term or trunc
    assert mem.events[-1].type == "episode_end"
    assert all(e.episode_id == mem.events[0].episode_id for e in mem.events)


def test_heuristic_beats_random():
    r = evaluate(RandomPolicy(), 30)
    h = evaluate(HeuristicPolicy(), 30)
    assert h["success_rate"] > r["success_rate"]
