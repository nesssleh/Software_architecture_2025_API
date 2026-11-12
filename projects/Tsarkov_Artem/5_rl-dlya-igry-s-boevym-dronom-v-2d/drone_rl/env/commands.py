"""Действия агента — паттерн Command.

Дискретное действие политики (int) превращается в объект-команду, который
применяется к дрону. Команды логируются в телеметрию и могут быть
воспроизведены (replay) — это удобно для отладки эпизодов.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .drone_env import Drone


class DroneCommand(ABC):
    name: str = "abstract"

    @abstractmethod
    def execute(self, drone: "Drone") -> None: ...

    def __repr__(self) -> str:
        return f"<{self.name}>"


class HoverCommand(DroneCommand):
    name = "hover"

    def execute(self, drone: "Drone") -> None:
        drone.speed *= 0.6  # активное торможение


class ForwardCommand(DroneCommand):
    name = "forward"

    def execute(self, drone: "Drone") -> None:
        drone.speed = min(drone.speed + drone.accel, drone.max_speed)


class TurnCommand(DroneCommand):
    def __init__(self, direction: int) -> None:
        self.direction = direction
        self.name = "turn_left" if direction > 0 else "turn_right"

    def execute(self, drone: "Drone") -> None:
        drone.heading = (drone.heading + self.direction * drone.turn_rate) % (2 * math.pi)
        drone.speed *= 0.95


class FireCommand(DroneCommand):
    name = "fire"

    def execute(self, drone: "Drone") -> None:
        drone.fired_this_step = drone.weapon.fire()


ACTION_TO_COMMAND: dict[int, DroneCommand] = {
    0: HoverCommand(),
    1: ForwardCommand(),
    2: TurnCommand(+1),
    3: TurnCommand(-1),
    4: FireCommand(),
}
ACTION_NAMES = {k: v.name for k, v in ACTION_TO_COMMAND.items()}
