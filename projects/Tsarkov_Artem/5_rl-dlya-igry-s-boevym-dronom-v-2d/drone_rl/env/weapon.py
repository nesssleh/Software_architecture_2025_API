"""Состояние оружия дрона — паттерн State.

Оружие может находиться в одном из состояний: Ready, Reloading, Empty.
Каждое состояние само решает, что произойдёт при попытке выстрела и при
очередном такте симуляции, поэтому в коде среды нет разветвлённых if/else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class WeaponState(ABC):
    name: str = "abstract"

    @abstractmethod
    def fire(self, weapon: "Weapon") -> bool:
        """Попытка выстрела. Возвращает True, если выстрел произведён."""

    def tick(self, weapon: "Weapon") -> None:  # noqa: B027 - по умолчанию ничего
        pass


class ReadyState(WeaponState):
    name = "ready"

    def fire(self, weapon: "Weapon") -> bool:
        weapon.ammo -= 1
        weapon.cooldown_left = weapon.cooldown_steps
        weapon.state = EmptyState() if weapon.ammo <= 0 else ReloadingState()
        return True


class ReloadingState(WeaponState):
    name = "reloading"

    def fire(self, weapon: "Weapon") -> bool:
        return False

    def tick(self, weapon: "Weapon") -> None:
        weapon.cooldown_left -= 1
        if weapon.cooldown_left <= 0:
            weapon.cooldown_left = 0
            weapon.state = ReadyState()


class EmptyState(WeaponState):
    name = "empty"

    def fire(self, weapon: "Weapon") -> bool:
        return False


@dataclass
class Weapon:
    ammo: int = 6
    cooldown_steps: int = 8
    cooldown_left: int = 0
    state: WeaponState = None  # type: ignore[assignment]
    max_ammo: int = 6

    def __post_init__(self) -> None:
        self.max_ammo = self.ammo
        if self.state is None:
            self.state = ReadyState()

    def fire(self) -> bool:
        return self.state.fire(self)

    def tick(self) -> None:
        self.state.tick(self)

    @property
    def ready(self) -> bool:
        return isinstance(self.state, ReadyState)
