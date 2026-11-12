"""Действия над тикетом — паттерн Command.

Маршрутизатор не меняет тикет напрямую, а формирует список команд
(назначить команду, поднять приоритет, добавить тег, открыть инцидент).
Команды исполняются Invoker'ом, пишутся в журнал аудита и поддерживают undo —
оператор может отменить авто-маршрутизацию одним действием.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..datagen import PRIORITIES


@dataclass
class Ticket:
    ticket_id: str
    text: str
    customer_tier: str = "free"
    channel: str = "web"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    category: str | None = None
    category_conf: float = 0.0
    priority: str | None = None
    priority_conf: float = 0.0
    team: str | None = None
    status: str = "new"
    tags: list[str] = field(default_factory=list)
    incident_id: str | None = None
    model: str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class TicketCommand(ABC):
    name = "abstract"

    @abstractmethod
    def execute(self, t: Ticket) -> None: ...

    @abstractmethod
    def undo(self, t: Ticket) -> None: ...

    def describe(self) -> dict:
        return {"command": self.name, **{k: v for k, v in self.__dict__.items() if not k.startswith("_")}}


class AssignTeamCommand(TicketCommand):
    name = "assign_team"

    def __init__(self, team: str, reason: str) -> None:
        self.team, self.reason, self._prev = team, reason, None

    def execute(self, t):
        self._prev, t.team, t.status = t.team, self.team, "assigned"

    def undo(self, t):
        t.team, t.status = self._prev, "new" if self._prev is None else t.status


class SetPriorityCommand(TicketCommand):
    name = "set_priority"

    def __init__(self, priority: str, reason: str) -> None:
        assert priority in PRIORITIES
        self.priority, self.reason, self._prev = priority, reason, None

    def execute(self, t):
        self._prev, t.priority = t.priority, self.priority

    def undo(self, t):
        t.priority = self._prev


class AddTagCommand(TicketCommand):
    name = "add_tag"

    def __init__(self, tag: str) -> None:
        self.tag, self._added = tag, False

    def execute(self, t):
        if self.tag not in t.tags:
            t.tags.append(self.tag)
            self._added = True

    def undo(self, t):
        if self._added:
            t.tags.remove(self.tag)


class LinkIncidentCommand(TicketCommand):
    name = "link_incident"

    def __init__(self, incident_id: str) -> None:
        self.incident_id, self._prev = incident_id, None

    def execute(self, t):
        self._prev, t.incident_id = t.incident_id, self.incident_id

    def undo(self, t):
        t.incident_id = self._prev


class CommandInvoker:
    def __init__(self) -> None:
        self.audit: list[dict] = []
        self._history: dict[str, list[TicketCommand]] = {}

    def run(self, t: Ticket, cmds: list[TicketCommand], actor: str = "router") -> None:
        for c in cmds:
            c.execute(t)
            self._history.setdefault(t.ticket_id, []).append(c)
            self.audit.append({"ticket_id": t.ticket_id, "actor": actor, "at": datetime.now(timezone.utc).isoformat(),
                               **c.describe()})

    def undo_all(self, t: Ticket, actor: str = "operator") -> int:
        cmds = self._history.pop(t.ticket_id, [])
        for c in reversed(cmds):
            c.undo(t)
            self.audit.append({"ticket_id": t.ticket_id, "actor": actor, "at": datetime.now(timezone.utc).isoformat(),
                               "command": f"undo_{c.name}"})
        return len(cmds)
