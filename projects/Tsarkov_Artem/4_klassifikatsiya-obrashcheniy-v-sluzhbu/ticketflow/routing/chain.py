"""Маршрутизация тикета — паттерн Chain of Responsibility.

SecurityHandler → OutageHandler → VipHandler → LowConfidenceHandler → CategoryHandler

Каждый обработчик может добавить команды и/или остановить цепочку. Это позволяет
бизнесу добавлять правила (например, «всё про утечки — в security, даже если модель
сказала billing») без переобучения модели.
"""
from __future__ import annotations

import re
import time
from abc import ABC
from collections import deque
from dataclasses import dataclass, field

from ..datagen import PRIORITIES, TEAM_BY_CATEGORY
from .commands import (AddTagCommand, AssignTeamCommand, LinkIncidentCommand, SetPriorityCommand, Ticket,
                       TicketCommand)


@dataclass
class RoutingContext:
    ticket: Ticket
    commands: list[TicketCommand] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    done: bool = False


class RouteHandler(ABC):
    name = "handler"

    def __init__(self) -> None:
        self._next: RouteHandler | None = None

    def set_next(self, h: "RouteHandler") -> "RouteHandler":
        self._next = h
        return h

    def handle(self, ctx: RoutingContext) -> RoutingContext:
        if not ctx.done and self._next:
            return self._next.handle(ctx)
        return ctx


class SecurityHandler(RouteHandler):
    name = "security"
    RX = re.compile(r"взлом|фишинг|уязвим|утечк|чуж(ие|ой) (устройств|вход)|незнаком\w* ip", re.I)

    def handle(self, ctx):
        t = ctx.ticket
        if t.category == "security" or self.RX.search(t.text):
            ctx.commands += [AssignTeamCommand("security_team", "security signal"), AddTagCommand("security")]
            if PRIORITIES.index(t.priority or "low") < 2:
                ctx.commands.append(SetPriorityCommand("high", "security floor"))
            ctx.trace.append(self.name)
            ctx.done = True
        return super().handle(ctx)


class CounterStore:
    """Скользящее окно событий (Redis ZSET в проде; in-memory здесь)."""

    def __init__(self, redis=None) -> None:
        self.redis = redis
        self.mem: dict[str, deque] = {}

    def hit(self, key: str, now: float, window: float) -> int:
        if self.redis is not None:
            k = f"win:{key}"
            p = self.redis.pipeline()
            p.zadd(k, {f"{now}": now})
            p.zremrangebyscore(k, 0, now - window)
            p.zcard(k)
            p.expire(k, int(window) + 60)
            return int(p.execute()[2])
        q = self.mem.setdefault(key, deque())
        q.append(now)
        while q and q[0] < now - window:
            q.popleft()
        return len(q)


class OutageHandler(RouteHandler):
    """Всплеск однотипных технических тикетов → массовый инцидент: приоритет critical, привязка к инциденту."""

    name = "outage"

    def __init__(self, counters: CounterStore, threshold: int = 8, window_s: float = 600, clock=time.time) -> None:
        super().__init__()
        self.counters, self.threshold, self.window, self.clock = counters, threshold, window_s, clock
        self.incidents: dict[str, str] = {}

    def handle(self, ctx):
        t = ctx.ticket
        if t.category == "technical":
            now = self.clock()
            n = self.counters.hit("technical", now, self.window)
            if n >= self.threshold:
                bucket = int(now // self.window)
                inc = self.incidents.setdefault(f"technical:{bucket}", f"INC-{bucket}")
                ctx.commands += [LinkIncidentCommand(inc), SetPriorityCommand("critical", f"{n} tech tickets / {int(self.window)}s"),
                                 AddTagCommand("mass_incident")]
                ctx.trace.append(self.name)
        return super().handle(ctx)


class VipHandler(RouteHandler):
    name = "vip"

    def handle(self, ctx):
        t = ctx.ticket
        if t.customer_tier == "enterprise":
            ctx.commands.append(AddTagCommand("enterprise"))
            if PRIORITIES.index(t.priority or "low") < 2:
                ctx.commands.append(SetPriorityCommand("high", "enterprise SLA"))
            ctx.trace.append(self.name)
        return super().handle(ctx)


class LowConfidenceHandler(RouteHandler):
    name = "low_confidence"

    def __init__(self, min_conf: float = 0.55) -> None:
        super().__init__()
        self.min_conf = min_conf

    def handle(self, ctx):
        if ctx.ticket.category_conf < self.min_conf:
            ctx.commands += [AssignTeamCommand("l1_triage", f"model confidence {ctx.ticket.category_conf:.2f}"),
                             AddTagCommand("needs_triage")]
            ctx.trace.append(self.name)
            ctx.done = True
        return super().handle(ctx)


class CategoryHandler(RouteHandler):
    name = "category"

    def handle(self, ctx):
        ctx.commands.append(AssignTeamCommand(TEAM_BY_CATEGORY[ctx.ticket.category], f"category={ctx.ticket.category}"))
        ctx.trace.append(self.name)
        ctx.done = True
        return ctx


def build_router(counters: CounterStore | None = None, min_conf: float = 0.55, outage_threshold: int = 8,
                 clock=time.time) -> RouteHandler:
    head = SecurityHandler()
    head.set_next(OutageHandler(counters or CounterStore(), outage_threshold, clock=clock)).set_next(VipHandler()) \
        .set_next(LowConfidenceHandler(min_conf)).set_next(CategoryHandler())
    return head
