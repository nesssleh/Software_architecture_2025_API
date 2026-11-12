"""Цепочка принятия решения по возврату — паттерн Chain of Responsibility.

Запрос на возврат проходит через обработчики по порядку; каждый может
вынести финальное решение или передать дальше:

  AllowlistHandler → HardRuleHandler → VelocityHandler → ModelHandler → ThresholdHandler

Так дешёвые детерминированные проверки отрабатывают до вызова модели,
а правила бизнеса меняются без переобучения.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from .strategies import ScoringStrategy


@dataclass
class ReturnContext:
    return_id: str
    customer_id: str
    features: dict
    amount: float
    score: float | None = None
    decision: str | None = None  # approve | review | block
    decided_by: str | None = None
    reasons: list[str] = field(default_factory=list)


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Handler | None = None

    def set_next(self, h: "Handler") -> "Handler":
        self._next = h
        return h

    def handle(self, ctx: ReturnContext) -> ReturnContext:
        if ctx.decision is None and self._next:
            return self._next.handle(ctx)
        return ctx


class AllowlistHandler(Handler):
    """VIP/сотрудники и мелкие возвраты — автоодобрение."""

    def __init__(self, allowlist: set[str] | None = None, min_amount: float = 10.0) -> None:
        super().__init__()
        self.allowlist = allowlist or set()
        self.min_amount = min_amount

    def handle(self, ctx: ReturnContext) -> ReturnContext:
        if ctx.customer_id in self.allowlist:
            ctx.decision, ctx.decided_by, ctx.score = "approve", "allowlist", 0.0
            ctx.reasons.append("customer in allowlist")
        elif ctx.amount < self.min_amount:
            ctx.decision, ctx.decided_by, ctx.score = "approve", "small_amount", 0.0
            ctx.reasons.append(f"amount < {self.min_amount}")
        return super().handle(ctx)


class HardRuleHandler(Handler):
    """Жёсткие правила: повторные заявления «не получил» при подтверждённой доставке."""

    def __init__(self, max_inr_claims: int = 2) -> None:
        super().__init__()
        self.max_inr = max_inr_claims

    def handle(self, ctx: ReturnContext) -> ReturnContext:
        f = ctx.features
        if f["ret_reason"] == "not_received" and f["n_not_received_claims"] >= self.max_inr:
            ctx.decision, ctx.decided_by, ctx.score = "block", "hard_rule", 1.0
            ctx.reasons.append(f"{f['n_not_received_claims']} prior 'not received' claims on delivered orders")
        return super().handle(ctx)


class VelocityHandler(Handler):
    """Всплеск возвратов за 30 дней — минимум ручная проверка (решение не финальное: добавляет флаг)."""

    def __init__(self, max_30d: int = 6) -> None:
        super().__init__()
        self.max_30d = max_30d

    def handle(self, ctx: ReturnContext) -> ReturnContext:
        if ctx.features["returns_30d"] >= self.max_30d:
            ctx.reasons.append(f"velocity: {ctx.features['returns_30d']} returns in 30d")
        return super().handle(ctx)


class ModelHandler(Handler):
    def __init__(self, strategy: ScoringStrategy) -> None:
        super().__init__()
        self.strategy = strategy

    def handle(self, ctx: ReturnContext) -> ReturnContext:
        ctx.score = self.strategy.score(ctx.features)
        return super().handle(ctx)


class ThresholdHandler(Handler):
    def __init__(self, block: float = 0.85, review: float = 0.5) -> None:
        super().__init__()
        self.block, self.review = block, review

    def handle(self, ctx: ReturnContext) -> ReturnContext:
        s = ctx.score or 0.0
        velocity = any(r.startswith("velocity") for r in ctx.reasons)
        if s >= self.block:
            ctx.decision = "block"
        elif s >= self.review or velocity:
            ctx.decision = "review"
        else:
            ctx.decision = "approve"
        ctx.decided_by = "model"
        ctx.reasons.append(f"score={s:.3f}")
        return ctx


def build_chain(strategy: ScoringStrategy, allowlist: set[str] | None = None,
                block: float = 0.85, review: float = 0.5) -> Handler:
    head = AllowlistHandler(allowlist)
    head.set_next(HardRuleHandler()).set_next(VelocityHandler()).set_next(ModelHandler(strategy)).set_next(
        ThresholdHandler(block, review))
    return head
