"""Синтетический поток обращений в поддержку SaaS/e-commerce сервиса (RU) с разметкой.

Разметка: category (7 классов), priority (low/medium/high/critical), team (маршрут).
Тексты собираются из шаблонов с вариативностью: разговорный стиль, опечатки,
лишние детали, смешанные темы (например, «не проходит оплата и ещё не могу войти»),
что делает задачу неидеально разделимой.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

CATEGORIES = ["billing", "technical", "account_access", "delivery", "refund", "feature_request", "security"]
PRIORITIES = ["low", "medium", "high", "critical"]
TEAM_BY_CATEGORY = {"billing": "billing_team", "technical": "l2_tech", "account_access": "identity_team",
                    "delivery": "logistics_team", "refund": "billing_team", "feature_request": "product_team",
                    "security": "security_team"}

T = {
    "billing": ["Списали деньги дважды за подписку {plan}", "Не проходит оплата картой, ошибка {code}",
                "Почему в счёте за {month} сумма больше чем обычно?", "Не пришёл чек об оплате заказа {order}",
                "Как сменить способ оплаты на {pay}?", "Выставили счёт за тариф {plan}, хотя я его не подключал",
                "Автопродление списало оплату, а я отключал", "Нужны закрывающие документы и акт сверки за {month}"],
    "technical": ["Приложение вылетает при открытии раздела {section}", "Ошибка {code} при сохранении {section}",
                  "Страница {section} грузится по минуте, всё тормозит", "Не работает выгрузка отчёта в {fmt}",
                  "После обновления пропали данные в {section}", "API возвращает {code} на запрос {endpoint}",
                  "Белый экран в мобильном приложении на {os}", "Не приходят push-уведомления на {os}"],
    "account_access": ["Не могу войти в аккаунт, пишет неверный пароль", "Не приходит смс с кодом для входа",
                       "Заблокировали учётную запись без объяснения", "Как восстановить доступ, потерял телефон",
                       "Не приходит письмо для сброса пароля на {email}", "Двухфакторка не принимает код из приложения",
                       "Хочу сменить email в профиле, а старый уже не доступен"],
    "delivery": ["Где мой заказ {order}? Статус не меняется {days} дней", "Курьер не приехал в назначенное время",
                 "Заказ {order} привезли не тот", "Посылка застряла на сортировке в {city}",
                 "Можно ли изменить адрес доставки для {order}?", "Трек-номер {order} не отслеживается",
                 "Пункт выдачи в {city} закрыт, как забрать заказ"],
    "refund": ["Хочу вернуть деньги за заказ {order}", "Оформил возврат {days} дней назад, деньги так и не пришли",
               "Верните средства за отменённую подписку {plan}", "Товар бракованный, прошу возврат",
               "Отказался от заказа {order} при получении, когда вернут оплату?",
               "Возврат одобрили, но сумма меньше чем я платил"],
    "feature_request": ["Добавьте, пожалуйста, тёмную тему", "Было бы удобно выгружать отчёты в {fmt}",
                        "Предлагаю сделать интеграцию с {integration}", "Не хватает фильтра по дате в разделе {section}",
                        "Хотелось бы API для {section}", "Сделайте возможность нескольких пользователей в одном аккаунте"],
    "security": ["В аккаунт кто-то зашёл не я, вижу чужие устройства", "Пришло письмо от вашего имени с просьбой ввести пароль, это фишинг?",
                 "С моей карты списали деньги после входа с незнакомого IP", "Подозреваю утечку данных, мне звонили и знали номер заказа {order}",
                 "Нашёл уязвимость в {endpoint}, можно получить доступ к чужим данным",
                 "Взломали аккаунт, сменили пароль и почту"],
}
URGENT = ["СРОЧНО!", "Очень срочно,", "Помогите срочно,", "Горит!", "Уже третий раз пишу!", "Работа стоит,"]
CALM = ["Здравствуйте.", "Добрый день!", "Привет,", "Подскажите,", "Вопрос:", ""]
TAILS = ["Спасибо.", "Жду ответа.", "Заранее спасибо!", "Это уже не первый раз.", "Иначе пойду в суд.",
         "У нас вся команда не может работать.", "Не критично, но неудобно.", "", "", ""]
FILL = dict(plan=["Pro", "Business", "Basic", "Premium"], code=["500", "403", "E-1042", "502", "timeout", "404"],
            month=["март", "апрель", "май", "июнь"], order=["#48213", "№1931-77", "A-55120", "#90012"],
            pay=["СБП", "корпоративную карту", "счёт для юрлица"], section=["Отчёты", "Склад", "Клиенты", "Аналитика", "Настройки"],
            fmt=["Excel", "CSV", "PDF"], endpoint=["/api/v2/orders", "/export", "/users/me"], os=["Android", "iOS"],
            email=["почту gmail", "рабочую почту"], days=["5", "10", "14", "3"], city=["Казани", "Екатеринбурге", "Москве"],
            integration=["1С", "Битрикс24", "amoCRM", "Telegram"])


def _typo(rng, s: str, p: float) -> str:
    chars = list(s)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and rng.random() < p:
            op = rng.integers(3)
            if op == 0:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == 1:
                chars[i] = ""
            else:
                chars[i] = chars[i] * 2
    return "".join(chars)


def _fill(rng, tpl: str) -> str:
    return tpl.format(**{k: str(rng.choice(v)) for k, v in FILL.items()})


def label_priority(rng, cat: str, urgent: bool, tier: str, outage_hint: bool) -> str:
    base = {"security": 2, "technical": 1, "account_access": 1, "billing": 1, "refund": 1, "delivery": 1,
            "feature_request": 0}[cat]
    base += urgent + (tier == "enterprise") + outage_hint
    if rng.random() < 0.08:  # шум разметки операторов
        base += rng.choice([-1, 1])
    return PRIORITIES[int(np.clip(base, 0, 3))]


def make_ticket(rng, i: int, ts: datetime) -> dict:
    cat = CATEGORIES[rng.choice(len(CATEGORIES), p=[.2, .24, .14, .16, .12, .08, .06])]
    urgent = rng.random() < 0.22
    tier = str(rng.choice(["free", "pro", "enterprise"], p=[.5, .35, .15]))
    outage = cat == "technical" and rng.random() < 0.15
    tpl = int(rng.integers(len(T[cat])))
    parts = [str(rng.choice(URGENT if urgent else CALM)), _fill(rng, T[cat][tpl])]
    if outage:
        parts.append(str(rng.choice(["У коллег то же самое.", "Похоже, лежит весь сервис.", "Ничего не работает ни у кого в офисе."])))
    if rng.random() < 0.15:  # смешанная тема — вторичная категория
        other = CATEGORIES[rng.integers(len(CATEGORIES))]
        parts.append("И ещё: " + _fill(rng, str(rng.choice(T[other]))).lower())
    parts.append(str(rng.choice(TAILS)))
    text = " ".join(p for p in parts if p)
    text = _typo(rng, text, 0.015) if rng.random() < 0.5 else text
    if rng.random() < 0.3:
        text = text.lower()
    pr = label_priority(rng, cat, urgent, tier, outage)
    return {"ticket_id": f"T{i:06d}", "created_at": ts.isoformat(), "channel": str(rng.choice(["email", "chat", "web", "app"])),
            "customer_tier": tier, "subject": text[:60], "text": text, "category": cat, "priority": pr,
            "team": TEAM_BY_CATEGORY[cat], "tpl": f"{cat}:{tpl}"}


def generate(n: int = 12000, seed: int = 3) -> list[dict]:
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [make_ticket(rng, i, t0 + timedelta(minutes=float(i * rng.uniform(1, 6)))) for i in range(n)]


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    Path("data/tickets.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in generate()))
