"""Clear Telegram rendering for forward paper results."""
from __future__ import annotations

from datetime import datetime


def _signed(value: float, digits: int = 2) -> str:
    return f"{float(value):+.{digits}f}"


def _period_range(stats: dict, tz) -> str:
    start = datetime.fromtimestamp(float(stats.get("since") or 0), tz)
    end = datetime.now(tz)
    if start.date() == end.date():
        return f"{start:%d.%m %H:%M}–{end:%H:%M}"
    return f"{start:%d.%m %H:%M}–{end:%d.%m %H:%M}"


def _period(stats: dict, label: str, tz) -> str:
    total = int(stats.get("total") or 0)
    closed = int(stats.get("closed") or 0)
    opened = int(stats.get("open") or 0) + int(stats.get("tp1_partial_open") or 0)
    title = label
    if stats.get("clamped"):
        title += " · только текущая версия"
    lines = [f"*{title}*  ·  {_period_range(stats, tz)}"]
    if total == 0:
        lines.append("Сигналов за этот период пока нет.")
        return "\n".join(lines)

    lines.append(f"Сигналы: *{total}*  ·  закрыто: *{closed}*  ·  в работе: *{opened}*")
    if closed == 0:
        lines.append("Результат появится после закрытия первой сделки.")
        return "\n".join(lines)

    wins = int(stats.get("wins") or 0)
    losses = int(stats.get("losses") or 0)
    flat = int(stats.get("flat") or 0)
    lines.append(
        f"Исходы: 🟢 {wins}  ·  🔴 {losses}  ·  ⚪ {flat}"
    )
    lines.append(
        f"Доля прибыльных: *{float(stats.get('win_rate') or 0):.1f}%* "
        f"из {closed} закрытых"
    )
    lines.append(
        f"TP1 достигнут: {int(stats.get('tp1_hit') or 0)}/{total} "
        f"({float(stats.get('tp1_rate') or 0):.1f}% от всех сигналов)"
    )
    unit_r = float(stats.get("unit_total_r", stats.get("total_r", 0)) or 0)
    avg_r = float(stats.get("unit_r_per_trade", stats.get("r_per_trade", 0)) or 0)
    lines.append(f"Результат сигналов: *{_signed(unit_r)}R*  ·  среднее {_signed(avg_r, 3)}R")
    model_n = int(stats.get("account_model_n") or 0)
    missing = int(stats.get("account_model_missing") or 0)
    if model_n:
        ret = float(stats.get("account_return_pct") or 0)
        dd = float(stats.get("account_max_drawdown_pct") or 0)
        lines.append(
            f"Модель депозита: *{_signed(ret)}%*  ·  макс. просадка {dd:.2f}%"
        )
    if missing:
        lines.append(f"⚠️ Без модели депозита: {missing} сделок без полной геометрии.")
    loss_streak = int(stats.get("max_loss_streak") or 0)
    if loss_streak >= 2:
        lines.append(f"Максимум стопов подряд: {loss_streak}")
    return "\n".join(lines)


def format_results_message(today: dict, week: dict, month: dict, tz,
                           margin_pct: float, leverage: float = 10.0,
                           configured: bool = False) -> str:
    basis = "ваша настройка" if configured else "пример"
    header = (
        "📈 *Результаты PAPER-сигналов*\n"
        f"🧮 Модель депозита ({basis}): {float(margin_pct):g}% маржи на сделку, "
        f"плечо ×{float(leverage):g}.\n"
        "Открытые сделки не входят в результат и долю прибыльных. "
        "Проценты считаются последовательным реинвестированием закрытых сигналов."
    )
    return "\n\n".join([
        header,
        _period(today, "Сегодня", tz),
        _period(week, "Последние 7 дней", tz),
        _period(month, "Последние 30 дней", tz),
        "R — движение сделки относительно её стопа. "
        "Это PAPER-модель, а не фактический PnL счёта.",
    ])
