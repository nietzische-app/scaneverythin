"""Executive top-3 digest. Daily at 20:00 Europe/Istanbul, or POST /run-digest."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from database.models import InsightView
from textutil import ISTANBUL, clip

logger = logging.getLogger(__name__)

DIGEST_HOUR = 20
_RETRY_SECONDS = 15 * 60


def digest_due(now: datetime, last_day: str | None, hour: int = DIGEST_HOUR) -> bool:
    local = _local(now)
    if local.hour < hour:
        return False
    return last_day != local.date().isoformat()


def seconds_until_digest(now: datetime | None = None, hour: int = DIGEST_HOUR) -> float:
    local = _local(now or datetime.now(ISTANBUL))
    target = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if local >= target:
        target = target + timedelta(days=1)
    return max(1.0, (target - local).total_seconds())


def format_digest(views: list[InsightView], *, days: int) -> str:
    window = "son 24 saat" if days <= 1 else f"son {days} gün"
    titles = []
    for index, view in enumerate(views, 1):
        title = str((view.analysis or {}).get("title_tr") or view.title or "—")
        titles.append(f"{index}. {clip(title, 24)}")
    lines = [f"📊 RADAR ÖZETİ | {window}", " | ".join(titles), "—" * 28]

    def cells(pick) -> str:
        return " | ".join(clip(pick(view), 28) for view in views)

    lines.append("Skor: " + cells(_score_cell))
    lines.append("Problem: " + cells(lambda view: _field(view, "net_problem", "summary_tr")))
    lines.append("Çözüm: " + cells(lambda view: _field(view, "solution_mechanism")))
    lines.append("Gelir: " + cells(lambda view: _field(view, "business_model")))
    lines.append("Fark: " + cells(lambda view: _field(view, "our_differentiator")))
    lines.append("")
    for index, view in enumerate(views, 1):
        if view.url:
            lines.append(f"{index}. {view.url}")
    return "\n".join(lines).strip()


async def send_digest(db, notifier, *, days: int = 1) -> bool:
    """Send the matrix. True when it was sent or there was nothing to send."""
    since = (datetime.now(timezone.utc) - timedelta(days=max(1, days))).isoformat(timespec="seconds")
    views = await db.top_digest(since_iso=since, limit=3)
    if not views:
        logger.info("digest empty for the last %s day(s)", days)
        return True
    sender = getattr(notifier, "send_text", None)
    if sender is None or not getattr(notifier, "available", False):
        logger.warning("digest not sent: telegram is not configured")
        return False
    try:
        sent = await sender(format_digest(views, days=days))
    except Exception as exc:  # noqa: BLE001
        logger.warning("digest send failed: %s", exc)
        return False
    return bool(sent)


async def digest_loop(settings, db, notifier, stop: asyncio.Event) -> None:
    if not settings.telegram_configured:
        return
    while not stop.is_set():
        delay = seconds_until_digest()
        now = datetime.now(ISTANBUL)
        last = await db.get_state("digest_day")
        if digest_due(now, last):
            ok = False
            try:
                ok = await send_digest(db, notifier, days=1)
            except Exception:  # noqa: BLE001
                logger.exception("digest failed")
            if ok:
                await db.set_state("digest_day", now.date().isoformat())
            else:
                delay = _RETRY_SECONDS
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except asyncio.TimeoutError:
            continue


def _local(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=ISTANBUL)
    return now.astimezone(ISTANBUL)


def _field(view: InsightView, *keys: str) -> str:
    analysis = view.analysis or {}
    for key in keys:
        value = str(analysis.get(key) or "").strip()
        if value:
            return value
    return "—"


def _score_cell(view: InsightView) -> str:
    mark = " ❤" if view.user_liked else ""
    return f"{int(view.opportunity_score)}/100{mark}"
