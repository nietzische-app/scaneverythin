"""Telegram cards for briefs that cleared both stages.

Messages are MarkdownV2. Every dynamic character is escaped. If Telegram
rejects the markup, the same card is sent once as plain text.
"""

from __future__ import annotations

import logging
import re

import httpx

from config import Settings
from database.models import InsightView

logger = logging.getLogger(__name__)

_MD_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")
_API = "https://api.telegram.org/bot{token}/sendMessage"


def escape_md(value: object) -> str:
    return _MD_SPECIAL.sub(r"\\\1", str(value or ""))


def md_link(label: str, url: str) -> str:
    safe = (url or "").replace("\\", "\\\\").replace(")", "\\)")
    return f"[{escape_md(label)}]({safe})"


def format_alert(view: InsightView) -> str:
    analysis = view.analysis or {}
    market = analysis.get("turkish_market") or {}
    blueprint = analysis.get("blueprint") or {}
    vibe = analysis.get("vibe_coding") or {}
    points = [str(point) for point in (blueprint.get("points") or [])[:5]]
    point_lines = "\n".join(f"• {escape_md(point)}" for point in points) or escape_md("—")
    days = vibe.get("difficulty_days", "?")
    lines = [
        f"*{escape_md('Yeni fırsat')}* · {int(view.opportunity_score)}/100",
        escape_md(analysis.get("title_tr") or view.title),
        f"{escape_md('Kaynak')}: {escape_md(view.source)} · {escape_md('güven')} {int(view.confidence)}",
        "",
        f"*{escape_md('Özet')}*",
        escape_md(analysis.get("summary_tr") or ""),
        "",
        f"*{escape_md('Çekirdek')}*",
        escape_md(analysis.get("core_problem_or_mechanic") or ""),
        "",
        f"*{escape_md('Türkiye')}*",
        escape_md(market.get("adaptation") or ""),
        escape_md(f"Ödeme: {market.get('payments') or '—'}"),
        escape_md(f"Kanal: {market.get('commerce_or_messaging') or '—'}"),
        escape_md(f"KVKK: {market.get('kvkk') or '—'}"),
        "",
        f"*{escape_md('Plan')}*",
        point_lines,
        escape_md(blueprint.get("mvp_scope") or ""),
        "",
        escape_md(f"Vibe coding: {days} gün. {vibe.get('why') or ''}"),
        escape_md(vibe.get("first_slice") or ""),
        "",
        md_link("Kaynağı aç", view.url),
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def plain_alert(view: InsightView) -> str:
    analysis = view.analysis or {}
    market = analysis.get("turkish_market") or {}
    vibe = analysis.get("vibe_coding") or {}
    return "\n".join(
        [
            f"Yeni fırsat · {view.opportunity_score}/100",
            str(analysis.get("title_tr") or view.title),
            f"Kaynak: {view.source}",
            "",
            str(analysis.get("summary_tr") or ""),
            "",
            str(analysis.get("core_problem_or_mechanic") or ""),
            "",
            str(market.get("adaptation") or ""),
            f"Vibe coding: {vibe.get('difficulty_days', '?')} gün",
            view.url,
        ]
    ).strip()[:3900]


class TelegramNotifier:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None

    @property
    def available(self) -> bool:
        return self.settings.telegram_configured

    async def send(self, view: InsightView) -> bool:
        if not self.available:
            return False
        token = self.settings.telegram_bot_token.strip()
        url = _API.format(token=token)
        client = self._client
        close_after = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
            close_after = True
        try:
            sent = await self._post(client, url, format_alert(view), "MarkdownV2")
            if sent:
                return True
            logger.info("telegram markdown rejected for insight %s; sending plain text", view.insight_id)
            return await self._post(client, url, plain_alert(view), None)
        finally:
            if close_after:
                await client.aclose()

    async def _post(self, client: httpx.AsyncClient, url: str, text: str, parse_mode: str | None) -> bool:
        payload: dict[str, object] = {
            "chat_id": self.settings.telegram_chat_id,
            "text": text[:4096],
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            logger.warning("telegram request failed: %s", exc)
            return False
        if response.status_code == 200:
            return True
        logger.warning("telegram status %s: %s", response.status_code, response.text[:300])
        return False
