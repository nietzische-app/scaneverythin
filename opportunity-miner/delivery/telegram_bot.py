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


def _field(analysis: dict, key: str) -> str:
    value = str(analysis.get(key) or "").strip()
    return value or "—"


def _summary(analysis: dict) -> str:
    summary = str(analysis.get("summary_tr") or "").strip()
    problem = str(analysis.get("core_problem_or_mechanic") or "").strip()
    if summary and problem and problem not in summary:
        return f"{summary}\n{problem}"
    return summary or problem or "—"


def format_alert(view: InsightView) -> str:
    analysis = view.analysis or {}
    title = analysis.get("title_tr") or view.title
    lines = [
        f"🚀 *FIRSAT RADARI* \\| Skor: {int(view.opportunity_score)}/100",
        "━━━━━━━━━━━━━━━━━━━━",
        f"📌 *Proje / Fikir:* {escape_md(title)}",
        f"🔗 {md_link('Kaynak Bağlantısı', view.url)}",
        "",
        "💡 *Özet & Problem:*",
        escape_md(_summary(analysis)),
        "",
        "🇹🇷 *TR Rekabet Durumu:*",
        escape_md(_field(analysis, "tr_market_analysis")),
        "",
        "⚡ *Bizim Farkımız Ne Olabilir?*",
        escape_md(_field(analysis, "our_differentiator")),
        "",
        "🎯 *Hedef Kitle & Gelir Modeli:*",
        escape_md(_field(analysis, "business_model")),
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines).strip()


def plain_alert(view: InsightView) -> str:
    analysis = view.analysis or {}
    title = analysis.get("title_tr") or view.title
    return "\n".join(
        [
            f"🚀 FIRSAT RADARI | Skor: {int(view.opportunity_score)}/100",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📌 Proje / Fikir: {title}",
            f"🔗 {view.url}",
            "",
            "💡 Özet & Problem:",
            _summary(analysis),
            "",
            "🇹🇷 TR Rekabet Durumu:",
            _field(analysis, "tr_market_analysis"),
            "",
            "⚡ Bizim Farkımız Ne Olabilir?",
            _field(analysis, "our_differentiator"),
            "",
            "🎯 Hedef Kitle & Gelir Modeli:",
            _field(analysis, "business_model"),
            "━━━━━━━━━━━━━━━━━━━━",
        ]
    ).strip()[:3900]


class TelegramNotifier:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        if not settings.telegram_bot_token.strip():
            logger.error("TELEGRAM_BOT_TOKEN is missing from the environment")
        if not settings.telegram_chat_id.strip():
            logger.error("TELEGRAM_CHAT_ID is missing from the environment")

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
