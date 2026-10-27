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


def _summary_blocks(analysis: dict, *, markdown: bool) -> list[str]:
    parts = (
        ("🤕", "Net Problem", analysis.get("net_problem")),
        ("🛠️", "Çözüm Mekanizması", analysis.get("solution_mechanism")),
        ("⏳", "Neden Şimdi?", analysis.get("why_now")),
    )
    filled = [str(value or "").strip() for _emoji, _label, value in parts]
    if not any(filled):
        text = _summary(analysis)
        return [escape_md(text) if markdown else text]
    lines: list[str] = []
    for emoji, label, value in parts:
        body = str(value or "").strip() or "—"
        if markdown:
            lines.append(f"{emoji} *{label}:* {escape_md(body)}")
        else:
            lines.append(f"{emoji} {label}: {body}")
    return lines


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
        *_summary_blocks(analysis, markdown=True),
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
            *_summary_blocks(analysis, markdown=False),
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

    async def send(self, view: InsightView) -> int | bool:
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
                return sent
            logger.info("telegram markdown rejected for insight %s; sending plain text", view.insight_id)
            return await self._post(client, url, plain_alert(view), None)
        finally:
            if close_after:
                await client.aclose()

    async def _post(self, client: httpx.AsyncClient, url: str, text: str, parse_mode: str | None) -> int:
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
            return 0
        if response.status_code == 200:
            result = response.json().get("result") or {}
            try:
                return int(result.get("message_id") or 0)
            except (TypeError, ValueError):
                return 0
        logger.warning("telegram status %s: %s", response.status_code, response.text[:300])
        return 0


_LIKE_EMOJI = {"❤", "❤️", "👍", "🔥"}


def _is_like(item: object) -> bool:
    if not isinstance(item, dict) or item.get("type") != "emoji":
        return False
    emoji = str(item.get("emoji") or "").replace("\ufe0f", "")
    return emoji in {value.replace("\ufe0f", "") for value in _LIKE_EMOJI}


def liked_message_id(update: dict, chat_id: str) -> int | None:
    """Return the message id when this chat adds ❤, 👍, or 🔥."""
    reaction = update.get("message_reaction")
    if not isinstance(reaction, dict):
        return None
    chat = reaction.get("chat") or {}
    if str(chat.get("id")) != str(chat_id).strip():
        return None
    new_reaction = reaction.get("new_reaction") or []
    if not any(_is_like(item) for item in new_reaction):
        return None
    try:
        return int(reaction["message_id"])
    except (KeyError, TypeError, ValueError):
        return None


async def apply_reaction(db, settings: Settings, update: dict) -> bool:
    message_id = liked_message_id(update, settings.telegram_chat_id)
    if message_id is None:
        return False
    liked = await db.mark_liked(message_id)
    if liked:
        logger.info("telegram reaction marked message %s as a favorite", message_id)
    return liked


async def poll_reactions(settings: Settings, db, stop) -> None:
    """Long-poll message_reaction updates until stop is set."""
    if not settings.telegram_configured:
        return
    import asyncio
    import json

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token.strip()}/getUpdates"
    offset: int | None = None
    logger.info("listening for telegram reactions")
    async with httpx.AsyncClient(timeout=40) as client:
        while not stop.is_set():
            params: dict[str, object] = {
                "timeout": 25,
                "allowed_updates": json.dumps(["message_reaction"]),
            }
            if offset is not None:
                params["offset"] = offset
            try:
                response = await client.get(url, params=params)
            except httpx.HTTPError as exc:
                logger.warning("telegram reaction poll failed: %s", exc)
                await asyncio.sleep(5)
                continue
            if response.status_code == 409:
                logger.error("telegram getUpdates conflict; reaction polling stopped")
                return
            if response.status_code != 200:
                logger.warning("telegram reaction poll status %s", response.status_code)
                await asyncio.sleep(15)
                continue
            for update in response.json().get("result") or []:
                try:
                    offset = int(update["update_id"]) + 1
                except (KeyError, TypeError, ValueError):
                    continue
                await apply_reaction(db, settings, update)
