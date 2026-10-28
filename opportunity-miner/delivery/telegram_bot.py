"""Telegram cards for briefs that cleared both stages.

Messages are MarkdownV2. Every dynamic character is escaped. If Telegram
rejects the markup, the same card is sent once as plain text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

from config import Settings
from database.models import InsightView
from ingestion.domain_checker import LABELS

logger = logging.getLogger(__name__)

_MD_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")
_API = "https://api.telegram.org/bot{token}/{method}"
_MVP_LABEL = "🚀 Cursor MVP Prompt"


def escape_md(value: object) -> str:
    return _MD_SPECIAL.sub(r"\\\1", str(value or ""))


def md_link(label: str, url: str) -> str:
    safe = (url or "").replace("\\", "\\\\").replace(")", "\\)")
    return f"[{escape_md(label)}]({safe})"


def _field(analysis: dict, key: str) -> str:
    value = str(analysis.get(key) or "").strip()
    return value or "—"


_JARGON_ANALOGIES = (
    ("kubernetes", "Kubernetes = Binlerce sunucuyu otomatik yöneten dijital kargo limanı müdürü"),
    ("vektör veritaban", "Vektör veritabanı = Anlamca benzeyen notları yan yana koyan bir dosya dolabı"),
    ("vector database", "Vektör veritabanı = Anlamca benzeyen notları yan yana koyan bir dosya dolabı"),
    ("vector db", "Vektör veritabanı = Anlamca benzeyen notları yan yana koyan bir dosya dolabı"),
    ("ci/cd", "CI/CD = Her kaydedişte paketi kontrol edip yerine koyan kurye"),
    ("rag", "RAG = Her seferinde bütün kitabı okumak yerine, önce doğru sayfayı bulan yardımcı"),
    ("mcp", "MCP = Yapay zekanın dışarıdaki araçları tek fişle kullanmasını sağlayan priz"),
)


def _two_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    kept = [part.strip() for part in parts if part.strip()][:2]
    return " ".join(kept)


def _analogy_line(analysis: dict, summary: str) -> str:
    given = str(analysis.get("jargon_analogy") or "").strip()
    if given:
        return given
    blob = " ".join(
        str(analysis.get(name) or "")
        for name in ("net_problem", "solution_mechanism", "core_problem_or_mechanic", "summary_tr")
    )
    blob = f"{summary} {blob}".casefold()
    for needle, line in _JARGON_ANALOGIES:
        if needle in {"rag", "mcp"}:
            if re.search(rf"\b{needle}\b", blob):
                return line
            continue
        if needle in blob:
            return line
    return ""


def _summary(analysis: dict) -> str:
    summary = str(analysis.get("summary_tr") or "").strip()
    problem = str(analysis.get("core_problem_or_mechanic") or "").strip()
    if summary and problem and problem not in summary:
        return f"{summary}\n{problem}"
    return summary or problem or "—"


def _summary_blocks(analysis: dict, *, markdown: bool) -> list[str]:
    parts = (
        ("🤕", "Net Problem", analysis.get("net_problem")),
        ("🛠️", "Çözüm & Mekanizma", analysis.get("solution_mechanism")),
        ("💰", "Gelir Modeli", analysis.get("business_model")),
        ("⏳", "Neden Şimdi?", analysis.get("why_now")),
    )
    summary = _two_sentences(str(analysis.get("summary_tr") or ""))
    analogy = _analogy_line(analysis, summary)
    filled = [str(value or "").strip() for _emoji, _label, value in parts]
    if not any(filled) and not summary and not analogy:
        text = _summary(analysis)
        return [escape_md(text) if markdown else text]
    lines: list[str] = []
    if summary:
        lines.append(escape_md(summary) if markdown else summary)
    if analogy:
        lines.append(escape_md(analogy) if markdown else analogy)
    for emoji, label, value in parts:
        body = str(value or "").strip() or "—"
        if markdown:
            lines.append(f"{emoji} *{label}:* {escape_md(body)}")
        else:
            lines.append(f"{emoji} {label}: {body}")
    return lines


def text_chunks(text: str, limit: int = 4000) -> list[str]:
    """Split on line breaks so a Telegram message is not cut mid-sentence."""
    rest = (text or "").strip()
    if not rest:
        return []
    if limit < 200:
        limit = 200
    chunks: list[str] = []
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        piece = rest[:cut].strip()
        rest = rest[cut:].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _domain_line(analysis: dict, *, markdown: bool) -> str:
    rows = analysis.get("_domains")
    if not rows:
        names = analysis.get("suggested_domains") or []
        if not isinstance(names, list) or not names:
            return ""
        rows = [{"name": name, "status": "unknown"} for name in names[:2]]
    parts = []
    for row in rows[:2]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        label = LABELS.get(str(row.get("status") or ""), LABELS["unknown"])
        parts.append(f"{name} ({label})")
    if not parts:
        return ""
    text = " | ".join(parts)
    if markdown:
        return f"🌐 *Önerilen Domainler:* {escape_md(text)}"
    return f"🌐 Önerilen Domainler: {text}"


_GROK_LABEL = "🤖 Grok ile İncele"


def mvp_keyboard(insight_id: int) -> dict:
    ident = int(insight_id)
    return {
        "inline_keyboard": [
            [
                {"text": _MVP_LABEL, "callback_data": f"mvp:{ident}"},
                {"text": _GROK_LABEL, "callback_data": f"grok:{ident}"},
            ]
        ]
    }


def mvp_prompt(view: InsightView) -> str:
    analysis = view.analysis or {}
    title = str(analysis.get("title_tr") or view.title or "Fırsat")
    blueprint = analysis.get("blueprint") if isinstance(analysis.get("blueprint"), dict) else {}
    points = [str(point).strip() for point in (blueprint.get("points") or []) if str(point).strip()]
    features = "\n".join(f"- {point}" for point in points[:6]) or "- One core user action"
    return (
        "You are a senior product engineer. Build a 1-day MVP for this specific idea.\n\n"
        f"Idea: {title}\n"
        f"Problem: {_pick(analysis, 'net_problem', 'core_problem_or_mechanic', 'summary_tr')}\n"
        f"How it works: {_pick(analysis, 'solution_mechanism')}\n"
        f"Why now: {_pick(analysis, 'why_now')}\n"
        f"Audience and revenue: {_pick(analysis, 'business_model')}\n"
        f"Differentiator: {_pick(analysis, 'our_differentiator')}\n"
        f"Non-obvious angle: {_pick(analysis, 'aha_angle')}\n"
        f"Unfair advantage: {_pick(analysis, 'unfair_advantage')}\n"
        f"Scope:\n{features}\n\n"
        "Stack (do not substitute):\n"
        "- Next.js App Router + Tailwind CSS for the UI\n"
        "- Supabase if auth or multi-user data is required, otherwise SQLite\n"
        "- FastAPI for any job, webhook, or integration the UI should not own\n\n"
        "Ship today:\n"
        "1. One screen that demonstrates the non-obvious angle, not a generic version of the idea.\n"
        "2. Persistence for the one record type that makes the loop real.\n"
        "3. A README with how to run it locally.\n\n"
        "Do not add billing, admin, or extra products."
    )


def grok_prompt(view: InsightView) -> str:
    analysis = view.analysis or {}
    title = str(analysis.get("title_tr") or view.title or "Fırsat")
    return (
        "You are Grok, a sharp product strategist. Brainstorm for this specific idea. "
        "Be concrete. Do not invent metrics.\n\n"
        f"Idea: {title}\n"
        f"Type and scale: {_pick(analysis, 'opportunity_type')} | {_pick(analysis, 'project_scale')}\n"
        f"Market scale: {_pick(analysis, 'market_scale')}\n"
        f"Problem: {_pick(analysis, 'net_problem', 'core_problem_or_mechanic', 'summary_tr')}\n"
        f"How it works: {_pick(analysis, 'solution_mechanism')}\n"
        f"Why now: {_pick(analysis, 'why_now')}\n"
        f"TR market: {_pick(analysis, 'tr_market_analysis')}\n"
        f"Differentiator: {_pick(analysis, 'our_differentiator')}\n"
        f"Audience and revenue: {_pick(analysis, 'business_model')}\n"
        f"Aha angle: {_pick(analysis, 'aha_angle')}\n"
        f"Unfair advantage: {_pick(analysis, 'unfair_advantage')}\n"
        f"Judge verdict: {_pick(analysis, 'judge_verdict')}\n"
        f"Score: {int(view.opportunity_score)}/100\n\n"
        "Answer only these two questions:\n"
        "1. How to exploit this non-obvious angle for viral distribution.\n"
        "2. How to build an uncopyable MVP in 24 hours using Cursor and AI agents."
    )


def _pick(analysis: dict, *keys: str) -> str:
    for key in keys:
        value = str(analysis.get(key) or "").strip()
        if value:
            return value
    return "—"


def _competitor_block(analysis: dict, *, markdown: bool) -> list[str]:
    rivals = _competitor_line(analysis.get("top_competitors"))
    weakness = _field(analysis, "competitor_weakness")
    trend = _field(analysis, "search_trend_signal")
    if markdown:
        rivals, weakness, trend = escape_md(rivals), escape_md(weakness), escape_md(trend)
        title = "🎯 *Pazar & Rakip Analizi:*"
    else:
        title = "🎯 Pazar & Rakip Analizi:"
    return [
        title,
        f"• ⚔️ Rakipler: {rivals}",
        f"• ⚠️ Rakiplerin Zayıf Karnı: {weakness}",
        f"• 📈 Arama Hacmi & Hype: {trend}",
    ]


def _competitor_line(raw: object) -> str:
    if isinstance(raw, list):
        names = [str(item).strip() for item in raw if str(item).strip()]
    else:
        names = [part.strip() for part in str(raw or "").split("|") if part.strip()]
    return " | ".join(names[:3]) or "—"


def _aha_lines(analysis: dict, *, markdown: bool) -> list[str]:
    angle = _field(analysis, "aha_angle")
    moat = _field(analysis, "unfair_advantage")
    if markdown:
        angle, moat = escape_md(angle), escape_md(moat)
        return [
            f"🧠 *Aha\\! / Zekice Yaklaşım:* {angle}",
            f"🛡️ *Haksız Avantaj / Moat:* {moat}",
        ]
    return [
        f"🧠 Aha! / Zekice Yaklaşım: {angle}",
        f"🛡️ Haksız Avantaj / Moat: {moat}",
    ]


def _type_scale(analysis: dict, *, markdown: bool) -> str:
    kind = str(analysis.get("opportunity_type") or "").strip() or "—"
    scale = str(analysis.get("project_scale") or "").strip() or "—"
    if markdown:
        return f"🏷️ *Tip & Ölçek:* {escape_md(kind)} \\| {escape_md(scale)}"
    return f"🏷️ Tip & Ölçek: {kind} | {scale}"


def format_alert(view: InsightView) -> str:
    analysis = view.analysis or {}
    title = analysis.get("title_tr") or view.title
    domain = _domain_line(analysis, markdown=True)
    lines = [
        f"🚀 *FIRSAT RADARI* \\| Skor: {int(view.opportunity_score)}/100",
        "━━━━━━━━━━━━━━━━━━━━",
        _type_scale(analysis, markdown=True),
        f"📌 *Proje / Fikir:* {escape_md(title)}",
        f"🔗 {md_link('Kaynak Bağlantısı', view.url)}",
        *([domain] if domain else []),
        "",
        "💡 *Özet & Problem:*",
        *_summary_blocks(analysis, markdown=True),
        "",
        "🇹🇷 *TR Rekabet Durumu:*",
        escape_md(_field(analysis, "tr_market_analysis")),
        "",
        *_competitor_block(analysis, markdown=True),
        "",
        "⚡ *Bizim Farkımız Ne Olabilir?*",
        escape_md(_field(analysis, "our_differentiator")),
        "",
        "🎯 *Hedef Kitle & Gelir Modeli:*",
        escape_md(_field(analysis, "business_model")),
        "",
        *_aha_lines(analysis, markdown=True),
        "",
        "⚖️ *Yargıç / Risk Notu:*",
        escape_md(_field(analysis, "judge_verdict")),
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines).strip()


def plain_alert(view: InsightView) -> str:
    analysis = view.analysis or {}
    title = analysis.get("title_tr") or view.title
    domain = _domain_line(analysis, markdown=False)
    return "\n".join(
        [
            f"🚀 FIRSAT RADARI | Skor: {int(view.opportunity_score)}/100",
            "━━━━━━━━━━━━━━━━━━━━",
            _type_scale(analysis, markdown=False),
            f"📌 Proje / Fikir: {title}",
            f"🔗 {view.url}",
            *([domain] if domain else []),
            "",
            "💡 Özet & Problem:",
            *_summary_blocks(analysis, markdown=False),
            "",
            "🇹🇷 TR Rekabet Durumu:",
            _field(analysis, "tr_market_analysis"),
            "",
            *_competitor_block(analysis, markdown=False),
            "",
            "⚡ Bizim Farkımız Ne Olabilir?",
            _field(analysis, "our_differentiator"),
            "",
            "🎯 Hedef Kitle & Gelir Modeli:",
            _field(analysis, "business_model"),
            "",
            *_aha_lines(analysis, markdown=False),
            "",
            "⚖️ Yargıç / Risk Notu:",
            _field(analysis, "judge_verdict"),
            "━━━━━━━━━━━━━━━━━━━━",
        ]
    ).strip()


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
        url = _API.format(token=token, method="sendMessage")
        client = self._client
        close_after = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
            close_after = True
        try:
            if int(view.opportunity_score) >= 80:
                photo = await self._card_png(view)
                if photo:
                    sent = await self._post_photo(client, token, view, photo)
                    if sent:
                        return sent
                    logger.warning("telegram photo rejected for insight %s; sending text", view.insight_id)
            markup = mvp_keyboard(view.insight_id)
            return await self._send_card_text(client, url, view, markup)
        finally:
            if close_after:
                await client.aclose()

    async def send_text(self, text: str) -> int | bool:
        if not self.available:
            return False
        url = _API.format(token=self.settings.telegram_bot_token.strip(), method="sendMessage")
        client = self._client
        close_after = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
            close_after = True
        try:
            return await self._post(client, url, text[:4096], None, None)
        finally:
            if close_after:
                await client.aclose()

    async def _card_png(self, view: InsightView) -> bytes:
        try:
            from delivery.card_generator import render_card

            png = await asyncio.to_thread(render_card, view)
        except Exception as exc:  # noqa: BLE001
            logger.warning("card image skipped: %s", exc)
            return b""
        if not png.startswith(b"\x89PNG"):
            logger.warning("card image skipped: renderer did not return a PNG")
            return b""
        return png

    async def _post_photo(self, client: httpx.AsyncClient, token: str, view: InsightView, png: bytes) -> int:
        url = _API.format(token=token, method="sendPhoto")
        markup = json.dumps(mvp_keyboard(view.insight_id), ensure_ascii=False)
        markdown = format_alert(view)
        attempts: list[tuple[str, str | None]] = [(plain_alert(view)[:1024], None)]
        if len(markdown) <= 1024:
            attempts.insert(0, (markdown, "MarkdownV2"))
        for caption, parse_mode in attempts:
            data: dict[str, str] = {
                "chat_id": self.settings.telegram_chat_id,
                "caption": caption,
                "reply_markup": markup,
            }
            if parse_mode:
                data["parse_mode"] = parse_mode
            try:
                response = await client.post(
                    url,
                    data=data,
                    files={"photo": ("card.png", png, "image/png")},
                )
            except httpx.HTTPError as exc:
                logger.warning("telegram photo request failed: %s", exc)
                return 0
            if response.status_code == 200:
                message_id = _message_id(response)
                if len(plain_alert(view)) > 1024:
                    text_url = _API.format(token=token, method="sendMessage")
                    await self._send_card_text(client, text_url, view, None)
                return message_id
            logger.warning("telegram photo status %s: %s", response.status_code, response.text[:300])
        return 0

    async def _send_card_text(
        self,
        client: httpx.AsyncClient,
        url: str,
        view: InsightView,
        reply_markup: dict | None,
    ) -> int:
        markdown_chunks = text_chunks(format_alert(view))
        first = await self._post(client, url, markdown_chunks[0] if markdown_chunks else "—", "MarkdownV2", reply_markup)
        if first:
            for extra in markdown_chunks[1:]:
                await self._post(client, url, extra, "MarkdownV2", None)
            return first
        logger.info("telegram markdown rejected for insight %s; sending plain text", view.insight_id)
        plain_chunks = text_chunks(plain_alert(view))
        first = await self._post(client, url, plain_chunks[0] if plain_chunks else "—", None, reply_markup)
        if first:
            for extra in plain_chunks[1:]:
                await self._post(client, url, extra, None, None)
        return first

    async def _post(
        self,
        client: httpx.AsyncClient,
        url: str,
        text: str,
        parse_mode: str | None,
        reply_markup: dict | None = None,
    ) -> int:
        payload: dict[str, object] = {
            "chat_id": self.settings.telegram_chat_id,
            "text": text[:4096],
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            logger.warning("telegram request failed: %s", exc)
            return 0
        if response.status_code == 200:
            return _message_id(response)
        logger.warning("telegram status %s: %s", response.status_code, response.text[:300])
        return 0


def _message_id(response: httpx.Response) -> int:
    result = response.json().get("result") or {}
    try:
        return int(result.get("message_id") or 0)
    except (TypeError, ValueError):
        return 0


def callback_action(update: dict, chat_id: str) -> tuple[str, int] | None:
    """Return `(mvp|grok, insight id)` for a button tap in the configured chat."""
    query = update.get("callback_query")
    if not isinstance(query, dict):
        return None
    message = query.get("message") or {}
    chat = message.get("chat") or {}
    if str(chat.get("id")) != str(chat_id).strip():
        return None
    kind, _, raw_id = str(query.get("data") or "").partition(":")
    if kind not in {"mvp", "grok"}:
        return None
    try:
        return kind, int(raw_id)
    except (TypeError, ValueError):
        return None


def callback_insight_id(update: dict, chat_id: str) -> int | None:
    """Return the insight id for an MVP button tap in the configured chat."""
    action = callback_action(update, chat_id)
    if action is None or action[0] != "mvp":
        return None
    return action[1]


async def handle_callback(settings: Settings, db, update: dict, client: httpx.AsyncClient) -> bool:
    """Answer the button and send a tailored MVP prompt. Failures stay in the log."""
    query = update.get("callback_query") if isinstance(update, dict) else None
    if not isinstance(query, dict):
        return False
    action = callback_action(update, settings.telegram_chat_id)
    if action is None and _callback_chat(query) != str(settings.telegram_chat_id).strip():
        return False
    token = settings.telegram_bot_token.strip()
    if not token:
        return False
    base = f"https://api.telegram.org/bot{token}"
    query_id = query.get("id")
    if query_id:
        try:
            await client.post(f"{base}/answerCallbackQuery", json={"callback_query_id": query_id})
        except httpx.HTTPError as exc:
            logger.warning("telegram callback answer failed: %s", exc)
    if action is None:
        return False
    kind, insight_id = action
    view = await db.get_insight(insight_id)
    if view is None:
        text = "Bu fırsat artık kayıtlı değil."
    elif kind == "grok":
        text = grok_prompt(view)
    else:
        text = mvp_prompt(view)
    try:
        response = await client.post(
            f"{base}/sendMessage",
            json={
                "chat_id": settings.telegram_chat_id,
                "text": text[:4096],
                "disable_web_page_preview": True,
            },
        )
    except httpx.HTTPError as exc:
        logger.warning("mvp prompt send failed: %s", exc)
        return False
    if response.status_code != 200:
        logger.warning("mvp prompt status %s: %s", response.status_code, response.text[:300])
        return False
    return view is not None


def _callback_chat(query: dict) -> str:
    message = query.get("message") or {}
    chat = message.get("chat") or {}
    return str(chat.get("id") or "")


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
    """Long-poll reaction and MVP-button updates until stop is set."""
    if not settings.telegram_configured:
        return
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token.strip()}/getUpdates"
    offset: int | None = None
    logger.info("listening for telegram reactions")
    async with httpx.AsyncClient(timeout=40) as client:
        while not stop.is_set():
            params: dict[str, object] = {
                "timeout": 25,
                "allowed_updates": json.dumps(["message_reaction", "callback_query"]),
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
                if isinstance(update, dict) and update.get("callback_query"):
                    await handle_callback(settings, db, update, client)
