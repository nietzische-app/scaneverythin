"""Stage-2 Turkish brief on OpenAI.

Only posts that already cleared Stage-1 should be passed in. The default
model is `gpt-4o-mini`. Set OPENAI_MODEL=gpt-4o for the larger model.
The model still returns one JSON object; Telegram renders that brief as Markdown.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from openai import APIStatusError, OpenAI

from config import Settings
from database.models import Opportunity, RawPost, opportunity_from_payload
from textutil import clip, parse_json_object

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "tr_analysis.txt"

SYSTEM_PROMPT = (
    "Sen Türkiye pazarı için ürün stratejisti ve oyun tasarımcısısın. "
    "Yalnızca tek bir JSON nesnesi döndür. Markdown kullanma. "
    "Uydurma metrik yazma. JSON anahtarlarını İngilizce bırak. "
    "Generate the output directly in natural, high-quality, professional Turkish. "
    "Do not write in English and translate literally. "
    "Strictly identify outdated or saturated post-AI wrapper ideas "
    "(standard PDF chat, basic text rephrasers, saturated AI wrappers). "
    "If the idea is already monopolized or trivialized post-2023, "
    "rate its opportunity_score strictly below 50 so it gets filtered out. "
    "tr_market_analysis: Türkiye pazarındaki yerel rekabet ve boşluklar. "
    "our_differentiator: Bizim ne farkımız olabilir, pazarda nasıl öne geçeriz (Moat / Valör). "
    "business_model: hedef kitle ve gelir modeli. "
    "The summary must be explicit, actionable, and easy to understand. Avoid abstract jargon. "
    "net_problem: Tam olarak kime, ne acı çektiriyor? "
    "solution_mechanism: Bu yazılım problemi tam olarak nasıl çözüyor? "
    "why_now: Pazardaki güncel fırsat veya trend nedir? "
    "Provide an 'opportunity_score' strictly between 0 and 100 based on SaaS / business potential. "
    "Return that score as an integer. "
    "suggested_domains: exactly 2 short catchy domains, each a single hostname such as slug.io, slug.app, or slug.com."
)


def favorite_clause(titles: list[str]) -> str:
    cleaned = [title.strip() for title in titles if title and title.strip()]
    if not cleaned:
        return ""
    return (
        "The user prefers ideas similar to these past favorites: "
        f"[{'; '.join(cleaned)}]. Give higher opportunity scores to projects with similar "
        "tech/business profiles (e.g., accessible B2C/B2B web & mobile tools, automation) "
        "and lower scores to overly niche low-level code/infrastructure utilities."
    )

_BLOCKED_STATUS = {401, 403, 429}


class OpenAIUnavailable(RuntimeError):
    """Auth or quota failure. The scan leaves the post queued and moves on."""


def render_prompt(template: str, post: RawPost, body_limit: int = 4000) -> str:
    return (
        template.replace("{{TITLE}}", post.title or "")
        .replace("{{URL}}", post.url or "")
        .replace("{{SOURCE}}", post.source or "")
        .replace("{{AUTHOR}}", post.author or "")
        .replace("{{BODY}}", clip(post.body or "", body_limit))
    )


def _blocked(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(exc, APIStatusError) and status in _BLOCKED_STATUS


def _unavailable(exc: Exception) -> OpenAIUnavailable:
    status = getattr(exc, "status_code", None)
    code = "openai_quota" if status == 429 else "openai_auth"
    return OpenAIUnavailable(code)


class OpenAIAnalyzer:
    def __init__(self, settings: Settings, prompt_path: Path | None = None, complete=None, db=None) -> None:
        self.settings = settings
        self.prompt_path = prompt_path or PROMPT_PATH
        self._complete_override = complete
        self._db = db
        self._client: OpenAI | None = None

    @property
    def available(self) -> bool:
        return self._complete_override is not None or self.settings.openai_configured

    async def analyze(self, post: RawPost) -> Opportunity:
        if not self.available:
            raise RuntimeError("openai_not_configured")
        template = self.prompt_path.read_text(encoding="utf-8")
        user_prompt = render_prompt(template, post)
        clause = ""
        if self._db is not None and self._complete_override is None:
            clause = favorite_clause(await self._db.liked_titles())
        content = await asyncio.wait_for(
            self._complete(user_prompt, clause),
            timeout=self.settings.analyzer_timeout_seconds,
        )
        return opportunity_from_payload(parse_json_object(content))

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            client.close()

    async def _complete(self, user_prompt: str, system_extra: str = "") -> str:
        if self._complete_override is not None:
            result = self._complete_override(user_prompt)
            if asyncio.iscoroutine(result):
                return await result
            return result
        try:
            return await self._request(user_prompt, json_mode=True, system_extra=system_extra)
        except OpenAIUnavailable:
            raise
        except Exception as exc:
            if _blocked(exc):
                raise _unavailable(exc) from exc
            message = str(exc).lower()
            if "response_format" not in message and "json" not in message:
                raise
            logger.info("retrying OpenAI without JSON mode: %s", exc)
            return await self._request(user_prompt, json_mode=False, system_extra=system_extra)

    async def _request(self, user_prompt: str, *, json_mode: bool, system_extra: str = "") -> str:
        try:
            return await asyncio.to_thread(self._request_sync, user_prompt, json_mode, system_extra)
        except OpenAIUnavailable:
            raise
        except Exception as exc:
            if _blocked(exc):
                raise _unavailable(exc) from exc
            raise

    def _request_sync(self, user_prompt: str, json_mode: bool, system_extra: str = "") -> str:
        system = SYSTEM_PROMPT if not system_extra else f"{SYSTEM_PROMPT}\n{system_extra}"
        kwargs = {
            "model": self.settings.openai_model,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client_or_create().chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("empty OpenAI response")
        return content

    def _client_or_create(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
