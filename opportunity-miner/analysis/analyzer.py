"""Stage-2 brief on DeepSeek (OpenAI-compatible).

Only posts that already cleared Stage-1 should be passed in. The default
model is `deepseek-chat` (V3). Set DEEPSEEK_MODEL=deepseek-reasoner for R1.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from config import Settings
from database.models import Opportunity, RawPost, opportunity_from_payload
from textutil import clip, parse_json_object

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "tr_analysis.txt"

SYSTEM_PROMPT = (
    "Sen Türkiye pazarı için ürün stratejisti ve oyun tasarımcısısın. "
    "Yalnızca tek bir JSON nesnesi döndür. Markdown kullanma. "
    "Uydurma metrik yazma. Kullanıcıya görünen metin alanları Türkçe olsun. "
    "JSON anahtarlarını İngilizce bırak."
)


def render_prompt(template: str, post: RawPost, body_limit: int = 4000) -> str:
    return (
        template.replace("{{TITLE}}", post.title or "")
        .replace("{{URL}}", post.url or "")
        .replace("{{SOURCE}}", post.source or "")
        .replace("{{AUTHOR}}", post.author or "")
        .replace("{{BODY}}", clip(post.body or "", body_limit))
    )


class DeepSeekAnalyzer:
    def __init__(self, settings: Settings, prompt_path: Path | None = None, complete=None) -> None:
        self.settings = settings
        self.prompt_path = prompt_path or PROMPT_PATH
        self._complete_override = complete
        self._client = None

    @property
    def available(self) -> bool:
        return self._complete_override is not None or self.settings.deepseek_configured

    async def analyze(self, post: RawPost) -> Opportunity:
        if not self.available:
            raise RuntimeError("deepseek_not_configured")
        template = self.prompt_path.read_text(encoding="utf-8")
        user_prompt = render_prompt(template, post)
        content = await asyncio.wait_for(
            self._complete(user_prompt),
            timeout=self.settings.analyzer_timeout_seconds,
        )
        return opportunity_from_payload(parse_json_object(content))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _complete(self, user_prompt: str) -> str:
        if self._complete_override is not None:
            result = self._complete_override(user_prompt)
            if asyncio.iscoroutine(result):
                return await result
            return result
        client = self._client_or_create()
        try:
            return await self._request(client, user_prompt, json_mode=True)
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" not in message and "json" not in message:
                raise
            logger.info("retrying DeepSeek without JSON mode: %s", exc)
            return await self._request(client, user_prompt, json_mode=False)

    async def _request(self, client, user_prompt: str, *, json_mode: bool) -> str:
        kwargs = {
            "model": self.settings.deepseek_model,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("empty DeepSeek response")
        return content

    def _client_or_create(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
            )
        return self._client
