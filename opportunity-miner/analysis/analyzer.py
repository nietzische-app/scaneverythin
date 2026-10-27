"""Stage-2 committee: a Scout extracts the brief, a Judge sets the score.

Only posts that already cleared Stage-1 should be passed in. The default
model is `gpt-4o-mini`. Set OPENAI_MODEL=gpt-4o for the larger model.
Both steps return one JSON object. The Judge's score is the one we store.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from openai import APIStatusError, OpenAI

from config import Settings
from database.models import Opportunity, RawPost, opportunity_from_payload
from textutil import clip, parse_json_object

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "tr_analysis.txt"

SCOUT_PROMPT = (
    "Sen Türkiye pazarı için ürün stratejisti ve oyun tasarımcısısın. Rolün Scout (Analizci). "
    "Yalnızca tek bir JSON nesnesi döndür. Markdown kullanma. "
    "Uydurma metrik yazma. JSON anahtarlarını İngilizce bırak. "
    "Generate the output directly in natural, high-quality, professional Turkish. "
    "Do not write in English and translate literally. "
    "Extract the core problem, the solution mechanism, the local Turkish adaptation, and the market scale. "
    "market_scale must be exactly one of: Micro, SMB, Enterprise B2B, B2C. "
    "tr_market_analysis: Türkiye pazarındaki yerel rekabet ve boşluklar. "
    "our_differentiator: Bizim ne farkımız olabilir, pazarda nasıl öne geçeriz (Moat / Valör). "
    "business_model: hedef kitle ve gelir modeli. "
    "The summary must be explicit, actionable, and easy to understand. Avoid abstract jargon. "
    "net_problem: Tam olarak kime, ne acı çektiriyor? "
    "solution_mechanism: Bu yazılım problemi tam olarak nasıl çözüyor? "
    "why_now: Pazardaki güncel fırsat veya trend nedir? "
    "Evaluate each idea as one of three opportunity types, not only as a local Turkish clone. "
    "opportunity_type must be exactly one of: "
    "Yeni & Kopyalanabilir (Fresh & Copyable: a modern product that is new globally and can be cloned and adapted quickly); "
    "Zamanının Ötesinde / Yeniden Canlandırılabilir (Underhyped & Revived: an older product that stayed dim because the technology or AI of its time was not enough, and that today's stack such as LLM, RAG, and automation can turn into a large opportunity); "
    "TR Pazar Fırsatı (Local Adaptation: a localization opportunity specific to the Turkish market). "
    "Give equal or higher weight to large-scale enterprise platforms, multi-tenant B2B SaaS, and infrastructure ecosystems, alongside ideas that can ship as a 1-day tool. "
    "project_scale: a short Turkish label such as Hızlı MVP, Büyük Çaplı B2B Platformu, or Yüksek Hype B2C. "
    "suggested_domains: exactly 2 short catchy domains, each a single hostname such as slug.io, slug.app, or slug.com. "
    "Leave the final gate to the Judge. A draft opportunity_score is allowed, and the Judge may replace it."
)

JUDGE_PROMPT = (
    "Sen acımasız bir yargıç ve risk kapıcısısın. Rolün Judge. "
    "Sana bir Scout özeti verilecek. Yalnızca tek bir JSON nesnesi döndür. Markdown kullanma. "
    "Uydurma metrik yazma. JSON anahtarlarını İngilizce bırak. judge_verdict Türkçe olsun. "
    "Critique the idea in judge_verdict. Answer all four questions in that commentary: "
    "Is this a saturated post-2023 AI wrapper (standard PDF chat, basic text rephrasers, saturated AI wrappers)? "
    "Does an easily accessible open-source alternative exist? "
    "Will customers actually pay monthly recurring revenue (MRR) for this? "
    "If it is a DevTool, is it a low-margin micro tool or a high-value B2B/Enterprise security or infrastructure solution? "
    "Strictly identify outdated or saturated post-AI wrapper ideas. "
    "If the idea is already monopolized or trivialized post-2023, set saturated_wrapper true and "
    "rate its opportunity_score strictly below 50 so it gets filtered out. "
    "Do not restrict opportunity_score to minor micro-tools. "
    "A high-value B2B or Enterprise security or infrastructure product is not a niche low-level utility. "
    "devtool_class is micro for a low-margin micro DevTool, enterprise for a high-value B2B or Enterprise "
    "security or infrastructure solution, and none otherwise. "
    "A micro DevTool must be rejected with opportunity_score strictly below 50. "
    "An enterprise DevTool passes only with opportunity_score >= 80; otherwise set it strictly below 50. "
    "Provide an 'opportunity_score' strictly between 0 and 100 based on SaaS / business potential. "
    "Return that score as an integer. "
    "Return JSON with opportunity_score, judge_verdict, saturated_wrapper, open_source_alternative, "
    "mrr_likely, and devtool_class."
)

# Scout owns the brief the rest of the app already reads. The Judge owns the score.
SYSTEM_PROMPT = SCOUT_PROMPT


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

        async def _run() -> Opportunity:
            scout_text = await self._complete(user_prompt, system=SCOUT_PROMPT)
            scout = parse_json_object(scout_text)
            judge_user = (
                "Scout brief JSON:\n"
                + clip(json.dumps(scout, ensure_ascii=False), 6000)
                + f"\n\nBaşlık: {post.title or ''}"
            )
            judge_text = await self._complete(judge_user, clause, system=JUDGE_PROMPT)
            return merge_committee(scout, parse_json_object(judge_text))

        return await asyncio.wait_for(_run(), timeout=self.settings.analyzer_timeout_seconds)

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            client.close()

    async def _complete(self, user_prompt: str, system_extra: str = "", system: str = SCOUT_PROMPT) -> str:
        if self._complete_override is not None:
            result = self._complete_override(user_prompt)
            if asyncio.iscoroutine(result):
                return await result
            return result
        try:
            return await self._request(user_prompt, json_mode=True, system_extra=system_extra, system=system)
        except OpenAIUnavailable:
            raise
        except Exception as exc:
            if _blocked(exc):
                raise _unavailable(exc) from exc
            message = str(exc).lower()
            if "response_format" not in message and "json" not in message:
                raise
            logger.info("retrying OpenAI without JSON mode: %s", exc)
            return await self._request(user_prompt, json_mode=False, system_extra=system_extra, system=system)

    async def _request(
        self,
        user_prompt: str,
        *,
        json_mode: bool,
        system_extra: str = "",
        system: str = SCOUT_PROMPT,
    ) -> str:
        try:
            return await asyncio.to_thread(
                self._request_sync, user_prompt, json_mode, system_extra, system
            )
        except OpenAIUnavailable:
            raise
        except Exception as exc:
            if _blocked(exc):
                raise _unavailable(exc) from exc
            raise

    def _request_sync(
        self,
        user_prompt: str,
        json_mode: bool,
        system_extra: str = "",
        system: str = SCOUT_PROMPT,
    ) -> str:
        system = system if not system_extra else f"{system}\n{system_extra}"
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


def merge_committee(scout: dict, judge: dict) -> Opportunity:
    """The Scout's brief stands. The Judge's score and verdict replace the draft."""
    payload = dict(scout)
    if judge.get("opportunity_score") is not None:
        payload["opportunity_score"] = judge.get("opportunity_score")
    payload["judge_verdict"] = str(judge.get("judge_verdict") or payload.get("judge_verdict") or "").strip()
    opportunity = opportunity_from_payload(payload)
    gated = apply_judge_gates(opportunity.opportunity_score, judge)
    if gated != opportunity.opportunity_score:
        return opportunity.model_copy(update={"opportunity_score": gated})
    return opportunity


def apply_judge_gates(score: int, judge: dict) -> int:
    """Enforce the reject rules in code so a soft model score cannot sneak through."""
    kind = str(judge.get("devtool_class") or "").strip().lower().replace("-", " ").replace("_", " ")
    micro = kind in {"micro", "low margin", "reject"}
    enterprise = kind in {"enterprise", "security", "infrastructure", "b2b security"}
    if _yes(judge.get("saturated_wrapper")) or micro:
        return min(score, 49)
    if enterprise and score < 80:
        return min(score, 49)
    return score


def _yes(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "evet"}
