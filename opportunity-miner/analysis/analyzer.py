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
from database.models import Opportunity, RawPost, normalize_concept_key, opportunity_from_payload
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
    "business_model: who pays, and the concrete price unit. "
    "Do not stop at SaaS aboneliği. Name the unit, for example Cluster başına aylık $49, lisanslama, or sunucu başı komisyon. "
    "top_competitors: 2 or 3 main global or local competitors. "
    "Each item names the competitor and an estimated model or presence, "
    "for example CompetitorA (Commercial SaaS) or CompetitorB (Open Source). "
    "competitor_weakness: the primary user complaint or the gap in those existing solutions. "
    "search_trend_signal: a short qualitative search-volume and hype trajectory in Turkish, "
    "for example Son 90 günde %120+ arama hacmi artışı or Stabil / Niş arama hacmi. "
    "Do not invent a percentage the source does not support. If the trajectory is unknown, write Sinyal belirsiz. "
    "Search for non-obvious, highly inventive, surprisingly clever B2C, B2B, or AI-native opportunities "
    "that provoke a Why didn't I think of that? reaction. "
    "aha_angle: 1 or 2 Turkish sentences on exactly why this solution is non-obvious and clever. "
    "unfair_advantage: the unique moat, growth hook, or distribution trick that makes this hard to beat. "
    "If the idea is obvious, boring, or generic, say that plainly in aha_angle instead of inventing cleverness. "
    "If the post is an old problem with no unlock from about the last 6 months, or a job a general chatbot already does "
    "(PDF chat, rewrite, meeting notes, generic support bot, generic image or video wrapper), do not dress it up as a startup. "
    "Say that in aha_angle and keep the draft score under 50. "
    "summary_tr: exactly 2 plain Turkish sentences. Explain like I'm 5. No DevOps or computer-science degree required. "
    "jargon_analogy: if the idea uses Kubernetes, RAG, a vector database, MCP, or CI/CD, add one Turkish line that turns that jargon into a physical analogy, "
    "for example Kubernetes = Binlerce sunucuyu otomatik yöneten dijital kargo limanı müdürü. Otherwise leave jargon_analogy empty. "
    "concept_key: one lowercase snake_case id for the underlying mechanic and problem. Ignore brand names and titles. "
    "Kupe Cloud and EasyKube both become kubernetes_deployment_simplifier. "
    "AI Video Gen from PR and RepoToVideo both become github_pr_to_video_generator. "
    "PDF Chatbot and AskPDF both become pdf_document_qna_bot. "
    "net_problem: Tam olarak kime, ne acı çektiriyor? Name the person, the place, and the concrete operational nightmare or the money they lose. "
    "Example shape: Şirketler karmaşık Kubernetes YAML dosyalarını yönetmek için ayda binlerce dolar DevOps mühendisine ödüyor. "
    "solution_mechanism: what the product automates in the background, or what it finishes in one click. "
    "why_now: Pazardaki güncel fırsat veya trend nedir? "
    "Strictly forbid these filler formulas, and do not paraphrase them: "
    "X yapmak isteyen kullanıcılara acı çektiriyor. "
    "Gelişmiş arayüz ve kullanıcı dostu yapısıyla ayrışır. "
    "Bu alanda önemli bir boşluk var. "
    "Evaluate each idea as one of three opportunity types, not only as a local Turkish clone. "
    "opportunity_type must be exactly one of: "
    "Yeni & Kopyalanabilir (Fresh & Copyable: a modern product that is new globally and can be cloned and adapted quickly); "
    "Zamanının Ötesinde / Yeniden Canlandırılabilir (Underhyped & Revived: an older product that stayed dim because the technology or AI of its time was not enough, and that today's stack such as LLM, RAG, and automation can turn into a large opportunity); "
    "TR Pazar Fırsatı (Local Adaptation: a localization opportunity specific to the Turkish market). "
    "Give equal or higher weight to large-scale enterprise platforms, multi-tenant B2B SaaS, and infrastructure ecosystems, alongside ideas that can ship as a 1-day tool. "
    "project_scale: a short Turkish label such as Hızlı MVP, Büyük Çaplı B2B Platformu, or Yüksek Hype B2C. "
    "If it is a developer tool, say whether it is a classical helper (linter, CLI, code cleanup, CRUD generator) "
    "or AI-native infrastructure (LLM observability, MCP, agent memory, guardrails, AI evals). "
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
    "If it is a DevTool, restrict the evaluation to AI Infrastructure and AI-Native Enablers. "
    "Strictly identify outdated or saturated post-AI wrapper ideas. "
    "If the idea is already monopolized or trivialized post-2023, set saturated_wrapper true and "
    "rate its opportunity_score strictly below 50 so it gets filtered out. "
    "Do not restrict opportunity_score to minor micro-tools. "
    "devtool_class is pre_ai, ai_infra, or none. "
    "pre_ai means a classical or pre-AI developer tool: traditional developer tools, code cleanups, "
    "basic linters, standard CLI helpers, or CRUD generators that a developer can easily write with "
    "Cursor, ChatGPT, or Claude. Force opportunity_score strictly below 50. "
    "In judge_verdict, explicitly include the words Pre-AI obsolete tool and REJECT. "
    "ai_infra means a high-value AI ecosystem tool. Actively seek these and score them from 80 to 95: "
    "LLM observability and cost tracking (token usage, latency, hallucination evaluation); "
    "MCP and agent connectors (custom MCP servers, agent tool-calling frameworks, agent memory or RAG infrastructure); "
    "AI safety and guardrails (prompt injection shields, output validators, automated red-teaming for LLM apps); "
    "AI-native testing and benchmark tools (automated evaluation harnesses for AI agents and workflows). "
    "An ai_infra DevTool passes only with opportunity_score from 80 to 95; otherwise set it strictly below 50. "
    "In judge_verdict, explicitly include the words High-Value AI Ecosystem Infrastructure and PASS. "
    "A favorite-based penalty for niche low-level utilities does not apply to ai_infra. "
    "The Scout brief may include top_competitors, competitor_weakness, and search_trend_signal. "
    "Use them when you judge saturation, an open-source alternative, and whether customers will pay. "
    "If the Scout left any of those three empty, fill them. Do not invent a search percentage. "
    "Apply the Aha! Factor filter. Seek non-obvious, highly inventive B2C, B2B, and AI-native ideas "
    "that provoke Why didn't I think of that? "
    "Set obvious true and opportunity_score strictly below 50 for obvious, boring, or generic ideas, even if useful. "
    "That includes generic PDF summarizers, basic social media schedulers, standard lead-gen scrapers, "
    "typical AI customer-service bots, and direct first-degree solutions many developers can build immediately. "
    "aha_pillars uses only these ids: cross_domain, friction_hack, viral, why_now. "
    "cross_domain: merging two seemingly unrelated workflows, such as automated video rendering triggered by a GitHub pull request. "
    "friction_hack: a painful, universally tolerated daily routine solved in a counter-intuitive way. "
    "viral: the product markets itself through its execution flow or a public data trigger. "
    "why_now: a technical unlock from about the last 6 months (MCP, a newly opened public API, or a recent regulation) "
    "that was not feasible before. "
    "Passing requires why_now plus at least one other pillar. Otherwise set opportunity_score strictly below 50. Do not leave it at 79. "
    "Set stale_problem true when the problem and the usual fix are old and nothing from about the last 6 months newly makes the product possible. "
    "Set ai_solved true when ChatGPT, Claude, Cursor, or another general AI product already does the job without this startup. "
    "Examples: PDF chat, summarizers, text rewriters, meeting notes, generic customer-service bots, basic image or video wrappers. "
    "Do not set ai_solved for a non-obvious mechanic that merely uses a model, and do not set it for AI-native infrastructure. "
    "Either flag forces opportunity_score strictly below 50. "
    "If the Scout left aha_angle or unfair_advantage empty, fill them in Turkish. Do not invent a clever angle for an obvious idea. "
    "concept_key: the mechanic id in lowercase snake_case, with no brand or product title. "
    "If the Scout key names a product instead of the mechanic, replace it. "
    "Examples: kubernetes_deployment_simplifier, github_pr_to_video_generator, pdf_document_qna_bot. "
    "If net_problem, summary_tr, solution_mechanism, or business_model is generic filler "
    "(acı çektiriyor as a formula, kullanıcı dostu arayüz, önemli bir boşluk var, or SaaS aboneliği with no price unit), "
    "set opportunity_score strictly below 50. "
    "Provide an 'opportunity_score' strictly between 0 and 100 based on SaaS / business potential. "
    "Return that score as an integer. "
    "Return JSON with opportunity_score, judge_verdict, saturated_wrapper, open_source_alternative, "
    "mrr_likely, devtool_class, top_competitors, competitor_weakness, search_trend_signal, "
    "obvious, stale_problem, ai_solved, aha_pillars, aha_angle, unfair_advantage, and concept_key."
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
        "and lower scores to overly niche low-level code/infrastructure utilities. "
        "Never use that preference to raise a stale problem or a job a general chatbot already does. Those stay under 50. "
        "Do not apply that penalty to AI-native infrastructure such as LLM observability, MCP servers, "
        "agent memory, guardrails, or AI evaluation harnesses."
    )

_BLOCKED_STATUS = {401, 403, 429}


class OpenAIUnavailable(RuntimeError):
    """Auth or quota failure. The scan leaves the post queued and moves on."""


class DuplicateConcept(Exception):
    """This mechanic was already stored in the last 30 days. Do not call the Judge."""

    def __init__(self, concept_key: str) -> None:
        super().__init__(concept_key)
        self.concept_key = concept_key


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
        self._concepts: set[str] = set()
        self._concept_lock = asyncio.Lock()

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
            scout_key = normalize_concept_key(scout.get("concept_key") if isinstance(scout, dict) else "")
            # shortcut: exact concept_key match, not a similarity search; a second brand name only collapses after the Judge rewrites it
            if scout_key and await self._concept_taken(scout_key):
                raise DuplicateConcept(scout_key)
            try:
                judge_user = (
                    "Scout brief JSON:\n"
                    + clip(json.dumps(scout, ensure_ascii=False), 14000)
                    + f"\n\nBaşlık: {post.title or ''}"
                )
                judge_text = await self._complete(judge_user, clause, system=JUDGE_PROMPT)
                opportunity = merge_committee(scout, parse_json_object(judge_text))
            except Exception:
                self._release_concept(scout_key)
                raise
            final = opportunity.concept_key
            if final and final != scout_key and await self._concept_taken(final):
                self._release_concept(scout_key)
                raise DuplicateConcept(final)
            return opportunity

        return await asyncio.wait_for(_run(), timeout=self.settings.analyzer_timeout_seconds)

    async def _concept_taken(self, key: str) -> bool:
        async with self._concept_lock:
            if key in self._concepts:
                return True
            if self._db is not None and await self._db.concept_seen(key):
                return True
            self._concepts.add(key)
            return False

    def _release_concept(self, key: str) -> None:
        if key:
            self._concepts.discard(key)

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
            "max_completion_tokens": 8192,
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
    for key in (
        "top_competitors",
        "competitor_weakness",
        "search_trend_signal",
        "aha_angle",
        "unfair_advantage",
    ):
        if _blank(payload.get(key)) and not _blank(judge.get(key)):
            payload[key] = judge.get(key)
    if not _blank(judge.get("aha_pillars")):
        payload["aha_pillars"] = judge.get("aha_pillars")
    if "obvious" in judge:
        payload["obvious"] = judge.get("obvious")
    if "stale_problem" in judge:
        payload["stale_problem"] = judge.get("stale_problem")
    if "ai_solved" in judge:
        payload["ai_solved"] = judge.get("ai_solved")
    if not _blank(judge.get("concept_key")):
        payload["concept_key"] = judge.get("concept_key")
    opportunity = opportunity_from_payload(payload)
    gated = apply_judge_gates(opportunity.opportunity_score, judge)
    verdict = _stamp_creativity_verdict(_stamp_devtool_verdict(opportunity.judge_verdict, judge, gated), judge)
    updates: dict = {}
    if gated != opportunity.opportunity_score:
        updates["opportunity_score"] = gated
    if verdict != opportunity.judge_verdict:
        updates["judge_verdict"] = verdict
    if updates:
        return opportunity.model_copy(update=updates)
    return opportunity


_PRE_AI = {"pre ai", "preai", "classical", "obsolete", "micro", "low margin", "reject"}
_AI_INFRA = {"ai infra", "ai infrastructure", "ai native", "enterprise", "security", "b2b security"}


def _devtool_kind(judge: dict) -> str:
    kind = str(judge.get("devtool_class") or "").strip().lower().replace("-", " ").replace("_", " ")
    kind = " ".join(kind.split())
    if kind in _PRE_AI:
        return "pre_ai"
    if kind in _AI_INFRA:
        return "ai_infra"
    return ""


def apply_judge_gates(score: int, judge: dict) -> int:
    """Enforce the reject rules in code so a soft model score cannot sneak through."""
    kind = _devtool_kind(judge)
    creative = aha_pillar_count(judge) >= 2 and _has_why_now(judge)
    if (
        _yes(judge.get("saturated_wrapper"))
        or kind == "pre_ai"
        or is_obvious(judge)
        or is_stale(judge)
        or is_ai_solved(judge)
        or not creative
    ):
        return min(score, 49)
    if kind == "ai_infra":
        if score < 80:
            return min(score, 49)
        return min(score, 95)
    return score


def clears_aha_bar(score: int, analysis: dict | None) -> bool:
    """Telegram only gets a current, non-obvious idea: why_now plus one more pillar.

    The numeric score is enforced by the notify threshold. This bar is the creative one.
    """
    del score
    payload = analysis or {}
    if is_obvious(payload) or is_stale(payload) or is_ai_solved(payload):
        return False
    return aha_pillar_count(payload) >= 2 and _has_why_now(payload)


def is_stale(source: object) -> bool:
    return isinstance(source, dict) and _yes(source.get("stale_problem"))


def is_ai_solved(source: object) -> bool:
    return isinstance(source, dict) and _yes(source.get("ai_solved"))


def is_obvious(source: object) -> bool:
    if not isinstance(source, dict):
        return False
    if _yes(source.get("obvious")):
        return True
    label = " ".join(str(source.get("aha_class") or "").casefold().replace("-", " ").split())
    return label in {"obvious", "boring", "generic"}


def aha_pillar_count(source: object) -> int:
    return len(_pillar_ids(source))


def _has_why_now(source: object) -> bool:
    return "why_now" in _pillar_ids(source)


def _pillar_ids(source: object) -> set[str]:
    raw = source.get("aha_pillars") if isinstance(source, dict) else source
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace(",", "|").split("|")]
    elif isinstance(raw, list):
        items = [str(item).strip() for item in raw]
    else:
        items = []
    found: set[str] = set()
    for item in items:
        pillar = _pillar_id(item)
        if pillar:
            found.add(pillar)
    return found


def _pillar_id(text: str) -> str:
    key = " ".join(text.casefold().replace("-", " ").replace("_", " ").replace("/", " ").split())
    if not key:
        return ""
    if "cross domain" in key or "arbitrage" in key:
        return "cross_domain"
    if "friction" in key:
        return "friction_hack"
    if "viral" in key or "organic" in key:
        return "viral"
    if "why now" in key or key in {"unlock", "regulation"}:
        return "why_now"
    return ""


def _stamp_creativity_verdict(verdict: str, judge: dict) -> str:
    if is_ai_solved(judge):
        tag = "AI devrimi çözmüş (REJECT)"
    elif is_stale(judge):
        tag = "Eski problem (REJECT)"
    else:
        return verdict
    if tag.split(" (", 1)[0] in verdict:
        return verdict
    return clip(f"{tag}. {verdict}".strip(), 1200)


def _stamp_devtool_verdict(verdict: str, judge: dict, score: int) -> str:
    kind = _devtool_kind(judge)
    if kind == "pre_ai":
        tag = "Pre-AI obsolete tool (REJECT)"
    elif kind == "ai_infra" and score >= 80:
        tag = "High-Value AI Ecosystem Infrastructure (PASS)"
    elif kind == "ai_infra":
        tag = "High-Value AI Ecosystem Infrastructure (REJECT)"
    else:
        return verdict
    if tag.split(" (", 1)[0] in verdict:
        return verdict
    return clip(f"{tag}. {verdict}".strip(), 1200)


def _blank(value: object) -> bool:
    if isinstance(value, list):
        return not any(str(item).strip() for item in value)
    return not str(value or "").strip()


def _yes(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "evet"}
