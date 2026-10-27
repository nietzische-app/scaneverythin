"""Stage-1 filter on Gemini Flash.

The five checks are atomic and independent. The default mode asks all five
in one JSON call so a 30-minute scan stays inside the free-tier daily quota.
Set CLASSIFIER_MODE=parallel to spend five calls per post instead.

Server-side scoring is authoritative. A post cannot reach Stage-2 unless the
required flags are true and the blended confidence is at least the threshold
(the pipeline applies that threshold; this module only computes the score).
"""

from __future__ import annotations

import asyncio
import logging

from config import Settings
from database.models import Classification, RawPost
from textutil import as_bool, clip, parse_json_object

logger = logging.getLogger(__name__)

ATOMIC_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "is_human",
        "Is this written by a real human (not a generic bot or pure ad)?",
    ),
    (
        "has_signal",
        "Does it express a real pain point, software/game request, or successful micro-SaaS launch?",
    ),
    (
        "has_mechanism",
        "Is there a potential business/software/game mechanism hidden here?",
    ),
    (
        "is_relevant",
        "Is this relevant for micro-SaaS, B2B tool, or indie mobile game development?",
    ),
    (
        "should_forward",
        "Should this be forwarded to deep analysis?",
    ),
)

_REQUIRED = ("is_human", "has_signal", "is_relevant", "should_forward")
_FLAG_KEYS = tuple(key for key, _question in ATOMIC_QUESTIONS)
FLASH_MODELS = ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash")


def gemini_model_id(name: str | None) -> str:
    """Bare id. google.genai adds `models/` once; a second prefix 404s."""
    raw = (name or "").strip()
    while raw.startswith("models/"):
        raw = raw.removeprefix("models/")
    return raw or FLASH_MODELS[0]


def gemini_model_candidates(name: str | None) -> list[str]:
    chosen = gemini_model_id(name)
    ordered = [chosen]
    for fallback in FLASH_MODELS:
        if fallback not in ordered:
            ordered.append(fallback)
    return ordered


def render_post(post: RawPost, limit: int = 3500) -> str:
    body = clip((post.body or "").strip(), limit)
    author = post.author or "-"
    return (
        f"Source: {post.source}\n"
        f"Author: {author}\n"
        f"Score: {post.score if post.score is not None else '-'}\n"
        f"Comments: {post.comment_count if post.comment_count is not None else '-'}\n"
        f"Title: {post.title}\n"
        f"URL: {post.url}\n\n"
        f"{body}"
    )


def build_batched_prompt(post: RawPost) -> str:
    lines = [
        "You are a harsh noise filter for an opportunity radar.",
        "Politics, memes, generic news, hiring posts, and pure ads are NO.",
        "Answer these 5 atomic yes/no questions about the post.",
        "Return one JSON object and nothing else.",
        "",
        render_post(post),
        "",
        "Questions:",
    ]
    for index, (key, question) in enumerate(ATOMIC_QUESTIONS, start=1):
        lines.append(f"{index}. {key}: {question}")
    lines.extend(
        [
            "",
            "JSON schema:",
            "{",
            '  "is_human": true,',
            '  "has_signal": false,',
            '  "has_mechanism": false,',
            '  "is_relevant": false,',
            '  "should_forward": false,',
            '  "model_confidence": 0,',
            '  "reasons": {"is_human": "", "has_signal": "", "has_mechanism": "", "is_relevant": "", "should_forward": ""}',
            "}",
            "model_confidence is an integer from 0 to 100. Be conservative.",
        ]
    )
    return "\n".join(lines)


def build_atomic_prompt(post: RawPost, key: str, question: str) -> str:
    return (
        "You are a harsh noise filter. Answer exactly one question.\n"
        "Return JSON only: {\"answer\": true or false, \"note\": \"short reason\"}\n\n"
        f"{render_post(post)}\n\n"
        f"Question id: {key}\n"
        f"{question}"
    )


def score_from_flags(flags: dict[str, bool], model_confidence: int | None) -> int:
    """Blend the five answers with the model's own confidence, then cap rejects."""
    earned = sum(1 for key in _FLAG_KEYS if flags.get(key))
    flag_score = int(round(earned / len(_FLAG_KEYS) * 100))
    if model_confidence is None:
        score = flag_score
    else:
        bounded = max(0, min(100, int(model_confidence)))
        score = min(flag_score, bounded)
    if not all(flags.get(key) for key in _REQUIRED):
        score = min(score, 69)
    elif not flags.get("has_mechanism"):
        score = min(score, 84)
    return max(0, min(100, score))


def classification_from_payload(payload: dict) -> Classification:
    nested = payload.get("questions") if isinstance(payload.get("questions"), dict) else {}
    flags: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    raw_reasons = payload.get("reasons") if isinstance(payload.get("reasons"), dict) else {}
    for key, _question in ATOMIC_QUESTIONS:
        if key in payload:
            flags[key] = as_bool(payload.get(key))
        elif key in nested:
            flags[key] = as_bool(nested.get(key))
        else:
            flags[key] = False
        reason = raw_reasons.get(key) or ""
        reasons[key] = clip(str(reason), 240)
    model_confidence = payload.get("model_confidence", payload.get("confidence"))
    parsed_confidence: int | None
    try:
        parsed_confidence = int(model_confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed_confidence = None
    confidence = score_from_flags(flags, parsed_confidence)
    return Classification(
        is_human=flags["is_human"],
        has_signal=flags["has_signal"],
        has_mechanism=flags["has_mechanism"],
        is_relevant=flags["is_relevant"],
        should_forward=flags["should_forward"],
        confidence=confidence,
        reasons=reasons,
    )


_NON_TEXT = ("image", "imagen", "tts", "embed", "live", "audio", "video", "veo")


def _text_model(model_id: str) -> bool:
    name = model_id.lower()
    return bool(name) and not any(part in name for part in _NON_TEXT)


class GeminiClassifier:
    def __init__(self, settings: Settings, generate=None) -> None:
        self.settings = settings
        self._generate_override = generate
        self._client = None
        self._dead: set[str] = set()
        self._working: str | None = None
        self._catalog: list | None = None

    @property
    def available(self) -> bool:
        return self._generate_override is not None or self.settings.gemini_configured

    async def classify(self, post: RawPost) -> Classification:
        if not self.available:
            return Classification(error="gemini_not_configured")
        try:
            if self.settings.classifier_mode == "parallel":
                return await self._classify_parallel(post)
            return await self._classify_batched(post)
        except Exception as exc:  # noqa: BLE001 — one bad response must not kill the scan
            logger.warning("gemini classification failed: %s", exc)
            return Classification(error=clip(str(exc), 300))

    async def _classify_batched(self, post: RawPost) -> Classification:
        raw = await self._generate(build_batched_prompt(post))
        return classification_from_payload(parse_json_object(raw))

    async def _classify_parallel(self, post: RawPost) -> Classification:
        async def ask(key: str, question: str) -> tuple[str, bool, str]:
            raw = await self._generate(build_atomic_prompt(post, key, question))
            payload = parse_json_object(raw)
            note = clip(str(payload.get("note") or payload.get("reason") or ""), 240)
            return key, as_bool(payload.get("answer")), note

        rows = await asyncio.gather(*(ask(key, question) for key, question in ATOMIC_QUESTIONS))
        flags = {key: answer for key, answer, _note in rows}
        reasons = {key: note for key, _answer, note in rows}
        confidence = score_from_flags(flags, None)
        return Classification(
            is_human=flags["is_human"],
            has_signal=flags["has_signal"],
            has_mechanism=flags["has_mechanism"],
            is_relevant=flags["is_relevant"],
            should_forward=flags["should_forward"],
            confidence=confidence,
            reasons=reasons,
        )

    async def _generate(self, prompt: str) -> str:
        if self._generate_override is not None:
            result = self._generate_override(prompt)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return await asyncio.wait_for(
            asyncio.to_thread(self._generate_sync, prompt),
            timeout=self.settings.llm_timeout_seconds,
        )

    def _generate_sync(self, prompt: str) -> str:
        client = self._sdk()
        if self._working:
            text = self._accept(client, self._working, prompt)
            if text:
                return text
            self._working = None
        last: Exception | None = None
        tried: list[str] = []
        all_missing = True
        for model in gemini_model_candidates(self.settings.gemini_model):
            model_id = gemini_model_id(model)
            if model_id in tried or model_id in self._dead or not _text_model(model_id):
                continue
            tried.append(model_id)
            text, exc, missing = self._take(client, model_id, prompt)
            if text:
                return text
            last = exc
            all_missing = all_missing and missing
        if all_missing:
            logger.warning("hardcoded Gemini models returned 404; listing models")
            for discovered in self._listed_models(client, tried):
                text, exc, _missing = self._take(client, discovered, prompt)
                if text:
                    return text
                last = exc
        if last is not None:
            raise last
        raise RuntimeError("no Gemini model id to call")

    def _accept(self, client, model_id: str, prompt: str) -> str | None:
        text, exc, _missing = self._take(client, model_id, prompt)
        if text:
            return text
        if exc is not None and not _skip_model(exc):
            if not _empty(exc):
                raise exc
        return None

    def _take(
        self, client, model_id: str, prompt: str
    ) -> tuple[str | None, Exception | None, bool]:
        text, exc = self._one(client, model_id, prompt)
        if text:
            self._working = model_id
            return text, None, False
        if _skip_model(exc):
            self._dead.add(model_id)
            if exc is not None and not _model_missing(exc):
                logger.warning(
                    "Gemini model %s has free-tier quota 0; trying the next id",
                    model_id,
                )
            return None, exc, True
        if exc is not None and not _empty(exc):
            raise exc
        return None, exc, False

    def _one(self, client, model_id: str, prompt: str) -> tuple[str | None, Exception | None]:
        logger.warning("requesting Gemini model %s on api %s", model_id, _api_version(client))
        try:
            text = self._complete(client, model_id, prompt)
        except Exception as exc:
            if _model_missing(exc):
                logger.warning(
                    "Gemini model %s returned 404 NOT_FOUND; trying the next id",
                    model_id,
                )
            return None, exc
        if text.strip():
            return text, None
        return None, RuntimeError(f"empty response from {model_id}")

    def _listed_models(self, client, tried: list[str]) -> list[str]:
        if self._catalog is None:
            try:
                self._catalog = list(client.models.list())
            except Exception as exc:
                logger.warning("Gemini models.list failed: %s", exc)
                return []
        usable: list[str] = []
        for item in self._catalog:
            actions = getattr(item, "supported_actions", None) or []
            if actions and not any(str(action).lower() == "generatecontent" for action in actions):
                continue
            model_id = gemini_model_id(str(getattr(item, "name", "") or ""))
            if (
                _text_model(model_id)
                and model_id not in tried
                and model_id not in self._dead
                and model_id not in usable
            ):
                usable.append(model_id)
        flash = [model_id for model_id in usable if "flash" in model_id]
        rest = [model_id for model_id in usable if model_id not in flash]
        return (flash + rest)[:4]

    def _complete(self, client, model: str, prompt: str) -> str:
        model_id = gemini_model_id(model)
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
        except Exception as exc:
            if _model_missing(exc):
                raise
            message = str(exc).lower()
            if "mime" not in message and "json" not in message and "schema" not in message:
                raise
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={"temperature": 0},
            )
        return getattr(response, "text", None) or ""

    def _sdk(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                api_key=self.settings.gemini_api_key,
                http_options={"api_version": "v1"},
            )
        return self._client


def _api_version(client) -> str:
    api_client = getattr(client, "_api_client", None)
    options = getattr(api_client, "_http_options", None)
    return str(getattr(options, "api_version", None) or "default")


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _model_missing(exc: Exception | None) -> bool:
    if exc is None:
        return False
    if _status_code(exc) == 404:
        return True
    message = str(exc).lower()
    return "not_found" in message or "not found" in message


def _skip_model(exc: Exception | None) -> bool:
    """404, or a model whose free-tier quota is permanently 0."""
    if _model_missing(exc) or exc is None:
        return _model_missing(exc)
    message = str(exc).lower()
    if "limit: 0" not in message and "limit:0" not in message:
        return False
    return _status_code(exc) == 429 or "resource_exhausted" in message


def _empty(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and str(exc).startswith("empty response")
