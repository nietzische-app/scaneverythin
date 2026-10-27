"""Domain records shared by the pipeline, API, and dashboard."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from textutil import clamp_int, clip


class RawPost(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    external_id: str
    url: str
    title: str
    body: str = ""
    author: str | None = None
    score: int | None = None
    comment_count: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        key = (self.external_id or "").strip() or (self.url or "").strip()
        basis = f"{self.source.strip().lower()}:{key}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()


class Classification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_human: bool = False
    has_signal: bool = False
    has_mechanism: bool = False
    is_relevant: bool = False
    should_forward: bool = False
    confidence: int = 0
    reasons: dict[str, str] = Field(default_factory=dict)
    error: str | None = None

    def passes(self, threshold: int) -> bool:
        return self.error is None and self.should_forward and self.confidence >= threshold


class TurkishMarket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    adaptation: str = ""
    payments: str = ""
    commerce_or_messaging: str = ""
    kvkk: str = ""
    fit_score: int = 0


class Blueprint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format: str = "mvp_features"
    points: list[str] = Field(default_factory=list)
    mvp_scope: str = ""


class VibeCoding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    difficulty_days: int = 3
    why: str = ""
    first_slice: str = ""


class Opportunity(BaseModel):
    """Stage-2 brief. Prose fields are Turkish; keys stay stable for the API."""

    model_config = ConfigDict(extra="ignore")

    title_tr: str
    summary_tr: str
    core_problem_or_mechanic: str
    opportunity_type: str = "other"
    project_scale: str = ""
    turkish_market: TurkishMarket = Field(default_factory=TurkishMarket)
    blueprint: Blueprint = Field(default_factory=Blueprint)
    vibe_coding: VibeCoding = Field(default_factory=VibeCoding)
    risks: list[str] = Field(default_factory=list)
    tr_market_analysis: str = ""
    our_differentiator: str = ""
    business_model: str = ""
    net_problem: str = ""
    solution_mechanism: str = ""
    why_now: str = ""
    suggested_domains: list[str] = Field(default_factory=list)
    top_competitors: list[str] = Field(default_factory=list)
    competitor_weakness: str = ""
    search_trend_signal: str = ""
    market_scale: str = ""
    judge_verdict: str = ""
    market_dial: float | None = None
    opportunity_score: int = 0


class ItemRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    source: str
    external_id: str
    url: str
    content_hash: str
    title: str
    body: str
    author: str | None = None
    score: int | None = None
    comment_count: int | None = None
    classification_json: str | None = None
    status: str
    note: str | None = None
    created_at: str
    updated_at: str

    def to_raw_post(self) -> RawPost:
        return RawPost(
            source=self.source,
            external_id=self.external_id,
            url=self.url,
            title=self.title,
            body=self.body,
            author=self.author,
            score=self.score,
            comment_count=self.comment_count,
        )

    def classification(self) -> Classification | None:
        if not self.classification_json:
            return None
        return Classification.model_validate_json(self.classification_json)


class InsightView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    insight_id: int
    item_id: int
    source: str
    title: str
    url: str
    author: str | None = None
    confidence: int
    opportunity_score: int
    analysis: dict[str, Any]
    telegram_sent: int
    user_liked: int = 0
    created_at: str


class ScanReport(BaseModel):
    started_at: str
    finished_at: str
    fetched: int = 0
    enqueued: int = 0
    duplicates: int = 0
    classified: int = 0
    filtered_out: int = 0
    analyzed: int = 0
    notified: int = 0
    deferred: int = 0
    source_errors: list[str] = Field(default_factory=list)
    item_errors: list[str] = Field(default_factory=list)


def opportunity_from_payload(payload: dict[str, Any]) -> Opportunity:
    """Normalize an OpenAI JSON object into the brief we store and notify."""
    market_raw = payload.get("turkish_market") if isinstance(payload.get("turkish_market"), dict) else {}
    blueprint_raw = payload.get("blueprint") if isinstance(payload.get("blueprint"), dict) else {}
    vibe_raw = payload.get("vibe_coding") if isinstance(payload.get("vibe_coding"), dict) else {}
    points_raw = blueprint_raw.get("points") or payload.get("points") or []
    risks_raw = payload.get("risks") or []
    if isinstance(points_raw, str):
        points_raw = [points_raw]
    if isinstance(risks_raw, str):
        risks_raw = [risks_raw]
    points = [clip(str(point).strip(), 240) for point in points_raw if str(point).strip()][:6]
    risks = [clip(str(risk).strip(), 240) for risk in risks_raw if str(risk).strip()][:4]
    raw_type = str(payload.get("opportunity_type") or "").strip()
    slug = raw_type.lower().replace("-", "_").replace(" ", "_")
    blueprint_format = str(blueprint_raw.get("format") or "").strip().lower()
    if blueprint_format not in {"core_loop", "mvp_features"}:
        blueprint_format = "core_loop" if slug == "indie_game" else "mvp_features"
    kind = _opportunity_type(raw_type)
    market = TurkishMarket(
        adaptation=clip(str(market_raw.get("adaptation") or ""), 1200),
        payments=clip(str(market_raw.get("payments") or ""), 400),
        commerce_or_messaging=clip(str(market_raw.get("commerce_or_messaging") or ""), 400),
        kvkk=clip(str(market_raw.get("kvkk") or ""), 400),
        fit_score=clamp_int(market_raw.get("fit_score"), 0, 100, 0),
    )
    vibe = VibeCoding(
        difficulty_days=clamp_int(vibe_raw.get("difficulty_days"), 1, 5, 3),
        why=clip(str(vibe_raw.get("why") or ""), 400),
        first_slice=clip(str(vibe_raw.get("first_slice") or ""), 400),
    )
    score = clamp_int(payload.get("opportunity_score"), 0, 100, market.fit_score)
    domains = _suggested_domains(payload)
    title = clip(str(payload.get("title_tr") or "").strip(), 180) or "İsimsiz fırsat"
    return Opportunity(
        title_tr=title,
        summary_tr=clip(str(payload.get("summary_tr") or "").strip(), 900),
        core_problem_or_mechanic=clip(str(payload.get("core_problem_or_mechanic") or "").strip(), 600),
        opportunity_type=kind,
        project_scale=clip(str(payload.get("project_scale") or "").strip(), 80),
        turkish_market=market,
        blueprint=Blueprint(
            format=blueprint_format,
            points=points,
            mvp_scope=clip(str(blueprint_raw.get("mvp_scope") or ""), 600),
        ),
        vibe_coding=vibe,
        risks=risks,
        tr_market_analysis=clip(str(payload.get("tr_market_analysis") or "").strip(), 1200),
        our_differentiator=clip(str(payload.get("our_differentiator") or "").strip(), 900),
        business_model=clip(str(payload.get("business_model") or "").strip(), 600),
        net_problem=clip(str(payload.get("net_problem") or "").strip(), 600),
        solution_mechanism=clip(str(payload.get("solution_mechanism") or "").strip(), 600),
        why_now=clip(str(payload.get("why_now") or "").strip(), 600),
        suggested_domains=domains,
        top_competitors=_top_competitors(payload.get("top_competitors")),
        competitor_weakness=clip(str(payload.get("competitor_weakness") or "").strip(), 900),
        search_trend_signal=clip(str(payload.get("search_trend_signal") or "").strip(), 240),
        market_scale=_market_scale(payload.get("market_scale")),
        judge_verdict=clip(str(payload.get("judge_verdict") or "").strip(), 1200),
        market_dial=_market_dial(payload.get("market_dial")),
        opportunity_score=score,
    )


_TYPE_LABELS = (
    "Yeni & Kopyalanabilir",
    "Zamanının Ötesinde / Yeniden Canlandırılabilir",
    "TR Pazar Fırsatı",
)
_TYPE_BY_FOLD = {label.casefold(): label for label in _TYPE_LABELS}
_LEGACY_TYPES = {"micro_saas", "b2b_tool", "indie_game", "content_tool", "other"}


_MARKET_SCALES = {
    "micro": "Micro",
    "smb": "SMB",
    "enterprise b2b": "Enterprise B2B",
    "enterprise_b2b": "Enterprise B2B",
    "b2c": "B2C",
}


def _market_scale(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    key = " ".join(text.casefold().split()).replace("-", " ").replace("_", " ")
    if key in _MARKET_SCALES:
        return _MARKET_SCALES[key]
    slug = key.replace(" ", "_")
    if slug in _MARKET_SCALES:
        return _MARKET_SCALES[slug]
    return clip(text, 40)


def _market_dial(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.50, min(1.00, round(value, 2)))


def _opportunity_type(raw: str) -> str:
    if not raw:
        return "other"
    folded = " ".join(raw.casefold().split())
    if folded in _TYPE_BY_FOLD:
        return _TYPE_BY_FOLD[folded]
    slug = raw.lower().replace("-", "_").replace(" ", "_")
    if slug in _LEGACY_TYPES:
        return slug
    return clip(raw, 80)


def _top_competitors(raw: object) -> list[str]:
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace("\n", "|").split("|")]
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("competitor") or "").strip()
                presence = str(item.get("presence") or item.get("model") or "").strip()
                text = f"{name} ({presence})" if name and presence else name or presence
            else:
                text = str(item).strip()
            items.append(text)
    else:
        items = []
    found: list[str] = []
    for item in items:
        name = clip(item, 120)
        if name and name not in found:
            found.append(name)
        if len(found) == 3:
            break
    return found


def _suggested_domains(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("suggested_domains") or []
    if isinstance(raw, str):
        raw = [part for chunk in raw.split(",") for part in chunk.split("|")]
    found: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        for prefix in ("https://", "http://"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        name = name.split("/")[0].strip().strip(".")
        if not name or " " in name or "." not in name or len(name) > 80:
            continue
        if name not in found:
            found.append(name)
        if len(found) == 2:
            break
    return found


def dumps(payload: dict[str, Any] | BaseModel) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return json.dumps(payload, ensure_ascii=False)
