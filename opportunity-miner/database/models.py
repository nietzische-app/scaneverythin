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
    turkish_market: TurkishMarket = Field(default_factory=TurkishMarket)
    blueprint: Blueprint = Field(default_factory=Blueprint)
    vibe_coding: VibeCoding = Field(default_factory=VibeCoding)
    risks: list[str] = Field(default_factory=list)
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
    """Normalize a DeepSeek JSON object into the brief we store and notify."""
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
    kind = str(payload.get("opportunity_type") or "other").strip().lower().replace("-", "_").replace(" ", "_")
    if kind not in {"micro_saas", "b2b_tool", "indie_game", "content_tool", "other"}:
        kind = "other"
    blueprint_format = str(blueprint_raw.get("format") or "").strip().lower()
    if blueprint_format not in {"core_loop", "mvp_features"}:
        blueprint_format = "core_loop" if kind == "indie_game" else "mvp_features"
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
    title = clip(str(payload.get("title_tr") or "").strip(), 180) or "İsimsiz fırsat"
    return Opportunity(
        title_tr=title,
        summary_tr=clip(str(payload.get("summary_tr") or "").strip(), 900),
        core_problem_or_mechanic=clip(str(payload.get("core_problem_or_mechanic") or "").strip(), 600),
        opportunity_type=kind,
        turkish_market=market,
        blueprint=Blueprint(
            format=blueprint_format,
            points=points,
            mvp_scope=clip(str(blueprint_raw.get("mvp_scope") or ""), 600),
        ),
        vibe_coding=vibe,
        risks=risks,
        opportunity_score=score,
    )


def dumps(payload: dict[str, Any] | BaseModel) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return json.dumps(payload, ensure_ascii=False)
