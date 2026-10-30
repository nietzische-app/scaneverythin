"""Shared fixtures for the radar tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.models import (  # noqa: E402
    Blueprint,
    Classification,
    Opportunity,
    RawPost,
    TurkishMarket,
    VibeCoding,
)
from ingestion.base import FetchBatch  # noqa: E402


def sample_post(**overrides) -> RawPost:
    payload = {
        "source": "hn",
        "external_id": "101",
        "url": "https://news.ycombinator.com/item?id=101",
        "title": "Show HN: Invoices that chase themselves",
        "body": (
            "Freelancers keep a spreadsheet of who has not paid. "
            "I want a tool that sends a polite reminder and stops when the invoice is marked paid."
        ),
        "author": "ada",
        "score": 42,
        "comment_count": 11,
    }
    payload.update(overrides)
    return RawPost(**payload)


def gold_classification() -> Classification:
    return Classification(
        is_human=True,
        has_signal=True,
        has_mechanism=True,
        is_relevant=True,
        should_forward=True,
        confidence=91,
        reasons={"is_human": "specific complaint"},
    )


def brief(score: int = 82, title: str = "Tahsilat hatırlatıcısı") -> Opportunity:
    return Opportunity(
        title_tr=title,
        summary_tr="Serbest çalışanlar ödenmemiş faturaları tek tek kovalıyor.",
        core_problem_or_mechanic="Ödenmemiş fatura için nazik ama ısrarlı bir hatırlatma döngüsü.",
        opportunity_type="micro_saas",
        turkish_market=TurkishMarket(
            adaptation="Küçük ajanslar e-fatura sonrası WhatsApp ile hatırlatma ister.",
            payments="gerekmez",
            commerce_or_messaging="WhatsApp otomasyonu",
            kvkk="Fatura üzerindeki kişisel veri için aydınlatma gerekir.",
            fit_score=80,
        ),
        blueprint=Blueprint(
            format="mvp_features",
            points=["CSV içe aktar", "Tek hatırlatma", "Ödendi işareti"],
            mvp_scope="Muhasebe entegrasyonu yok",
        ),
        vibe_coding=VibeCoding(
            difficulty_days=3,
            why="Form, kuyruk ve bir mesaj şablonu yeter.",
            first_slice="Tek fatura, tek hatırlatma",
        ),
        risks=["İletiler spam klasörüne düşebilir"],
        aha_angle="Ödeme işareti gelince hatırlatma kendi kendine susuyor.",
        unfair_advantage="Her giden mesaj faturanın linkini taşır ve ajans içinde yayılır.",
        aha_pillars=["friction_hack", "viral", "why_now"],
        current_problem=True,
        opportunity_score=score,
    )


class StaticFetcher:
    def __init__(self, posts, errors=None, source: str = "hn") -> None:
        self.source = source
        self.posts = posts
        self.errors = errors or []
        self.calls = 0

    async def fetch(self) -> FetchBatch:
        self.calls += 1
        return FetchBatch(source=self.source, posts=list(self.posts), errors=list(self.errors))


class ScriptedClassifier:
    def __init__(self, result, available: bool = True) -> None:
        self.result = result
        self.available = available
        self.calls = 0

    async def classify(self, post: RawPost) -> Classification:
        self.calls += 1
        if callable(self.result):
            return self.result(post)
        return self.result


class ScriptedAnalyzer:
    def __init__(self, opportunity: Opportunity, available: bool = True) -> None:
        self.opportunity = opportunity
        self.available = available
        self.calls = 0

    async def analyze(self, post: RawPost) -> Opportunity:
        self.calls += 1
        if isinstance(self.opportunity, Exception):
            raise self.opportunity
        return self.opportunity

    async def aclose(self) -> None:
        return None


class ScriptedNotifier:
    def __init__(self, available: bool = True, ok: bool = True) -> None:
        self.available = available
        self.ok = ok
        self.sent = []

    async def send(self, view) -> bool:
        self.sent.append(view)
        return self.ok
