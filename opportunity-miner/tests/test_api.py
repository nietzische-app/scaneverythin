"""HTTP surface: dashboard, manual scan, and the bearer gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from tests.support import (
    ScriptedAnalyzer,
    ScriptedClassifier,
    ScriptedNotifier,
    StaticFetcher,
    brief,
    gold_classification,
    sample_post,
)

from config import Settings
from delivery.telegram_bot import format_alert, plain_alert, text_chunks
from main import create_app
from pipeline import ScanService


def _service_factory(notifier: ScriptedNotifier, title: str):
    def factory(settings, db):
        return ScanService(
            settings,
            db,
            fetchers=[StaticFetcher([sample_post(title="<img src=x onerror=alert(1)> Invoices")])],
            classifier=ScriptedClassifier(gold_classification()),
            analyzer=ScriptedAnalyzer(brief(title=title)),
            notifier=notifier,
        )

    return factory


class ApiTests(unittest.TestCase):
    def test_dashboard_escapes_markup_and_lists_the_brief(self) -> None:
        notifier = ScriptedNotifier()
        title = "<script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                database_path=str(Path(tmp) / "radar.sqlite"),
                scheduler_enabled=False,
                api_token="",
            )
            app = create_app(settings, service_factory=_service_factory(notifier, title))
            with TestClient(app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertFalse(health.json()["providers"]["gemini"])
                denied = client.post("/run-scan?wait=true")
                self.assertEqual(denied.status_code, 200)
                page = client.get("/")
                self.assertEqual(page.status_code, 200)
                self.assertIn("Fırsat Radarı", page.text)
                self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page.text)
                self.assertNotIn("<script>alert(1)</script>", page.text)
                insights = client.get("/api/insights")
                self.assertEqual(insights.status_code, 200)
                self.assertEqual(insights.json()[0]["analysis"]["title_tr"], title)
                self.assertIn("<img src=x", insights.json()[0]["title"])
                self.assertEqual(len(notifier.sent), 1)

    def test_bearer_token_gates_the_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                database_path=str(Path(tmp) / "radar.sqlite"),
                scheduler_enabled=False,
                api_token="secret-token",
            )
            app = create_app(settings, service_factory=_service_factory(ScriptedNotifier(), "Kapı"))
            with TestClient(app) as client:
                blocked = client.post("/run-scan?wait=true")
                self.assertEqual(blocked.status_code, 401)
                allowed = client.post(
                    "/run-scan?wait=true",
                    headers={"Authorization": "Bearer secret-token"},
                )
                self.assertEqual(allowed.status_code, 200)
                self.assertEqual(allowed.json()["report"]["analyzed"], 1)

    def test_telegram_card_escapes_markdown(self) -> None:
        from database.models import InsightView

        view = InsightView(
            insight_id=1,
            item_id=1,
            source="hn",
            title="v1.2!",
            url="https://news.ycombinator.com/item?id=5",
            confidence=91,
            opportunity_score=82,
            analysis=brief(title="v1.2!").model_dump(),
            telegram_sent=0,
            created_at="2026-10-27T00:00:00+00:00",
        )
        card = format_alert(view)
        self.assertIn(r"v1\.2\!", card)
        self.assertIn("Kaynak Bağlantısı", card)
        self.assertIn("FIRSAT RADARI", card)
        self.assertIn("TR Rekabet Durumu", card)
        self.assertNotIn("v1.2!", card)
        analysis = brief(title="v1.2!").model_dump()
        analysis["net_problem"] = "Ajanslar tahsilatı kaçırıyor"
        analysis["solution_mechanism"] = "Tek hatırlatma gönderir"
        analysis["why_now"] = "E-fatura zorunlu"
        view.analysis = analysis
        detailed = format_alert(view)
        self.assertIn("Net Problem", detailed)
        self.assertIn("Çözüm & Mekanizma", detailed)
        self.assertIn("Neden Şimdi?", detailed)
        self.assertIn("Gelir Modeli", detailed)
        self.assertIn("Ajanslar tahsilatı kaçırıyor", detailed)
        analysis["summary_tr"] = "Ajans her faturayı elle kovalıyor. Bu araç ödenince susuyor."
        analysis["jargon_analogy"] = "Kubernetes = Binlerce sunucuyu otomatik yöneten dijital kargo limanı müdürü"
        analysis["business_model"] = "Ajans başına aylık $49"
        view.analysis = analysis
        concrete = format_alert(view)
        self.assertIn("elle kovalıyor", concrete)
        self.assertIn("kargo limanı", concrete)
        self.assertIn("Ajans başına aylık $49", concrete)
        from database.models import InsightView

        long_view = InsightView(
            insight_id=1,
            item_id=1,
            source="hn",
            title="uzun",
            url="https://news.ycombinator.com/item?id=5",
            confidence=91,
            opportunity_score=82,
            analysis=brief().model_dump(),
            telegram_sent=0,
            created_at="2026-10-27T00:00:00+00:00",
        )
        long_view.analysis["judge_verdict"] = "SON " + ("kelime " * 900)
        card = plain_alert(long_view)
        self.assertIn("SON", card)
        self.assertTrue(card.rstrip().endswith("━"))
        chunks = text_chunks(card, limit=700)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 700 for chunk in chunks))
        self.assertIn("SON", "".join(chunks))


if __name__ == "__main__":
    unittest.main()
