"""MVP button, domain line, digest matrix, and the score card."""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from analysis.analyzer import PROMPT_PATH, SYSTEM_PROMPT, apply_judge_gates
from config import Settings
from database.db import Database
from database.models import InsightView, opportunity_from_payload
from delivery.card_generator import render_card
from delivery.digest import digest_due, format_digest, seconds_until_digest, send_digest
from delivery.telegram_bot import (
    TelegramNotifier,
    callback_insight_id,
    format_alert,
    grok_prompt,
    handle_callback,
    mvp_keyboard,
    mvp_prompt,
    plain_alert,
)
from ingestion.domain_checker import check_domains
from main import create_app
from pipeline import ScanService, attach_domains, market_dial, notify_threshold
from tests.support import (
    ScriptedAnalyzer,
    ScriptedClassifier,
    ScriptedNotifier,
    StaticFetcher,
    brief,
    gold_classification,
    sample_post,
)
from textutil import ISTANBUL


def _view(score: int = 82, title: str = "Nodify") -> InsightView:
    analysis = brief(score=score, title=title).model_dump()
    return InsightView(
        insight_id=7,
        item_id=7,
        source="hn",
        title=title,
        url="https://news.ycombinator.com/item?id=7",
        confidence=90,
        opportunity_score=score,
        analysis=analysis,
        telegram_sent=0,
        created_at="2026-10-27T12:00:00+00:00",
    )


class UpgradeTests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_asks_for_two_domains_and_the_parser_keeps_two(self) -> None:
        self.assertIn("suggested_domains", SYSTEM_PROMPT)
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("suggested_domains", prompt)
        self.assertIn("top_competitors", prompt)
        self.assertIn("competitor_weakness", prompt)
        self.assertIn("search_trend_signal", prompt)
        opportunity = opportunity_from_payload(
            {
                "title_tr": "Ad",
                "suggested_domains": [
                    "HTTPS://NodifyRadar.io/path",
                    "not a domain",
                    "nodify.app",
                    "extra.com",
                ],
            }
        )
        self.assertEqual(opportunity.suggested_domains, ["nodifyradar.io", "nodify.app"])
        scaled = opportunity_from_payload(
            {
                "title_tr": "Platform",
                "opportunity_type": "zamanının ötesinde / yeniden canlandırılabilir",
                "project_scale": "Büyük Çaplı B2B Platformu",
            }
        )
        self.assertEqual(scaled.opportunity_type, "Zamanının Ötesinde / Yeniden Canlandırılabilir")
        self.assertEqual(scaled.project_scale, "Büyük Çaplı B2B Platformu")
        legacy = opportunity_from_payload({"title_tr": "Oyun", "opportunity_type": "indie_game"})
        self.assertEqual(legacy.opportunity_type, "indie_game")
        self.assertEqual(legacy.blueprint.format, "core_loop")

    def test_card_shows_type_and_scale(self) -> None:
        view = _view()
        view.analysis["opportunity_type"] = "Yeni & Kopyalanabilir"
        view.analysis["project_scale"] = "Yüksek Hype B2C"
        card = plain_alert(view)
        self.assertIn("🏷️ Tip & Ölçek: Yeni & Kopyalanabilir | Yüksek Hype B2C", card)
        view.analysis["judge_verdict"] = "Açık kaynak alternatifi var, MRR zayıf."
        self.assertIn("⚖️ Yargıç / Risk Notu:", plain_alert(view))
        self.assertIn("Açık kaynak alternatifi var, MRR zayıf.", plain_alert(view))
        view.analysis["top_competitors"] = [
            "RakipA (Commercial SaaS)",
            "RakipB (Open Source)",
        ]
        view.analysis["competitor_weakness"] = "Kurulum günler sürüyor."
        view.analysis["search_trend_signal"] = "Stabil / Niş arama hacmi"
        card = plain_alert(view)
        self.assertIn("🎯 Pazar & Rakip Analizi:", card)
        self.assertIn("• ⚔️ Rakipler: RakipA (Commercial SaaS) | RakipB (Open Source)", card)
        self.assertIn("• ⚠️ Rakiplerin Zayıf Karnı: Kurulum günler sürüyor.", card)
        self.assertIn("• 📈 Arama Hacmi & Hype: Stabil / Niş arama hacmi", card)
        self.assertIn(r"RakipA \(Commercial SaaS\)", format_alert(view))

    async def test_competitor_columns_are_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "radar.sqlite"))
            await db.connect()
            await db.enqueue(sample_post())
            item = (await db.list_items(status="new"))[0]
            opportunity = opportunity_from_payload(
                {
                    "title_tr": "Radar",
                    "top_competitors": [
                        {"name": "RakipA", "presence": "Commercial SaaS"},
                        "RakipB (Open Source)",
                    ],
                    "competitor_weakness": "Kurulum günler sürüyor.",
                    "search_trend_signal": "Stabil / Niş arama hacmi",
                }
            )
            await db.add_insight(
                item_id=item.id,
                confidence=90,
                opportunity_score=80,
                flags_json="{}",
                analysis_json=opportunity.model_dump_json(),
            )
            async with db.conn.execute(
                "SELECT top_competitors, competitor_weakness, search_trend_signal FROM insights"
            ) as cursor:
                row = await cursor.fetchone()
            self.assertEqual(row["top_competitors"], "RakipA (Commercial SaaS) | RakipB (Open Source)")
            self.assertEqual(row["competitor_weakness"], "Kurulum günler sürüyor.")
            self.assertEqual(row["search_trend_signal"], "Stabil / Niş arama hacmi")
            await db.close()

    def test_judge_gates_and_quiet_day_dial(self) -> None:
        self.assertEqual(apply_judge_gates(90, {"devtool_class": "micro"}), 49)
        self.assertEqual(apply_judge_gates(90, {"saturated_wrapper": True}), 49)
        self.assertEqual(apply_judge_gates(70, {"devtool_class": "enterprise"}), 49)
        self.assertEqual(apply_judge_gates(85, {"devtool_class": "enterprise"}), 85)
        self.assertEqual(apply_judge_gates(92, {"devtool_class": "ai_infra"}), 92)
        self.assertEqual(apply_judge_gates(99, {"devtool_class": "ai_infra"}), 95)
        self.assertEqual(apply_judge_gates(90, {"devtool_class": "pre_ai"}), 49)
        from analysis.analyzer import merge_committee

        rejected = merge_committee(
            {"title_tr": "Lint"},
            {"opportunity_score": 90, "devtool_class": "pre_ai", "judge_verdict": "Cursor ile yazılır."},
        )
        self.assertEqual(rejected.opportunity_score, 49)
        self.assertIn("Pre-AI obsolete tool", rejected.judge_verdict)
        self.assertIn("REJECT", rejected.judge_verdict)
        passed = merge_committee(
            {"title_tr": "Token izleme"},
            {
                "opportunity_score": 91,
                "devtool_class": "ai_infra",
                "judge_verdict": "Halüsinasyon ve maliyet izler.",
            },
        )
        self.assertEqual(passed.opportunity_score, 91)
        self.assertIn("High-Value AI Ecosystem Infrastructure", passed.judge_verdict)
        self.assertIn("PASS", passed.judge_verdict)
        kept = merge_committee(
            {
                "title_tr": "Radar",
                "top_competitors": [
                    "RakipA (Commercial SaaS)",
                    "RakipB (Open Source)",
                    "RakipC (Local TR)",
                    "RakipD (Extra)",
                ],
                "competitor_weakness": "Kurulum uzun.",
                "search_trend_signal": "Stabil / Niş arama hacmi",
            },
            {"opportunity_score": 71, "judge_verdict": "MRR mümkün.", "top_competitors": ["Yok"]},
        )
        self.assertEqual(
            kept.top_competitors,
            ["RakipA (Commercial SaaS)", "RakipB (Open Source)", "RakipC (Local TR)"],
        )
        self.assertEqual(kept.competitor_weakness, "Kurulum uzun.")
        filled = merge_committee(
            {"title_tr": "Radar"},
            {
                "opportunity_score": 71,
                "top_competitors": "RakipA (Commercial SaaS) | RakipB (Open Source)",
                "competitor_weakness": "Pahalı.",
                "search_trend_signal": "Sinyal belirsiz",
            },
        )
        self.assertEqual(
            filled.top_competitors,
            ["RakipA (Commercial SaaS)", "RakipB (Open Source)"],
        )
        self.assertEqual(filled.search_trend_signal, "Sinyal belirsiz")
        quiet = market_dial(fetched=8, sources=1, engaged=0)
        self.assertLess(quiet, 0.60)
        self.assertGreaterEqual(quiet, 0.50)
        self.assertEqual(notify_threshold(60, quiet), 75)
        self.assertEqual(notify_threshold(60, 0.80), 60)
        self.assertEqual(market_dial(0, 0, 0), 0.75)
        subs = Settings().reddit_subs
        for name in (
            "SaaS",
            "Entrepreneur",
            "SmallBusiness",
            "SideProject",
            "Ecommerce",
            "digitalmarketing",
            "gamedev",
        ):
            self.assertIn(name, subs)
        self.assertEqual(Settings().producthunt_feed_url, "https://www.producthunt.com/feed")
        prompt = grok_prompt(_view(title="Fatura hatırlatıcı"))
        self.assertIn("Fatura hatırlatıcı", prompt)
        self.assertIn("Viral loop", prompt)
        self.assertIn("Competitor vulnerabilities", prompt)
        buttons = mvp_keyboard(7)["inline_keyboard"][0]
        self.assertEqual(buttons[1]["text"], "🤖 Grok ile İncele")
        self.assertEqual(buttons[1]["callback_data"], "grok:7")

    async def test_domain_check_uses_the_injected_resolver(self) -> None:
        def resolve(name: str) -> str:
            if name.endswith(".io"):
                raise socket.gaierror("absent")
            if name.endswith(".com"):
                return "taken"
            raise OSError("timeout")

        rows = await check_domains(["nodifyradar.io", "nodifyapp.com", "slow.dev"], resolve=resolve)
        self.assertEqual(
            rows,
            [
                {"name": "nodifyradar.io", "status": "open"},
                {"name": "nodifyapp.com", "status": "taken"},
            ],
        )

    def test_card_prints_the_domain_line(self) -> None:
        view = _view()
        view.analysis["_domains"] = [
            {"name": "nodifyradar.io", "status": "open"},
            {"name": "nodifyapp.com", "status": "taken"},
        ]
        card = plain_alert(view)
        self.assertIn(
            "🌐 Önerilen Domainler: nodifyradar.io (BOŞTA 🟢) | nodifyapp.com (DOLU 🔴)",
            card,
        )

    def test_mvp_prompt_names_the_stack_and_the_idea(self) -> None:
        view = _view(title="Fatura hatırlatıcı")
        text = mvp_prompt(view)
        self.assertIn("Fatura hatırlatıcı", text)
        for token in ("Next.js", "Tailwind", "Supabase", "SQLite", "FastAPI", "1-day"):
            self.assertIn(token, text)
        self.assertEqual(mvp_keyboard(7)["inline_keyboard"][0][0]["text"], "🚀 Cursor MVP Prompt")
        self.assertEqual(mvp_keyboard(7)["inline_keyboard"][0][0]["callback_data"], "mvp:7")
        update = {
            "callback_query": {
                "id": "q",
                "data": "mvp:7",
                "message": {"chat": {"id": 99}},
            }
        }
        self.assertEqual(callback_insight_id(update, "99"), 7)
        self.assertIsNone(callback_insight_id(update, "1"))

    def test_png_card_for_a_high_score(self) -> None:
        png = render_card(_view(score=91, title="Tahsilat hatırlatıcısı"))
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_digest_clock_and_matrix(self) -> None:
        afternoon = datetime(2026, 10, 27, 19, 0, tzinfo=ISTANBUL)
        evening = datetime(2026, 10, 27, 20, 30, tzinfo=ISTANBUL)
        self.assertFalse(digest_due(afternoon, None))
        self.assertTrue(digest_due(evening, None))
        self.assertFalse(digest_due(evening, "2026-10-27"))
        self.assertGreaterEqual(seconds_until_digest(afternoon), 59 * 60)
        self.assertLessEqual(seconds_until_digest(afternoon), 60 * 60)
        first, second, third = _view(91, "Alfa"), _view(88, "Beta"), _view(40, "Gama")
        third.user_liked = 1
        text = format_digest([first, second, third], days=1)
        self.assertIn("son 24 saat", text)
        self.assertIn("1. Alfa", text)
        self.assertIn("3. Gama", text)
        self.assertIn("40/100 ❤", text)
        week = format_digest([first], days=7)
        self.assertIn("son 7 gün", week)

    async def test_digest_query_keeps_the_top_three_qualifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "radar.sqlite"))
            await db.connect()
            specs = [
                ("1", 90, "Alfa", 0, None),
                ("2", 82, "Beta", 0, None),
                ("3", 40, "Gama", 1, None),
                ("4", 70, "Dusuk", 0, None),
                ("5", 99, "Eski", 0, "2020-01-01T00:00:00+00:00"),
            ]
            for external_id, score, title, liked, created in specs:
                await db.enqueue(
                    sample_post(
                        external_id=external_id,
                        url=f"https://news.ycombinator.com/item?id={external_id}",
                        title=title,
                    )
                )
                item = next(
                    row
                    for row in await db.list_items(status="new", limit=20)
                    if row.external_id == external_id
                )
                insight_id = await db.add_insight(
                    item_id=item.id,
                    confidence=90,
                    opportunity_score=score,
                    flags_json="{}",
                    analysis_json=brief(score=score, title=title).model_dump_json(),
                )
                if liked:
                    await db.conn.execute(
                        "UPDATE insights SET user_liked = 1 WHERE id = ?",
                        (insight_id,),
                    )
                if created:
                    await db.conn.execute(
                        "UPDATE insights SET created_at = ? WHERE id = ?",
                        (created, insight_id),
                    )
                await db.conn.commit()
            since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
            rows = await db.top_digest(since_iso=since, limit=3)
            self.assertEqual([row.analysis["title_tr"] for row in rows], ["Alfa", "Beta", "Gama"])
            loaded = await db.get_insight(rows[0].insight_id)
            self.assertEqual(loaded.analysis["title_tr"], "Alfa")

            class _Notifier:
                available = True

                def __init__(self) -> None:
                    self.text = ""

                async def send_text(self, text: str) -> int:
                    self.text = text
                    return 5

            notifier = _Notifier()
            sent = await send_digest(db, notifier, days=1)
            self.assertTrue(sent)
            self.assertIn("Alfa", notifier.text)
            self.assertNotIn("Eski", notifier.text)
            self.assertNotIn("Dusuk", notifier.text)
            await db.close()

    async def test_button_sends_the_prompt_for_that_insight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "radar.sqlite"))
            await db.connect()
            await db.enqueue(sample_post())
            item = (await db.list_items(status="new"))[0]
            insight_id = await db.add_insight(
                item_id=item.id,
                confidence=90,
                opportunity_score=80,
                flags_json="{}",
                analysis_json=brief(title="Fatura hatırlatıcı").model_dump_json(),
            )
            seen: list[dict] = []

            def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("sendMessage"):
                    seen.append(json.loads(request.content.decode()))
                return httpx.Response(200, json={"ok": True, "result": {}})

            settings = Settings(telegram_bot_token="123:abc", telegram_chat_id="99")
            update = {
                "callback_query": {
                    "id": "q1",
                    "data": f"mvp:{insight_id}",
                    "message": {"chat": {"id": 99}},
                }
            }
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                ok = await handle_callback(settings, db, update, client)
            self.assertTrue(ok)
            self.assertIn("Fatura hatırlatıcı", seen[-1]["text"])
            self.assertIn("Next.js", seen[-1]["text"])
            await db.close()

    async def test_high_score_tries_a_photo_then_falls_back_with_the_button(self) -> None:
        calls: list[str] = []
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path.endswith("sendPhoto"):
                return httpx.Response(400, json={"ok": False})
            bodies.append(json.loads(request.content.decode()))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})

        settings = Settings(telegram_bot_token="123:abc", telegram_chat_id="99")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            notifier = TelegramNotifier(settings, client=client)
            sent = await notifier.send(_view(score=88))
        self.assertEqual(sent, 9)
        self.assertTrue(any(path.endswith("sendPhoto") for path in calls))
        self.assertEqual(bodies[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"], "mvp:7")

    async def test_domain_lookup_failure_still_sends_the_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "radar.sqlite"))
            await db.connect()
            opportunity = brief(score=80)
            opportunity.suggested_domains = ["nodifyradar.io"]

            async def boom(names, resolve=None):
                raise RuntimeError("dns down")

            notifier = ScriptedNotifier()
            service = ScanService(
                Settings(
                    database_path=db.path,
                    scheduler_enabled=False,
                    notify_min_opportunity_score=60,
                ),
                db,
                fetchers=[StaticFetcher([sample_post()])],
                classifier=ScriptedClassifier(gold_classification()),
                analyzer=ScriptedAnalyzer(opportunity),
                notifier=notifier,
            )
            import pipeline

            original = pipeline.check_domains
            pipeline.check_domains = boom
            try:
                report = await service.run()
            finally:
                pipeline.check_domains = original
            self.assertEqual(report.notified, 1)
            self.assertNotIn("_domains", notifier.sent[0].analysis)
            view = _view()
            view.analysis["suggested_domains"] = ["nodifyradar.io", "nodify.app"]

            async def fake(names, resolve=None):
                return [{"name": names[0], "status": "open"}, {"name": names[1], "status": "taken"}]

            pipeline.check_domains = fake
            try:
                await attach_domains(view)
            finally:
                pipeline.check_domains = original
            self.assertEqual(view.analysis["_domains"][0]["status"], "open")
            await db.close()

    async def test_quiet_day_raises_the_notify_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "radar.sqlite"))
            await db.connect()
            posts = [
                sample_post(
                    external_id=str(index),
                    url=f"https://news.ycombinator.com/item?id={index}",
                    score=0,
                    comment_count=0,
                )
                for index in range(8)
            ]
            service = ScanService(
                Settings(
                    database_path=db.path,
                    scheduler_enabled=False,
                    notify_min_opportunity_score=60,
                    max_analyses_per_scan=1,
                ),
                db,
                fetchers=[StaticFetcher(posts)],
                classifier=ScriptedClassifier(gold_classification()),
                analyzer=ScriptedAnalyzer(brief(score=70)),
                notifier=ScriptedNotifier(),
            )
            with self.assertLogs("pipeline", level="INFO") as logs:
                report = await service.run()
            self.assertIn("required: 75", "\n".join(logs.output))
            self.assertEqual(report.notified, 0)
            async with db.conn.execute("SELECT market_dial FROM insights") as cursor:
                row = await cursor.fetchone()
            self.assertLess(row["market_dial"], 0.60)
            await db.close()

    def test_run_digest_uses_the_same_bearer_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                database_path=str(Path(tmp) / "radar.sqlite"),
                scheduler_enabled=False,
                api_token="secret-token",
            )
            app = create_app(settings)
            with TestClient(app) as client:
                blocked = client.post("/run-digest")
                self.assertEqual(blocked.status_code, 401)
                allowed = client.post(
                    "/run-digest?days=7",
                    headers={"Authorization": "Bearer secret-token"},
                )
                self.assertEqual(allowed.status_code, 200)
                self.assertEqual(allowed.json()["sent"], True)
                self.assertEqual(allowed.json()["days"], 7)
