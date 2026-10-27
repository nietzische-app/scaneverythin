"""Queue, gate, budget, and notification behavior of one scan."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.support import (
    ScriptedAnalyzer,
    ScriptedClassifier,
    ScriptedNotifier,
    StaticFetcher,
    brief,
    gold_classification,
    sample_post,
)

from analysis.analyzer import OpenAIUnavailable
from config import Settings
from delivery.telegram_bot import TelegramNotifier
from database.db import Database
from database.models import Classification
from pipeline import ScanCoordinator, ScanService


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self.tmp.name) / "radar.sqlite"))
        await self.db.connect()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.tmp.cleanup()

    def settings(self, **overrides) -> Settings:
        payload = {
            "database_path": self.db.path,
            "scheduler_enabled": False,
            "max_classifications_per_scan": 10,
            "max_analyses_per_scan": 5,
            "notify_min_opportunity_score": 60,
        }
        payload.update(overrides)
        return Settings(**payload)

    def test_missing_telegram_env_logs_an_error(self) -> None:
        with self.assertLogs("delivery.telegram_bot", level="ERROR") as logs:
            TelegramNotifier(Settings())
        text = "\n".join(logs.output)
        self.assertIn("TELEGRAM_BOT_TOKEN is missing from the environment", text)
        self.assertIn("TELEGRAM_CHAT_ID is missing from the environment", text)
        with self.assertLogs("delivery.telegram_bot", level="ERROR") as logs:
            TelegramNotifier(Settings(telegram_bot_token="123:abc"))
        text = "\n".join(logs.output)
        self.assertNotIn("TELEGRAM_BOT_TOKEN", text)
        self.assertIn("TELEGRAM_CHAT_ID is missing from the environment", text)

    async def test_high_score_without_two_aha_pillars_is_not_sent(self) -> None:
        dull = brief(score=88)
        dull.aha_pillars = []
        notifier = ScriptedNotifier()
        service = ScanService(
            self.settings(),
            self.db,
            fetchers=[StaticFetcher([sample_post()])],
            classifier=ScriptedClassifier(gold_classification()),
            analyzer=ScriptedAnalyzer(dull),
            notifier=notifier,
        )
        with self.assertLogs("pipeline", level="INFO") as logs:
            report = await service.run()
        text = "\n".join(logs.output)
        self.assertIn("Item 1 score: 88, required: 60 - SKIPPED", text)
        self.assertIn("Aha filter blocked Telegram", text)
        self.assertEqual(report.notified, 0)
        self.assertEqual(notifier.sent, [])

    async def test_same_target_is_not_queued_twice(self) -> None:
        from ingestion.base import FetchBatch

        service = ScanService(
            self.settings(),
            self.db,
            fetchers=[],
            classifier=ScriptedClassifier(gold_classification()),
            analyzer=ScriptedAnalyzer(brief()),
            notifier=ScriptedNotifier(),
        )
        from database.models import ScanReport

        report = ScanReport(started_at="t", finished_at="t")
        tool = "https://example.com/tool"
        await service._store_batch(
            FetchBatch(
                source="mix",
                posts=[
                    sample_post(source="hn", external_id="1", url=tool),
                    sample_post(source="github", external_id="99", url=tool + "/"),
                ],
            ),
            report,
        )
        self.assertEqual(report.enqueued, 1)
        self.assertEqual(report.duplicates, 1)
        nitter = sample_post(source="x", external_id="55", url="https://nitter.example/ada/status/55")
        direct = sample_post(source="x-direct", external_id="55", url="https://x.com/i/status/55")
        await service._store_batch(FetchBatch(source="x", posts=[nitter, direct]), report)
        self.assertEqual(report.enqueued, 2)
        self.assertEqual(report.duplicates, 2)
        rows = await self.db.list_items(limit=10)
        self.assertEqual(len(rows), 2)

    async def test_duplicate_hash_and_terminal_status(self) -> None:
        post = sample_post()
        self.assertTrue(await self.db.enqueue(post))
        self.assertFalse(await self.db.enqueue(post))
        rows = await self.db.list_items(status="new")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].content_hash, post.content_hash)

    async def test_gold_post_is_analyzed_and_notified_once(self) -> None:
        notifier = ScriptedNotifier()
        analyzer = ScriptedAnalyzer(brief())
        service = ScanService(
            self.settings(),
            self.db,
            fetchers=[StaticFetcher([sample_post()])],
            classifier=ScriptedClassifier(gold_classification()),
            analyzer=analyzer,
            notifier=notifier,
        )
        with self.assertLogs("pipeline", level="INFO") as logs:
            first = await service.run()
        self.assertIn("Item 1 score: 82, required: 60 - PASSED", "\n".join(logs.output))
        self.assertNotIn("Aha filter blocked", "\n".join(logs.output))
        self.assertNotIn("no ideas met the minimum score", "\n".join(logs.output))
        self.assertEqual(first.analyzed, 1)
        self.assertEqual(first.notified, 1)
        self.assertEqual(analyzer.calls, 1)
        second = await service.run()
        self.assertEqual(second.duplicates, 1)
        self.assertEqual(second.analyzed, 0)
        self.assertEqual(analyzer.calls, 1)
        self.assertEqual(len(notifier.sent), 1)
        insights = await self.db.list_insights()
        self.assertEqual(insights[0].analysis["title_tr"], "Tahsilat hatırlatıcısı")
        self.assertEqual(insights[0].telegram_sent, 1)

    async def test_low_confidence_never_reaches_the_deep_model(self) -> None:
        weak = Classification(
            is_human=True,
            has_signal=True,
            has_mechanism=True,
            is_relevant=True,
            should_forward=True,
            confidence=42,
        )
        analyzer = ScriptedAnalyzer(brief())
        service = ScanService(
            self.settings(),
            self.db,
            fetchers=[
                StaticFetcher(
                    [
                        sample_post(),
                        sample_post(
                            title="hi",
                            body="",
                            external_id="short",
                            url="https://news.ycombinator.com/item?id=short",
                        ),
                    ]
                )
            ],
            classifier=ScriptedClassifier(weak),
            analyzer=analyzer,
            notifier=ScriptedNotifier(),
        )
        report = await service.run()
        self.assertEqual(report.filtered_out, 2)
        self.assertEqual(report.analyzed, 0)
        self.assertEqual(analyzer.calls, 0)
        self.assertEqual(service.classifier.calls, 1)

    async def test_classification_budget_leaves_the_rest_queued(self) -> None:
        posts = [sample_post(external_id=str(index), url=f"https://news.ycombinator.com/item?id={index}") for index in range(3)]
        classifier = ScriptedClassifier(gold_classification())
        service = ScanService(
            self.settings(max_classifications_per_scan=1, max_analyses_per_scan=1),
            self.db,
            fetchers=[StaticFetcher(posts)],
            classifier=classifier,
            analyzer=ScriptedAnalyzer(brief()),
            notifier=ScriptedNotifier(),
        )
        report = await service.run()
        self.assertEqual(classifier.calls, 1)
        self.assertEqual(report.analyzed, 1)
        self.assertGreaterEqual(report.deferred, 2)
        waiting = await self.db.count_status("new")
        self.assertEqual(waiting, 2)

    async def test_missing_gemini_key_does_not_drop_the_queue(self) -> None:
        classifier = ScriptedClassifier(gold_classification(), available=False)
        service = ScanService(
            self.settings(),
            self.db,
            fetchers=[StaticFetcher([sample_post()])],
            classifier=classifier,
            analyzer=ScriptedAnalyzer(brief()),
            notifier=ScriptedNotifier(),
        )
        report = await service.run()
        self.assertEqual(classifier.calls, 0)
        self.assertIn("gemini_not_configured", report.source_errors)
        self.assertEqual(await self.db.count_status("new"), 1)

    async def test_low_opportunity_score_is_stored_but_not_sent(self) -> None:
        notifier = ScriptedNotifier()
        service = ScanService(
            self.settings(),
            self.db,
            fetchers=[StaticFetcher([sample_post()])],
            classifier=ScriptedClassifier(gold_classification()),
            analyzer=ScriptedAnalyzer(brief(score=20)),
            notifier=notifier,
        )
        with self.assertLogs("pipeline", level="INFO") as logs:
            report = await service.run()
        text = "\n".join(logs.output)
        self.assertIn("Item 1 score: 20, required: 60 - SKIPPED", text)
        self.assertIn(
            "Scan completed successfully, but no ideas met the minimum score threshold for Telegram notification.",
            text,
        )
        self.assertEqual(report.analyzed, 1)
        self.assertEqual(report.notified, 0)
        self.assertEqual(notifier.sent, [])
        insights = await self.db.list_insights()
        self.assertEqual(insights[0].telegram_sent, 2)

    async def test_repeated_model_errors_become_terminal(self) -> None:
        broken = Classification(error="gemini exploded")
        service = ScanService(
            self.settings(),
            self.db,
            fetchers=[StaticFetcher([sample_post()])],
            classifier=ScriptedClassifier(broken),
            analyzer=ScriptedAnalyzer(brief()),
            notifier=ScriptedNotifier(),
        )
        for _ in range(3):
            await service.run()
        self.assertEqual(await self.db.count_status("error"), 1)
        again = await service.run()
        self.assertEqual(again.classified, 0)
        self.assertEqual(service.classifier.calls, 3)

    async def test_openai_quota_leaves_the_queue_and_finishes_the_scan(self) -> None:
        posts = [
            sample_post(external_id=str(index), url=f"https://news.ycombinator.com/item?id={index}")
            for index in range(2)
        ]
        analyzer = ScriptedAnalyzer(OpenAIUnavailable("openai_quota"))
        service = ScanService(
            self.settings(classifier_concurrency=1),
            self.db,
            fetchers=[StaticFetcher(posts)],
            classifier=ScriptedClassifier(gold_classification()),
            analyzer=analyzer,
            notifier=ScriptedNotifier(),
        )
        report = await service.run()
        self.assertEqual(report.analyzed, 0)
        self.assertEqual(analyzer.calls, 1)
        self.assertIn("openai_quota", report.source_errors)
        self.assertEqual(await self.db.count_status("pending_analysis"), 2)
        self.assertEqual(await self.db.count_status("error"), 0)

    async def test_coordinator_reports_busy_and_errors(self) -> None:
        class Boom:
            async def run(self):
                raise RuntimeError("disk full")

        coordinator = ScanCoordinator(Boom())  # type: ignore[arg-type]
        await coordinator._lock.acquire()
        busy = await coordinator.try_run()
        self.assertEqual(busy.state, "busy")
        coordinator._lock.release()
        failed = await coordinator.try_run()
        self.assertEqual(failed.state, "error")
        self.assertIn("disk full", failed.error or "")


if __name__ == "__main__":
    unittest.main()
