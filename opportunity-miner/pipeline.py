"""One scan: fetch, drop duplicates, classify a budget, analyze the survivors, notify."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import httpx

from analysis.analyzer import OpenAIAnalyzer, OpenAIUnavailable
from config import Settings
from database.db import Database
from database.models import Classification, InsightView, ItemRecord, Opportunity, RawPost, ScanReport
from filtering.classifier import GeminiClassifier
from filtering.heuristic import heuristic_reject
from ingestion import build_fetchers
from ingestion.base import FetchBatch, Fetcher
from textutil import attempt_count, attempt_note, clip, utcnow_iso

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
_CLASSIFY_STATUSES = ["new", "pending_classification"]


class SupportsClassify(Protocol):
    available: bool

    async def classify(self, post: RawPost) -> Classification:
        """Return the five flags and a confidence score."""


class SupportsAnalyze(Protocol):
    available: bool

    async def analyze(self, post: RawPost) -> Opportunity:
        """Return a Turkish-market brief."""

    async def aclose(self) -> None:
        """Release the API client."""


class SupportsNotify(Protocol):
    available: bool

    async def send(self, view: InsightView) -> bool:
        """Deliver one card. False means try again next scan."""


class ScanService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        fetchers: list[Fetcher] | None = None,
        classifier: SupportsClassify | None = None,
        analyzer: SupportsAnalyze | None = None,
        notifier: SupportsNotify | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.fetchers = fetchers
        self.classifier = classifier or GeminiClassifier(settings)
        self.analyzer = analyzer or OpenAIAnalyzer(settings)
        self.notifier = notifier
        if self.notifier is None:
            from delivery.telegram_bot import TelegramNotifier

            self.notifier = TelegramNotifier(settings)
        self._analysis_slots = 0
        self._openai_blocked = False
        self._slot_lock = asyncio.Lock()
        self._report_lock = asyncio.Lock()

    async def aclose(self) -> None:
        closer = getattr(self.analyzer, "aclose", None)
        if closer is not None:
            await closer()

    async def run(self) -> ScanReport:
        started = utcnow_iso()
        report = ScanReport(started_at=started, finished_at=started)
        run_id = await self.db.start_run(started)
        self._analysis_slots = self.settings.max_analyses_per_scan
        self._openai_blocked = False
        try:
            await self._fetch(report)
            await self._analyze_pending(report)
            await self._classify_pending(report)
            await self._notify(report)
            report.finished_at = utcnow_iso()
            await self.db.finish_run(run_id, "ok", report)
            logger.info(
                "scan done fetched=%s enqueued=%s classified=%s analyzed=%s notified=%s",
                report.fetched,
                report.enqueued,
                report.classified,
                report.analyzed,
                report.notified,
            )
            return report
        except Exception as exc:
            report.finished_at = utcnow_iso()
            await self.db.finish_run(run_id, "error", report, error=clip(str(exc), 500))
            raise

    async def _fetch(self, report: ScanReport) -> None:
        if self.fetchers is not None:
            await self._consume(self.fetchers, report)
            return
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        headers = {"User-Agent": self.settings.user_agent}
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            await self._consume(build_fetchers(self.settings, client), report)

    async def _consume(self, fetchers: list[Fetcher], report: ScanReport) -> None:
        batches = await asyncio.gather(*(fetcher.fetch() for fetcher in fetchers), return_exceptions=True)
        for batch in batches:
            if isinstance(batch, BaseException):
                await self._note(report, f"fetch: {batch}")
                continue
            await self._store_batch(batch, report)

    async def _store_batch(self, batch: FetchBatch, report: ScanReport) -> None:
        async with self._report_lock:
            report.fetched += len(batch.posts)
            for error in batch.errors:
                if error not in report.source_errors:
                    report.source_errors.append(error)
        for post in batch.posts:
            try:
                created = await self.db.enqueue(post)
            except Exception as exc:  # noqa: BLE001
                await self._item_error(report, f"{post.source}:{post.external_id}: {exc}")
                continue
            async with self._report_lock:
                if created:
                    report.enqueued += 1
                else:
                    report.duplicates += 1

    async def _analyze_pending(self, report: ScanReport) -> None:
        waiting = await self.db.count_status("pending_analysis")
        if waiting == 0:
            return
        if not self.analyzer.available:
            await self._note(report, "openai_not_configured")
            await self._inc(report, "deferred", waiting)
            return
        rows = await self.db.list_queue(
            ["pending_analysis"],
            self.settings.max_analyses_per_scan,
            oldest_first=True,
        )
        await self._inc(report, "deferred", max(0, waiting - len(rows)))
        for row in rows:
            if not await self._take_analysis_slot():
                await self._inc(report, "deferred")
                continue
            await self._analyze_one(row, report)

    async def _classify_pending(self, report: ScanReport) -> None:
        waiting = 0
        for status in _CLASSIFY_STATUSES:
            waiting += await self.db.count_status(status)
        if waiting == 0:
            return
        if not self.classifier.available:
            await self._note(report, "gemini_not_configured")
            await self._inc(report, "deferred", waiting)
            return
        rows = await self.db.list_queue(
            _CLASSIFY_STATUSES,
            self.settings.max_classifications_per_scan,
            oldest_first=True,
        )
        await self._inc(report, "deferred", max(0, waiting - len(rows)))
        semaphore = asyncio.Semaphore(self.settings.classifier_concurrency)

        async def one(row: ItemRecord) -> None:
            async with semaphore:
                await self._classify_one(row, report)

        await asyncio.gather(*(one(row) for row in rows))

    async def _classify_one(self, row: ItemRecord, report: ScanReport) -> None:
        post = row.to_raw_post()
        reason = heuristic_reject(post)
        if reason:
            await self.db.update_item(row.id, status="filtered_out", note=reason)
            await self._inc(report, "filtered_out")
            return
        result = await self.classifier.classify(post)
        if result.error:
            await self._fail(row, report, result.error, "pending_classification")
            return
        payload = result.model_dump_json()
        await self._inc(report, "classified")
        if not result.passes(self.settings.confidence_threshold):
            await self.db.update_item(
                row.id,
                status="filtered_out",
                note="below_threshold",
                classification_json=payload,
            )
            await self._inc(report, "filtered_out")
            return
        await self.db.update_item(
            row.id,
            status="pending_analysis",
            note=None,
            classification_json=payload,
        )
        if not self.analyzer.available:
            await self._note(report, "openai_not_configured")
            await self._inc(report, "deferred")
            return
        if not await self._take_analysis_slot():
            await self._inc(report, "deferred")
            return
        refreshed = row.model_copy(update={"note": None, "classification_json": payload})
        await self._analyze_one(refreshed, report)

    async def _analyze_one(self, row: ItemRecord, report: ScanReport) -> None:
        post = row.to_raw_post()
        if self._openai_blocked:
            await self._inc(report, "deferred")
            return
        try:
            opportunity = await self.analyzer.analyze(post)
        except OpenAIUnavailable as exc:
            logger.warning("OpenAI stopped analysis for %s: %s", row.external_id, exc)
            async with self._slot_lock:
                self._openai_blocked = True
                self._analysis_slots = 0
            await self._note(report, str(exc) or "openai_unavailable")
            await self._inc(report, "deferred")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("analysis failed for %s: %s", row.external_id, exc)
            await self._fail(row, report, clip(str(exc), 300), "pending_analysis")
            return
        classification = row.classification()
        confidence = classification.confidence if classification else 0
        flags = classification.model_dump_json() if classification else "{}"
        await self.db.add_insight(
            item_id=row.id,
            confidence=confidence,
            opportunity_score=opportunity.opportunity_score,
            flags_json=flags,
            analysis_json=opportunity.model_dump_json(),
        )
        await self.db.update_item(row.id, status="analyzed", note=None)
        await self._inc(report, "analyzed")

    async def _notify(self, report: ScanReport) -> None:
        pending = await self.db.pending_notifications(limit=20)
        if not pending:
            return
        assert self.notifier is not None
        if not self.notifier.available:
            await self._note(report, "telegram_not_configured")
            return
        for view in pending:
            if view.opportunity_score < self.settings.notify_min_opportunity_score:
                await self.db.set_telegram_state(view.insight_id, 2)
                continue
            try:
                sent = await self.notifier.send(view)
            except Exception as exc:  # noqa: BLE001
                logger.warning("telegram send failed: %s", exc)
                sent = False
            if sent:
                await self.db.set_telegram_state(view.insight_id, 1)
                await self._inc(report, "notified")

    async def _fail(self, row: ItemRecord, report: ScanReport, message: str, retry_status: str) -> None:
        note = attempt_note(row.note, message)
        status = "error" if attempt_count(note) >= MAX_ATTEMPTS else retry_status
        await self.db.update_item(row.id, status=status, note=note)
        await self._item_error(report, f"{row.source}:{row.external_id}: {message}")

    async def _take_analysis_slot(self) -> bool:
        async with self._slot_lock:
            if self._analysis_slots <= 0:
                return False
            self._analysis_slots -= 1
            return True

    async def _inc(self, report: ScanReport, field: str, amount: int = 1) -> None:
        async with self._report_lock:
            setattr(report, field, getattr(report, field) + amount)

    async def _note(self, report: ScanReport, message: str) -> None:
        async with self._report_lock:
            if message not in report.source_errors:
                report.source_errors.append(message)

    async def _item_error(self, report: ScanReport, message: str) -> None:
        async with self._report_lock:
            if len(report.item_errors) < 40:
                report.item_errors.append(clip(message, 300))


class RunOutcome:
    def __init__(self, state: str, report: ScanReport | None = None, error: str | None = None) -> None:
        self.state = state
        self.report = report
        self.error = error

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "report": None if self.report is None else self.report.model_dump(),
            "error": self.error,
        }


class ScanCoordinator:
    """Makes sure a manual scan and the scheduler never run together."""

    def __init__(self, service: ScanService) -> None:
        self.service = service
        self.last_report: ScanReport | None = None
        self.last_error: str | None = None
        self.running = False
        self._pending = 0
        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        return self.running or self._pending > 0 or self._lock.locked()

    def arm(self) -> None:
        self._pending += 1

    def disarm(self) -> None:
        self._pending = max(0, self._pending - 1)

    async def try_run(self) -> RunOutcome:
        if self._lock.locked():
            return RunOutcome("busy")
        async with self._lock:
            self.running = True
            try:
                report = await self.service.run()
                self.last_report = report
                self.last_error = None
                return RunOutcome("done", report=report)
            except Exception as exc:  # noqa: BLE001
                logger.exception("scan failed")
                self.last_error = clip(f"{type(exc).__name__}: {exc}", 500)
                return RunOutcome("error", error=self.last_error)
            finally:
                self.running = False
