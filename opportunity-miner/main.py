"""Opportunity Radar.

Run from this directory:

    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from config import Settings, load_settings
from database.db import Database
from pipeline import ScanCoordinator, ScanService

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _authorized(settings: Settings, authorization: str | None) -> bool:
    expected = settings.api_token.strip()
    if not expected:
        return True
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    return hmac.compare_digest(provided.encode(), expected.encode())


def create_app(
    settings: Settings | None = None,
    service_factory=None,
) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_path)
        await db.connect()
        factory = service_factory or (lambda current, database: ScanService(current, database))
        service = factory(settings, db)
        coordinator = ScanCoordinator(service)
        app.state.settings = settings
        app.state.db = db
        app.state.service = service
        app.state.coordinator = coordinator
        app.state.tasks = set()
        scheduler = None
        if settings.scheduler_enabled:
            scheduler = asyncio.create_task(_scheduler(coordinator, settings))
        from delivery.telegram_bot import poll_reactions

        reaction_stop = asyncio.Event()
        reactions = asyncio.create_task(poll_reactions(settings, db, reaction_stop))
        try:
            yield
        finally:
            reaction_stop.set()
            reactions.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reactions
            if scheduler is not None:
                scheduler.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await scheduler
            for task in list(app.state.tasks):
                task.cancel()
            closer = getattr(service, "aclose", None)
            if closer is not None:
                await closer()
            await db.close()

    app = FastAPI(title="Opportunity Radar", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict:
        current: Settings = request.app.state.settings
        return {
            "status": "ok",
            "providers": _providers(current),
        }

    @app.get("/api/status")
    async def status(request: Request) -> dict:
        coordinator: ScanCoordinator = request.app.state.coordinator
        db: Database = request.app.state.db
        current: Settings = request.app.state.settings
        last = None
        if coordinator.last_report is not None:
            last = {
                "started_at": coordinator.last_report.started_at,
                "finished_at": coordinator.last_report.finished_at,
                "status": "ok" if coordinator.last_error is None else "error",
                "stats": coordinator.last_report.model_dump(),
                "error": coordinator.last_error,
            }
        else:
            last = await db.latest_run()
        return {
            "running": coordinator.busy,
            "scheduler_enabled": current.scheduler_enabled,
            "interval_seconds": current.scan_interval_seconds,
            "providers": _providers(current),
            "counts": await db.status_counts(),
            "notified": await db.notification_count(1),
            "last_run": last,
            "last_error": coordinator.last_error,
        }

    @app.get("/api/insights")
    async def insights(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
        rows = await request.app.state.db.list_insights(limit=limit)
        return [row.model_dump() for row in rows]

    @app.get("/api/items")
    async def items(
        request: Request,
        status: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[dict]:
        rows = await request.app.state.db.list_items(status=status, limit=limit)
        return [row.model_dump() for row in rows]

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        from delivery.dashboard import render_dashboard

        db: Database = request.app.state.db
        coordinator: ScanCoordinator = request.app.state.coordinator
        current: Settings = request.app.state.settings
        counts = await db.status_counts()
        last = await db.latest_run()
        if coordinator.last_report is not None:
            last = {
                "started_at": coordinator.last_report.started_at,
                "finished_at": coordinator.last_report.finished_at,
                "status": "ok",
                "stats": coordinator.last_report.model_dump(),
                "error": coordinator.last_error,
            }
        page = render_dashboard(
            {
                "counts": counts,
                "insights": await db.list_insights(limit=12),
                "noise": await db.list_items(status="filtered_out", limit=10),
                "queue": await _queue(db),
                "last_run": last,
                "running": coordinator.busy,
                "providers": _providers(current),
                "token_required": bool(current.api_token.strip()),
            }
        )
        return HTMLResponse(page)

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request) -> dict:
        from delivery.telegram_bot import apply_reaction

        update = await request.json()
        liked = await apply_reaction(request.app.state.db, request.app.state.settings, update)
        return {"ok": True, "liked": liked}

    @app.post("/run-scan")
    async def run_scan(
        request: Request,
        wait: bool = False,
        authorization: str | None = Header(default=None),
    ):
        current: Settings = request.app.state.settings
        if not _authorized(current, authorization):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")
        coordinator: ScanCoordinator = request.app.state.coordinator
        if coordinator.busy:
            return JSONResponse({"status": "already_running"}, status_code=409)
        if wait:
            outcome = await coordinator.try_run()
            if outcome.state == "busy":
                return JSONResponse({"status": "already_running"}, status_code=409)
            status_code = 200 if outcome.state == "done" else 500
            return JSONResponse(outcome.as_dict(), status_code=status_code)
        # Arm before the task is scheduled so a fast poll waits for the scan
        # instead of reloading the dashboard before the fetch starts.
        coordinator.arm()

        async def _background():
            try:
                await coordinator.try_run()
            finally:
                coordinator.disarm()

        task = asyncio.create_task(_background())
        request.app.state.tasks.add(task)
        task.add_done_callback(request.app.state.tasks.discard)
        return JSONResponse({"status": "started"}, status_code=202)

    return app


def _providers(settings: Settings) -> dict[str, bool]:
    return {
        "gemini": settings.gemini_configured,
        "openai": settings.openai_configured,
        "telegram": settings.telegram_configured,
    }


async def _queue(db: Database):
    rows = []
    for status in ("new", "pending_classification", "pending_analysis"):
        rows.extend(await db.list_items(status=status, limit=8))
    return rows[:8]


async def _scheduler(coordinator: ScanCoordinator, settings: Settings) -> None:
    if settings.scan_on_startup:
        await coordinator.try_run()
    while True:
        await asyncio.sleep(settings.scan_interval_seconds)
        outcome = await coordinator.try_run()
        if outcome.state == "error":
            logger.error("scheduled scan failed: %s", outcome.error)


app = create_app()


if __name__ == "__main__":
    import uvicorn

    current = load_settings()
    uvicorn.run("main:app", host=current.host, port=current.port, reload=False)
