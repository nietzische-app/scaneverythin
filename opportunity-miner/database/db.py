"""Async SQLite log of everything the radar has already seen.

`content_hash` is the duplicate gate. Terminal rows (`filtered_out`,
`analyzed`, `error`) are left alone on later scans. Rows still in the queue
are retried without inserting a second copy.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from database.models import InsightView, ItemRecord, RawPost, ScanReport, normalize_concept_key
from ingestion.base import canonical_key
from textutil import clip, utcnow_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS scanned_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    author TEXT,
    score INTEGER,
    comment_count INTEGER,
    raw_json TEXT,
    classification_json TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_status_created
    ON scanned_items(status, created_at);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL UNIQUE REFERENCES scanned_items(id),
    confidence INTEGER NOT NULL,
    opportunity_score INTEGER NOT NULL,
    flags_json TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    telegram_sent INTEGER NOT NULL DEFAULT 0,
    user_liked INTEGER NOT NULL DEFAULT 0,
    telegram_message_id INTEGER,
    judge_verdict TEXT NOT NULL DEFAULT '',
    market_dial REAL,
    top_competitors TEXT NOT NULL DEFAULT '',
    competitor_weakness TEXT NOT NULL DEFAULT '',
    search_trend_signal TEXT NOT NULL DEFAULT '',
    aha_angle TEXT NOT NULL DEFAULT '',
    unfair_advantage TEXT NOT NULL DEFAULT '',
    concept_key TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_insights_score
    ON insights(opportunity_score DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    stats_json TEXT,
    error TEXT
);
"""

_INSIGHT_SELECT = """
    SELECT
        i.id AS insight_id,
        i.item_id AS item_id,
        s.source AS source,
        s.title AS title,
        s.url AS url,
        s.author AS author,
        i.confidence AS confidence,
        i.opportunity_score AS opportunity_score,
        i.analysis_json AS analysis_json,
        i.telegram_sent AS telegram_sent,
        i.user_liked AS user_liked,
        i.created_at AS created_at
    FROM insights i
    JOIN scanned_items s ON s.id = i.item_id
"""

_ITEM_COLUMNS = (
    "id",
    "source",
    "external_id",
    "url",
    "content_hash",
    "title",
    "body",
    "author",
    "score",
    "comment_count",
    "classification_json",
    "status",
    "note",
    "created_at",
    "updated_at",
)


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA)
        await self._ensure_insight_columns()
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_insights_concept ON insights(concept_key, created_at)"
        )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("database is not connected")
        return self._conn

    async def enqueue(self, post: RawPost) -> bool:
        """Insert a post. Return False when its hash is already stored."""
        now = utcnow_iso()
        raw = clip(json.dumps(post.raw, ensure_ascii=False, default=str), 8000)
        try:
            await self.conn.execute(
                """
                INSERT INTO scanned_items (
                    source, external_id, url, content_hash, title, body, author,
                    score, comment_count, raw_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
                """,
                (
                    post.source,
                    clip(post.external_id, 300),
                    clip(post.url, 1000),
                    post.content_hash,
                    clip(post.title, 500),
                    clip(post.body, 20000),
                    clip(post.author, 200) or None,
                    post.score,
                    post.comment_count,
                    raw,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            return False
        await self.conn.commit()
        return True

    async def already_seen(self, post: RawPost) -> bool:
        """True when this URL, or this X status id, is already stored."""
        key = canonical_key(post)
        if key.startswith("url:"):
            url = clip((post.url or "").strip(), 1000)
            if not url:
                return False
            sql = "SELECT 1 AS hit FROM scanned_items WHERE lower(rtrim(url, '/')) = lower(rtrim(?, '/')) LIMIT 1"
            args: tuple = (url,)
        elif key.startswith("status:"):
            status_id = key.split(":", 1)[1]
            sql = """
                SELECT 1 AS hit FROM scanned_items
                WHERE (source IN ('x', 'x-direct') AND external_id = ?)
                   OR url LIKE ?
                LIMIT 1
            """
            args = (status_id, f"%/status/{status_id}")
        else:
            return False
        async with self.conn.execute(sql, args) as cursor:
            return await cursor.fetchone() is not None

    async def list_queue(self, statuses: list[str], limit: int, *, oldest_first: bool = True) -> list[ItemRecord]:
        if limit <= 0 or not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        direction = "ASC" if oldest_first else "DESC"
        sql = f"""
            SELECT {", ".join(_ITEM_COLUMNS)}
            FROM scanned_items
            WHERE status IN ({placeholders})
            ORDER BY created_at {direction}, id {direction}
            LIMIT ?
        """
        async with self.conn.execute(sql, (*statuses, limit)) as cursor:
            rows = await cursor.fetchall()
        return [_item(row) for row in rows]

    async def list_items(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        oldest_first: bool = False,
    ) -> list[ItemRecord]:
        direction = "ASC" if oldest_first else "DESC"
        if status is None:
            sql = f"""
                SELECT {", ".join(_ITEM_COLUMNS)}
                FROM scanned_items
                ORDER BY created_at {direction}, id {direction}
                LIMIT ?
            """
            params: tuple[Any, ...] = (limit,)
        else:
            sql = f"""
                SELECT {", ".join(_ITEM_COLUMNS)}
                FROM scanned_items
                WHERE status = ?
                ORDER BY created_at {direction}, id {direction}
                LIMIT ?
            """
            params = (status, limit)
        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_item(row) for row in rows]

    async def count_status(self, status: str) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS c FROM scanned_items WHERE status = ?",
            (status,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["c"] if row else 0)

    async def status_counts(self) -> dict[str, int]:
        async with self.conn.execute(
            "SELECT status, COUNT(*) AS c FROM scanned_items GROUP BY status"
        ) as cursor:
            rows = await cursor.fetchall()
        return {str(row["status"]): int(row["c"]) for row in rows}

    async def update_item(
        self,
        item_id: int,
        *,
        status: str,
        note: str | None = None,
        classification_json: str | None = None,
    ) -> None:
        await self.conn.execute(
            """
            UPDATE scanned_items
            SET status = ?,
                note = ?,
                classification_json = COALESCE(?, classification_json),
                updated_at = ?
            WHERE id = ?
            """,
            (status, note, classification_json, utcnow_iso(), item_id),
        )
        await self.conn.commit()

    async def add_insight(
        self,
        *,
        item_id: int,
        confidence: int,
        opportunity_score: int,
        flags_json: str,
        analysis_json: str,
    ) -> int:
        verdict, dial, rivals, weakness, trend, aha, moat, concept = _brief_columns(analysis_json)
        try:
            cursor = await self.conn.execute(
                """
                INSERT INTO insights (
                    item_id, confidence, opportunity_score, flags_json,
                    analysis_json, telegram_sent, judge_verdict, market_dial,
                    top_competitors, competitor_weakness, search_trend_signal,
                    aha_angle, unfair_advantage, concept_key, created_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    confidence,
                    opportunity_score,
                    flags_json,
                    analysis_json,
                    verdict,
                    dial,
                    rivals,
                    weakness,
                    trend,
                    aha,
                    moat,
                    concept,
                    utcnow_iso(),
                ),
            )
        except sqlite3.IntegrityError:
            async with self.conn.execute(
                "SELECT id FROM insights WHERE item_id = ?",
                (item_id,),
            ) as existing:
                row = await existing.fetchone()
            return int(row["id"])
        await self.conn.commit()
        return int(cursor.lastrowid or 0)

    async def list_insights(self, limit: int = 20) -> list[InsightView]:
        sql = _INSIGHT_SELECT + """
            ORDER BY i.opportunity_score DESC, i.created_at DESC
            LIMIT ?
        """
        async with self.conn.execute(sql, (limit,)) as cursor:
            rows = await cursor.fetchall()
        return [_insight(row) for row in rows]

    async def get_insight(self, insight_id: int) -> InsightView | None:
        sql = _INSIGHT_SELECT + " WHERE i.id = ?"
        async with self.conn.execute(sql, (insight_id,)) as cursor:
            row = await cursor.fetchone()
        return None if row is None else _insight(row)

    async def top_digest(self, *, since_iso: str, limit: int = 3) -> list[InsightView]:
        """Liked rows, or any row scored above 75, inside the window. Highest score first."""
        sql = _INSIGHT_SELECT + """
            WHERE (i.user_liked = 1 OR i.opportunity_score > 75)
              AND i.created_at >= ?
            ORDER BY i.opportunity_score DESC, i.user_liked DESC, i.created_at DESC
            LIMIT ?
        """
        async with self.conn.execute(sql, (since_iso, limit)) as cursor:
            rows = await cursor.fetchall()
        return [_insight(row) for row in rows]

    async def get_state(self, key: str) -> str | None:
        async with self.conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
        return None if row is None else str(row["value"])

    async def set_state(self, key: str, value: str) -> None:
        await self.conn.execute(
            """
            INSERT INTO app_state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        await self.conn.commit()

    async def pending_notifications(self, limit: int = 20) -> list[InsightView]:
        sql = _INSIGHT_SELECT + """
            WHERE i.telegram_sent = 0
            ORDER BY i.opportunity_score DESC, i.id ASC
            LIMIT ?
        """
        async with self.conn.execute(sql, (limit,)) as cursor:
            rows = await cursor.fetchall()
        return [_insight(row) for row in rows]

    async def set_telegram_state(self, insight_id: int, state: int) -> None:
        await self.conn.execute(
            "UPDATE insights SET telegram_sent = ? WHERE id = ?",
            (state, insight_id),
        )
        await self.conn.commit()

    async def set_telegram_message(self, insight_id: int, message_id: int) -> None:
        await self.conn.execute(
            "UPDATE insights SET telegram_message_id = ? WHERE id = ?",
            (message_id, insight_id),
        )
        await self.conn.commit()

    async def mark_liked(self, message_id: int) -> bool:
        cursor = await self.conn.execute(
            "UPDATE insights SET user_liked = 1 WHERE telegram_message_id = ?",
            (message_id,),
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    async def concept_seen(self, concept_key: str, *, prior_to: int | None = None) -> bool:
        """True when this mechanic was stored in the last 30 days.

        prior_to is an insight id. Then only an earlier row, or one already sent or skipped, counts.
        """
        key = normalize_concept_key(concept_key)
        if not key:
            return False
        since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
        if prior_to is None:
            sql = "SELECT 1 FROM insights WHERE concept_key = ? AND created_at >= ? LIMIT 1"
            args: tuple = (key, since)
        else:
            sql = """
                SELECT 1 FROM insights
                WHERE concept_key = ? AND created_at >= ? AND id != ?
                  AND (id < ? OR telegram_sent != 0)
                LIMIT 1
            """
            args = (key, since, prior_to, prior_to)
        async with self.conn.execute(sql, args) as cursor:
            return await cursor.fetchone() is not None

    async def liked_titles(self, limit: int = 8) -> list[str]:
        sql = """
            SELECT i.analysis_json AS analysis_json, s.title AS title
            FROM insights i
            JOIN scanned_items s ON s.id = i.item_id
            WHERE i.user_liked = 1
            ORDER BY i.created_at DESC
            LIMIT ?
        """
        async with self.conn.execute(sql, (limit,)) as cursor:
            rows = await cursor.fetchall()
        titles: list[str] = []
        for row in rows:
            title = ""
            try:
                payload = json.loads(row["analysis_json"] or "{}")
                title = str(payload.get("title_tr") or "").strip()
            except json.JSONDecodeError:
                title = ""
            titles.append(title or str(row["title"] or "").strip())
        return [title for title in titles if title]

    async def _ensure_insight_columns(self) -> None:
        async with self.conn.execute("PRAGMA table_info(insights)") as cursor:
            rows = await cursor.fetchall()
        names = {row["name"] for row in rows}
        if "user_liked" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN user_liked INTEGER NOT NULL DEFAULT 0"
            )
        if "telegram_message_id" not in names:
            await self.conn.execute("ALTER TABLE insights ADD COLUMN telegram_message_id INTEGER")
        if "judge_verdict" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN judge_verdict TEXT NOT NULL DEFAULT ''"
            )
        if "market_dial" not in names:
            await self.conn.execute("ALTER TABLE insights ADD COLUMN market_dial REAL")
        if "top_competitors" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN top_competitors TEXT NOT NULL DEFAULT ''"
            )
        if "competitor_weakness" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN competitor_weakness TEXT NOT NULL DEFAULT ''"
            )
        if "search_trend_signal" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN search_trend_signal TEXT NOT NULL DEFAULT ''"
            )
        if "aha_angle" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN aha_angle TEXT NOT NULL DEFAULT ''"
            )
        if "unfair_advantage" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN unfair_advantage TEXT NOT NULL DEFAULT ''"
            )
        if "concept_key" not in names:
            await self.conn.execute(
                "ALTER TABLE insights ADD COLUMN concept_key TEXT NOT NULL DEFAULT ''"
            )

    async def notification_count(self, state: int) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS c FROM insights WHERE telegram_sent = ?",
            (state,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["c"] if row else 0)

    async def start_run(self, started_at: str) -> int:
        cursor = await self.conn.execute(
            "INSERT INTO scan_runs (started_at, status) VALUES (?, 'running')",
            (started_at,),
        )
        await self.conn.commit()
        return int(cursor.lastrowid or 0)

    async def finish_run(
        self,
        run_id: int,
        status: str,
        report: ScanReport,
        error: str | None = None,
    ) -> None:
        await self.conn.execute(
            """
            UPDATE scan_runs
            SET finished_at = ?, status = ?, stats_json = ?, error = ?
            WHERE id = ?
            """,
            (report.finished_at, status, report.model_dump_json(), error, run_id),
        )
        await self.conn.commit()

    async def latest_run(self) -> dict[str, Any] | None:
        async with self.conn.execute(
            """
            SELECT id, started_at, finished_at, status, stats_json, error
            FROM scan_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        stats = None
        if row["stats_json"]:
            stats = ScanReport.model_validate_json(row["stats_json"]).model_dump()
        return {
            "id": row["id"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "status": row["status"],
            "stats": stats,
            "error": row["error"],
        }


def _brief_columns(analysis_json: str) -> tuple[str, float | None, str, str, str, str, str, str]:
    empty = ("", None, "", "", "", "", "", "")
    try:
        payload = json.loads(analysis_json or "{}")
    except json.JSONDecodeError:
        return empty
    if not isinstance(payload, dict):
        return empty
    verdict = clip(str(payload.get("judge_verdict") or "").strip(), 2000)
    rivals = _competitor_text(payload.get("top_competitors"))
    weakness = clip(str(payload.get("competitor_weakness") or "").strip(), 1200)
    trend = clip(str(payload.get("search_trend_signal") or "").strip(), 400)
    aha = clip(str(payload.get("aha_angle") or "").strip(), 500)
    moat = clip(str(payload.get("unfair_advantage") or "").strip(), 400)
    concept = normalize_concept_key(payload.get("concept_key"))
    raw_dial = payload.get("market_dial")
    if raw_dial is None or raw_dial == "":
        return verdict, None, rivals, weakness, trend, aha, moat, concept
    try:
        dial = float(raw_dial)
    except (TypeError, ValueError):
        return verdict, None, rivals, weakness, trend, aha, moat, concept
    return verdict, max(0.50, min(1.00, dial)), rivals, weakness, trend, aha, moat, concept


def _competitor_text(raw: object) -> str:
    if isinstance(raw, list):
        names = [str(item).strip() for item in raw if str(item).strip()]
    else:
        names = [part.strip() for part in str(raw or "").split("|") if part.strip()]
    return clip(" | ".join(names[:3]), 400)


def _item(row: aiosqlite.Row) -> ItemRecord:
    return ItemRecord(
        id=row["id"],
        source=row["source"],
        external_id=row["external_id"],
        url=row["url"],
        content_hash=row["content_hash"],
        title=row["title"],
        body=row["body"],
        author=row["author"],
        score=row["score"],
        comment_count=row["comment_count"],
        classification_json=row["classification_json"],
        status=row["status"],
        note=row["note"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _insight(row: aiosqlite.Row) -> InsightView:
    analysis = json.loads(row["analysis_json"] or "{}")
    return InsightView(
        insight_id=row["insight_id"],
        item_id=row["item_id"],
        source=row["source"],
        title=row["title"],
        url=row["url"],
        author=row["author"],
        confidence=row["confidence"],
        opportunity_score=row["opportunity_score"],
        analysis=analysis,
        telegram_sent=row["telegram_sent"],
        user_liked=int(row["user_liked"] or 0),
        created_at=row["created_at"],
    )
