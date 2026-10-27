"""Telegram reactions become favorites that the next brief can see."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.support import brief, sample_post

from analysis.analyzer import favorite_clause
from config import Settings
from database.db import Database
from delivery.telegram_bot import apply_reaction, liked_message_id


def _reaction(chat_id: int, message_id: int, emoji: str) -> dict:
    return {
        "update_id": 7,
        "message_reaction": {
            "chat": {"id": chat_id},
            "message_id": message_id,
            "new_reaction": [{"type": "emoji", "emoji": emoji}],
        },
    }


class ReactionTests(unittest.IsolatedAsyncioTestCase):
    def test_like_emojis_match_the_configured_chat(self) -> None:
        self.assertEqual(liked_message_id(_reaction(99, 42, "❤️"), "99"), 42)
        self.assertEqual(liked_message_id(_reaction(99, 42, "👍"), "99"), 42)
        self.assertIsNone(liked_message_id(_reaction(99, 42, "👎"), "99"))
        self.assertIsNone(liked_message_id(_reaction(1, 42, "🔥"), "99"))

    async def test_reaction_marks_the_insight_and_titles_come_back(self) -> None:
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
            await db.set_telegram_message(insight_id, 42)
            settings = Settings(telegram_chat_id="99")
            liked = await apply_reaction(db, settings, _reaction(99, 42, "🔥"))
            self.assertTrue(liked)
            self.assertEqual(await db.liked_titles(), ["Fatura hatırlatıcı"])
            clause = favorite_clause(await db.liked_titles())
            self.assertIn("Fatura hatırlatıcı", clause)
            self.assertIn("past favorites", clause)
            await db.close()

    async def test_old_insight_table_gains_the_like_columns(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "radar.sqlite")
            raw = sqlite3.connect(path)
            raw.execute(
                """
                CREATE TABLE insights (
                    id INTEGER PRIMARY KEY,
                    item_id INTEGER NOT NULL UNIQUE,
                    confidence INTEGER NOT NULL,
                    opportunity_score INTEGER NOT NULL,
                    flags_json TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    telegram_sent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            raw.commit()
            raw.close()
            db = Database(path)
            await db.connect()
            async with db.conn.execute("PRAGMA table_info(insights)") as cursor:
                names = {row["name"] for row in await cursor.fetchall()}
            self.assertIn("user_liked", names)
            self.assertIn("telegram_message_id", names)
            await db.close()
