"""Telegram reactions become favorites that the next brief can see."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.support import brief, sample_post

import httpx

from analysis.analyzer import dismissed_clause, favorite_clause
from config import Settings
from database.db import Database
from delivery.telegram_bot import apply_reaction, handle_callback, liked_message_id


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
            self.assertIn("judge_verdict", names)
            self.assertIn("market_dial", names)
            self.assertIn("top_competitors", names)
            self.assertIn("competitor_weakness", names)
            self.assertIn("search_trend_signal", names)
            self.assertIn("aha_angle", names)
            self.assertIn("unfair_advantage", names)
            self.assertIn("concept_key", names)
            self.assertIn("not_interested", names)
            await db.close()

    async def test_not_interested_hides_the_mechanic_until_a_heart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "radar.sqlite"))
            await db.connect()
            await db.enqueue(sample_post())
            item = (await db.list_items(status="new"))[0]
            idea = brief(title="Gümrük masası")
            idea.concept_key = "customs_declaration_desk"
            insight_id = await db.add_insight(
                item_id=item.id,
                confidence=90,
                opportunity_score=80,
                flags_json="{}",
                analysis_json=idea.model_dump_json(),
            )
            await db.set_telegram_message(insight_id, 42)
            await apply_reaction(db, Settings(telegram_chat_id="99"), _reaction(99, 42, "❤️"))
            seen: list[dict] = []

            def handler(request: httpx.Request) -> httpx.Response:
                seen.append(json.loads(request.content.decode()))
                return httpx.Response(200, json={"ok": True, "result": {}})

            update = {
                "callback_query": {
                    "id": "q1",
                    "data": f"nope:{insight_id}",
                    "message": {"chat": {"id": 99}},
                }
            }
            settings = Settings(telegram_bot_token="123:abc", telegram_chat_id="99")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                ok = await handle_callback(settings, db, update, client)
            self.assertTrue(ok)
            self.assertEqual(seen, [{"callback_query_id": "q1", "text": "Bu tür bir daha gösterilmeyecek"}])
            self.assertTrue(await db.concept_dismissed("customs_declaration_desk"))
            self.assertEqual(await db.liked_titles(), [])
            self.assertEqual(await db.dismissed_titles(), ["Gümrük masası"])
            self.assertIn("Gümrük masası", dismissed_clause(await db.dismissed_titles()))
            self.assertIn("not interested", dismissed_clause(["Gümrük masası"]))
            liked = await apply_reaction(db, settings, _reaction(99, 42, "❤️"))
            self.assertTrue(liked)
            self.assertFalse(await db.concept_dismissed("customs_declaration_desk"))
            self.assertEqual(await db.liked_titles(), ["Gümrük masası"])
            await db.close()
