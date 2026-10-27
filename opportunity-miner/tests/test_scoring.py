"""Stage-1 scoring, prompt shape, and the Turkish brief parser."""

from __future__ import annotations

import json
import unittest

from tests.support import gold_classification, sample_post

from analysis.analyzer import PROMPT_PATH, render_prompt
from config import Settings, load_settings
from filtering.classifier import (
    ATOMIC_QUESTIONS,
    GeminiClassifier,
    build_batched_prompt,
    classification_from_payload,
    gemini_model_candidates,
    gemini_model_id,
    score_from_flags,
)
from filtering.heuristic import heuristic_reject
from ingestion.accounts import ACCOUNTS
from ingestion.x_fetcher import rotating_slice
from textutil import parse_json_object
from datetime import datetime, timezone


class ScoringTests(unittest.IsolatedAsyncioTestCase):
    def test_five_atomic_questions_are_in_the_batched_prompt(self) -> None:
        self.assertEqual(len(ATOMIC_QUESTIONS), 5)
        prompt = build_batched_prompt(sample_post())
        for _key, question in ATOMIC_QUESTIONS:
            self.assertIn(question, prompt)

    def test_required_flags_cannot_clear_the_gate(self) -> None:
        flags = {
            "is_human": False,
            "has_signal": True,
            "has_mechanism": True,
            "is_relevant": True,
            "should_forward": True,
        }
        self.assertLessEqual(score_from_flags(flags, 99), 69)
        result = classification_from_payload({**flags, "model_confidence": 99})
        self.assertFalse(result.passes(70))

    def test_low_model_confidence_is_kept(self) -> None:
        payload = {
            "is_human": True,
            "has_signal": True,
            "has_mechanism": True,
            "is_relevant": True,
            "should_forward": True,
            "model_confidence": 40,
        }
        result = classification_from_payload(payload)
        self.assertEqual(result.confidence, 40)
        self.assertFalse(result.passes(70))

    def test_fenced_json_parses(self) -> None:
        payload = parse_json_object("```json\n{\"is_human\": true}\n```")
        self.assertTrue(payload["is_human"])

    async def test_parallel_mode_asks_five_times_and_batched_asks_once(self) -> None:
        calls: list[str] = []

        async def generate(prompt: str) -> str:
            calls.append(prompt)
            if "Question id:" in prompt:
                return json.dumps({"answer": True, "note": "yes"})
            return json.dumps(
                {
                    "is_human": True,
                    "has_signal": True,
                    "has_mechanism": True,
                    "is_relevant": True,
                    "should_forward": True,
                    "model_confidence": 88,
                    "reasons": {"is_human": "specific"},
                }
            )

        parallel = GeminiClassifier(Settings(classifier_mode="parallel"), generate=generate)
        result = await parallel.classify(sample_post())
        self.assertEqual(len(calls), 5)
        self.assertEqual(result.confidence, 100)
        self.assertTrue(result.passes(70))

        calls.clear()
        batched = GeminiClassifier(Settings(classifier_mode="batched"), generate=generate)
        result = await batched.classify(sample_post())
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.confidence, 88)
        self.assertTrue(result.passes(70))

    def test_heuristic_keeps_a_real_request_and_drops_garbage(self) -> None:
        self.assertIsNone(heuristic_reject(sample_post()))
        self.assertEqual(heuristic_reject(sample_post(title="hi", body="")), "too_short")
        self.assertEqual(
            heuristic_reject(sample_post(title="We're hiring a generalist", body="Come work with us on the team.")),
            "hiring",
        )

    def test_prompt_covers_the_turkish_brief(self) -> None:
        text = PROMPT_PATH.read_text(encoding="utf-8")
        for token in ("iyzico", "PayTR", "KVKK", "WhatsApp", "core_loop", "{{BODY}}"):
            self.assertIn(token, text)
        rendered = render_prompt(text, sample_post(body="keep {braces}"))
        self.assertIn("keep {braces}", rendered)
        self.assertNotIn("{{BODY}}", rendered)

    def test_rotation_covers_the_account_list_without_repeating_inside_a_slice(self) -> None:
        self.assertGreaterEqual(len(ACCOUNTS), 50)
        self.assertLessEqual(len(ACCOUNTS), 100)
        morning = datetime(2026, 10, 27, 0, 0, tzinfo=timezone.utc)
        half = datetime(2026, 10, 27, 0, 30, tzinfo=timezone.utc)
        first = rotating_slice(ACCOUNTS, 8, morning)
        second = rotating_slice(ACCOUNTS, 8, half)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)
        self.assertNotEqual(first, second)

    def test_settings_clamp_interval_and_read_env(self) -> None:
        self.assertEqual(Settings(scan_interval_seconds=5).scan_interval_seconds, 60)
        with self.assertRaises(ValueError):
            Settings(classifier_mode="yolo")
        import os

        previous = os.environ.get("GEMINI_MODEL")
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        try:
            loaded = load_settings()
        finally:
            if previous is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = previous
        self.assertEqual(loaded.gemini_model, "gemini-2.0-flash")
        self.assertEqual(gemini_model_id("models/gemini-1.5-flash"), "gemini-1.5-flash")
        self.assertEqual(Settings(gemini_model="models/gemini-2.5-flash").gemini_model, "gemini-2.5-flash")
        self.assertEqual(Settings(gemini_model="").gemini_model, "gemini-2.5-flash")
        self.assertEqual(
            gemini_model_candidates(""),
            ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
        )
        self.assertEqual(
            gemini_model_candidates("models/gemini-2.0-flash"),
            ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"],
        )

    def test_generate_sync_calls_genai_and_drops_json_mime_when_rejected(self) -> None:
        calls: list[dict] = []

        class Models:
            def generate_content(self, *, model, contents, config):
                calls.append({"model": model, "contents": contents, "config": dict(config)})
                if "response_mime_type" in config:
                    raise RuntimeError("response mime type is not supported")
                return type("Response", (), {"text": "{\"is_human\": true}"})()

        classifier = GeminiClassifier(Settings(gemini_api_key="k", gemini_model="gemini-2.0-flash"))
        classifier._client = type("Client", (), {"models": Models()})()
        self.assertEqual(classifier._generate_sync("hello"), "{\"is_human\": true}")
        self.assertEqual(calls[0]["model"], "gemini-2.0-flash")
        self.assertEqual(calls[0]["contents"], "hello")
        self.assertEqual(calls[0]["config"]["response_mime_type"], "application/json")
        self.assertNotIn("response_mime_type", calls[1]["config"])

    def test_missing_model_strips_prefix_and_tries_flash_fallback(self) -> None:
        calls: list[str] = []

        class Missing(Exception):
            status_code = 404

        class Models:
            def generate_content(self, *, model, contents, config):
                calls.append(model)
                if model.startswith("models/"):
                    raise AssertionError(model)
                if model != "gemini-2.0-flash":
                    raise Missing("404 NOT_FOUND")
                return type("Response", (), {"text": "{\"is_human\": true}"})()

        classifier = GeminiClassifier(Settings(gemini_api_key="k", gemini_model="models/gemini-1.5-flash"))
        classifier._client = type("Client", (), {"models": Models()})()
        self.assertEqual(classifier._generate_sync("hello"), "{\"is_human\": true}")
        self.assertEqual(calls, ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"])

    def test_gold_fixture_clears_the_default_gate(self) -> None:
        self.assertTrue(gold_classification().passes(70))


if __name__ == "__main__":
    unittest.main()
