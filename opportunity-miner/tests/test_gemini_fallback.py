"""Fallback across bare Flash model ids and parse the JSON the client returns."""

from __future__ import annotations

import asyncio
import json
import logging

from google.genai.errors import ClientError

from config import Settings
from filtering.classifier import FLASH_MODELS, GeminiClassifier
from tests.support import sample_post


def test_client_keeps_the_sdk_default_endpoint():
    from google import genai

    plain = genai.Client(api_key="k")
    client = GeminiClassifier(Settings(gemini_api_key="k"))._sdk()
    assert client._api_client._http_options.api_version == plain._api_client._http_options.api_version
    assert client._api_client._http_options.base_url == plain._api_client._http_options.base_url


def test_generate_content_is_called_with_gemini_2_0_flash():
    seen: list[str] = []

    class Models:
        def generate_content(self, *, model, contents, config):
            seen.append(model)
            return type("Response", (), {"text": "{\"is_human\": true}"})()

    classifier = GeminiClassifier(Settings(gemini_api_key="k", gemini_model="gemini-2.0-flash"))
    classifier._client = type("Client", (), {"models": Models()})()
    assert classifier._generate_sync("ping") == "{\"is_human\": true}"
    assert seen == ["gemini-2.0-flash"]


def test_flash_fallback_order_is_bare_ids():
    assert list(FLASH_MODELS) == ["gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash"]


def test_not_found_falls_through_and_parses_the_client_payload(caplog):
    payload = {
        "is_human": True,
        "has_signal": True,
        "has_mechanism": True,
        "is_relevant": True,
        "should_forward": True,
        "model_confidence": 90,
        "reasons": {"is_human": "named a real workflow"},
    }
    seen: list[str] = []

    class Models:
        def generate_content(self, *, model, contents, config):
            seen.append(model)
            assert not str(model).startswith("models/")
            assert model in FLASH_MODELS
            if model == "gemini-2.0-flash":
                raise ClientError(
                    404,
                    {
                        "error": {
                            "code": 404,
                            "message": "gemini-2.0-flash is not found",
                            "status": "NOT_FOUND",
                        }
                    },
                )
            assert model == "gemini-1.5-flash-8b"
            return type("Response", (), {"text": json.dumps(payload)})()

    classifier = GeminiClassifier(Settings(gemini_api_key="k", gemini_model="gemini-2.0-flash"))
    classifier._client = type("Client", (), {"models": Models()})()
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(classifier.classify(sample_post()))

    assert seen == ["gemini-2.0-flash", "gemini-1.5-flash-8b"]
    assert "requesting Gemini model gemini-2.0-flash" in caplog.text
    assert "requesting Gemini model gemini-1.5-flash-8b" in caplog.text
    assert "requesting Gemini model models/" not in caplog.text
    assert result.error is None
    assert result.is_human is True
    assert result.has_signal is True
    assert result.has_mechanism is True
    assert result.is_relevant is True
    assert result.should_forward is True
    assert result.confidence == 90
    assert result.reasons["is_human"] == "named a real workflow"
