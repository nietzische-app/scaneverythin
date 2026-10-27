"""Fallback across bare Flash model ids and parse the JSON the client returns."""

from __future__ import annotations

import asyncio
import json
import logging

from google.genai.errors import ClientError

from config import Settings
from filtering.classifier import FLASH_MODELS, GeminiClassifier
from tests.support import sample_post


def test_client_stays_on_v1beta_without_a_model_prefix():
    from google.genai._transformers import t_model

    classifier = GeminiClassifier(Settings(gemini_api_key="k"))
    client = classifier._sdk()
    assert client._api_client._http_options.api_version == "v1beta"
    wired = t_model(client._api_client, "gemini-2.5-flash")
    assert wired == "models/gemini-2.5-flash"


def test_flash_fallback_order_is_bare_ids():
    assert list(FLASH_MODELS) == ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash"]


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
            if model == "gemini-2.5-flash":
                raise ClientError(
                    404,
                    {
                        "error": {
                            "code": 404,
                            "message": "gemini-2.5-flash is not found",
                            "status": "NOT_FOUND",
                        }
                    },
                )
            assert model == "gemini-2.0-flash-exp"
            return type("Response", (), {"text": json.dumps(payload)})()

    classifier = GeminiClassifier(Settings(gemini_api_key="k", gemini_model="gemini-2.5-flash"))
    classifier._client = type("Client", (), {"models": Models()})()
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(classifier.classify(sample_post()))

    assert seen == ["gemini-2.5-flash", "gemini-2.0-flash-exp"]
    assert "requesting Gemini model gemini-2.5-flash" in caplog.text
    assert "requesting Gemini model gemini-2.0-flash-exp" in caplog.text
    assert "requesting Gemini model models/" not in caplog.text
    assert result.error is None
    assert result.is_human is True
    assert result.has_signal is True
    assert result.has_mechanism is True
    assert result.is_relevant is True
    assert result.should_forward is True
    assert result.confidence == 90
    assert result.reasons["is_human"] == "named a real workflow"
