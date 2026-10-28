"""Stage-2 calls OpenAI and turns quota or auth failures into a queued skip."""

from __future__ import annotations

import asyncio

import httpx
from openai import AuthenticationError, RateLimitError

from analysis.analyzer import OpenAIAnalyzer, OpenAIUnavailable
from config import Settings, load_settings
from tests.support import sample_post

_BODY = (
    '{"title_tr":"Rapor","opportunity_score":71,'
    '"tr_market_analysis":"Yerel boşluk var.",'
    '"our_differentiator":"Sektöre özel akış.",'
    '"business_model":"KOBİ aboneliği."}'
)


def _status_error(cls, status: int, message: str):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body={"error": {"message": message}})


class _Client:
    def __init__(self, create) -> None:
        self.chat = self
        self.completions = self
        self._create = create
        self.closed = False

    def create(self, **kwargs):
        return self._create(**kwargs)

    def close(self) -> None:
        self.closed = True


def test_default_model_is_gpt_4o_mini_and_parses_the_brief():
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        message = type("Message", (), {"content": _BODY})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()

    analyzer = OpenAIAnalyzer(Settings(openai_api_key="sk-test"))
    analyzer._client = _Client(create)
    result = asyncio.run(analyzer.analyze(sample_post()))

    assert len(seen) == 2
    assert seen[0]["model"] == "gpt-4o-mini"
    assert seen[0]["max_completion_tokens"] == 8192
    assert seen[1]["max_completion_tokens"] == 8192
    assert seen[1]["response_format"] == {"type": "json_object"}
    system = "\n".join(call["messages"][0]["content"] for call in seen)
    assert seen[0]["messages"][0]["role"] == "system"
    assert "Scout brief JSON" in seen[1]["messages"][1]["content"]
    assert (
        "Provide an 'opportunity_score' strictly between 0 and 100 based on SaaS / business potential."
        in system
    )
    assert "integer" in system
    assert result.title_tr == "Rapor"
    assert result.opportunity_score == 71
    assert result.tr_market_analysis == "Yerel boşluk var."
    assert result.our_differentiator == "Sektöre özel akış."
    assert result.business_model == "KOBİ aboneliği."
    assert "professional Turkish" in system
    assert "strictly below 50" in system
    assert "tr_market_analysis" in system
    assert "our_differentiator" in system
    assert "Tam olarak kime, ne acı çektiriyor?" in system
    assert "solution_mechanism" in system
    assert "why_now" in system
    assert "Yeni & Kopyalanabilir" in system
    assert "Zamanının Ötesinde / Yeniden Canlandırılabilir" in system
    assert "TR Pazar Fırsatı" in system
    assert "minor micro-tools" in system
    assert "Pre-AI obsolete tool" in system
    assert "High-Value AI Ecosystem Infrastructure" in system
    assert "MCP" in system
    assert "project_scale" in system
    assert "top_competitors" in system
    assert "competitor_weakness" in system
    assert "search_trend_signal" in system
    assert "Sinyal belirsiz" in system
    assert "Why didn't I think of that?" in system
    assert "aha_angle" in system
    assert "unfair_advantage" in system
    assert "cross_domain" in system
    assert "exactly 2 plain Turkish sentences" in system
    assert "X yapmak isteyen kullanıcılara acı çektiriyor." in system
    assert "Gelişmiş arayüz ve kullanıcı dostu yapısıyla ayrışır." in system
    assert "Bu alanda önemli bir boşluk var." in system
    assert "Cluster başına aylık $49" in system
    assert "jargon_analogy" in system
    assert "strictly below 50" in system
    assert Settings(openai_model="gpt-4o").openai_model == "gpt-4o"
    assert Settings(openai_model="").openai_model == "gpt-4o-mini"


def test_client_is_constructed_with_openai_api_key(monkeypatch):
    created: dict = {}

    class Fake:
        def __init__(self, *, api_key: str) -> None:
            created["api_key"] = api_key

        def close(self) -> None:
            return None

    monkeypatch.setattr("analysis.analyzer.OpenAI", Fake)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    loaded = load_settings()
    client = OpenAIAnalyzer(loaded)._client_or_create()
    assert isinstance(client, Fake)
    assert created["api_key"] == "sk-from-env"
    assert loaded.openai_model == "gpt-4o"


def test_auth_and_quota_do_not_escape_as_sdk_errors():
    def boom(error):
        def create(**kwargs):
            raise error

        return create

    auth = OpenAIAnalyzer(Settings(openai_api_key="sk-test"))
    auth._client = _Client(boom(_status_error(AuthenticationError, 401, "invalid api key")))
    try:
        asyncio.run(auth.analyze(sample_post()))
    except OpenAIUnavailable as exc:
        assert str(exc) == "openai_auth"
    else:
        raise AssertionError("auth error escaped")

    quota = OpenAIAnalyzer(Settings(openai_api_key="sk-test", openai_model="gpt-4o"))
    quota._client = _Client(boom(_status_error(RateLimitError, 429, "insufficient_quota")))
    try:
        asyncio.run(quota.analyze(sample_post()))
    except OpenAIUnavailable as exc:
        assert str(exc) == "openai_quota"
    else:
        raise AssertionError("quota error escaped")
