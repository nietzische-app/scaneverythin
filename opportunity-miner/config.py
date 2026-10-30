"""Environment-backed settings.

The process boots with empty API keys. Ingestion and the dashboard still work;
Gemini, OpenAI, and Telegram stages no-op until their variables are set.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

_DEFAULT_SUBS = [
    "SaaS",
    "Entrepreneur",
    "SmallBusiness",
    "SideProject",
    "Ecommerce",
    "digitalmarketing",
    "IndieHackers",
    "gamedev",
]
_DEFAULT_NITTER = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.space",
    "https://xcancel.com",
]
_DEFAULT_GITHUB_TOPICS: list[str] = []
_DEFAULT_TREND_KEYWORDS = ["B2B software", "AI tools", "workflow automation"]
_DEFAULT_X_QUERIES = ["#buildinpublic", "#saas", "mcp server", "ai agent SaaS"]
_DEFAULT_FUNDING_FEEDS = [
    "https://news.google.com/rss/search?q=%22series%20A%22%20OR%20%22seed%20round%22%20OR%20%22yat%C4%B1r%C4%B1m%20ald%C4%B1%22%20OR%20%22yat%C4%B1r%C4%B1m%20turu%22%20when%3A7d&hl=en-US&gl=US&ceid=US:en",
    "https://webrazzi.com/feed/",
    "https://www.eu-startups.com/feed/",
    "https://techcrunch.com/category/venture/feed/",
]


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw.strip())


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    api_token: str = ""

    database_path: str = "data/radar.sqlite"
    scan_interval_seconds: int = 1800
    scheduler_enabled: bool = True
    scan_on_startup: bool = False
    confidence_threshold: int = 70
    classifier_mode: str = "batched"
    classifier_concurrency: int = 2
    analyzer_concurrency: int = 1
    max_classifications_per_scan: int = 25
    max_analyses_per_scan: int = 5
    notify_min_opportunity_score: int = 60
    request_timeout_seconds: float = 20.0
    llm_timeout_seconds: float = 45.0
    analyzer_timeout_seconds: float = 90.0

    hn_story_limit: int = 40
    hn_min_comments: int = 40
    hn_comment_sample: int = 2
    reddit_subs: list[str] = Field(default_factory=lambda: list(_DEFAULT_SUBS))
    reddit_limit: int = 15
    producthunt_feed_url: str = "https://www.producthunt.com/feed"
    x_max_accounts_per_scan: int = 8
    x_posts_per_account: int = 2
    x_hashtags_per_scan: int = 2
    nitter_instances: list[str] = Field(default_factory=lambda: list(_DEFAULT_NITTER))
    github_topics: list[str] = Field(default_factory=lambda: list(_DEFAULT_GITHUB_TOPICS))
    github_min_stars: int = 30
    github_min_stars_per_day: float = 3.0
    github_created_within_days: int = 180
    github_token: str = ""
    acquire_sitemap_url: str = "https://app.acquire.com/sitemap.xml"
    indiehackers_feed_url: str = "https://www.indiehackers.com/feed.rss"
    flippa_url: str = "https://flippa.com/buy/saas"
    saas_listing_limit: int = 5
    trends_keywords: list[str] = Field(default_factory=lambda: list(_DEFAULT_TREND_KEYWORDS))
    trends_min_growth: int = 200
    x_bearer_token: str = ""
    x_direct_queries: list[str] = Field(default_factory=lambda: list(_DEFAULT_X_QUERIES))
    funding_feeds: list[str] = Field(default_factory=lambda: list(_DEFAULT_FUNDING_FEEDS))
    funding_item_limit: int = 8

    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    user_agent: str = "OpportunityRadar/0.1 (public-feed trend research)"

    @field_validator("scan_interval_seconds")
    @classmethod
    def _interval(cls, value: int) -> int:
        return max(60, value)

    @field_validator("confidence_threshold", "notify_min_opportunity_score")
    @classmethod
    def _score_range(cls, value: int) -> int:
        return max(0, min(100, value))

    @field_validator("classifier_concurrency", "analyzer_concurrency")
    @classmethod
    def _concurrency(cls, value: int) -> int:
        return max(1, value)

    @field_validator("max_classifications_per_scan", "max_analyses_per_scan")
    @classmethod
    def _budgets(cls, value: int) -> int:
        return max(0, value)

    @field_validator("port")
    @classmethod
    def _port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return value

    @field_validator("gemini_model")
    @classmethod
    def _gemini_model(cls, value: str) -> str:
        raw = (value or "").strip()
        while raw.startswith("models/"):
            raw = raw.removeprefix("models/")
        return raw or "gemini-2.0-flash"

    @field_validator("classifier_mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in {"batched", "parallel"}:
            raise ValueError("classifier_mode must be 'batched' or 'parallel'")
        return mode

    @field_validator("producthunt_feed_url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("openai_model")
    @classmethod
    def _openai_model(cls, value: str) -> str:
        return (value or "").strip() or "gpt-4o-mini"

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token.strip() and self.telegram_chat_id.strip())


def load_settings() -> Settings:
    """Load `.env` without overriding variables already present in the process."""
    load_dotenv(override=False)
    return Settings(
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_model=_env("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
        api_token=_env("API_TOKEN"),
        database_path=_env("DATABASE_PATH", "data/radar.sqlite"),
        scan_interval_seconds=_int("SCAN_INTERVAL_SECONDS", 1800),
        scheduler_enabled=_bool("SCAN_SCHEDULER_ENABLED", True),
        scan_on_startup=_bool("SCAN_ON_STARTUP", False),
        confidence_threshold=_int("CONFIDENCE_THRESHOLD", 70),
        classifier_mode=_env("CLASSIFIER_MODE", "batched"),
        classifier_concurrency=_int("CLASSIFIER_CONCURRENCY", 2),
        analyzer_concurrency=_int("ANALYZER_CONCURRENCY", 1),
        max_classifications_per_scan=_int("MAX_CLASSIFICATIONS_PER_SCAN", 25),
        max_analyses_per_scan=_int("MAX_ANALYSES_PER_SCAN", 5),
        notify_min_opportunity_score=_int("NOTIFY_MIN_OPPORTUNITY_SCORE", 60),
        request_timeout_seconds=_float("REQUEST_TIMEOUT_SECONDS", 20.0),
        llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 45.0),
        analyzer_timeout_seconds=_float("ANALYZER_TIMEOUT_SECONDS", 90.0),
        hn_story_limit=_int("HN_STORY_LIMIT", 40),
        hn_min_comments=_int("HN_MIN_COMMENTS", 40),
        hn_comment_sample=_int("HN_COMMENT_SAMPLE", 2),
        reddit_subs=_csv("REDDIT_SUBS", _DEFAULT_SUBS),
        reddit_limit=_int("REDDIT_LIMIT", 15),
        producthunt_feed_url=_env("PRODUCTHUNT_FEED_URL", "https://www.producthunt.com/feed"),
        x_max_accounts_per_scan=_int("X_MAX_ACCOUNTS_PER_SCAN", 8),
        x_posts_per_account=_int("X_POSTS_PER_ACCOUNT", 2),
        x_hashtags_per_scan=_int("X_HASHTAGS_PER_SCAN", 2),
        nitter_instances=_csv("NITTER_INSTANCES", _DEFAULT_NITTER),
        github_topics=_csv("GITHUB_TOPICS", _DEFAULT_GITHUB_TOPICS),
        github_min_stars=_int("GITHUB_MIN_STARS", 30),
        github_min_stars_per_day=_float("GITHUB_MIN_STARS_PER_DAY", 3.0),
        github_created_within_days=_int("GITHUB_CREATED_WITHIN_DAYS", 180),
        github_token=_env("GITHUB_TOKEN"),
        acquire_sitemap_url=_env("ACQUIRE_SITEMAP_URL", "https://app.acquire.com/sitemap.xml"),
        indiehackers_feed_url=_env(
            "INDIEHACKERS_FEED_URL", "https://www.indiehackers.com/feed.rss"
        ),
        flippa_url=_env("FLIPPA_URL", "https://flippa.com/buy/saas"),
        saas_listing_limit=_int("SAAS_LISTING_LIMIT", 5),
        trends_keywords=_csv("TRENDS_KEYWORDS", _DEFAULT_TREND_KEYWORDS),
        trends_min_growth=_int("TRENDS_MIN_GROWTH", 200),
        x_bearer_token=_env("X_BEARER_TOKEN"),
        x_direct_queries=_csv("X_DIRECT_QUERIES", _DEFAULT_X_QUERIES),
        funding_feeds=_csv("FUNDING_FEEDS", _DEFAULT_FUNDING_FEEDS),
        funding_item_limit=_int("FUNDING_ITEM_LIMIT", 8),
        log_level=_env("LOG_LEVEL", "INFO"),
        host=_env("HOST", "0.0.0.0"),
        port=_int("PORT", 8000),
    )
