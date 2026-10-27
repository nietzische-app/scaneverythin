"""Public, unauthenticated sources for the radar."""

from __future__ import annotations

import httpx

from config import Settings
from ingestion.hn_fetcher import HackerNewsFetcher
from ingestion.producthunt_fetcher import ProductHuntFetcher
from ingestion.reddit_fetcher import RedditFetcher
from ingestion.x_fetcher import XFetcher

__all__ = [
    "HackerNewsFetcher",
    "ProductHuntFetcher",
    "RedditFetcher",
    "XFetcher",
    "build_fetchers",
]


def build_fetchers(settings: Settings, client: httpx.AsyncClient) -> list:
    return [
        HackerNewsFetcher(settings, client),
        RedditFetcher(settings, client),
        ProductHuntFetcher(settings, client),
        XFetcher(settings, client),
    ]
