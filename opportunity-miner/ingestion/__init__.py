"""Public, unauthenticated sources for the radar."""

from __future__ import annotations

import httpx

from config import Settings
from ingestion.github_fetcher import GitHubFetcher
from ingestion.hn_fetcher import HackerNewsFetcher
from ingestion.producthunt_fetcher import ProductHuntFetcher
from ingestion.reddit_fetcher import RedditFetcher
from ingestion.saas_pioneer_fetcher import SaasPioneerFetcher
from ingestion.trends_fetcher import TrendsFetcher
from ingestion.x_direct_fetcher import XDirectFetcher
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
        GitHubFetcher(settings, client),
        SaasPioneerFetcher(settings, client),
        TrendsFetcher(settings, client),
        XDirectFetcher(settings, client),
    ]
