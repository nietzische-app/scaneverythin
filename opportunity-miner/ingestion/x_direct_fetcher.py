"""Direct X recent search for funding announcements, with Nitter as fallback.

Recent search is the official API v2 endpoint and needs X_BEARER_TOKEN.
Queries are English and Turkish funding phrases. When the token is missing
or the API call fails, those same phrases are read through Nitter RSS.
"""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from ingestion.funding_fetcher import annotate_funding
from ingestion.x_fetcher import XFetcher, rotating_slice
from textutil import clip


class XDirectFetcher(BaseFetcher):
    source = "x-direct"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.5)

    async def fetch(self) -> FetchBatch:
        queries = [item.strip() for item in self.settings.x_direct_queries if item.strip()]
        if not queries:
            return FetchBatch(source=self.source, errors=["x-direct: no queries configured"])
        token = self.settings.x_bearer_token.strip()
        if token:
            posts, error = await self._search_api(token, queries)
            if error is None:
                return FetchBatch(source=self.source, posts=posts)
        else:
            error = "x-direct: X_BEARER_TOKEN is missing; trying Nitter"
        fallback = await self._search_nitter(queries)
        return FetchBatch(
            source=self.source,
            posts=fallback.posts,
            errors=[error, *fallback.errors],
        )

    async def _search_api(self, token: str, queries: list[str]) -> tuple[list[RawPost], str | None]:
        posts: list[RawPost] = []
        for query in query_groups(queries):
            url = "https://api.twitter.com/2/tweets/search/recent?" + urlencode(
                {
                    "query": query,
                    "max_results": "10",
                    "tweet.fields": "public_metrics,author_id",
                }
            )
            try:
                payload = await self.get_json(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                )
            except FetchError as exc:
                return [], f"x-direct: {exc}"
            if not isinstance(payload, dict):
                return [], "x-direct: search did not return an object"
            posts.extend(posts_from_search(payload, queries))
        return posts, None

    async def _search_nitter(self, queries: list[str]) -> FetchBatch:
        limit = self.settings.x_funding_queries_per_scan
        chosen = rotating_slice(queries, limit, datetime.now(timezone.utc))
        mirror = XFetcher(self.settings, self.client, accounts=[], hashtags=[], min_interval=0.5)
        batch = await mirror.search(chosen, source=self.source)
        return FetchBatch(
            source=self.source,
            posts=[annotate_funding(post) for post in batch.posts],
            errors=batch.errors,
        )


def direct_query(queries: list[str]) -> str:
    parts: list[str] = []
    for query in queries:
        text = query.strip()
        if not text:
            continue
        parts.append(f'"{text}"' if " " in text else text)
    return "(" + " OR ".join(parts) + ") -is:retweet"


def query_groups(queries: list[str], limit: int = 480) -> list[str]:
    """Split a long OR into recent-search calls. One short list stays one query."""
    groups: list[list[str]] = []
    current: list[str] = []
    for query in queries:
        text = query.strip()
        if not text:
            continue
        trial = current + [text]
        if current and len(direct_query(trial)) > limit:
            groups.append(current)
            current = [text]
        else:
            current = trial
    if current:
        groups.append(current)
    return [direct_query(group) for group in groups]


def posts_from_search(payload: dict, queries: list[str]) -> list[RawPost]:
    posts: list[RawPost] = []
    for tweet in payload.get("data") or []:
        if not isinstance(tweet, dict):
            continue
        text = str(tweet.get("text") or "").strip()
        status_id = str(tweet.get("id") or "").strip()
        if not text or not status_id:
            continue
        metrics = tweet.get("public_metrics") if isinstance(tweet.get("public_metrics"), dict) else {}
        matched = _matched_query(text, queries)
        title = clip(text.split("\n", 1)[0], 180)
        posts.append(
            annotate_funding(
                RawPost(
                    source="x-direct",
                    external_id=status_id,
                    url=f"https://x.com/i/status/{status_id}",
                    title=title,
                    body=f"Query: {matched}\n{text}",
                    author=str(tweet.get("author_id") or "") or None,
                    score=int(metrics.get("like_count") or 0),
                    comment_count=int(metrics.get("reply_count") or 0),
                    raw={"query": matched},
                )
            )
        )
    return posts


def _matched_query(text: str, queries: list[str]) -> str:
    folded = text.casefold()
    for query in queries:
        needle = query.lstrip("#").casefold()
        if needle and needle in folded:
            return query
    return queries[0] if queries else ""
