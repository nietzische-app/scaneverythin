"""Direct X recent search. This path does not call Nitter.

Recent search is the official API v2 endpoint and needs X_BEARER_TOKEN.
Without that token the fetcher records one error and the rest of the scan
continues. Queries default to #buildinpublic, #saas, "mcp server", and
"ai agent SaaS".
"""

from __future__ import annotations

from urllib.parse import urlencode

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from textutil import clip


class XDirectFetcher(BaseFetcher):
    source = "x-direct"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.5)

    async def fetch(self) -> FetchBatch:
        token = self.settings.x_bearer_token.strip()
        queries = [item.strip() for item in self.settings.x_direct_queries if item.strip()]
        if not queries:
            return FetchBatch(source=self.source, errors=["x-direct: no queries configured"])
        if not token:
            return FetchBatch(
                source=self.source,
                errors=["x-direct: X_BEARER_TOKEN is missing; Nitter was not used"],
            )
        query = direct_query(queries)
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
            return FetchBatch(source=self.source, errors=[f"x-direct: {exc}"])
        if not isinstance(payload, dict):
            return FetchBatch(source=self.source, errors=["x-direct: search did not return an object"])
        return FetchBatch(source=self.source, posts=posts_from_search(payload, queries))


def direct_query(queries: list[str]) -> str:
    parts: list[str] = []
    for query in queries:
        text = query.strip()
        if not text:
            continue
        parts.append(f'"{text}"' if " " in text else text)
    return "(" + " OR ".join(parts) + ") -is:retweet"


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
    return posts


def _matched_query(text: str, queries: list[str]) -> str:
    folded = text.casefold()
    for query in queries:
        needle = query.lstrip("#").casefold()
        if needle and needle in folded:
            return query
    return queries[0] if queries else ""
