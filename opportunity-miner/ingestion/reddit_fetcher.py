"""Public Reddit listings. No OAuth; one polite request per subreddit.

The Atom feed is the primary source. Reddit's `.json` listing often answers
403 to datacenter addresses, so it is only the fallback.
"""

from __future__ import annotations

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from ingestion.feeds import parse_feed


def parse_reddit_listing(payload: object, subreddit: str) -> list[RawPost]:
    if not isinstance(payload, dict):
        return []
    children = (payload.get("data") or {}).get("children") or []
    posts: list[RawPost] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        data = child.get("data") or {}
        if not isinstance(data, dict) or data.get("stickied"):
            continue
        title = str(data.get("title") or "").strip()
        if not title or title.lower() in {"[deleted]", "[removed]"}:
            continue
        selftext = str(data.get("selftext") or "")
        if selftext.strip().lower() in {"[removed]", "[deleted]"}:
            selftext = ""
        permalink = str(data.get("permalink") or "")
        if permalink.startswith("/"):
            url = f"https://www.reddit.com{permalink}"
        else:
            url = str(data.get("url") or "")
        external_id = str(data.get("id") or permalink or title)
        posts.append(
            RawPost(
                source="reddit",
                external_id=external_id,
                url=url,
                title=title,
                body=selftext,
                author=str(data.get("author") or "") or None,
                score=_int_or_none(data.get("score")),
                comment_count=_int_or_none(data.get("num_comments")),
                raw={
                    "subreddit": data.get("subreddit") or subreddit,
                    "link_url": data.get("url") or "",
                },
            )
        )
    return posts


class RedditFetcher(BaseFetcher):
    source = "reddit"

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        *,
        min_interval: float = 2.0,
    ) -> None:
        super().__init__(settings, client, min_interval=min_interval)

    async def fetch(self) -> FetchBatch:
        posts: list[RawPost] = []
        errors: list[str] = []
        for subreddit in self.settings.reddit_subs:
            parsed, error = await self._load_subreddit(subreddit)
            if error:
                errors.append(error)
            posts.extend(parsed)
        return FetchBatch(source=self.source, posts=posts, errors=errors)

    async def _load_subreddit(self, subreddit: str) -> tuple[list[RawPost], str | None]:
        rss_url = f"https://www.reddit.com/r/{subreddit}/.rss?limit={self.settings.reddit_limit}"
        try:
            response = await self.get(
                rss_url,
                headers={"Accept": "application/atom+xml, application/rss+xml, application/xml, text/xml, */*"},
            )
            return parse_feed(response.text, source=self.source), None
        except FetchError as rss_error:
            json_url = (
                f"https://www.reddit.com/r/{subreddit}/hot.json"
                f"?limit={self.settings.reddit_limit}&raw_json=1"
            )
            try:
                payload = await self.get_json(json_url, headers={"Accept": "application/json"})
            except FetchError as json_error:
                return [], f"reddit:{subreddit}: {rss_error}; {json_error}"
            return parse_reddit_listing(payload, subreddit), None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
