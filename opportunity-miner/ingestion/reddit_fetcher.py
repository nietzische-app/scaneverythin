"""Public Reddit listings. No OAuth; one polite request per subreddit.

The Atom feed is the primary source. Reddit's `.json` listing is the fallback.
Every request sends a browser-like User-Agent; Reddit answers 403 or 429
without one.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from ingestion.feeds import parse_feed

logger = logging.getLogger(__name__)

REDDIT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpportunityMiner/1.0"


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
        min_interval: float = 3.0,
    ) -> None:
        super().__init__(settings, client, min_interval=min_interval)

    async def get(self, url: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
        merged = {"User-Agent": REDDIT_USER_AGENT}
        if headers:
            merged.update(headers)
        merged["User-Agent"] = REDDIT_USER_AGENT
        return await super().get(url, headers=merged)

    async def fetch(self) -> FetchBatch:
        posts: list[RawPost] = []
        errors: list[str] = []
        seen: set[str] = set()
        subs = list(self.settings.reddit_subs)
        for index, subreddit in enumerate(subs):
            try:
                parsed = await self._rss(subreddit)
            except FetchError as exc:
                if "429" not in str(exc):
                    parsed, json_error = await self._json_or_error(subreddit, exc)
                    if json_error:
                        errors.append(json_error)
                        continue
                else:
                    try:
                        parsed = await self._combined(subs[index:])
                    except FetchError as combined_error:
                        if posts and "429" in str(combined_error):
                            logger.warning(
                                "Reddit rate limit after %s posts; stopping this pass.",
                                len(posts),
                            )
                            break
                        errors.append(f"reddit: {combined_error}")
                        break
                self._add(posts, seen, parsed)
                if "429" in str(exc):
                    break
                continue
            self._add(posts, seen, parsed)
        return FetchBatch(source=self.source, posts=posts, errors=errors)

    async def _rss(self, subreddit: str) -> list[RawPost]:
        url = f"https://www.reddit.com/r/{subreddit}/.rss?limit={self.settings.reddit_limit}"
        response = await self._send(
            url,
            "application/atom+xml, application/rss+xml, application/xml, text/xml, */*",
        )
        return parse_feed(response.text, source=self.source)

    async def _combined(self, subreddits: list[str]) -> list[RawPost]:
        names = "+".join(subreddits)
        limit = min(25, max(self.settings.reddit_limit, 1) * max(len(subreddits), 1))
        url = f"https://www.reddit.com/r/{names}/.rss?limit={limit}"
        response = await self._send(
            url,
            "application/atom+xml, application/rss+xml, application/xml, text/xml, */*",
        )
        return parse_feed(response.text, source=self.source)

    async def _json_or_error(self, subreddit: str, rss_error: FetchError) -> tuple[list[RawPost], str | None]:
        url = (
            f"https://www.reddit.com/r/{subreddit}/hot.json"
            f"?limit={self.settings.reddit_limit}&raw_json=1"
        )
        try:
            response = await self._send(url, "application/json")
            payload = response.json()
        except (FetchError, ValueError) as exc:
            return [], f"reddit:{subreddit}: {rss_error}; {exc}"
        return parse_reddit_listing(payload, subreddit), None

    async def _send(self, url: str, accept: str) -> httpx.Response:
        await self.limiter.wait()
        try:
            response = await self.client.get(
                url,
                headers={"User-Agent": REDDIT_USER_AGENT, "Accept": accept},
                timeout=self.settings.request_timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise FetchError(str(exc)) from exc
        if response.status_code >= 400:
            raise FetchError(f"{response.status_code} for {url}")
        return response

    @staticmethod
    def _add(posts: list[RawPost], seen: set[str], extra: list[RawPost]) -> None:
        for post in extra:
            if post.external_id in seen:
                continue
            seen.add(post.external_id)
            posts.append(post)


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    settings = Settings(reddit_limit=5, request_timeout_seconds=20)
    headers = {"User-Agent": REDDIT_USER_AGENT}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        batch = await RedditFetcher(settings, client, min_interval=3.0).fetch()
    if batch.errors:
        for error in batch.errors:
            logger.error(error)
        raise SystemExit(1)
    logger.info("fetched %s reddit posts", len(batch.posts))
    for post in batch.posts:
        logger.info("%s %s", post.external_id, post.title)
    if not batch.posts:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
