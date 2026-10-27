"""Hacker News stories from the public Firebase API.

Index calls hit ``showstories``, ``askstories``, and ``newstories`` together.
Item payloads are then loaded in parallel. Callers receive a flat list of
normalized story dicts; deleted and empty items are dropped.
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
from textutil import strip_html

logger = logging.getLogger(__name__)

API_ROOT = "https://hacker-news.firebaseio.com/v0"
FEEDS = ("showstories", "askstories", "newstories")
DISCUSSION_URL = "https://news.ycombinator.com/item?id={item_id}"
DEFAULT_HEADERS = {"User-Agent": "OpportunityRadar/0.1 (public-feed trend research)"}
_DETAIL_CONCURRENCY = 12


def dedupe_ids(buckets: list[object], limit_per_feed: int) -> list[int]:
    """Keep the first sighting of each id, capped per feed, in feed order."""
    seen: set[int] = set()
    ordered: list[int] = []
    cap = max(0, limit_per_feed)
    for bucket in buckets:
        if not isinstance(bucket, list):
            continue
        taken = 0
        for value in bucket:
            if taken >= cap:
                break
            try:
                item_id = int(value)
            except (TypeError, ValueError):
                continue
            taken += 1
            if item_id in seen:
                continue
            seen.add(item_id)
            ordered.append(item_id)
    return ordered


def normalize_story(item: dict | None) -> dict | None:
    """Map one Firebase item onto the radar story schema. ``None`` means skip."""
    if not isinstance(item, dict):
        return None
    if item.get("deleted") or item.get("dead"):
        return None
    if item.get("type") not in {None, "story"}:
        return None
    try:
        item_id = int(item["id"])
    except (KeyError, TypeError, ValueError):
        return None
    title = str(item.get("title") or "").strip()
    body = strip_html(str(item.get("text") or ""))
    outbound = str(item.get("url") or "").strip()
    text = body or outbound
    if not title and not text:
        return None
    return {
        "id": f"hn_{item_id}",
        "source": "HackerNews",
        "title": title,
        "text": text,
        "author": str(item.get("by") or ""),
        "url": DISCUSSION_URL.format(item_id=item_id),
        "score": _as_int(item.get("score")),
        "comments_count": _as_int(item.get("descendants")),
        "timestamp": _as_int(item.get("time")),
    }


def story_to_raw_post(story: dict) -> RawPost:
    """Adapt the public schema to the record the scan queue stores."""
    return RawPost(
        source="hn",
        external_id=str(story["id"]),
        url=str(story["url"]),
        title=str(story["title"]),
        body=str(story["text"]),
        author=str(story["author"] or "") or None,
        score=int(story["score"]),
        comment_count=int(story["comments_count"]),
        raw=dict(story),
    )


async def fetch_hn_stories(
    client: httpx.AsyncClient,
    *,
    limit_per_feed: int = 40,
    timeout: float = 20.0,
    errors: list[str] | None = None,
) -> list[dict]:
    """Load deduped show, ask, and new stories. Item failures are skipped."""
    sink = errors if errors is not None else []
    id_lists = await asyncio.gather(
        *(_get_json(client, f"{API_ROOT}/{name}.json", timeout, sink, name) for name in FEEDS)
    )
    story_ids = dedupe_ids(list(id_lists), limit_per_feed)
    if not story_ids:
        return []
    semaphore = asyncio.Semaphore(_DETAIL_CONCURRENCY)

    async def load(item_id: int) -> dict | None:
        async with semaphore:
            payload = await _get_json(
                client,
                f"{API_ROOT}/item/{item_id}.json",
                timeout,
                sink,
                f"item:{item_id}",
            )
        if not isinstance(payload, dict):
            return None
        return normalize_story(payload)

    loaded = await asyncio.gather(*(load(item_id) for item_id in story_ids))
    return [story for story in loaded if story is not None]


class HackerNewsFetcher(BaseFetcher):
    source = "hn"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.0)

    async def fetch(self) -> FetchBatch:
        errors: list[str] = []
        try:
            stories = await fetch_hn_stories(
                self.client,
                limit_per_feed=self.settings.hn_story_limit,
                timeout=self.settings.request_timeout_seconds,
                errors=errors,
            )
        except Exception as exc:  # noqa: BLE001 — one source must not abort the scan
            logger.warning("hacker news fetch failed: %s", exc)
            return FetchBatch(source=self.source, errors=[f"hn: {exc}"])
        posts = [story_to_raw_post(story) for story in stories]
        return FetchBatch(source=self.source, posts=posts, errors=errors)


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
    errors: list[str],
    label: str,
) -> object | None:
    """GET JSON. Timeouts and 429s are retried; a bad item becomes ``None``."""
    last = "failed"
    for attempt in range(3):
        try:
            response = await client.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = str(exc)
            await asyncio.sleep(0.25 * (2**attempt))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            last = f"{response.status_code} for {url}"
            delay = _retry_after(response)
            if delay is None:
                delay = 0.4 * (2**attempt)
            await asyncio.sleep(min(delay, 5.0))
            continue
        if response.status_code >= 400:
            last = f"{response.status_code} for {url}"
            break
        if not response.content or response.content.strip() == b"null":
            return None
        try:
            return response.json()
        except ValueError:
            last = f"invalid json from {url}"
            break
    message = f"hn:{label}: {last}"
    logger.warning(message)
    if message not in errors:
        errors.append(message)
    return None


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        stories = await fetch_hn_stories(client, limit_per_feed=12, timeout=20.0)
    logger.info("fetched %s Hacker News stories", len(stories))
    for story in stories:
        logger.info(
            "%s score=%s comments=%s %s",
            story["id"],
            story["score"],
            story["comments_count"],
            story["title"],
        )


if __name__ == "__main__":
    asyncio.run(main())
