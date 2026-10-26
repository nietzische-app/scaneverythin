"""Hacker News top and new stories, narrowed to Show HN, Ask HN, and busy threads."""

from __future__ import annotations

import asyncio

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from textutil import strip_html

API_ROOT = "https://hacker-news.firebaseio.com/v0"
STORY_URL = "https://news.ycombinator.com/item?id={item_id}"


def is_focus_story(item: dict, min_comments: int) -> bool:
    title = str(item.get("title") or "").strip().lower()
    if title.startswith("show hn") or title.startswith("ask hn"):
        return True
    try:
        descendants = int(item.get("descendants") or 0)
    except (TypeError, ValueError):
        descendants = 0
    return descendants >= min_comments


def story_to_post(item: dict, comments: list[str]) -> RawPost | None:
    item_id = item.get("id")
    title = str(item.get("title") or "").strip()
    if item_id is None or not title:
        return None
    parts: list[str] = []
    outbound = str(item.get("url") or "").strip()
    if outbound:
        parts.append(f"Outbound: {outbound}")
    text = strip_html(str(item.get("text") or ""))
    if text:
        parts.append(text)
    if comments:
        parts.append("Top comments:\n" + "\n---\n".join(comments))
    try:
        score = int(item["score"]) if item.get("score") is not None else None
    except (TypeError, ValueError):
        score = None
    try:
        comments_n = int(item.get("descendants") or 0)
    except (TypeError, ValueError):
        comments_n = 0
    return RawPost(
        source="hn",
        external_id=str(item_id),
        url=STORY_URL.format(item_id=item_id),
        title=title,
        body="\n\n".join(parts),
        author=str(item.get("by") or "") or None,
        score=score,
        comment_count=comments_n,
        raw={"id": item_id, "url": outbound},
    )


class HackerNewsFetcher(BaseFetcher):
    source = "hn"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.0)

    async def fetch(self) -> FetchBatch:
        errors: list[str] = []
        try:
            top = await self.get_json(f"{API_ROOT}/topstories.json")
            new = await self.get_json(f"{API_ROOT}/newstories.json")
        except FetchError as exc:
            return FetchBatch(source=self.source, errors=[f"hn: {exc}"])
        ids = _merge_ids(top, new, self.settings.hn_story_limit)
        loaded, load_errors = await self._load_items(ids)
        errors.extend(load_errors)
        selected = [
            item
            for item in loaded
            if item
            and not item.get("deleted")
            and not item.get("dead")
            and item.get("type") == "story"
            and is_focus_story(item, self.settings.hn_min_comments)
        ][: self.settings.hn_story_limit]
        posts = await self._with_comments(selected, errors)
        return FetchBatch(source=self.source, posts=posts, errors=errors)

    async def _load_items(self, ids: list[int]) -> tuple[list[dict | None], list[str]]:
        semaphore = asyncio.Semaphore(8)
        errors: list[str] = []

        async def load(item_id: int) -> dict | None:
            async with semaphore:
                try:
                    payload = await self.get_json(f"{API_ROOT}/item/{item_id}.json")
                except FetchError as exc:
                    errors.append(f"hn:{item_id}: {exc}")
                    return None
            if not isinstance(payload, dict):
                return None
            return payload

        items = await asyncio.gather(*(load(item_id) for item_id in ids))
        return list(items), errors

    async def _with_comments(self, items: list[dict], errors: list[str]) -> list[RawPost]:
        semaphore = asyncio.Semaphore(8)
        sample = max(0, self.settings.hn_comment_sample)

        async def build(item: dict) -> RawPost | None:
            comments: list[str] = []
            async with semaphore:
                for kid in list(item.get("kids") or [])[:sample]:
                    try:
                        payload = await self.get_json(f"{API_ROOT}/item/{kid}.json")
                    except FetchError as exc:
                        errors.append(f"hn:comment:{kid}: {exc}")
                        continue
                    if not isinstance(payload, dict):
                        continue
                    text = strip_html(str(payload.get("text") or ""))
                    if text and not payload.get("deleted"):
                        comments.append(text[:500])
            return story_to_post(item, comments)

        built = await asyncio.gather(*(build(item) for item in items))
        return [post for post in built if post is not None]


def _merge_ids(top: object, new: object, limit: int) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    buckets = []
    if isinstance(top, list):
        buckets.append(top[:limit])
    if isinstance(new, list):
        buckets.append(new[:limit])
    for bucket in buckets:
        for value in bucket:
            try:
                item_id = int(value)
            except (TypeError, ValueError):
                continue
            if item_id in seen:
                continue
            seen.add(item_id)
            ids.append(item_id)
    return ids
