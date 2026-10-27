"""Rising Google Trends queries for B2B, AI tools, and workflow automation.

This uses the public explore and related-queries endpoints that PyTrends wraps.
The pandas package is not installed. A query is kept when Google marks it
Breakout or its formatted growth is at least the configured percent (default 200).
"""

from __future__ import annotations

import json
import re
from urllib.parse import quote

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from textutil import clip

_PREFIX = ")]}'"
_GROWTH = re.compile(r"(\d+)\s*%")
# shortcut: widget token shape, switch if Google changes the explore payload
_EXPLORE = "https://trends.google.com/trends/api/explore"
_RELATED = "https://trends.google.com/trends/api/widgetdata/relatedsearches"


class TrendsFetcher(BaseFetcher):
    source = "trends"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.8)

    async def fetch(self) -> FetchBatch:
        keywords = [item.strip() for item in self.settings.trends_keywords if item.strip()]
        if not keywords:
            return FetchBatch(source=self.source, errors=["trends: no keywords configured"])
        posts: list[RawPost] = []
        errors: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            try:
                rows = await self._rising(keyword)
            except FetchError as exc:
                errors.append(f"trends:{keyword}: {exc}")
                continue
            for query, label in rows:
                key = query.casefold()
                if key in seen:
                    continue
                seen.add(key)
                posts.append(_post(keyword, query, label))
        return FetchBatch(source=self.source, posts=posts, errors=errors)

    async def _rising(self, keyword: str) -> list[tuple[str, str]]:
        request = {
            "comparisonItem": [{"keyword": keyword, "geo": "", "time": "today 3-m"}],
            "category": 0,
            "property": "",
        }
        explore_url = f"{_EXPLORE}?hl=en-US&tz=0&req={quote(json.dumps(request, separators=(',', ':')))}"
        explore = _loads(await self._read(explore_url))
        widget = _related_widget(explore)
        if widget is None:
            raise FetchError(f"no related-query widget for {keyword}")
        related_url = (
            f"{_RELATED}?hl=en-US&tz=0&req={quote(json.dumps(widget['request'], separators=(',', ':')))}"
            f"&token={quote(str(widget['token']))}"
        )
        payload = _loads(await self._read(related_url))
        return rising_queries(payload, min_growth=self.settings.trends_min_growth)

    async def _read(self, url: str) -> str:
        response = await self.get(
            url,
            headers={
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "Mozilla/5.0 (compatible; OpportunityRadar/0.1)",
            },
        )
        return response.text


def rising_queries(payload: dict, *, min_growth: int) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    default = payload.get("default") if isinstance(payload, dict) else None
    blocks = default.get("rankedList") if isinstance(default, dict) else None
    if not isinstance(blocks, list):
        return []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for row in block.get("rankedKeyword") or []:
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            label = str(row.get("formattedValue") or "").strip()
            if query and _surge(label, row.get("value"), min_growth):
                found.append((query, label or "Breakout"))
    return found


def _surge(label: str, value: object, min_growth: int) -> bool:
    if "breakout" in label.casefold():
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= min_growth:
        return True
    match = _GROWTH.search(label)
    return bool(match and int(match.group(1)) >= min_growth)


def _loads(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith(_PREFIX):
        raw = raw[len(_PREFIX) :].lstrip()
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise FetchError("trends payload was not json") from exc
    if not isinstance(payload, dict):
        raise FetchError("trends payload was not an object")
    return payload


def _related_widget(payload: dict) -> dict | None:
    for widget in payload.get("widgets") or []:
        if not isinstance(widget, dict):
            continue
        if widget.get("id") == "RELATED_QUERIES" and widget.get("token") and isinstance(widget.get("request"), dict):
            return widget
    return None


def _post(keyword: str, query: str, label: str) -> RawPost:
    growth = label if label.endswith("%") or "breakout" in label.casefold() else label
    title = clip(f"Search spike: {query}", 180)
    body = (
        f"Rising search: {query}\n"
        f"Growth: {growth}\n"
        f"Seed topic: {keyword}\n"
        "Problem / opportunity: people are suddenly searching for this. "
        "Treat the query as the unmet job, not as an existing product."
    )
    url = "https://trends.google.com/trends/explore?q=" + quote(query)
    return RawPost(
        source="trends",
        external_id=clip(query.casefold(), 300),
        url=url,
        title=title,
        body=body,
        score=_score(label),
        raw={"keyword": keyword, "growth": growth},
    )


def _score(label: str) -> int | None:
    if "breakout" in label.casefold():
        return 100
    match = _GROWTH.search(label)
    if not match:
        return None
    return min(100, int(match.group(1)) // 10)
