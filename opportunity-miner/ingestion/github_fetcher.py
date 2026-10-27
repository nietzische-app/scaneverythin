"""Fast-growing GitHub repos from the public Search API.

There is no official trending endpoint. A repo qualifies when it was created
inside the configured window and its star count divided by age clears both a
floor and a stars-per-day rate. That is the acceleration signal passed on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from textutil import clip


class GitHubFetcher(BaseFetcher):
    source = "github"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.5)

    async def fetch(self) -> FetchBatch:
        topics = [topic.strip() for topic in self.settings.github_topics if topic.strip()]
        if not topics:
            return FetchBatch(source=self.source, errors=["github: no topics configured"])
        now = datetime.now(timezone.utc)
        day = (now - timedelta(days=max(1, self.settings.github_created_within_days))).date().isoformat()
        query = "(" + " OR ".join(f"topic:{topic}" for topic in topics) + f") created:>{day}"
        url = "https://api.github.com/search/repositories?" + urlencode(
            {"q": query, "sort": "stars", "order": "desc", "per_page": "20"}
        )
        headers = {"Accept": "application/vnd.github+json"}
        token = self.settings.github_token.strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            payload = await self.get_json(url, headers=headers)
        except FetchError as exc:
            return FetchBatch(source=self.source, errors=[f"github: {exc}"])
        if not isinstance(payload, dict):
            return FetchBatch(source=self.source, errors=["github: search did not return an object"])
        posts = select_repos(
            payload.get("items") or [],
            now=now,
            min_stars=self.settings.github_min_stars,
            min_per_day=self.settings.github_min_stars_per_day,
        )
        return FetchBatch(source=self.source, posts=posts)


def select_repos(
    items: list,
    *,
    now: datetime,
    min_stars: int,
    min_per_day: float,
) -> list[RawPost]:
    posts: list[RawPost] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        stars = int(item.get("stargazers_count") or 0)
        created = _parse_time(str(item.get("created_at") or ""))
        if created is None or stars < min_stars:
            continue
        age_days = max((now - created).total_seconds() / 86400, 1.0)
        per_day = stars / age_days
        if per_day < min_per_day:
            continue
        name = str(item.get("full_name") or item.get("name") or "").strip()
        url = str(item.get("html_url") or "").strip()
        if not name or not url:
            continue
        description = str(item.get("description") or "").strip()
        topics = [str(topic) for topic in (item.get("topics") or []) if str(topic).strip()]
        language = str(item.get("language") or "").strip()
        owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
        body = (
            f"Repository: {name}\n"
            f"What it does: {description or 'No description.'}\n"
            f"Topics: {', '.join(topics) or 'none'}\n"
            f"Language: {language or 'unknown'}\n"
            f"Stars: {stars} ({per_day:.1f}/day since {created.date().isoformat()})"
        )
        posts.append(
            RawPost(
                source="github",
                external_id=str(item.get("id") or name)[:300],
                url=url,
                title=clip(f"{name}: {description or 'fast-growing repo'}", 180),
                body=body,
                author=str(owner.get("login") or "") or None,
                score=stars,
                raw={"stars_per_day": round(per_day, 2), "topics": topics},
            )
        )
    return posts


def _parse_time(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
