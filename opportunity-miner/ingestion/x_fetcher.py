"""Public X posts through Nitter-compatible RSS. No cookies, tokens, or login.

Each scan takes a small rotating slice of the account list so a full day of
30-minute runs covers the catalog without hammering one mirror.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

from config import Settings
from database.models import RawPost
from ingestion.accounts import ACCOUNTS, HASHTAGS
from ingestion.base import FetchBatch, FetchError, RateLimiter
from ingestion.feeds import parse_feed


def rotating_slice(items: list[str] | tuple[str, ...], batch: int, now: datetime) -> list[str]:
    """Pick `batch` items, advancing twice an hour so the list turns over the day."""
    pool = list(items)
    if batch <= 0 or not pool:
        return []
    if batch >= len(pool):
        return pool
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    slot = now.hour * 2 + (now.minute // 30)
    start = (slot * batch) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(batch)]


class XFetcher:
    source = "x"

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        *,
        accounts: tuple[str, ...] | list[str] | None = None,
        hashtags: tuple[str, ...] | list[str] | None = None,
        clock: Callable[[], datetime] | None = None,
        min_interval: float = 0.8,
    ) -> None:
        self.settings = settings
        self.client = client
        self.accounts = list(accounts or ACCOUNTS)
        self.hashtags = list(hashtags or HASHTAGS)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.limiter = RateLimiter(min_interval)
        self._down: set[str] = set()
        self._warned = False

    async def fetch(self) -> FetchBatch:
        try:
            return await self._fetch()
        except Exception as exc:  # noqa: BLE001 — X is optional and must not abort the scan
            self._warn_exhausted(str(exc))
            return FetchBatch(source=self.source, posts=[], errors=[f"x: {exc}"])

    async def _fetch(self) -> FetchBatch:
        posts: list[RawPost] = []
        errors: list[str] = []
        now = self.clock()
        handles = rotating_slice(self.accounts, self.settings.x_max_accounts_per_scan, now)
        tags = rotating_slice(self.hashtags, self.settings.x_hashtags_per_scan, now)
        for handle in handles:
            if self._exhausted():
                errors.append("x: public RSS mirrors are down; remaining accounts skipped")
                break
            try:
                xml_text = await self._read(f"/{quote(handle)}/rss")
                parsed = parse_feed(xml_text, source=self.source, author=handle)
                posts.extend(parsed[: self.settings.x_posts_per_account])
            except FetchError as exc:
                errors.append(f"x:{handle}: {exc}")
        for tag in tags:
            if self._exhausted():
                errors.append("x: public RSS mirrors are down; remaining hashtags skipped")
                break
            query = quote(f"#{tag}")
            try:
                xml_text = await self._read(f"/search/rss?f=tweets&q={query}")
                parsed = parse_feed(xml_text, source=self.source, author=f"#{tag}")
                posts.extend(parsed[: self.settings.x_posts_per_account])
            except FetchError as exc:
                errors.append(f"x:#{tag}: {exc}")
        return FetchBatch(source=self.source, posts=posts, errors=errors)

    def _exhausted(self) -> bool:
        instances = [item for item in self.settings.nitter_instances if item]
        return bool(instances) and len(self._down) >= len(instances)

    async def _read(self, path: str) -> str:
        candidates = [base for base in self.settings.nitter_instances if base and base not in self._down]
        if not candidates:
            self._warn_exhausted("no live mirrors left")
            raise FetchError("all Nitter instances failed")
        last: Exception | None = None
        headers = {
            "User-Agent": self.settings.user_agent,
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        for index, base in enumerate(candidates):
            if index == 0:
                await self.limiter.wait()
            url = base.rstrip("/") + path
            try:
                response = await self.client.get(
                    url,
                    headers=headers,
                    timeout=min(self.settings.request_timeout_seconds, 8.0),
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self._down.add(base)
                last = exc
                logger.info("Nitter instance %s failed (%s); trying the next mirror", base, exc)
                continue
            if response.status_code in {403, 451}:
                self._down.add(base)
                last = FetchError(f"{response.status_code} from {base}")
                logger.info(
                    "Nitter instance %s returned %s; trying the next mirror",
                    base,
                    response.status_code,
                )
                continue
            if response.status_code in {401, 429} or response.status_code >= 500:
                self._down.add(base)
                last = FetchError(f"{response.status_code} from {base}")
                continue
            if response.status_code >= 400:
                last = FetchError(f"{response.status_code} from {base}")
                continue
            text = response.text or ""
            lowered = text.lower()
            if "<rss" not in lowered and "<feed" not in lowered:
                self._down.add(base)
                last = FetchError(f"{base} did not return a feed")
                continue
            return text
        detail = str(last) if last else "no feed"
        self._warn_exhausted(detail)
        raise FetchError(detail)

    def _warn_exhausted(self, detail: str) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "All Nitter instances failed (%s). X posts are skipped for this pass.",
            detail,
        )
