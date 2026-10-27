"""Shared HTTP behavior for the public-feed fetchers."""

from __future__ import annotations

import asyncio
import re
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from config import Settings
from database.models import RawPost


_STATUS_ID = re.compile(r"/status/(\d+)")


def canonical_key(post: RawPost) -> str:
    """Identity used to drop the same target before a second Scout pass.

    An X status id collapses Nitter and direct copies. Any other http(s) URL
    collapses trailing-slash and case variants. Everything else keeps the hash.
    """
    match = _STATUS_ID.search(post.url or "")
    if match is None and post.source in {"x", "x-direct"}:
        match = _STATUS_ID.search(f"/status/{post.external_id}")
    if match:
        return f"status:{match.group(1)}"
    url = (post.url or "").strip().rstrip("/").lower()
    if url:
        return f"url:{url}"
    return f"hash:{post.content_hash}"


class FetchError(RuntimeError):
    """A source failed after the retries we are willing to spend."""


class FetchBatch(BaseModel):
    source: str
    posts: list[RawPost] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Fetcher(Protocol):
    source: str

    async def fetch(self) -> FetchBatch:
        """Return whatever this source could read. Errors stay inside the batch."""


class RateLimiter:
    """Serial gap between requests. The first call does not sleep."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self._next:
                await asyncio.sleep(self._next - now)
            self._next = loop.time() + self.min_interval


def retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class BaseFetcher:
    source = "base"

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        *,
        min_interval: float = 0.0,
    ) -> None:
        self.settings = settings
        self.client = client
        self.limiter = RateLimiter(min_interval)

    async def get(self, url: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
        await self.limiter.wait()
        timeout = self.settings.request_timeout_seconds
        merged = {"User-Agent": self.settings.user_agent}
        if headers:
            merged.update(headers)
        last: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.client.get(url, headers=merged, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                await asyncio.sleep(0.25 * (2**attempt))
                continue
            if response.status_code == 429:
                last = FetchError(f"{response.status_code} for {url}")
                if attempt >= 1:
                    break
                delay = retry_after_seconds(response)
                if delay is None:
                    delay = 3.0
                await asyncio.sleep(min(delay, 8.0))
                continue
            if response.status_code in {500, 502, 503, 504}:
                last = FetchError(f"{response.status_code} for {url}")
                delay = retry_after_seconds(response)
                if delay is None:
                    delay = 0.25 * (2**attempt)
                await asyncio.sleep(min(delay, 5.0))
                continue
            if response.status_code >= 400:
                raise FetchError(f"{response.status_code} for {url}")
            return response
        raise FetchError(str(last) if last else f"failed {url}")

    async def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> object:
        response = await self.get(url, headers=headers)
        try:
            return response.json()
        except ValueError as exc:
            raise FetchError(f"invalid json from {url}") from exc
