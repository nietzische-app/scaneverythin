"""Product Hunt daily launches via the public RSS feed.

Every scan polls `producthunt_feed_url` (default https://www.producthunt.com/feed).
"""

from __future__ import annotations

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from ingestion.feeds import parse_feed


class ProductHuntFetcher(BaseFetcher):
    source = "producthunt"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.0)

    async def fetch(self) -> FetchBatch:
        url = self.settings.producthunt_feed_url
        headers = {"Accept": "application/rss+xml, application/xml, text/xml, */*"}
        try:
            response = await self.get(url, headers=headers)
            posts = parse_feed(response.text, source=self.source)
        except FetchError as exc:
            return FetchBatch(source=self.source, errors=[f"producthunt: {exc}"])
        return FetchBatch(source=self.source, posts=posts, errors=[])


def launches_from_feed(xml_text: str) -> list[RawPost]:
    return parse_feed(xml_text, source="producthunt")
