"""Public micro-SaaS listings from Acquire, Indie Hackers, and Flippa.

Flippa is the marketplace polled for listed SaaS (the public buy/saas page).
Each kept row carries the problem text, a revenue tier, and a business-model
label in the body so Scout can score it. A failed feed does not drop the others.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from ingestion.feeds import parse_feed
from textutil import clip, strip_html

_HTML_HEADERS = {
    "Accept": "text/html, application/xhtml+xml, */*",
    "User-Agent": "Mozilla/5.0 (compatible; OpportunityRadar/0.1; +https://github.com)",
}
_MONEY = re.compile(
    r"(?:mrr|arr|monthly revenue|monthly profit|revenue)?[^$]{0,24}\$\s*([0-9][0-9,]*(?:\.\d+)?)\s*([kKmM])?",
    re.IGNORECASE,
)
_MONEY_WORD = re.compile(
    r"\$\s*([0-9][0-9,]*(?:\.\d+)?)\s*([kKmM])?\s*(?:/\s*mo|per month|a month|mrr|arr)",
    re.IGNORECASE,
)


class SaasPioneerFetcher(BaseFetcher):
    source = "saas"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.4)

    async def fetch(self) -> FetchBatch:
        results = await asyncio.gather(
            self._acquire(),
            self._indiehackers(),
            self._flippa(),
            return_exceptions=True,
        )
        posts: list[RawPost] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(f"saas: {result}")
                continue
            posts.extend(result[0])
            errors.extend(result[1])
        return FetchBatch(source=self.source, posts=posts, errors=errors)

    async def _acquire(self) -> tuple[list[RawPost], list[str]]:
        limit = max(1, self.settings.saas_listing_limit)
        try:
            response = await self.get(self.settings.acquire_sitemap_url, headers={"Accept": "application/xml, text/xml"})
        except FetchError as exc:
            return [], [f"acquire: {exc}"]
        urls = newest_startup_urls(response.text, limit)
        posts: list[RawPost] = []
        errors: list[str] = []
        for url in urls:
            try:
                page = await self.get(url, headers=_HTML_HEADERS)
                html = page.text
            except FetchError as exc:
                errors.append(f"acquire: {exc}")
                html = ""
            posts.append(listing_post("acquire", url, html, fallback_title=_slug_title(url)))
        return posts, errors

    async def _indiehackers(self) -> tuple[list[RawPost], list[str]]:
        try:
            response = await self.get(
                self.settings.indiehackers_feed_url,
                headers={"Accept": "application/rss+xml, application/xml, text/xml"},
            )
            parsed = parse_feed(response.text, source="indiehackers", limit=self.settings.saas_listing_limit)
        except FetchError as exc:
            return [], [f"indiehackers: {exc}"]
        return [annotate_listing(post, "indiehackers") for post in parsed], []

    async def _flippa(self) -> tuple[list[RawPost], list[str]]:
        try:
            response = await self.get(self.settings.flippa_url, headers=_HTML_HEADERS)
        except FetchError as exc:
            return [], [f"flippa: {exc}"]
        # shortcut: public card markup, switch when Flippa stops putting listing ids in hrefs
        posts = flippa_posts(response.text, self.settings.saas_listing_limit)
        return posts, []


def newest_startup_urls(xml_text: str, limit: int) -> list[str]:
    raw = (xml_text or "").strip()
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    rows: list[tuple[str, str]] = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1].lower() != "url":
            continue
        loc = ""
        lastmod = ""
        for child in list(node):
            name = child.tag.rsplit("}", 1)[-1].lower()
            text = "".join(child.itertext()).strip()
            if name == "loc":
                loc = text
            elif name == "lastmod":
                lastmod = text
        if "/startup/" in loc:
            rows.append((lastmod, loc))
    rows.sort(reverse=True)
    found: list[str] = []
    for _stamp, url in rows:
        if url not in found:
            found.append(url)
        if len(found) >= limit:
            break
    return found


def listing_post(source: str, url: str, html: str, *, fallback_title: str) -> RawPost:
    title = _meta(html, "og:title") or _tag_text(html, "title") or fallback_title
    description = _meta(html, "og:description") or _meta(html, "description") or strip_html(html)[:500]
    return _pack(source, url, title, description)


def annotate_listing(post: RawPost, marketplace: str) -> RawPost:
    text = f"{post.title}\n{post.body}"
    packed = _pack(marketplace, post.url or post.external_id, post.title, text)
    packed.external_id = post.external_id
    packed.author = post.author
    packed.score = post.score
    packed.comment_count = post.comment_count
    return packed


def flippa_posts(html: str, limit: int) -> list[RawPost]:
    posts: list[RawPost] = []
    seen: set[str] = set()
    for match in re.finditer(r'href="(https://flippa\.com/(\d+)[^"]*|/\d+[^"]*)"', html or "", re.IGNORECASE):
        href = match.group(1)
        listing_id = match.group(2) or href.strip("/").split("/")[0]
        if not listing_id.isdigit() or listing_id in seen:
            continue
        seen.add(listing_id)
        url = href if href.startswith("http") else f"https://flippa.com/{listing_id}"
        window = html[match.start() : match.start() + 700]
        title = _link_text(window) or f"Flippa listing {listing_id}"
        posts.append(_pack("flippa", url.split("?")[0], title, strip_html(window)))
        if len(posts) >= limit:
            break
    return posts


def revenue_tier(text: str) -> str:
    amount = _monthly_amount(text)
    if amount is None:
        return "unknown"
    if amount < 1000:
        return "under $1k MRR"
    if amount < 10000:
        return "$1k–$10k MRR"
    return "$10k+ MRR"


def business_model(text: str) -> str:
    folded = text.casefold()
    if "marketplace" in folded:
        return "marketplace"
    if any(token in folded for token in ("subscription", "saas", "mrr", "arr")):
        return "subscription SaaS"
    if "usage" in folded:
        return "usage-based"
    if any(token in folded for token in ("one-time", "one time", "paid app")):
        return "one-time purchase"
    return "unspecified"


def _pack(source: str, url: str, title: str, description: str) -> RawPost:
    clean_title = clip(strip_html(title).strip() or url, 180)
    clean_body = strip_html(description).strip()
    tier = revenue_tier(f"{clean_title}\n{clean_body}")
    model = business_model(f"{clean_title}\n{clean_body}")
    body = (
        f"Problem / mechanism: {clip(clean_body or clean_title, 600)}\n"
        f"Revenue tier: {tier}\n"
        f"Business model: {model}\n"
        f"Marketplace: {source}"
    )
    slug = url.rstrip("/").rsplit("/", 1)[-1][:300] or clean_title
    return RawPost(source=source, external_id=slug, url=url, title=clean_title, body=body)


def _monthly_amount(text: str) -> float | None:
    match = _MONEY_WORD.search(text)
    if match is None:
        for candidate in _MONEY.finditer(text):
            prefix = text[max(0, candidate.start() - 48) : candidate.start()].casefold()
            if any(token in prefix for token in ("mrr", "arr", "monthly", "revenue", "profit")):
                match = candidate
                break
    if match is None:
        return None
    number = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        number *= 1000
    elif suffix == "m":
        number *= 1_000_000
    return number


def _meta(html: str, key: str) -> str:
    pattern = (
        rf'<meta[^>]+(?:property|name)="{re.escape(key)}"[^>]+content="([^"]*)"'
        rf'|<meta[^>]+content="([^"]*)"[^>]+(?:property|name)="{re.escape(key)}"'
    )
    match = re.search(pattern, html or "", re.IGNORECASE)
    if match is None:
        return ""
    return strip_html(match.group(1) or match.group(2) or "").strip()


def _tag_text(html: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html or "", re.IGNORECASE | re.DOTALL)
    return strip_html(match.group(1)).strip() if match else ""


def _link_text(fragment: str) -> str:
    match = re.search(r">([^<]{3,180})<", fragment)
    return strip_html(match.group(1)).strip() if match else ""


def _slug_title(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"^[a-z0-9]{6,}-", "", slug)
    return clip(slug.replace("-", " ").strip() or url, 180)
