"""Public funding headlines from Google News, Webrazzi, EU-Startups, and TechCrunch.

A raise is kept only when the headline names a company and a round or an amount.
The same company and round from two outlets becomes one post. A dead feed does
not drop the others.
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import urlparse

import httpx

from config import Settings
from database.models import RawPost
from ingestion.base import BaseFetcher, FetchBatch, FetchError
from ingestion.feeds import parse_feed
from textutil import clip

# shortcut: headline regex only, switch to a model extract when a real raise has no verb or amount in the title
_RAISE = re.compile(
    r"\b(?:raises?|raised|secures?|secured|bags?|bagged|closes?|closed|lands?)\b"
    r"|yatırım aldı|yatırım turu|yatırım yaptı|turunda",
    re.IGNORECASE,
)
_ROUND = re.compile(
    r"pre-seed|series\s+[a-e]\b|seri\s+[a-e]\b|growth round|büyüme turu|\bseed\b|\btohum\b",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"\$\s?\d[\d.,]*\s?(?:million|billion|mn|m|b|k)?"
    r"|\d[\d.,]*\s?(?:million|billion|milyon|milyar)",
    re.IGNORECASE,
)
_OUTLET_TAIL = re.compile(r"\s+[-–|]\s+([^-–|]{2,40})$")
_PREFIX = re.compile(r"^(?:exclusive|report|breaking)\s*:?\s*", re.IGNORECASE)
_PER_FEED = 40
_RSS_HEADERS = {"Accept": "application/rss+xml, application/xml, text/xml, */*"}


class FundingFetcher(BaseFetcher):
    source = "funding"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client, min_interval=0.4)

    async def fetch(self) -> FetchBatch:
        feeds = [url.strip() for url in self.settings.funding_feeds if url.strip()]
        if not feeds:
            return FetchBatch(source=self.source, errors=["funding: no feeds configured"])
        results = await asyncio.gather(*(self._read(url) for url in feeds), return_exceptions=True)
        posts: list[RawPost] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(f"funding: {result}")
                continue
            posts.extend(result[0])
            errors.extend(result[1])
        limit = max(1, self.settings.funding_item_limit)
        return FetchBatch(source=self.source, posts=collapse_funding(posts)[:limit], errors=errors)

    async def _read(self, url: str) -> tuple[list[RawPost], list[str]]:
        try:
            response = await self.get(url, headers=_RSS_HEADERS)
            parsed = parse_feed(response.text, source="funding", limit=_PER_FEED)
        except FetchError as exc:
            return [], [f"funding: {exc}"]
        return [item for post in parsed if (item := funding_post(post))], []


def funding_post(post: RawPost) -> RawPost | None:
    parsed = extract_funding(post.title, post.body)
    if parsed is None:
        return None
    company, rnd, amount = parsed
    what = _what(post.body, post.title)
    ident = f"{_slug(company)}:{_slug(rnd or amount)}"
    return RawPost(
        source="funding",
        external_id=ident[:300],
        url=post.url,
        title=post.title,
        body=_body(company, rnd, amount, what, post.body),
        author=_outlet(post.url),
    )


def extract_funding(title: str, summary: str = "") -> tuple[str, str, str] | None:
    """Company, display round, display amount. None when this is not a raise."""
    headline = _clean_title(title)
    blob = f"{headline}\n{summary or ''}"
    verb = _RAISE.search(blob)
    rnd = _normalize_round(_ROUND.search(blob))
    amount = _normalize_amount(_AMOUNT.search(blob))
    if not ((verb and (rnd or amount)) or (rnd and amount)):
        return None
    company = _company_name(headline)
    if not company:
        return None
    return company, rnd, amount


def collapse_funding(posts: list[RawPost]) -> list[RawPost]:
    """One row per company and round. Prefer the outlet URL over a Google News wrapper."""
    order: list[str] = []
    chosen: dict[str, RawPost] = {}
    for post in posts:
        key = post.external_id
        current = chosen.get(key)
        if current is None:
            chosen[key] = post
            order.append(key)
            continue
        if _google(current.url) and not _google(post.url):
            chosen[key] = post
    return [chosen[key] for key in order]


def _body(company: str, rnd: str, amount: str, what: str, description: str) -> str:
    lines = [
        "Signal: funding",
        f"Company: {company}",
        f"Round: {rnd}",
        f"Amount: {amount}",
        f"What: {what}",
        "",
        clip(description, 500),
    ]
    return "\n".join(lines).strip()


def _clean_title(title: str) -> str:
    text = " ".join((title or "").split())
    tail = _OUTLET_TAIL.search(text)
    if tail:
        outlet = tail.group(1)
        if (
            not _RAISE.search(outlet)
            and not _AMOUNT.search(outlet)
            and not _ROUND.search(outlet)
            and len(outlet.split()) <= 4
        ):
            text = text[: tail.start()].strip()
    return _PREFIX.sub("", text).strip()


def _company_name(headline: str) -> str:
    match = _RAISE.search(headline)
    head = headline[: match.start()] if match else headline
    head = _AMOUNT.sub(" ", head)
    head = _ROUND.sub(" ", head)
    head = re.sub(r"\s+", " ", head).strip(" -–|:;,'\"")
    words = head.split()
    while words and words[-1].casefold() in {"in", "a", "an", "its", "their", "the", "for", "to", "ve"}:
        words.pop()
    if not words or words[0].casefold() in {"how", "why", "what", "when", "tips", "guide"}:
        return ""
    if len(words) > 8 or len(" ".join(words)) > 60:
        return ""
    return " ".join(words)


def _normalize_round(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    folded = " ".join(match.group(0).casefold().split())
    if folded.startswith("series"):
        return f"Series {folded.split()[-1].upper()}"
    if folded.startswith("seri"):
        return f"Seri {folded.split()[-1].upper()}"
    labels = {
        "pre-seed": "Pre-seed",
        "seed": "Seed",
        "tohum": "Tohum",
        "growth round": "Growth round",
        "büyüme turu": "Büyüme turu",
    }
    return labels.get(folded, "")


def _normalize_amount(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    text = " ".join(match.group(0).split())
    return re.sub(r"^\$\s+", "$", text)


def _what(description: str, title: str) -> str:
    text = " ".join((description or "").split())
    if not text or text.casefold() == " ".join((title or "").split()).casefold():
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return clip(sentence, 240)


def _slug(value: str) -> str:
    text = value.casefold().replace("$", "")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:60] or "unknown"


def _outlet(url: str) -> str:
    host = urlparse(url or "").netloc.casefold().removeprefix("www.")
    if "news.google." in host:
        return "google-news"
    if "webrazzi" in host:
        return "webrazzi"
    if "eu-startups" in host:
        return "eu-startups"
    if "techcrunch" in host:
        return "techcrunch"
    return host.split(".")[0] or "funding"


def _google(url: str) -> bool:
    return "news.google." in (url or "").casefold()
