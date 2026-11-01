"""Public funding headlines from Google News, Webrazzi, EU-Startups, and TechCrunch.

A raise is kept when the headline names a company and at least one of a round,
an amount, or a funding verb together with a funding noun. The company may sit
before the verb or after "for" / "startup". The same company and round from two
outlets becomes one post. A dead feed does not drop the others.
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

# Longer phrases first: alternation is leftmost-first, not longest-match.
_VERB = re.compile(
    r"yatırım turunu kapattı|yatırım açıkladı|yatırım duyurdu|yatırım aldı|yatırım turu|yatırım yaptı|"
    r"fonlama aldı|fon aldı|turunu kapattı|turunda|"
    r"\b(?:raises?|raised|secured|secures?|bagged|bags?|closed|closes?|landed|lands?|"
    r"announced|announces?|nabbed|nabs?|received|receives?|collected|collects?|"
    r"scooped|scoops?|grabbed|grabs?|snagged|snags?|banked|banks?|netted|nets?|"
    r"hauled|hauls?|won|wins?|got|gets?|pulls?\s+in)\b",
    re.IGNORECASE,
)
_ROUND = re.compile(
    r"pre[-\s]seed|series\s+[a-f]\b|seri\s+[a-f]\b|growth round|bridge round|extension round|"
    r"büyüme turu|köprü turu|\bseed\b|\btohum\b",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"[$€£]\s?\d[\d.,]*\s?(?:million|billion|mn|bn|m|b|k)?"
    r"|\d[\d.,]*\s?(?:million|billion|milyon|milyar)",
    re.IGNORECASE,
)
_FUNDING_NOUN = re.compile(
    r"\b(?:funding|financing|investment|venture)\b|yatırım(?!cı)|fonlama",
    re.IGNORECASE,
)
_AFTER_KIND = re.compile(
    r"\b(?:startup|company|girişimi|sirketi|şirketi)\s+"
    r"([A-Z0-9][\w.&'-]*(?:\s+[A-Z0-9][\w.&'-]*){0,4})",
)
_LEAD_SKIP = {
    "the",
    "a",
    "an",
    "its",
    "their",
    "exclusive",
    "report",
    "breaking",
    "logistics",
    "fintech",
    "turkish",
    "türk",
    "turk",
    "new",
    "us",
    "uk",
}
_STOP_WORDS = {
    "to",
    "for",
    "in",
    "on",
    "with",
    "from",
    "after",
    "as",
    "at",
    "by",
    "of",
    "and",
    "or",
    "its",
    "their",
    "the",
    "a",
    "an",
    "ve",
    "ile",
    "için",
    "icin",
    "led",
    "that",
    "which",
    "who",
}
_GENERIC_COMPANY = {
    "funding",
    "startup",
    "startups",
    "founders",
    "founder",
    "investors",
    "investor",
    "round",
    "rounds",
    "series",
    "seed",
    "venture",
    "capital",
    "investment",
    "company",
    "guide",
    "tips",
    "report",
    "playbook",
    "yatırım",
    "yatirim",
    "fon",
    "fonlama",
    "tur",
    "turu",
    "seri",
    "girişim",
    "girisimi",
    "şirketi",
    "sirketi",
}
_QUESTION = {"how", "why", "what", "when", "tips", "guide", "nasıl", "neden", "nasil"}
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
    """Company, display round, display amount. None when this is not a raise.

    A company plus a round, a company plus an amount, or a company plus a
    funding verb and a funding noun is enough. The company can lead the
    headline, follow "startup", or follow "for".
    """
    headline = _clean_title(title)
    blob = f"{headline}\n{summary or ''}"
    verb = _VERB.search(blob)
    noun = _FUNDING_NOUN.search(blob)
    rnd = _normalize_round(_ROUND.search(blob))
    amount = _normalize_amount(_AMOUNT.search(blob))
    if not (rnd or amount or (verb and noun)):
        return None
    company = _company_name(headline)
    if not company:
        return None
    return company, rnd, amount


def annotate_funding(post: RawPost) -> RawPost:
    """Prefix a post whose text is a raise. Source, url, and id stay put."""
    if (post.body or "").lstrip().startswith("Signal: funding"):
        return post
    parsed = extract_funding(post.title, post.body)
    if parsed is None:
        return post
    company, rnd, amount = parsed
    return post.model_copy(
        update={"body": _body(company, rnd, amount, _what(post.body, post.title), post.body)}
    )


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
            not _VERB.search(outlet)
            and not _AMOUNT.search(outlet)
            and not _ROUND.search(outlet)
            and len(outlet.split()) <= 4
        ):
            text = text[: tail.start()].strip()
    return _PREFIX.sub("", text).strip()


def _company_name(headline: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", headline)
    named = _AFTER_KIND.search(text)
    if named:
        company = _finish_company(named.group(1))
        if company:
            return company
    verb = _VERB.search(text)
    head = text[: verb.start()] if verb else text
    company = _finish_company(_strip_money_and_round(head))
    if company:
        return company
    return _finish_company(_strip_money_and_round(text))


def _strip_money_and_round(text: str) -> str:
    text = _AMOUNT.sub(" ", text)
    text = _ROUND.sub(" ", text)
    text = _FUNDING_NOUN.sub(" ", text)
    text = re.sub(r"[$€£₺]", " ", text)
    text = re.sub(
        r"\b(?:dolar|dollar|dollars|euro|euros|lira|tl|usd|eur)\w*\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"['’](?:s|in|ın|un|ün|a|e|i|ı|u|ü|n)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" -–|:;,'\"")


def _finish_company(raw: str) -> str:
    words = [word.strip(".,;:|") for word in raw.split() if word.strip(".,;:|")]
    while words and words[0].casefold() in _LEAD_SKIP:
        words.pop(0)
    kept: list[str] = []
    for word in words:
        folded = word.casefold().strip(".'’")
        if kept and (folded in _STOP_WORDS or (word[:1].islower() and folded not in {"&", "and", "ve"})):
            break
        if not kept and (folded in _STOP_WORDS or word[:1].islower()):
            continue
        kept.append(word.strip(".'’"))
    while kept and kept[-1].casefold().strip(".") in _STOP_WORDS:
        kept.pop()
    if not kept or kept[0].casefold() in _QUESTION:
        return ""
    if len(kept) > 6:
        return ""
    name = " ".join(kept)
    if len(name) < 2 or len(name) > 60 or name.casefold() in _GENERIC_COMPANY:
        return ""
    if not re.search(r"[A-Za-zÀ-ÿ]", name):
        return ""
    return name


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
        "pre seed": "Pre-seed",
        "seed": "Seed",
        "tohum": "Tohum",
        "growth round": "Growth round",
        "bridge round": "Bridge round",
        "extension round": "Extension round",
        "büyüme turu": "Büyüme turu",
        "köprü turu": "Köprü turu",
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
