"""Free local rejects. These never call a model.

The heuristic is intentionally narrow. Gemini does the real noise cut;
this only drops posts that cannot be an opportunity.
"""

from __future__ import annotations

import re

from database.models import RawPost

_HIRING = re.compile(
    r"\b(we['’]re hiring|now hiring|job opening|hiring:)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+")


def heuristic_reject(post: RawPost) -> str | None:
    title = (post.title or "").strip()
    body = (post.body or "").strip()
    if title.lower() in {"[deleted]", "[removed]"}:
        return "deleted"
    text = f"{title}\n{body}".strip()
    if len(text) < 25:
        return "too_short"
    if _HIRING.search(text):
        return "hiring"
    urls = _URL.findall(text)
    words = _URL.sub(" ", text).split()
    if len(urls) >= 6 and len(words) < 30:
        return "link_dump"
    return None
