"""Minimal RSS/Atom item parser. Keeps the dependency list free of feedparser."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from database.models import RawPost
from ingestion.base import FetchError
from textutil import strip_html

_STATUS_MARK = "/status/"


def parse_feed(
    xml_text: str,
    *,
    source: str,
    author: str | None = None,
    limit: int | None = None,
) -> list[RawPost]:
    raw = (xml_text or "").strip()
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise FetchError(f"invalid feed: {exc}") from exc
    posts: list[RawPost] = []
    for item in root.iter():
        if _local(item.tag) not in {"item", "entry"}:
            continue
        fields: dict[str, str] = {}
        link = ""
        for child in list(item):
            name = _local(child.tag)
            text = "".join(child.itertext()).strip()
            if text and name not in fields:
                fields[name] = text
            if name == "link":
                rel = (child.attrib.get("rel") or "alternate").lower()
                if rel == "self":
                    continue
                href = (child.attrib.get("href") or text).strip()
                if href and not link:
                    link = href
        title = fields.get("title") or ""
        description = (
            fields.get("description")
            or fields.get("summary")
            or fields.get("encoded")
            or fields.get("content")
            or ""
        )
        guid = fields.get("guid") or fields.get("id") or link or title
        if not title and not description:
            continue
        status_id = _status_id(link) or _status_id(guid)
        external_id = status_id or guid
        item_author = author or _nested_author(item) or fields.get("creator") or None
        body = strip_html(description)
        if body == strip_html(title):
            body = ""
        posts.append(
            RawPost(
                source=source,
                external_id=external_id[:300],
                url=link or guid,
                title=title,
                body=body,
                author=item_author,
                raw={"guid": guid[:300]},
            )
        )
        if limit is not None and len(posts) >= limit:
            break
    return posts


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _status_id(value: str) -> str:
    if _STATUS_MARK not in value:
        return ""
    tail = value.split(_STATUS_MARK, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return "".join(digits)


def _nested_author(item: ET.Element) -> str | None:
    for child in list(item):
        if _local(child.tag) != "author":
            continue
        for grand in list(child):
            if _local(grand.tag) == "name":
                text = "".join(grand.itertext()).strip()
                if text:
                    return text
        text = "".join(child.itertext()).strip()
        if text:
            return text.split()[0]
    return None
