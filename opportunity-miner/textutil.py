"""Small text helpers shared by ingestion, scoring, and delivery."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ISTANBUL = ZoneInfo("Europe/Istanbul")

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TAG_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
_HTML_RE = re.compile(r"(?s)<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_ATTEMPT_RE = re.compile(r"^attempts=(\d+)")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_istanbul(value: str | None) -> str:
    """Render a UTC ISO timestamp in the operator's local zone."""
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone(ISTANBUL)
    return local.strftime("%d.%m.%Y %H:%M TR")


def strip_html(value: str | None) -> str:
    text = _TAG_RE.sub(" ", value or "")
    text = _HTML_RE.sub(" ", text)
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", text).strip()


def clip(value: str | None, limit: int) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from a model response, including fenced blocks."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    raw = _FENCE_RE.sub("", raw).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("model response was not a JSON object")
    return payload


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "evet"}
    return False


def clamp_int(value: object, low: int, high: int, default: int) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def attempt_count(note: str | None) -> int:
    if not note:
        return 0
    match = _ATTEMPT_RE.match(note)
    if match is None:
        return 0
    return int(match.group(1))


def attempt_note(previous: str | None, message: str, limit: int = 400) -> str:
    count = attempt_count(previous) + 1
    text = f"attempts={count}; {message}"
    return text[:limit]
