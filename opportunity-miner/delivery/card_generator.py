"""1200×630 dark PNG for opportunities that scored 80 or above."""

from __future__ import annotations

import io
import os

from database.models import InsightView

_W, _H = 1200, 630
_BG = (11, 18, 32)
_INK = (241, 245, 249)
_MUTED = (148, 163, 184)
_BADGE = (16, 185, 129)
_CHIP = (30, 41, 59)
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def render_card(view: InsightView) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (_W, _H), _BG)
    draw = ImageDraw.Draw(image)
    title_font = _font(ImageFont, 54, bold=True)
    body_font = _font(ImageFont, 28, bold=False)
    badge_font = _font(ImageFont, 42, bold=True)
    analysis = view.analysis or {}
    title = str(analysis.get("title_tr") or view.title or "Fırsat")
    draw.rectangle((0, 0, 14, _H), fill=_BADGE)
    draw.rounded_rectangle((980, 48, 1120, 148), radius=22, fill=_BADGE)
    score = str(int(view.opportunity_score))
    draw.text((1012, 74), score, font=badge_font, fill=(255, 255, 255))
    _write_lines(draw, _wrap(draw, title, title_font, 860), title_font, 64, 170, _INK, 64)
    x = 64
    y = 430
    for tag in _tags(analysis):
        width = int(_text_width(draw, tag, body_font)) + 36
        if x + width > 1100:
            break
        draw.rounded_rectangle((x, y, x + width, y + 48), radius=16, fill=_CHIP)
        draw.text((x + 18, y + 8), tag, font=body_font, fill=_INK)
        x += width + 16
    draw.text((64, 560), "Nodify Radar", font=body_font, fill=_MUTED)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _tags(analysis: dict) -> list[str]:
    kind = str(analysis.get("opportunity_type") or "").replace("_", " ").strip()
    blueprint = analysis.get("blueprint") if isinstance(analysis.get("blueprint"), dict) else {}
    points = [str(point).strip() for point in (blueprint.get("points") or []) if str(point).strip()]
    tags = ([kind] if kind else []) + points[:2]
    return [tag[:28] for tag in tags[:3]]


def _font(image_font, size: int, *, bold: bool):
    paths = _FONT_CANDIDATES if bold else _FONT_CANDIDATES[::-1]
    for path in paths:
        if os.path.exists(path):
            return image_font.truetype(path, size)
    return image_font.load_default()


def _text_width(draw, text: str, font) -> float:
    try:
        return float(draw.textlength(text, font=font))
    except Exception:  # noqa: BLE001
        return float(len(text) * 12)


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return ["Fırsat"]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if _text_width(draw, trial, font) <= max_width:
            current = trial
            continue
        lines.append(current)
        current = word
        if len(lines) == 3:
            break
    if len(lines) < 3:
        lines.append(current)
    return lines[:3]


def _write_lines(draw, lines: list[str], font, x: int, y: int, fill, gap: int) -> None:
    for index, line in enumerate(lines):
        draw.text((x, y + index * gap), line, font=font, fill=fill)
