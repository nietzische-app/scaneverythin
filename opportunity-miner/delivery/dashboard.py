"""Server-rendered preview of the queue, the rejects, and the gold briefs."""

from __future__ import annotations

import html
from typing import Any

from database.models import InsightView, ItemRecord
from textutil import clip, format_istanbul

_SOURCE = {
    "hn": "Hacker News",
    "reddit": "Reddit",
    "producthunt": "Product Hunt",
    "x": "X",
}

_REASON = {
    "too_short": "çok kısa",
    "deleted": "silinmiş",
    "hiring": "iş ilanı",
    "link_dump": "link yığını",
    "below_threshold": "eşik altı",
}


def render_dashboard(context: dict[str, Any]) -> str:
    counts = context["counts"]
    gold = context["insights"]
    noise = context["noise"]
    queue = context["queue"]
    last = context.get("last_run")
    running = bool(context.get("running"))
    providers = context["providers"]
    token_required = bool(context.get("token_required"))
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fırsat Radarı</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' fill='%2314160f'/%3E%3Crect x='2' y='2' width='12' height='12' fill='%23e4ff57'/%3E%3C/svg%3E">
  <style>
    :root {{
      --bg: #14160f;
      --bg-2: #1c1f14;
      --ink: #f4f1e4;
      --muted: #a7ab93;
      --line: #343824;
      --lime: #e4ff57;
      --lime-ink: #17190a;
      --coral: #ff8d6c;
      --paper: #23271a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(900px 420px at 0% -10%, #30381c 0%, transparent 60%),
        var(--bg);
      font-family: "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      padding: 28px 7vw 18px;
      border-bottom: 1px solid var(--line);
    }}
    .mark {{
      margin: 0 0 8px;
      color: var(--lime);
      font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(40px, 6vw, 72px);
      font-weight: 560;
      letter-spacing: -0.04em;
      line-height: 0.9;
    }}
    .lede {{
      margin: 10px 0 0;
      max-width: 46ch;
      color: var(--muted);
      font-size: 18px;
    }}
    .actions {{ display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }}
    button {{
      appearance: none;
      border: 0;
      background: var(--lime);
      color: var(--lime-ink);
      font: 600 15px/1 ui-monospace, Menlo, Consolas, monospace;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      padding: 14px 18px;
      cursor: pointer;
    }}
    button:disabled {{ opacity: 0.55; cursor: wait; }}
    button:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 3px; }}
    .status {{
      color: var(--muted);
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    main {{ padding: 22px 7vw 64px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border: 1px solid var(--line);
      margin-bottom: 28px;
    }}
    .stat {{ padding: 14px 16px; border-right: 1px solid var(--line); }}
    .stat:last-child {{ border-right: 0; }}
    .stat b {{
      display: block;
      font-size: 28px;
      letter-spacing: -0.04em;
    }}
    .stat span {{
      color: var(--muted);
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .providers {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 22px;
    }}
    .pill {{
      border: 1px solid var(--line);
      padding: 4px 8px;
      color: var(--muted);
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .pill.on {{ color: var(--lime); border-color: #5d6a22; }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 28px;
      font-weight: 560;
      letter-spacing: -0.03em;
    }}
    .card {{
      background: var(--paper);
      border-top: 3px solid var(--lime);
      padding: 18px 18px 16px;
      margin-bottom: 14px;
    }}
    .card h3 {{ margin: 8px 0; font-size: 26px; letter-spacing: -0.03em; }}
    .meta, .quiet {{
      color: var(--muted);
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .score {{ color: var(--lime); }}
    p {{ margin: 8px 0; font-size: 18px; line-height: 1.45; }}
    ul {{ margin: 8px 0 0; padding-left: 18px; }}
    li {{ margin: 4px 0; }}
    a {{ color: var(--lime); }}
    .split {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 28px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 92px 1fr auto;
      gap: 12px;
      align-items: baseline;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
    }}
    .row a {{ color: var(--ink); text-decoration: none; }}
    .row a:hover {{ color: var(--lime); }}
    .tag {{ color: var(--coral); }}
    .empty {{
      border: 1px dashed var(--line);
      padding: 22px;
      color: var(--muted);
    }}
    .errors {{
      margin: 0 0 18px;
      color: var(--coral);
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 860px) {{
      header, .split, .stats {{ display: block; }}
      .actions {{ align-items: flex-start; margin-top: 18px; }}
      .stat {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .row {{ grid-template-columns: 1fr; gap: 4px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <p class="mark">Opportunity radar</p>
      <h1>Fırsat Radarı</h1>
      <p class="lede">Topluluk gürültüsünü ele, kalan az sayıdaki fikri Türkiye pazarına çevir.</p>
    </div>
    <div class="actions">
      <button id="scan" type="button">{'Taranıyor' if running else 'Şimdi tara'}</button>
      <div class="status" id="scan-status">{_last_line(last, running)}</div>
    </div>
  </header>
  <main>
    <div class="providers">
      {_pill('Gemini', providers['gemini'])}
      {_pill('DeepSeek', providers['deepseek'])}
      {_pill('Telegram', providers['telegram'])}
      {_pill('Belirteç', token_required)}
    </div>
    <section class="stats">
      <div class="stat"><span>Kuyruk</span><b>{_n(counts, 'new') + _n(counts, 'pending_classification') + _n(counts, 'pending_analysis')}</b></div>
      <div class="stat"><span>Elenen</span><b>{_n(counts, 'filtered_out')}</b></div>
      <div class="stat"><span>Altın</span><b>{_n(counts, 'analyzed')}</b></div>
      <div class="stat"><span>Hata</span><b>{_n(counts, 'error')}</b></div>
    </section>
    {_errors(last)}
    <h2>Altın fikirler</h2>
    {_gold(gold)}
    <div class="split">
      <section>
        <h2>Kuyruk</h2>
        {_rows(queue, empty="Kuyruk boş. Bir tarama yeni gönderileri buraya alır.")}
      </section>
      <section>
        <h2>Elenen gürültü</h2>
        {_rows(noise, empty="Henüz elenen gönderi yok.", reason=True)}
      </section>
    </div>
  </main>
  <script>
    const button = document.querySelector("#scan");
    const status = document.querySelector("#scan-status");
    button.disabled = {str(running).lower()};
    button.addEventListener("click", async () => {{
      button.disabled = true;
      status.textContent = "Tarama başladı";
      const response = await fetch("/run-scan", {{ method: "POST" }});
      if (response.status === 401) {{
        status.textContent = "Bu uç nokta belirteç istiyor";
        button.disabled = false;
        return;
      }}
      if (response.status === 409) {{
        status.textContent = "Tarama zaten çalışıyor";
      }}
      const poll = async () => {{
        const body = await fetch("/api/status").then((res) => res.json());
        if (body.running) {{
          status.textContent = "Taranıyor";
          setTimeout(poll, 1500);
          return;
        }}
        window.location.reload();
      }};
      poll();
    }});
  </script>
</body>
</html>
"""


def _pill(label: str, on: bool) -> str:
    state = "açık" if on else "kapalı"
    klass = "pill on" if on else "pill"
    return f'<span class="{klass}">{html.escape(label)} {state}</span>'


def _n(counts: dict[str, int], key: str) -> int:
    return int(counts.get(key, 0))


def _last_line(last: dict | None, running: bool) -> str:
    if running:
        return "Tarama sürüyor"
    if not last or not last.get("finished_at"):
        return "Henüz tarama yok"
    return f"Son tarama {format_istanbul(last.get('finished_at'))}"


def _errors(last: dict | None) -> str:
    if not last:
        return ""
    stats = last.get("stats") or {}
    errors = list(stats.get("source_errors") or [])
    if last.get("error"):
        errors.append(str(last["error"]))
    if not errors:
        return ""
    items = "".join(f"<div>{html.escape(clip(str(item), 180))}</div>" for item in errors[:8])
    return f'<div class="errors">{items}</div>'


def _gold(insights: list[InsightView]) -> str:
    if not insights:
        return '<div class="empty">Eşik üstü fikir henüz yok. Anahtarlar tanımlıysa bir sonraki tarama burayı doldurur.</div>'
    return "".join(_card(view) for view in insights)


def _card(view: InsightView) -> str:
    analysis = view.analysis or {}
    market = analysis.get("turkish_market") or {}
    blueprint = analysis.get("blueprint") or {}
    vibe = analysis.get("vibe_coding") or {}
    points = "".join(f"<li>{html.escape(str(point))}</li>" for point in (blueprint.get("points") or [])[:5])
    href = _safe_url(view.url)
    link = f'<a href="{href}" rel="noopener noreferrer">{html.escape(_SOURCE.get(view.source, view.source))}</a>' if href else html.escape(view.source)
    sent = {0: "bildirim bekliyor", 1: "telegram gitti", 2: "puan düşük, bildirilmedi"}.get(view.telegram_sent, "")
    return f"""
    <article class="card">
      <div class="meta"><span class="score">{int(view.opportunity_score)}/100</span> · güven {int(view.confidence)} · {link} · {html.escape(sent)} · {html.escape(format_istanbul(view.created_at))}</div>
      <h3>{html.escape(str(analysis.get("title_tr") or view.title))}</h3>
      <p>{html.escape(str(analysis.get("summary_tr") or ""))}</p>
      <p><strong>Çekirdek.</strong> {html.escape(str(analysis.get("core_problem_or_mechanic") or ""))}</p>
      <p><strong>Türkiye.</strong> {html.escape(str(market.get("adaptation") or ""))}</p>
      <p class="quiet">Ödeme: {html.escape(str(market.get("payments") or "—"))} · Kanal: {html.escape(str(market.get("commerce_or_messaging") or "—"))} · KVKK: {html.escape(str(market.get("kvkk") or "—"))}</p>
      <ul>{points}</ul>
      <p class="quiet">Vibe coding: {html.escape(str(vibe.get("difficulty_days", "—")))} gün. {html.escape(str(vibe.get("why") or ""))}</p>
    </article>
    """


def _rows(items: list[ItemRecord], *, empty: str, reason: bool = False) -> str:
    if not items:
        return f'<div class="empty">{html.escape(empty)}</div>'
    blocks = []
    for item in items:
        href = _safe_url(item.url)
        title = html.escape(item.title or item.external_id)
        label = f'<a href="{href}" rel="noopener noreferrer">{title}</a>' if href else title
        extra = ""
        if reason:
            why = _REASON.get(item.note or "", item.note or item.status)
            extra = f'<span class="tag">{html.escape(str(why))}</span>'
        else:
            extra = f'<span class="quiet">{html.escape(item.status)}</span>'
        blocks.append(
            "<div class=\"row\">"
            f"<span class=\"meta\">{html.escape(_SOURCE.get(item.source, item.source))}</span>"
            f"<span>{label}</span>"
            f"{extra}"
            "</div>"
        )
    return "".join(blocks)


def _safe_url(url: str | None) -> str:
    value = url or ""
    if value.startswith("https://") or value.startswith("http://"):
        return html.escape(value, quote=True)
    return ""
