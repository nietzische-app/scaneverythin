"""Fast DNS availability check for suggested hostnames.

A name that resolves is treated as taken. A resolver error that means
"no such name" is treated as open. This is not a registrar WHOIS query.
shortcut: any gaierror counts as open, so a temporary resolver failure can
look available; switch to registrar WHOIS when a false BOŞTA matters.
"""

from __future__ import annotations

import asyncio
import logging
import socket

logger = logging.getLogger(__name__)

LABELS = {
    "open": "BOŞTA 🟢",
    "taken": "DOLU 🔴",
    "unknown": "BELİRSİZ ⚪",
}


async def check_domains(names: list[str], resolve=None) -> list[dict]:
    """Return `{name, status}` for up to two hostnames. Never raises."""
    cleaned = [str(name).strip() for name in names if str(name).strip()][:2]

    async def one(name: str) -> dict:
        try:
            if resolve is not None:
                status = resolve(name)
                if asyncio.iscoroutine(status):
                    status = await status
            else:
                status = await _resolve(name)
        except socket.gaierror:
            status = "open"
        except Exception as exc:  # noqa: BLE001
            logger.warning("domain check failed for %s: %s", name, exc)
            status = "unknown"
        if status not in LABELS:
            status = "unknown"
        return {"name": name, "status": status}

    return list(await asyncio.gather(*(one(name) for name in cleaned)))


async def _resolve(name: str) -> str:
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(loop.getaddrinfo(name, None), timeout=2)
    except asyncio.TimeoutError:
        return "unknown"
    except socket.gaierror:
        return "open"
    except OSError:
        return "unknown"
    return "taken"
