"""
Official MCP fetch server (stdio): PyPI mcp-server-fetch, driven by MCP Python SDK client.

Enable with MCP_FETCH_ENABLED=true. Optional host allowlist: MCP_FETCH_ALLOWLIST (comma-separated;
patterns may use *.example.com suffix form).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from os import getenv
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def mcp_fetch_enabled() -> bool:
    return getenv("MCP_FETCH_ENABLED", "false").lower() in ("1", "true", "yes")


def _allowlist_entries() -> list[str]:
    raw = (getenv("MCP_FETCH_ALLOWLIST") or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _host_matches_pattern(host: str, pattern: str) -> bool:
    h = host.lower()
    p = pattern.strip().lower()
    if p.startswith("*."):
        base = p[2:]
        if not base:
            return False
        return h == base or h.endswith("." + base)
    return h == p


def url_host_allowed(url: str) -> tuple[bool, str]:
    """Returns (ok, reason). Empty allowlist allows all hosts (demo only — SSRF risk)."""
    entries = _allowlist_entries()
    if not entries:
        return True, ""
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"invalid URL: {e}"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "URL has no host"
    for pat in entries:
        if _host_matches_pattern(host, pat):
            return True, ""
    return False, f"host {host!r} not in MCP_FETCH_ALLOWLIST"


def _server_command() -> str:
    return (getenv("MCP_FETCH_COMMAND") or "python").strip() or "python"


def _server_args() -> list[str]:
    argv = getenv("MCP_FETCH_ARGV")
    if argv and argv.strip():
        return [x.strip() for x in argv.split(",") if x.strip()]
    return ["-m", "mcp_server_fetch"]


def _fetch_timeout_sec() -> float:
    try:
        v = float(getenv("MCP_FETCH_TIMEOUT_SEC") or "60")
    except ValueError:
        return 60.0
    return max(5.0, min(v, 300.0))


def _max_retries() -> int:
    try:
        n = int(getenv("MCP_FETCH_RETRIES") or "3")
    except ValueError:
        return 3
    return max(1, min(n, 5))


def _tool_result_to_text(result) -> str:
    parts: list[str] = []
    for c in result.content or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
    body = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return body or "MCP fetch failed (no details)"
    return body or "(empty response)"


async def _call_fetch_once(
    url: str,
    max_length: int | None,
    start_index: int | None,
    raw: bool | None,
) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cmd = _server_command()
    args = _server_args()
    env = os.environ.copy()
    params = StdioServerParameters(command=cmd, args=args, env=env)
    timeout = _fetch_timeout_sec()

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            arguments: dict = {"url": url}
            if max_length is not None:
                arguments["max_length"] = max_length
            if start_index is not None:
                arguments["start_index"] = start_index
            if raw is not None:
                arguments["raw"] = raw
            result = await asyncio.wait_for(
                session.call_tool("fetch", arguments),
                timeout=timeout,
            )
            return _tool_result_to_text(result)


def call_fetch_sync(
    url: str,
    max_length: int | None = None,
    start_index: int | None = None,
    raw: bool | None = None,
) -> str:
    """
    Run one stdio MCP session, call tool ``fetch``, return markdown or error text.
    Retries on failure (transport / init races in Docker).
    """
    last_err: Exception | None = None
    delays = (0.15, 0.35, 0.7)
    for attempt in range(_max_retries()):
        try:
            return asyncio.run(
                _call_fetch_once(url, max_length, start_index, raw),
            )
        except Exception as e:
            last_err = e
            logger.warning(
                "mcp fetch attempt %s/%s failed: %s",
                attempt + 1,
                _max_retries(),
                e,
            )
            if attempt < _max_retries() - 1:
                time.sleep(delays[min(attempt, len(delays) - 1)])
    return f"MCP fetch failed after {_max_retries()} attempts: {last_err}"
