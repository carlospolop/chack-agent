from __future__ import annotations

import asyncio
import json
import shutil
import threading
from functools import lru_cache
from typing import Any

from agents.mcp import MCPServerStdio


_PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@latest"


@lru_cache(maxsize=1)
def playwright_mcp_is_available() -> bool:
    return bool(shutil.which("npx"))


def playwright_mcp_server_config() -> dict[str, Any]:
    return {
        "command": "npx",
        "args": [_PLAYWRIGHT_MCP_PACKAGE],
    }


def playwright_mcp_server_instance() -> MCPServerStdio:
    return MCPServerStdio(
        params=playwright_mcp_server_config(),
        cache_tools_list=True,
        name="playwright",
    )


def playwright_mcp_server_json() -> str:
    return json.dumps(playwright_mcp_server_config(), ensure_ascii=False)


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    box: dict[str, Any] = {"result": None, "error": None}

    def _target() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if box["error"] is not None:
        raise box["error"]
    return box["result"]


def playwright_mcp_list_tools() -> list[Any]:
    async def _list_tools() -> list[Any]:
        server = playwright_mcp_server_instance()
        async with server:
            return await server.list_tools()

    return list(_run_coro_sync(_list_tools()))


def playwright_mcp_call_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    async def _call_tool() -> Any:
        server = playwright_mcp_server_instance()
        async with server:
            return await server.call_tool(tool_name, arguments or {})

    return _run_coro_sync(_call_tool())


def playwright_mcp_result_to_text(result: Any) -> str:
    content = getattr(result, "content", None)
    blocks: list[str] = []
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if text:
                blocks.append(str(text))
                continue
            try:
                blocks.append(json.dumps(item.model_dump(), ensure_ascii=False))
            except Exception:
                blocks.append(str(item))
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        blocks.append(json.dumps(structured, ensure_ascii=False))
    if blocks:
        return "\n\n".join(blocks)
    try:
        return json.dumps(result.model_dump(), ensure_ascii=False)
    except Exception:
        return str(result)
