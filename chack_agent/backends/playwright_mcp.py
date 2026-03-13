from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from agents.mcp import MCPServerStdio


_PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@latest"


@lru_cache(maxsize=1)
def playwright_mcp_is_available() -> bool:
    return bool(shutil.which("npx")) and bool(playwright_mcp_browser_executable_path())


@lru_cache(maxsize=1)
def playwright_mcp_browser_executable_path() -> str | None:
    explicit = str(os.environ.get("CHACK_PLAYWRIGHT_MCP_EXECUTABLE_PATH", "") or "").strip()
    if explicit and Path(explicit).is_file():
        return explicit

    commands = [
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for command in commands:
        if command:
            return command

    home = Path.home()
    candidates = [
        home / ".cache/ms-playwright",
        home / "Library/Caches/ms-playwright",
    ]
    patterns = [
        "chromium-*/chrome-linux/chrome",
        "chromium-*/chrome-linux/headless_shell",
        "chromium_headless_shell-*/chrome-linux/headless_shell",
        "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
        "chromium_headless_shell-*/chrome-mac/headless_shell",
    ]
    for root in candidates:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return str(matches[0])
    return None


@lru_cache(maxsize=1)
def playwright_mcp_needs_no_sandbox() -> bool:
    explicit = str(os.environ.get("CHACK_PLAYWRIGHT_MCP_NO_SANDBOX", "") or "").strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    return Path("/.dockerenv").exists()


def playwright_mcp_server_config() -> dict[str, Any]:
    args: list[str] = ["-y", _PLAYWRIGHT_MCP_PACKAGE]
    browser_executable = playwright_mcp_browser_executable_path()
    if browser_executable:
        args.extend(["--executable-path", browser_executable])
    if playwright_mcp_needs_no_sandbox():
        args.append("--no-sandbox")
    return {
        "command": "npx",
        "args": args,
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
