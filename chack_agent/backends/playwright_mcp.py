from __future__ import annotations

import json
import shutil
from functools import lru_cache
from typing import Any


_PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@latest"


@lru_cache(maxsize=1)
def playwright_mcp_is_available() -> bool:
    return bool(shutil.which("npx"))


def playwright_mcp_server_config() -> dict[str, Any]:
    return {
        "command": "npx",
        "args": [_PLAYWRIGHT_MCP_PACKAGE],
    }


def playwright_mcp_server_json() -> str:
    return json.dumps(playwright_mcp_server_config(), ensure_ascii=False)
