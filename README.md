# chack-tools

Reusable tools package extracted from Chack so multiple bots can share the same tooling.

## Install

```bash
pip install chack-tools
```

or from GitHub:

```bash
pip install "git+https://github.com/carlospolop/chack-tools.git@main"
```

If you want the social/scientific sub-agents (OpenAI Agents SDK):

```bash
pip install "chack-tools[openai_agents]"
```

## API keys needed

API keys depend on which tools you enable:

- `BRAVE_API_KEY` (optional): needed only for `brave_search`.
- `FORUMSCOUT_API_KEY` (optional): needed for ForumScout social/forum endpoints.
- `SERPAPI_API_KEY` (optional): needed for Google/Bing web search, Google Scholar/Patents/YouTube search, and ForumScout's SerpAPI fallback tools.
- `OPENAI_API_KEY` (optional but required if you use `social_network_research` or `scientific_research` or `websearcher_agent` sub-agents via `openai-agents`).

## How to pass API keys

You can pass keys directly in `ToolsConfig` (recommended for explicit setup):

```python
from chack_tools import Toolset, ToolsConfig

config = ToolsConfig(
    brave_api_key="...",
    forumscout_api_key="...",
    serpapi_api_key="...",
)

tools = Toolset(config, tool_profile="telegram").tools
```

You can also use environment variables:

```bash
export BRAVE_API_KEY="..."
export FORUMSCOUT_API_KEY="..."
export OPENAI_API_KEY="..."
```

Notes:
- `BRAVE_API_KEY` and `FORUMSCOUT_API_KEY` are read from env vars automatically.
- `SERPAPI_API_KEY` is currently read from `ToolsConfig.serpapi_api_key` (set it in code).
- Optional model overrides for sub-agents:
  - `CHACK_SOCIAL_AGENT_MODEL`
  - `CHACK_SCIENTIFIC_AGENT_MODEL`

## Minimal usage

```python
from chack_tools import Toolset, ToolsConfig

config = ToolsConfig(
    brave_enabled=True,
    forumscout_enabled=True,
    serpapi_api_key="...",  # required for SerpAPI-backed tools
)

tools = Toolset(config, tool_profile="all").tools
```
