# chack-agent

A configurable OpenAI Agents SDK runtime with rich tools and sub‑agent researchers (web, scientific, social). Designed for complex, multi‑turn investigations with usage/cost tracking.

## Installation

```bash
pip install chack-agent
```

## Quick Start

```python
import os
from chack_agent import (
    Chack,
    ChackConfig,
    ModelConfig,
    AgentConfig,
    SessionConfig,
    ToolsConfig,
    CredentialsConfig,
    LoggingConfig,
)

# 1. Configure the agent
config = ChackConfig(
    model=ModelConfig(
        primary="gpt-4o",
    # Defaults for specialized tools are CHEAP_BUT_QUALITY.
    social_network="CHEAP_BUT_QUALITY",
    scientific="CHEAP_BUT_QUALITY",
    websearcher="CHEAP_BUT_QUALITY",
    tester="CHEAP_BUT_QUALITY",
    provider="openai",  # use "openrouter", "codex" or "langgraph"
    ),
    agent=AgentConfig(
        self_critique_enabled=True,  # Agent critiques its own plan before acting
    ),
    session=SessionConfig(
        max_turns=30,
        memory_max_messages=20,          # Short-term context window
      memory_reset_to_messages=10,     # Messages kept after compaction/reset
        long_term_memory_enabled=True,   # Enable file-based long-term memory
        long_term_memory_dir="./memory", # Where to store session summaries
    ),
    tools=ToolsConfig(
        exec_enabled=True,
        brave_enabled=True,
        websearcher_enabled=True,
        scientific_enabled=True,
        social_network_enabled=True,
    ),
    credentials=CredentialsConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        # openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
    ),
    logging=LoggingConfig(level="INFO"),
    system_prompt="You are an advanced researcher agent.",
    user_prompt="Investigate {topic} with recent sources and summarize findings.",
    env={},
)

# 2. Initialize and run
agent = Chack(config)
result = agent.run(
    session_id="investigation-001",
    text="",  # empty => uses user_prompt from config (with template replacement)
    prompt_variables_override={"topic": "plastic-eating bacteria"},
)

print(result.output)
```

## Quick Start From YAML

You can initialize directly from a YAML path. The library will load and apply
all model/agent/session/tools/credentials/logging values from that file.

```python
from chack_agent import Chack

agent = Chack("./config/chack.yaml")
# equivalent explicit form:
# agent = Chack.from_config_path("./config/chack.yaml")

result = agent.run(
    session_id="investigation-001",
    text="Investigate this issue end-to-end."
)
print(result.output)
```

Optional: include `user_prompt` in YAML and let `agent.run(text="")` render it automatically.
Template values can come from:
- context object fields (e.g., `{repo_path}`, `{target_service}`)
- `prompt_variables_override` in `run(...)`
- optional `user_prompt_variables` in YAML (`context.<field>` / `env.<VAR>` sources)

## GitHub Action

You can run the agent in GitHub Actions by using the repo as an action.

```yaml
- name: Run chack-agent
  id: chack
  uses: carlospolop/chack-agent@v0
  with:
    provider: openai
    model_primary: gpt-5.2-codex
    system_prompt: You are an advanced research agent.
    prompt_file: codex_prompt.txt
    output_schema_file: .github/chack-agent/pr-merge-schema.json
    tools_config_json: "{\"exec_enabled\": true}"
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Use output
  run: echo "${{ steps.chack.outputs.final-message }}"
```

`output_schema_file` or `output_schema_json` will be forwarded to the OpenAI Agents SDK as an output schema so the model is constrained to return JSON that matches it.

## Key Features

### 1. Specialized Sub‑Agents
The agent can delegate to specialized sub‑agents. Sub‑agents run with restricted toolsets to reduce noise.

* **Web Research**: Brave + SerpAPI (Google/Bing + AI‑mode endpoints if enabled).
* **Scientific**: arXiv, Europe PMC, Semantic Scholar, OpenAlex, PLOS, Google Scholar/Patents, YouTube transcripts, PDF text.
* **Social Network**: ForumScout + SerpAPI forums/news.

### 2. Tool Ecosystem
`ToolsConfig` allows granular control over every tool. Note: tools are **disabled by default**.

* **System Tools**:
  * `exec`: Execute local shell commands (timeout/output limits from config).
  * `pdf_text`: Extract text from PDFs.
  * `task_steps_manager`: Maintain a dynamic task list.
* **Web Tools**:
  * `brave_search`: Brave Search API.
  * `serpapi`: Google/Bing web and AI‑mode endpoints.

### 3. Memory Architecture
* **Short‑Term Memory**: Compaction is driven by `max_context_tokens` and `compaction_threshold_ratio`.
* **Long‑Term Memory**: File-based persistence. The agent reads/writes summaries to a `long_term_memory_dir`.

## Configuration & Environment Variables

Most tools require API keys. Provide them via env vars (recommended) or your own config loader.

| Environment Variable | Description | Required For |
|----------------------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API Key | Core functionality (OpenAI/Codex) |
| `OPENROUTER_API_KEY` | OpenRouter API Key | Core functionality (OpenRouter/LangGraph) |
| `CODEX_PATH` | Path to Codex CLI binary | Optional override for provider=`codex` |
| `OPENROUTER_HTTP_REFERER` | App referer for OpenRouter attribution | Optional |
| `OPENROUTER_APP_NAME` | App name for OpenRouter attribution | Optional |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | Optional override |
| `BRAVE_API_KEY` | Brave Search API Key | `brave_search` |
| `SERPAPI_API_KEY` | SerpAPI Key | Google/Bing web + AI mode |
| `FORUMSCOUT_API_KEY` | ForumScout API Key | Social network tools |
| `FORUMSCOUT_BASE_URL` | ForumScout API base URL | Optional override |
| `CHACK_AWS_PROFILES` | Base64 of an AWS credentials file | AWS profile injection |

### Detailed Config Structure

* **`ModelConfig`**:
  * `primary`: Main model for the coordinator agent.
  * `social_network`, `scientific`, `websearcher`: Sub‑agent models (fallback to `primary`).
* **`ToolsConfig`**:
  * All tools are disabled by default. Enable only what you need.
  * `exec_timeout_seconds` defaults to **60** and is configurable via YAML/config (not via env).
  * Subtool flags exist for scientific, social, and websearcher toolsets.

## Development

**Project Structure**:
*   `chack_agent/`: Core runtime, memory management, and agent logic.
*   `chack_tools/`: Tool implementations and sub-agent definitions.

**Running Tests**:
```bash
# Run verifying import of the toolset
python3 -c "from chack_tools.agents_toolset import AgentsToolset; print('Import OK')"
```

## Extra tools
You can append tools at runtime without overriding the default set:

```python
result = agent.run(
    session_id="demo",
    text="Use my custom tool too.",
    tools_append=[my_tool],
)
```
