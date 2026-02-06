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
        social_network="gpt-4o",
        scientific="gpt-4o",
        websearcher="gpt-4o",
    ),
    agent=AgentConfig(
        self_critique_enabled=True,  # Agent critiques its own plan before acting
    ),
    session=SessionConfig(
        max_turns=30,
        memory_max_messages=20,          # Short-term context window
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
    ),
    logging=LoggingConfig(level="INFO"),
    system_prompt="You are an advanced researcher agent.",
    env={},
)

# 2. Initialize and run
agent = Chack(config)
result = agent.run(
    session_id="investigation-001",
    text="Find recent research on plastic-eating bacteria and what people are saying about it on Reddit."
)

print(result.output)
```

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
  * `task_list`: Maintain a dynamic task list.
* **Web Tools**:
  * `brave_search`: Brave Search API.
  * `serpapi`: Google/Bing web and AI‑mode endpoints.

### 3. Memory Architecture
*   **Short-Term Memory**: Managed via `memory_max_messages` in `SessionConfig`. Keeps the immediate context window efficient.
*   **Long-Term Memory**: File-based persistence. The agent reads/writes summaries to a `long_term_memory_dir`. This allows it to recall key facts across different runs of the same `session_id`.

## Configuration & Environment Variables

Most tools require API keys. Provide them via env vars (recommended) or your own config loader.

| Environment Variable | Description | Required For |
|----------------------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API Key | Core functionality |
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
