# chack-agent

A configurable OpenAI Agents SDK runtime with rich tools and sub‑agent researchers (web, scientific, social). Designed for complex, multi‑turn investigations with usage/cost tracking.

## Installation

```bash
pip install chack-agent
```

> **Naxus consumers track `master`, not a pinned commit.** AISecurityAuditor,
> Dynamic-AIgent, and the backend all install
> `chack-agent @ git+https://github.com/carlospolop/chack-agent.git@master`, and
> AISecurityAuditor and Dynamic-AIgent additionally reinstall from `master` at
> the start of every run. A commit merged to `master` therefore reaches every
> Naxus agent on its next build or scan, with no pin to advance and no window in
> which to catch a regression downstream. Land breaking changes behind a config
> flag and keep `master` green.

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
        cli="CHEAP_BUT_QUALITY",
        provider="openai",  # use "openrouter", "codex", "gemini", "claude", or "langgraph"
    ),
    agent=AgentConfig(
        self_critique_rounds=1,  # Optional try-harder passes; default 0 disables them
        max_runtime_minutes=120,  # Optional runtime limit (minutes), 0 means unlimited
        max_cost_usd=12.50,  # Optional spend limit (USD), 0 means unlimited
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
        exec_cwd="/path/to/repo",
        brave_enabled=True,
        playwright_enabled=True,
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
    exec_cwd="/path/to/repo",
)

print(result.output)

# Opt in only when this continuation should compact its existing native
# conversation before receiving the new instruction.
continued = agent.run(
    session_id="investigation-001",
    text="Find additional checks",
    compact_before_resume=True,
    resume_compaction_instructions=(
        "Preserve the supplied context, existing checks, prior conclusions, "
        "and unexplored hypotheses."
    ),
)
```

You can also replace the default toolset entirely:

```python
result = agent.run(
  session_id="demo",
  text="Use only my custom tool.",
  tools_override=[my_tool],
)
```

`tools_append` and `tools_override` work with both in-process backends and CLI backends such as `codex`, `claude`, and `gemini`.

By default, a tool override creates a fresh executor for that call. For
continuation turns that must retain both conversation memory and the overridden
tools, opt into a session-persistent executor:

```python
first = agent.run(
    session_id="demo",
    text="Inspect the repository.",
    tools_override=[my_tool],
    reuse_session_executor=True,
)
correction = agent.run(
    session_id="demo",
    text="Return the same conclusion in the required JSON shape.",
    tools_override=[my_tool],
    reuse_session_executor=True,
)
```

The first executor configuration for that session is reused until
`reset_session()` is called.

You can also require specific tools to be called before a run is accepted as
complete. This is useful when a workflow must persist a verdict or update a
database row:

```python
result = agent.run(
  session_id="cli-123",
  text="Test this vulnerability and save the verdict.",
  required_tool_names=["update_vulnerability"],
  required_tool_call_attempts=3,
)
```

If the model tries to finish without the required tool, Chack re-prompts it to
call the missing tool instead of accepting the final answer. Tool names are
matched suffix-aware, so `update_vulnerability`,
`chack_tools-update_vulnerability`, and similar MCP-prefixed names satisfy the
same requirement. After the retry budget is exhausted, the run returns an
explicit `missing_required_tool_call` error output.

The same defaults can be configured in YAML:

```yaml
tools:
  required_tool_names:
    - update_vulnerability
  required_tool_call_attempts: 3
```

## Quick Start From YAML

You can initialize directly from a YAML path. The library will load and apply
all agent/tool/credential/logging values from that file.

For YAML files, keep all agent settings inside a single top-level `agent` section:

```yaml
agent:
  primary: gpt-5
  provider: openai
  # Defaults to high for the main agent and every nested agent on every backend.
  thinking_effort: high
  social_network: CHEAP_BUT_QUALITY
  scientific: CHEAP_BUT_QUALITY
  websearcher: CHEAP_BUT_QUALITY
  cli: CHEAP_BUT_QUALITY
  # Optional independent effort per nested agent type.
  scientific_thinking_effort: extra_high
  websearcher_thinking_effort: medium
  subchack_thinking_effort: high
  researcher_administrator_thinking_effort: high
  max_turns: 50
  memory_max_messages: 20
  memory_reset_to_messages: 10
  memory_summary_max_chars: 4000
  long_term_memory_enabled: true
  long_term_memory_dir: ./memory
```

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

`thinking_effort` accepts `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`/`extra_high`, and `max`. The default is `high` everywhere. Per-agent
keys follow `<role>_thinking_effort`; supported roles are `social_network`,
`scientific`, `websearcher`, `business`, `product`, `travel`, `legal`,
`data_statistics`, `news_media`, `knowledge_graph`, `religious`, `cli`,
`subchack`, `researcher_administrator`, and `researcher_queue`.

You can also keep a setting beside a role's other advanced options. This form
takes precedence over the flat per-role key:

```yaml
tools:
  scientific_agent:
    thinking_effort: low
  researcher_administrator_agent:
    thinking_effort: max
  researcher_queue_agent:
    thinking_effort: medium
```

The runtime translates the common vocabulary to each backend's native control:
OpenAI/OpenRouter model settings, Codex CLI config, Claude Code `--effort`,
Copilot CLI `--reasoning-effort`, Gemini CLI model generation config, and
LangGraph/OpenRouter request parameters. Where a backend has fewer levels,
the nearest native level is used. Claude Code capabilities are detected from
the installed CLI, and Gemini 2.5/3 receive their respective `thinkingBudget`
or `thinkingLevel` control (never both).

### Validation against the selected model

Providers do not share one effort vocabulary, and levels differ between models
of the same provider. Every configured value is therefore checked against the
model it will actually run on — `agent.thinking_effort` against `agent.primary`,
and each `<role>_thinking_effort` (or `tools.<role>_agent.thinking_effort`)
against that role's own model. A mismatch fails when the config is loaded:

```
agent.thinking_effort='xhigh' is not supported by model 'claude-sonnet-4-6'.
Supported values for this model: low, medium, high, max
```

The levels come from `chack_agent/config/thinking_effort.yaml`, which the
Update OpenRouter Pricing workflow regenerates every night from OpenRouter's
published `reasoning.supported_efforts` — the same run that refreshes
`pricing.yaml`. Nothing has to be hand-maintained as models ship, and the list
covers every vendor OpenRouter carries, not just the first-party ones:

```yaml
models:
  claude-opus-4-6: [low, medium, high, max]
  claude-opus-4-7: [low, medium, high, xhigh, max]
  gpt-5-4: [none, low, medium, high, xhigh]
  gpt-5-4-pro: [medium, high, xhigh]
  gemini-3-flash-preview: [minimal, low, medium, high]
```

Keys are normalized, so the OpenRouter (`openrouter/anthropic/claude-opus-4.6`),
API (`claude-opus-4-6`), Copilot (`claude-sonnet-4.6`) and Bedrock
(`us.anthropic.claude-opus-4-6-v1:0`) spellings of one model all resolve to the
same entry, as do dated releases like `claude-opus-4-5-20251101`.

A small set of built-in family rules covers what OpenRouter does not publish:

| Model | Supported levels |
| --- | --- |
| Gemini 2.5 Flash / Flash-Lite | every level (token-budget based, so no effort enum is published) |
| Gemini 2.5 Pro | every level except `none` (thinking cannot be disabled) |
| Gemini 3 Pro | `low`, `high` |
| Claude Opus 4.5, Mythos Preview | `low`, `medium`, `high` / `+ max` |
| o1 / o3 / o4 series | `low`, `medium`, `high` |
| No effort control (Claude Haiku, Claude 3.x, GPT-4.x) | `high` only |

`high` is the one level those last models still accept, because every provider
defines it as "behave as if the parameter was never sent". A model neither
source knows is not validated at all, so a brand new model works before either
list catches up.

MCP-launched researchers receive the same per-role settings through the
serialized tools configuration. A standalone/shared MCP server can set
`CHACK_THINKING_EFFORT` for every MCP-created agent, or a role-specific value
such as `CHACK_SCIENTIFIC_THINKING_EFFORT` or
`CHACK_RESEARCHER_ADMINISTRATOR_THINKING_EFFORT`. Role-specific environment
values take precedence; when nothing is configured, every MCP-created agent
uses `high`.

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
* **Travel**: Google Flights and Hotels, Booking.com stay prices/reviews, Amadeus hotel offers/sentiments, vacation rentals, OpenTripMap attractions, Open-Meteo weather/air quality, Transitous public transport, Frankfurter exchange rates, Wikivoyage guides, travel-scoped Wikidata entity/relationship lookups, local listings, and traveler opinions.
* **Business / Product / Legal / Data & Statistics / News & Media / Knowledge Graph / Religious / CLI**: additional domain researchers.

#### Travel research

Enable `travel_research` for autonomous trip comparisons and itinerary research:

```yaml
agent:
  travel: CHEAP_BUT_QUALITY
  travel_thinking_effort: high
  travel_max_turns: 50

tools:
  travel_enabled: true
  travel_max_results: 10
  travel_max_tools_used: 40
  playwright_enabled: true
```

With `SERPAPI_API_KEY`, the researcher receives structured Google Flights, Google Travel Explore, Google Hotels/vacation-rental search, property details, and hotel reviews. It also cross-checks Maps, Yelp, Apple Maps, Tripadvisor, web/news, forums, and Reddit when their corresponding credentials are available. Add `BOOKING_API_TOKEN` plus `BOOKING_AFFILIATE_ID` for official Booking.com prices, property data, and reviews; review access depends on the affiliate agreement. Add `AMADEUS_CLIENT_ID` plus `AMADEUS_CLIENT_SECRET` for Amadeus hotel offers and aggregate review sentiments, and `OPENTRIPMAP_API_KEY` for attraction discovery.

The researcher always receives additional keyless sources: Open-Meteo weather, marine, and air-quality/pollen forecasts; Nager.Date public holidays; Frankfurter central-bank reference exchange rates; Wikivoyage destination guides; Wikidata entity search and SPARQL; and Transitous public-transport routing. Wikivoyage and Wikidata are orientation evidence rather than authority for volatile or consequential historical details. Transitous is a best-effort community service intended for free/open-source and non-profit use; its source attribution and usage policy must be respected.

Booking.com credentials require Managed Affiliate Partner access. Use `BOOKING_DEMAND_API_BASE_URL=https://demandapi-sandbox.booking.com/3.1` while testing and switch to the production URL only when the integration is approved. Amadeus defaults to its free test environment; use `AMADEUS_BASE_URL=https://api.amadeus.com` for production credentials. OpenTripMap offers a free non-commercial tier.

The same endpoints can be exposed directly instead of through the researcher:

```yaml
tools:
  travel_google_flights_enabled: true
  travel_google_travel_explore_enabled: true
  travel_google_hotels_enabled: true
  travel_google_hotels_reviews_enabled: true
  travel_booking_enabled: true
  travel_amadeus_enabled: true
  travel_opentripmap_enabled: true
  travel_open_meteo_enabled: true
  travel_open_meteo_air_quality_enabled: true
  travel_frankfurter_enabled: true
  travel_wikivoyage_enabled: true
  travel_transitous_enabled: true
```

Additional direct tools are `find_booking_cities`, `search_booking_stays`, `get_booking_stay_details`, `get_booking_stay_reviews`, `search_amadeus_hotel_prices`, `get_amadeus_hotel_sentiments`, `search_destination_places`, `get_destination_place_details`, `get_destination_air_quality`, `convert_travel_currency`, `search_wikivoyage_guides`, and `plan_public_transport_trip`. Existing tools remain `search_google_flights`, `explore_google_travel_destinations`, `search_google_stays`, `get_google_stay_details`, `get_google_stay_reviews`, and `get_destination_weather`.

Airbnb does not provide a generally available public shopping API. Set `vacation_rentals=true` on `search_google_stays` for Airbnb-like inventory aggregated by Google, and use the researcher's web access to verify a specific Airbnb page when necessary. Neither path claims a result is from Airbnb unless the source identifies it. All flight, hotel, and rental prices are time-sensitive snapshots and never booking guarantees; these tools do not book or purchase anything.

#### Research Administrator
As the number of `*_research` researchers grows, you can expose a single **`researcher_administrator`** tool instead of every researcher. The administrator is itself a Chack sub‑agent whose only tools are the researchers you enable for it. Given one research request it:

* decomposes the problem and launches the relevant researchers — as many runs as needed, including several runs of the same type with more focused prompts;
* reviews every returned review and **cross‑pollinates** leads between researchers (e.g. relaunches the scientific researcher with papers the web researcher surfaced), since each researcher runs blind to the others;
* keeps launching follow‑ups until coverage stops producing new findings, then returns **its own conclusions, every researcher's conclusions, and the path to a master evidence folder**.

The administrator creates one master temp folder with **one subfolder per researcher type**; every researcher of a given type writes its downloads into the shared type subfolder, so same‑type researchers can see and build on what earlier runs already found.

Enable and scope it via config:

```yaml
tools:
  researcher_administrator_enabled: true
  # Which researchers the administrator may launch. Accepts short names or
  # aliases (e.g. "web" == "websearcher"). Empty = every researcher enabled above.
  researcher_administrator_researchers: ["scientific", "websearcher", "business", "travel"]
  researcher_administrator_max_tools_used: 60
  # Manage, from the yaml, the model used by the administrator itself AND by the
  # researchers it launches. This block works on every backend and takes
  # precedence over the model.* keys below.
  researcher_administrator_agent:
    model: CHEAP_BUT_QUALITY          # the administrator's own model
    max_turns: 120
    researcher_models:                # per-researcher models for the runs it launches
      scientific: SMART
      websearcher: CHEAP_BUT_QUALITY
      business: CHEAP_BUT_QUALITY
    researcher_max_turns:
      scientific: 40
agent:
  # Simpler alternative to researcher_administrator_agent.model (in-process /
  # Codex backends). Defaults to the agent's primary model when empty.
  researcher_administrator: ""
  researcher_administrator_max_turns: 100
```

Researchers listed in `researcher_administrator_researchers` are force‑enabled for the administrator even if they are not exposed at the top level, so you can hide the individual researcher tools and surface only `researcher_administrator`. When `researcher_administrator_agent.researcher_models` omits a researcher, that researcher inherits the top‑level `model.<researcher>` value.

**Evidence retention.** When the administrator is called with `save_artifacts=false`, the researchers it launches still **do not delete their own data mid‑run** — the per‑type subfolders persist so later researchers of the same type can inspect what earlier runs already downloaded. Only the administrator itself removes the master folder, once, when the whole run finishes. Call it with `save_artifacts=true` to keep the master folder afterwards.

### 2. Tool Ecosystem
`ToolsConfig` allows granular control over every tool. Note: tools are **disabled by default**.

* **System Tools**:
  * `exec`: Execute local shell commands (timeout/output limits from config).
  * `pdf_text`: Extract text from PDFs.
  * `task_steps_manager`: Maintain a dynamic task list.
* **Web Tools**:
  * `brave_search`: Brave Search API.
  * `playwright_fetch`: Open a real Chromium browser to read rendered page content and save text/HTML locally.
  * `serpapi`: Google/Bing web and AI‑mode endpoints.

### 3. Memory Architecture
* **Short‑Term Memory**: Compaction is driven by `max_context_tokens` and `compaction_threshold_ratio`.
  - `compaction_threshold_ratio` is only the trigger point. For example,
    `max_context_tokens: 250000` with `compaction_threshold_ratio: 0.75`
    starts compaction at 187,500 active input tokens. It does **not** retain
    75% of the old conversation.
  - After the trigger, native backends replace old history with their compact
    summary. Summary-based backends retain the generated summary (bounded by
    `memory_summary_max_chars`) plus only the newest
    `memory_reset_to_messages`, so the resulting context is much smaller than
    the trigger context.
  - `memory_summary_max_chars` controls how long the running memory summary can be.
  - `run(..., compact_before_resume=True)` explicitly invokes the selected
    backend's compactor before that individual continuation. It is off by
    default and is not a session-wide setting: an ordinary resume never
    implicitly requests it.
  - `resume_compaction_instructions="..."` optionally tells compaction which
    prior conclusions, state, or unresolved work must survive.
  - `RunResult` reports whether compaction was attempted/succeeded, its backend
    method and duration, and any error. Backend-reported compaction tokens and
    cost are included in the run totals.
* **Long‑Term Memory**: File-based persistence. The agent reads/writes summaries to a `long_term_memory_dir`.

## Configuration & Environment Variables

Most tools require API keys. Provide them via env vars (recommended) or your own config loader.

| Environment Variable | Description | Required For |
|----------------------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API Key | Core functionality (OpenAI/Codex) |
| `OPENROUTER_API_KEY` | OpenRouter API Key | Core functionality (OpenRouter/LangGraph) |
| `CODEX_PATH` | Path to Codex CLI binary | Optional override for provider=`codex` |
| `GEMINI_CLI_PATH` | Path to Gemini CLI binary | Optional override for provider=`gemini` |
| `CLAUDE_CLI_PATH` | Path to Claude CLI binary | Optional override for provider=`claude` |
| `OPENROUTER_HTTP_REFERER` | App referer for OpenRouter attribution | Optional |
| `OPENROUTER_APP_NAME` | App name for OpenRouter attribution | Optional |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | Optional override |
| `GEMINI_API_KEY` | Gemini API key | Optional, required for provider=`gemini` unless `GOOGLE_API_KEY` used |
| `GOOGLE_API_KEY` | Google API key for Gemini auth | Optional, required for provider=`gemini` unless `GEMINI_API_KEY` used |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional, required for provider=`claude` unless `CLAUDE_API_KEY` is set |
| `CLAUDE_API_KEY` | Anthropic API key alias | Optional, required for provider=`claude` unless `ANTHROPIC_API_KEY` is set |
| `BRAVE_API_KEY` | Brave Search API Key | `brave_search` |
| `SERPAPI_API_KEY` | SerpAPI Key | Google/Bing web + AI mode |
| `FORUMSCOUT_API_KEY` | ForumScout API Key | Social network tools |
| `FORUMSCOUT_BASE_URL` | ForumScout API base URL | Optional override |
| `CHACK_AWS_PROFILES` | Base64 of an AWS credentials file | AWS profile injection |

### Remote ChatGPT Pro / Extra High / Deep worker

Cloud applications do not need Chrome and must not try to reach a workstation directly. Configure them with the
authenticated asynchronous broker's HTTPS origin and its client-only secret:

```bash
export CHACK_CHATGPT_ASYNC_API_URL="https://broker.example.com"
export CHACK_CHATGPT_ASYNC_API_SECRET="<client bearer secret>"
```

When either broker variable is present, the Pro, Extra High, and Deep researcher tools use only the broker. Incomplete broker
configuration is a hard failure; it never falls back to a local browser.

Run the separate outbound worker on the PC that has the authenticated ChatGPT Chrome profile. Give this process the
distinct worker secret—not the cloud client secret—and the local CDP address:

```bash
export CHACK_CHATGPT_ASYNC_API_URL="https://broker.example.com"
export CHACK_CHATGPT_ASYNC_WORKER_SECRET="<worker bearer secret>"
export CHACK_CHATGPT_CDP_URL="http://127.0.0.1:9226"
chack-chatgpt-worker
```

The worker only makes outbound HTTPS requests: it leases queued jobs, heartbeats while local browser research runs,
and posts the terminal result. It opens no inbound port and does not print prompts, answers, or credentials.

`playwright_fetch` also requires the Python `playwright` package plus installed browser binaries. Once installed, set `tools.playwright_enabled: true` to expose it. If Playwright is missing or Chromium cannot launch, the tool is not registered.

```bash
pip install playwright
python -m playwright install chromium
```

For MCP-capable CLI backends (`codex`, `claude`, `gemini`), enabling `tools.playwright_enabled` now also injects Microsoft’s official Playwright MCP server (`npx @playwright/mcp@latest`) when `npx` is available on the host. This lets those backends control the browser through [`microsoft/playwright-mcp`](https://github.com/microsoft/playwright-mcp) in addition to the local `playwright_fetch` tool.

### Detailed Config Structure

* **`agent`** (`ModelConfig` + `SessionConfig` + `AgentConfig`):
  * `primary`, `provider`: Main agent model settings.
  * `social_network`, `scientific`, `websearcher`, `business`, `product`, `travel`, `cli`: Sub‑agent models (fallback to `primary`).
  * `max_turns`, `memory_max_messages`, `memory_reset_to_messages`, `memory_summary_max_chars`: Session behavior.
  * `max_runtime_minutes`: Maximum runtime in minutes for the run. Set to `0` to disable.
  * If the budget is reached, the run raises `TimeoutError` and stops.
  * `max_cost_usd`: Maximum estimated spend in USD for the run. Set to `0` to disable.
  * If the spend budget is reached, the run raises `TimeoutError` and stops.
* **`ToolsConfig`**:
  * All tools are disabled by default. Enable only what you need.
  * `exec_cwd` binds the built-in `exec` tool to a default working directory.
  * `exec_timeout_seconds` defaults to **60** and is configurable via YAML/config (not via env).
  * Subtool flags exist for scientific, social, and websearcher toolsets.

The built-in `exec` tool also accepts an optional `cwd` argument per call, and falls back in this order:
1. tool call `cwd`
2. `tools.exec_cwd`
3. `CHACK_EXEC_CWD`

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
