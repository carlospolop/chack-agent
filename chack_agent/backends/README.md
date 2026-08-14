# Backends Overview

This folder contains 7 runtime backends:
- `openai_compaction_backend.py`
- `openrouter_openai_backend.py`
- `codex_backend.py`
- `langgraph_backend.py`
- `gemini_cli_backend.py`
- `claude_code_backend.py`
- `copilot_cli_backend.py`

## Prompt-cache boundary

Agent prompts that run repeatedly with a large immutable prefix may place one
visible `<!-- CHACK_PROMPT_CACHE_BREAKPOINT -->` marker between that prefix and
their changing suffix. Keep the real stable text and variables in the YAML;
do not hide them behind a generated prefix placeholder. Chack removes the
marker before inference and uses the same split for both CLI providers:

- Claude Code receives the stable side through its cached system layer and the
  changing side through stdin. No-tool agents replace the generic coding
  system prompt; tool-using agents append to it.
- Codex GPT-5.4 and older receive the stable side as developer instructions and
  the changing side through stdin for automatic prefix caching.
- No-tool Codex GPT-5.6+ agents use Chack's direct Responses transport. Public
  API-key requests use `prompt_cache_key`, an explicit breakpoint after the
  stable developer content, and explicit cache mode. ChatGPT/Codex subscription
  requests use the same deterministic `prompt_cache_key` and `session_id`
  across fresh agents and retain the official CLI's first-party transport
  classification because that endpoint rejects the public explicit-cache
  fields and routes unknown originators differently. Transient overload and
  rate-limit failures receive bounded jittered retries. Tool-using agents and
  terminal direct-transport failures safely retain the Codex CLI path.

Everything before the marker must be byte-identical for requests intended to
share a cache. Put check inventories, round notes, focus instructions,
timestamps, budgets, and retry data after it. A text marker is not itself a
provider cache directive; Chack converts it to the appropriate provider
request boundary. Set `CHACK_CODEX_DIRECT_CACHE_TRANSPORT=off` only to disable
the GPT-5.6+ direct path for diagnosis. Cache behavior must be verified from
reported `cached_prompt_tokens`/`cache_write_prompt_tokens`, not inferred from
the prompt layout.

Backends log a deterministic prefix key so cache-read/cache-write telemetry can
be grouped without logging the prompt. When an agent resolves to zero tools,
the Codex and Claude CLI backends also skip the Chack MCP server/tool registry;
this removes startup and schema-token overhead without changing agent
capabilities.

## Explicit pre-resume compaction

`Chack.run(..., compact_before_resume=True)` asks the live executor to compact
its existing conversation before the new top-level instruction is sent. This is
strictly opt-in: ordinary resumes, first turns, and internal retry attempts do
not trigger it. The choice is per `run`/`arun` call; there is no session-wide
configuration that silently compacts every continuation. Optional
`resume_compaction_instructions` focus the summary on state the next turn needs.

- OpenAI: `responses.compact(...)`
- Codex: app-server `thread/compact/start`
- Claude Code: `/compact [focus instructions]`
- Gemini CLI: `/compress`
- Copilot CLI: `/compact [focus instructions]`
- OpenRouter and LangGraph: summary plus server/checkpoint thread rotation

Compaction is fail-open: a backend compaction error is returned in `RunResult`
and the requested continuation still runs. When a backend reports usage, those
input, cached-input, output-token, and cost values are included in the run
totals.

## Current shared config defaults

- `agent.memory_max_messages`: `50`
- `agent.memory_reset_to_messages`: `20`
- Specialized tool models (`agent.social_network`, `agent.scientific`, `agent.websearcher`, `agent.cli`) default to `CHEAP_BUT_QUALITY`.

## Shared Architecture

- Entry point: `chack_agent/backends/__init__.py::build_executor(...)`
- Each backend returns an executor with:
  - `invoke(payload, context=None)`
  - `aget_memory_messages()`
- Main orchestration loop (including `min_tools_used` retries and self-critique passes) is implemented in `chack_agent/agent.py`, not in backend files.

## Guardrails and Tool Limits

### Where `min_tools_used` is enforced

- Enforced in `chack_agent/agent.py` via multi-attempt invocation logic.
- The agent checks tool count after each run and can re-run with extra guidance if below minimum.

### Where `max_tools_used` is enforced

- Enforced in two layers:
  - Backend tool-input guardrail `respect_max_tools_used` (hard reject once reached).
  - `agent.py` orchestration logic (checks when deciding retries/finalization).

### `task_steps_manager init first`

- For backends without native planning, Chack enforces this setting in its
  `task_steps_manager` tool-input guardrail.
- Codex and Claude Code do not receive that MCP tool. The same setting becomes
  prompt guidance to create the first plan with Codex `update_plan` or Claude
  Code `TodoWrite`/`Task*`; native plan updates are mirrored into Chack's shared
  plan store and its Telegram/Discord listener.

## `openai_compaction_backend.py`

### Library/SDK

- Uses `openai-agents` (`Agent`, `Runner`, `ModelSettings`, tool guardrails).
- Uses native OpenAI Responses API through default OpenAI client.

### Loop execution

- `invoke()` performs one backend run via `Runner.run_sync(...)`.
- Higher-level retries/min-tool behavior are driven by `agent.py`.

### Memory model

- Primary memory continuity: `previous_response_id` + `conversation_id` (server-side chain).
- Local `_conversation` transcript is kept for fallback/recovery and telemetry compatibility.
- Local memory is bounded by `agent.memory_max_messages` / `agent.memory_reset_to_messages`.

### Compaction

- Supports OpenAI Responses compaction (`client.responses.compact(...)`).
- Triggered when input tokens exceed `compaction_threshold_ratio * max_context_tokens`.
- On success, chain continues from compacted response id.

### Error recovery

- On sequence/tool-chain errors, backend drops `previous_response_id` and retries while preserving `conversation_id`.

---

## `openrouter_openai_backend.py`

### Library/SDK

- Uses `openai-agents`, but with custom model wrapper `_OpenRouterResponsesModel` over `OpenAIResponsesModel`.
- Uses `AsyncOpenAI` configured against OpenRouter base URL and headers.

### Loop execution

- `invoke()` calls `Runner.run_sync(...)` once per attempt.
- `agent.py` handles multi-attempt behavior (`min_tools_used`, self-critique).

### Memory model

- Uses `previous_response_id`/`conversation_id` when available.
- Maintains local `_conversation` transcript for fallback when a response chain is rejected.
- Input is sanitized to keep tool-call / tool-output consistency.
- Local memory is bounded by `agent.memory_max_messages` / `agent.memory_reset_to_messages`.

### OpenRouter-specific behavior

- Retries `RateLimitError` with backoff.
- Normalizes tool names from provider-specific variants (e.g., `tool_` prefixes/suffixes).
- Disables OpenAI tracing endpoints (`set_tracing_disabled(True)`) for compatibility.

### Error recovery

- On recoverable chain errors, clears server-side IDs and retries with full local history.

---

## `codex_backend.py`

### Library/SDK

- Does not use `openai-agents`.
- Uses local Codex CLI (`codex exec ...`) via subprocess.
- MCP tool server is launched from `chack_tools_mcp_server.py` via generated `CODEX_HOME/config.toml`.

### Loop execution

- One CLI invocation per `invoke()` call.
- Conversation continuity uses Codex thread resume (`codex exec resume ...`) via stored `_thread_id`.

### Memory model

- Codex manages active thread context internally (including its own context management/compaction behavior).
- Local `_conversation` stores user/assistant text history for Chack-level APIs/observability only.
- Bounded by `agent.memory_max_messages` / `agent.memory_reset_to_messages`.
- Tool events are parsed from Codex JSON output lines and mapped into intermediate steps.
- Codex `todo_list` events from native `update_plan` are mirrored on every
  `item.started`, `item.updated`, and `item.completed` event into the shared
  Chack plan board. Duplicate snapshots do not emit redundant chat edits.
- Top-level MCP calls are also counted in a tiny file-backed run-state counter
  at the MCP execution boundary. `agent.py` merges those counts with provider
  steps by taking the larger per-tool observation, so Codex transcript
  compaction, timeout, or a truncated event stream cannot erase earlier tool
  telemetry or cause required-tool enforcement to repeat completed work. The
  counter is deleted with the rest of the per-run state.

### Guardrails

- No backend tool-input guardrail hooks (since execution is external CLI-driven).
- The MCP server never exposes `task_steps_manager` to Codex. When init-first is
  configured, the prompt asks Codex to call native `update_plan` first and keep
  it current; this cannot be hard-enforced because it is a provider-native tool.
- MCP still hard-enforces `max_tools_used` for transported Chack tools.
- MCP does not expose command-execution tools (e.g. `exec`) to avoid duplication with Codex native command execution.

## `claude_code_backend.py`

### Library/SDK

- Does not use `openai-agents`.
- Uses local Claude Code CLI (`claude`) via subprocess (`-p` + `--output-format stream-json`).
- MCP tool server is launched from `chack_tools_mcp_server.py` via generated `~/.claude/chack/<session>/settings.json`.

### Loop execution

- One CLI invocation per `invoke()` call.
- Conversation continuity uses Claude Code resume (`--resume <session_id>`) via stored `_claude_session_id`.

### Memory model

- Claude maintains active session context internally (`--resume`).
- Local `_conversation` stores user/assistant text history for Chack-level APIs/observability only.
- Bounded by `agent.memory_max_messages` / `agent.memory_reset_to_messages`.
- Tool events are parsed from Claude JSON stream events and mapped into intermediate steps.
- Successful Claude native `TodoWrite`, `TaskCreate`, and `TaskUpdate` events are
  normalized into the same shared plan board and listener used by Chack's tool.
- The shared MCP-boundary counter supplies any calls missing from Claude's
  returned event stream without double-counting calls present in both sources.

### Guardrails

- Prompt-level policy for min/max tool usage is injected in the prompt.
- The MCP server never exposes `task_steps_manager` to Claude Code. When
  init-first is configured, Claude is asked to create and maintain the plan with
  its installed native `TodoWrite` or `TaskCreate`/`TaskUpdate` tools; this is
  advisory rather than hard-enforced.
- Native planning calls do not count toward Chack's non-task tool limits.

---

## `langgraph_backend.py`

### Library/SDK

- Uses **LangChain** (`langchain-openai`) and **LangGraph** (`StateGraph`).
- Model transport is OpenRouter via LangChain `ChatOpenAI` configured with OpenRouter base URL/headers.
- Requires `OPENROUTER_API_KEY` for this backend.
- Uses existing `AgentsToolset` so tool set and sub-agent behavior stay aligned with other backends.

### Loop execution

- Native autonomous ReAct-style loop in graph form: `llm_call -> tool_node -> llm_call ...` until no tool calls.
- Uses runtime `recursion_limit` to avoid infinite loops.

### Memory model

- Uses LangGraph checkpointer thread memory (`thread_id`) for short-term continuity.
- Adds summary-based context compaction in-model when message history grows.
- Summary thresholds are sourced directly from `agent.memory_max_messages` and `agent.memory_reset_to_messages`.
- Keeps local `_conversation` for Chack API compatibility.

### Guardrails

- Enforces `task_steps_manager init first` and `max_tools_used` in the graph tool node.

---

## `gemini_cli_backend.py`

### Library/SDK

- Does not use `openai-agents`.
- Uses local Gemini CLI (`gemini ...`) via subprocess.
- MCP tool server is configured from generated `~/.gemini/chack/<session>/settings.json`.

### Loop execution

- One CLI invocation per `invoke()` call.
- Conversation continuity uses Gemini session resume (`-r <session_id>`) via stored `_gemini_session_id`.

### Memory model

- Gemini manages active session context internally.
- Local `_conversation` stores user/assistant text history for Chack-level APIs/observability only.
- Bounded by `agent.memory_max_messages` / `agent.memory_reset_to_messages`.
- Tool events are parsed from Gemini `stream-json` events and mapped into intermediate steps.
- The shared MCP-boundary counter supplies any calls missing from Gemini's
  returned event stream without double-counting calls present in both sources.

### Guardrails

- Prompt-level policy for min/max tool usage and `task_steps_manager init first` is injected in the instructions.
- MCP server hard-enforces `task_steps_manager init first` and `max_tools_used` limits.
- `tools.core` is set to an empty allowlist to disable Gemini native built-ins, and the backend denylist removes duplicates from Chack tool names before exposing MCP tools.

## Notes for future changes

- If you add a new guardrail, attach it in both:
  - `openai_compaction_backend.py::_apply_guardrails`
  - `openrouter_openai_backend.py::_apply_guardrails`
- If you add new tool limits, check both backend builders and `agent.py` orchestration.
