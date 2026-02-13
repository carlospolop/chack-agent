# Backends Overview

This folder contains 4 runtime backends:
- `openai_compaction_backend.py`
- `openrouter_openai_backend.py`
- `codex_backend.py`
- `langgraph_backend.py`

## Current shared config defaults

- `session.memory_max_messages`: `50`
- `session.memory_reset_to_messages`: `20`
- Specialized tool models (`social_network`, `scientific`, `websearcher`, `tester`) default to `CHEAP_BUT_QUALITY`.

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

- Enforced in backend tool-input guardrail `require_task_steps_manager_init_first`.
- Codex and LangGraph backends enforce this in their tool execution layers.

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
- Local memory is bounded by `memory_max_messages` / `memory_reset_to_messages`.

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
- Local memory is bounded by `memory_max_messages` / `memory_reset_to_messages`.

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
- Bounded by `memory_max_messages` / `memory_reset_to_messages`.
- Tool events are parsed from Codex JSON output lines and mapped into intermediate steps.

### Guardrails

- No backend tool-input guardrail hooks (since execution is external CLI-driven).
- MCP server hard-enforces `task_steps_manager init first` and `max_tools_used` limits.
- MCP does not expose command-execution tools (e.g. `exec`) to avoid duplication with Codex native command execution.

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
- Summary thresholds are sourced directly from `memory_max_messages` and `memory_reset_to_messages`.
- Keeps local `_conversation` for Chack API compatibility.

### Guardrails

- Enforces `task_steps_manager init first` and `max_tools_used` in the graph tool node.

## Notes for future changes

- If you add a new guardrail, attach it in both:
  - `openai_compaction_backend.py::_apply_guardrails`
  - `openrouter_openai_backend.py::_apply_guardrails`
- If you add new tool profiles or tool limits, check both backend builders and `agent.py` orchestration.
