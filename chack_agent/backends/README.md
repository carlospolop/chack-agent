# Chack backends

`build_executor()` in [`__init__.py`](__init__.py) resolves `agent.provider` and constructs one of seven runtime routes. The common loop in [`../agent.py`](../agent.py) owns run limits, required-tool retries, self-critique, usage aggregation, and final `RunResult` construction.

## Runtime matrix

| Provider | Implementation | Transport and continuity | Tools |
| --- | --- | --- | --- |
| `openai` | `openai_compaction_backend.py` | OpenAI Responses through `openai-agents`; server response/conversation IDs | In-process and MCP tools with SDK guardrails |
| `openrouter` | `openrouter_openai_backend.py` | OpenRouter Responses through an `openai-agents` model adapter; response IDs with local fallback | In-process and MCP tools with SDK guardrails |
| `codex` | `codex_backend.py` | Codex CLI thread resume; eligible no-tool GPT-5.6+ runs may use direct Responses transport and fall back to the CLI | Codex native tools plus non-duplicate Chack tools over MCP |
| `claude` | `claude_code_backend.py` | Claude Code CLI session resume | Claude native tools plus non-duplicate Chack tools over MCP |
| `gemini` | `gemini_cli_backend.py` | Gemini CLI session resume | Chack tools over MCP; the generated settings turn Gemini's native built-ins off |
| `copilot` | `copilot_cli_backend.py` | GitHub Copilot CLI session resume | Copilot native tools plus Chack tools over MCP |
| `langgraph` | `langgraph_backend.py` | LangGraph checkpoint thread using OpenRouter `ChatOpenAI` | Chack tools in a ReAct graph |

Provider aliases and credentials are resolved before this router runs. A configured OpenRouter route can redirect an otherwise CLI-oriented model through an API transport.

## Shared contracts

Every executor exposes `invoke(payload, context=None)` and conversation-memory access. Backends also implement `compact_for_resume()` where their provider can compact or summarize active state.

Chack keeps immutable runtime instructions and the output contract in the provider's system or developer prefix where the transport permits it. Repeated prompts can place one `<!-- CHACK_PROMPT_CACHE_BREAKPOINT -->` marker between stable context and changing task data. Chack removes the marker and translates the split into the provider-specific cache boundary. Text before the marker must remain byte-identical for calls intended to share a cache; reported cache token counters are the evidence that reuse occurred.

The Codex direct transport is limited to eligible no-tool GPT-5.6+ runs. Public OpenAI-key calls use explicit cache fields; subscription-token calls retain the first-party transport shape accepted by that endpoint. Failures retry within bounds and then fall back to the Codex CLI. Set `CHACK_CODEX_DIRECT_CACHE_TRANSPORT=off` only when diagnosing that route.

`run(..., compact_before_resume=True)` compacts the live conversation before the new top-level instruction. The backend methods are OpenAI Responses compaction, Codex `thread/compact/start`, Claude `/compact`, Gemini `/compress`, Copilot `/compact`, and summary plus thread rotation for OpenRouter and LangGraph. Failure is recorded but does not cancel the continuation.

## Memory and schemas

OpenAI and OpenRouter prefer provider-side response chains and retain bounded local history for recovery. Codex, Claude, Gemini, and Copilot use their CLI session or thread state and retain local text for Chack APIs and telemetry. LangGraph uses checkpointed thread memory. Automatic compaction is driven by `max_context_tokens` and `compaction_threshold_ratio`; summary-backed routes use the configured message and summary bounds.

OpenAI/OpenRouter use their structured-output path. CLI backends pass supported schema controls to the CLI and validate returned JSON where needed. The output schema is part of cache identity when it changes the stable provider prefix.

## Planning and tool limits

The main loop checks `min_tools_used`, required tool names, maximum calls, and self-critique completion. OpenAI/OpenRouter and LangGraph can reject invalid tool calls in-process. MCP-backed CLI routes enforce transported-tool limits at the MCP boundary and merge those counters with provider events so truncated streams do not erase completed calls.

Codex and Claude do not receive Chack's `task_steps_manager`; the runtime maps planning guidance to Codex `update_plan` and Claude `TodoWrite`/`Task*`, then mirrors provider plan events into Chack's shared plan state. Gemini and other transported routes can use the Chack task manager directly. Native planning calls do not count toward non-planning tool requirements.

When changing a shared rule, verify the common loop, in-process SDK guardrails, LangGraph tool node, CLI prompt policy, MCP enforcement, and telemetry merge. Backend-specific tests live under [`../../tests`](../../tests).
