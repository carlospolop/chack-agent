# Chack backends

`build_executor()` in [`__init__.py`](__init__.py) resolves `agent.provider` and constructs a runtime route. The common loop in [`../agent.py`](../agent.py) owns run limits, required-tool retries, self-critique, usage aggregation, and final `RunResult` construction.

## Runtime matrix

| Provider | Implementation | Transport and continuity | Tools |
| --- | --- | --- | --- |
| `openai` | `openai_compaction_backend.py` | OpenAI Responses through `openai-agents`; server response/conversation IDs | In-process and MCP tools with SDK guardrails |
| `openrouter` | `gateway_responses_backend.py` | OpenRouter Responses through an `openai-agents` model adapter; response IDs with local fallback | In-process and MCP tools with SDK guardrails |
| `gateway` | `gateway_responses_backend.py` | Responses API of a custom LLM gateway; the full local transcript is resent each run because gateways rebuild chained responses lossily | In-process and MCP tools with SDK guardrails |
| `codex` | `codex_backend.py` | Codex CLI thread resume; eligible no-tool GPT-5.6+ runs may use direct Responses transport and fall back to the CLI | Codex native tools plus non-duplicate Chack tools over MCP |
| `claude` | `claude_code_backend.py` | Claude Code CLI session resume | Claude native tools plus non-duplicate Chack tools over MCP |
| `gemini` | `gemini_cli_backend.py` | Gemini CLI session resume | Chack tools over MCP; the generated settings turn Gemini's native built-ins off |
| `kimi` | `code_cli_backend.py` | Kimi Code CLI print-mode JSONL and session resume | Chack tools over MCP; native host tools off |
| `qwen` | `code_cli_backend.py` | Qwen Code CLI print-mode JSONL and session resume | Chack tools over MCP; native host tools off |
| `copilot` | `copilot_cli_backend.py` | GitHub Copilot CLI session resume | Copilot native tools plus Chack tools over MCP |
| `opencode` | `opencode_cli_backend.py` | opencode CLI session resume; a per-session config defines one provider (OpenAI Responses, Anthropic Messages, OpenRouter, or a gateway) and one agent carrying the system prompt and `steps` | opencode file, search and edit tools (shell only with `exec_enabled`) plus Chack tools over MCP |
| `langgraph` | `langgraph_backend.py` | LangGraph checkpoint thread using OpenRouter or gateway `ChatOpenAI` | Chack tools in a ReAct graph |

LangGraph summarizes older turns when its configured message or token threshold is reached, then removes those turns from checkpoint state. It retains the current request and complete assistant tool-call/result pairs so the next model request has a valid history. Its prompt states the remaining tool budget and asks for source verification when a web tool is available and the task calls for current external evidence.

Provider aliases and credentials are resolved before this router runs. A configured OpenRouter route can redirect an otherwise CLI-oriented model through an API transport.

[`../llm_gateway.py`](../llm_gateway.py) resolves OpenRouter and custom gateway routes. On a custom gateway route, Codex gets a `responses` model provider in its generated `config.toml` (key and headers passed by environment, native web search and Daybreak off) and Claude Code gets `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` with every model family mapped to gateway names and `WebSearch` disallowed. The Gemini CLI runs on the gateway's Gemini API when it serves one (LiteLLM at its root for every model, other gateways at `LLM_GATEWAY_GEMINI_BASE_URL`), through a per-run local relay ([`gemini_gateway_relay.py`](gemini_gateway_relay.py)) that sends streaming calls as `:generateContent` so the gateway reports their token usage; otherwise Gemini, like Copilot and OpenAI routes, moves to the `gateway` Responses runtime with a warning. The key's model list is read once per hour, and a failed refresh keeps the last list read; unavailable explicit models fail early, while built-in role aliases the key cannot use fall back to the primary model.

## Shared contracts

Every executor exposes `invoke(payload, context=None)` and conversation-memory access. Backends also implement `compact_for_resume()` where their provider can compact or summarize active state.

On Linux, every agent CLI (Codex, Claude Code, Gemini, opencode, Copilot, Kimi, Qwen) starts with a raised `oom_score_adj`, which the processes it starts inherit. When the machine runs out of memory, the kernel then kills one agent run instead of the process hosting all of them.

Chack keeps immutable runtime instructions and the output contract in the provider's system or developer prefix where the transport permits it. Repeated prompts can place one `<!-- CHACK_PROMPT_CACHE_BREAKPOINT -->` marker between stable context and changing task data. Chack removes the marker and translates the split into the provider-specific cache boundary. Text before the marker must remain byte-identical for calls intended to share a cache; reported cache token counters are the evidence that reuse occurred.

The Codex direct transport is limited to eligible no-tool GPT-5.6+ runs. Public OpenAI-key calls use explicit cache fields; subscription-token calls retain the first-party transport shape accepted by that endpoint. Failures retry within bounds and then fall back to the Codex CLI. Set `CHACK_CODEX_DIRECT_CACHE_TRANSPORT=off` only when diagnosing that route.

Codex launches opt into Daybreak, OpenAI's trusted-access program for cyber work, only when the credential's model catalog offers a Daybreak program for the agent's model. [`codex_daybreak.py`](codex_daybreak.py) asks `codex app-server` (`model/list`, no inference) once per credential identity, model, and Codex version, caches the answer under a SHA-256 fingerprint, and then adds `--enable cli_daybreak -c daybreak=true` (or `access_programs` on the direct transport). A run refused because of Daybreak is retried without it and the credential is remembered as unavailable. `CHACK_CODEX_DAYBREAK=off` disables it, `on` skips the probe; Codex 0.162 or newer is required.

`run(..., compact_before_resume=True)` compacts the live conversation before the new top-level instruction. The backend methods are OpenAI Responses compaction, Codex `thread/compact/start`, Claude `/compact`, Gemini `/compress`, Kimi `/compact`, Qwen `/compress`, Copilot `/compact`, and summary plus thread rotation for OpenRouter and LangGraph. Failure is recorded but does not cancel the continuation.

## Memory and schemas

OpenAI and OpenRouter prefer provider-side response chains and retain bounded local history for recovery. Codex, Claude, Gemini, and Copilot use their CLI session or thread state and retain local text for Chack APIs and telemetry. LangGraph uses checkpointed thread memory. Automatic compaction is driven by `max_context_tokens` and `compaction_threshold_ratio`; summary-backed routes use the configured message and summary bounds. At the threshold, OpenRouter rotates its provider response chain after summarizing, Gemini sends `/compress` after a turn whose reported input usage reaches the limit, Copilot sends `/compact` after its `session.usage_info` context measurement reaches the limit, and LangGraph summarizes and rotates its checkpoint thread. Copilot also retains its own native automatic compaction if usage events are unavailable.

OpenAI/OpenRouter use their structured-output path. CLI backends pass supported schema controls to the CLI and validate returned JSON where needed. The output schema is part of cache identity when it changes the stable provider prefix. Where the schema is only stated in the prompt (LangGraph, opencode, Kimi Code, Qwen Code, Gemini CLI), an answer that does not validate gets exactly one repair turn in the same session: a user message with the short validation error (message and JSON path, at most 400 characters) and the schema, asking for only the complete JSON object. A turn that ends without an answer returns an empty reply: neither the CLI's stream events nor what it printed outside them (warnings and logs, since stderr is merged into stdout) are taken as the answer. An empty reply, or a JSON object with none of the answer's fields (such as a tool call written as text), instead gets a turn telling the model that text does not run tools and to finish the task with its tools before answering. A top-level list the first answer filled and the repair emptied or left out is put back when the merged answer validates, because some models resend only the field they fixed. The agent prompt is unchanged, so the provider's prompt cache still applies, and the repair turn's tool calls and usage are counted with the run. If the repair still does not validate, the latest answer is returned as given; backend errors and runs without a resumable session are not repaired. See `validate_schema_output` and `repair_schema_output` in [`output_schema.py`](../output_schema.py).

After the backend returns, `Chack.run` checks the final answer once more against the output schema and the caller's optional `output_validator` (a callable that returns why an answer is unusable, or raises, for rules a schema cannot state). A rejected answer is sent back to the same session as a short user message with the reason, on every backend: one turn for GPT and Claude models, two for other models, which often fix one error and introduce another. Accepted, empty and backend-error answers add no turn, so prompts are unchanged whenever an answer is valid. See `final_answer_rejection` and `answer_repair_turns`.

## Planning and tool limits

The main loop checks `min_tools_used`, required tool names, maximum calls, and self-critique completion. OpenAI/OpenRouter and LangGraph can reject invalid tool calls in-process. MCP-backed CLI routes enforce transported-tool limits at the MCP boundary and merge those counters with provider events so truncated streams do not erase completed calls.

Codex and Claude do not receive Chack's `task_steps_manager`; the runtime maps planning guidance to Codex `update_plan` and Claude `TodoWrite`/`Task*`, then mirrors provider plan events into Chack's shared plan state. Gemini and other transported routes can use the Chack task manager directly. Native planning calls do not count toward non-planning tool requirements.

When changing a shared rule, verify the common loop, in-process SDK guardrails, LangGraph tool node, CLI prompt policy, MCP enforcement, and telemetry merge. Backend-specific tests live under [`../../tests`](../../tests).
