# chack-agent

`chack-agent` is the shared agent runtime used by Naxus. It provides one Python API over OpenAI, OpenRouter, Codex CLI, Claude Code, Gemini CLI, Kimi Code CLI, Qwen Code CLI, GitHub Copilot CLI, and LangGraph, with configurable tools, sessions, budgets, structured output, and usage accounting.

AISecurityAuditor and Dynamic-AIgent install the `master` branch directly. Keep `master` compatible and green because downstream runs receive merged changes without advancing a version pin.

## Install

```bash
pip install chack-agent
```

Python 3.10 or newer is required.

CLI backends also require their provider command on `PATH`: `codex`, `claude`, `gemini`, `kimi`, `qwen`, `copilot`, or `opencode`. Pin the Gemini CLI and opencode versions Chack is tested with: `npm i -g @google/gemini-cli@0.67.0` (set `GEMINI_CLI_PATH` when it is not on `PATH`) and `npm i -g opencode-ai@1.19.5` (set `OPENCODE_CLI_PATH` when it is not on `PATH`). Kimi Code 2.5.3 and Qwen Code 0.27.2 were exercised with `npm i -g @moonshot-ai/kimi-code@2.5.3 @qwen-code/qwen-code@0.27.2`; both require Node.js 22.19 or newer. The Python package includes Playwright, but browser-backed tools require an installed Chromium binary:

```bash
python -m playwright install chromium
```

## Configure an agent

The recommended configuration is YAML. `system_prompt`, `agent.primary`, `agent.provider`, `agent.main_action`, and `agent.sub_action` are required. Model, session, and run settings share one flat `agent` section; legacy top-level `model` and `session` sections are rejected.

```yaml
system_prompt: |
  You are a repository reviewer. Inspect the supplied code and return an evidence-based result.

user_prompt: |
  Review {repository_path} for the requested issue.

agent:
  primary: gpt-5.4
  provider: codex
  thinking_effort: high
  max_turns: 50
  max_context_tokens: 250000
  main_action: repository_review
  sub_action: inspect
  max_runtime_minutes: 60
  max_cost_usd: 5
  output_schema_file: output.schema.json

tools:
  task_steps_manager_enabled: false
  exec_enabled: true
  exec_cwd: /path/to/repository
  min_tools_used: 0

credentials:
  codex_access_token: ${CODEX_ACCESS_TOKEN}

logging:
  level: INFO
```

Model aliases such as `OPENAI_BEST_QUALITY` resolve through the central model aliases Lambda, which also publishes each tier's thinking effort. Set `thinking_effort: preset` (also accepted for `<role>_thinking_effort` and per-tool `thinking_effort`) to use the effort the Lambda assigns to that agent's model alias; `resolve_thinking_effort_alias()` returns it for other callers. `preset` with a concrete model name is rejected.

Environment placeholders use `${NAME}`. A prompt may contain `$$TOOLS$$`; Chack replaces it with `tools_prompt_file`, which defaults to `TOOLS.md`, `TOOLS_TELEGRAM.md`, or `TOOLS_DISCORD.md` according to the integration configured by the caller.

Run the configured agent with a stable logical session ID:

```python
from chack_agent import Chack

agent = Chack("chack.yaml")
result = agent.run(
    session_id="review-001",
    text="",
    prompt_variables_override={"repository_path": "/path/to/repository"},
)

print(result.output)
print(result.total_cost, result.tool_counts)
```

When `text` is empty, Chack renders `user_prompt`. Template values can come from fields on `context`, `prompt_variables_override`, or `user_prompt_variables` mappings that reference `context.<field>` and `env.<NAME>`.

## Providers

Set `agent.provider` to one of these values:

| Provider | Runtime | Main credential |
| --- | --- | --- |
| `openai` | OpenAI Responses through `openai-agents` | `OPENAI_API_KEY` |
| `openrouter` | OpenRouter Responses through `openai-agents` | `OPENROUTER_API_KEY` |
| `codex` | Codex CLI, with a direct Responses path for eligible no-tool GPT-5.6+ runs | `CODEX_ACCESS_TOKEN` or `OPENAI_API_KEY` |
| `claude` | Claude Code CLI | `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` |
| `gemini` | Gemini CLI | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `LLM_GATEWAY_API_KEY` for a gateway that serves the Gemini API |
| `copilot` | GitHub Copilot CLI | `COPILOT_GITHUB_TOKEN` or existing CLI authentication |
| `opencode` | opencode CLI (`opencode run`) with the model from any key type | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, or `LLM_GATEWAY_API_KEY` |
| `kimi` | Current Kimi Code CLI (`kimi -p`), not the older Python `kimi-cli` | `KIMI_API_KEY` or `LLM_GATEWAY_API_KEY` |
| `qwen` | Qwen Code CLI (`qwen -p`) | `QWEN_API_KEY` plus `QWEN_BASE_URL`, or `LLM_GATEWAY_API_KEY` |
| `langgraph` | LangGraph with an OpenRouter or gateway `ChatOpenAI` transport | `OPENROUTER_API_KEY` or `LLM_GATEWAY_API_KEY` |
| `gateway` | Responses API of a custom LLM gateway through `openai-agents` | `LLM_GATEWAY_API_KEY` |

Aliases are normalized when configuration loads. Unsupported providers and unsupported thinking-effort values for known models fail early. The accepted common effort vocabulary is `none`, `minimal`, `low`, `medium`, `high`, `xhigh`/`extra_high`, and `max`; each backend maps supported values to its native control.

`api_key_type` picks the key a backend uses. With `api_key_type: openrouter`, `codex` and `claude` still run their CLIs, against OpenRouter with `OPENROUTER_API_KEY`: Codex through a Responses model provider, Claude Code through `ANTHROPIC_BASE_URL=https://openrouter.ai/api`. Model names are mapped to OpenRouter IDs (`claude-sonnet-5-5` becomes `anthropic/claude-sonnet-5.5`) and priced from the OpenRouter table.

`opencode` runs the opencode CLI with a private config per session: OpenAI keys use the Responses API, Anthropic keys the Messages API, OpenRouter its own provider, and a gateway the Messages API for Claude models, the Responses API for OpenAI models and Chat Completions for the rest. `agent.thinking_effort` becomes the model's reasoning effort or adaptive thinking. Chack tools run over MCP next to opencode's file, search, edit and (with `exec_enabled`) shell tools; opencode's question, sub-agent, skill and todo tools are denied. opencode cannot use Codex or Claude Code subscription tokens. opencode 2 runs `opencode run` on one background service shared by every run, so Chack passes `--standalone` (a private server per run) to each opencode that offers it in `opencode run --help`; opencode 1 always uses a private server.

`kimi` and `qwen` use private CLI homes and empty launch workspaces. Their native shell and file tools are removed; Chack tools are provided over MCP, and `tools.exec_enabled` controls the only host shell. Both resume saved CLI sessions. Kimi uses its API key directly or a gateway's OpenAI-compatible API; Qwen needs an explicit OpenAI-compatible `QWEN_BASE_URL` with its direct key. For either CLI, set `api_key_type: gateway`, `LLM_GATEWAY_BASE_URL`, and `LLM_GATEWAY_API_KEY` to use a gateway model. CLI paths can be overridden with `KIMI_CLI_PATH` and `QWEN_CLI_PATH`. See [the CLI compatibility guide](docs/code-cli-compatibility.md).

See [the backend guide](chack_agent/backends/README.md) for transport, memory, compaction, and guardrail differences.

### Custom LLM gateways

A gateway is any service that serves vendor models behind one key through the OpenAI Responses API and, for Claude Code, the Anthropic Messages API: LiteLLM, Portkey, Kong, Cloudflare AI Gateway, or a company proxy. Set `LLM_GATEWAY_BASE_URL` (the OpenAI root, usually ending in `/v1`) and `LLM_GATEWAY_API_KEY`, then select the gateway with `provider: gateway`, `api_key_type: gateway`, or a `gateway/<model>` name. `codex` and `claude` keep running their CLIs against the gateway (OpenAI- and Anthropic-hosted web search are turned off because gateways cannot serve them); `copilot` and `openai` move to the `gateway` Responses runtime.

`gemini` keeps the Gemini CLI when the gateway serves Google's Gemini API (`<root>/v1beta/models/<model>:generateContent`, with the gateway key as `x-goog-api-key`). LiteLLM serves it at `LLM_GATEWAY_BASE_URL` without `/v1` and translates it for every model it serves, so with `LLM_GATEWAY_TYPE=litellm` the CLI runs Gemini and non-Gemini models alike; other gateways set `LLM_GATEWAY_GEMINI_BASE_URL` (or `credentials.llm_gateway_gemini_base_url`). Without either, `gemini` runs on the `gateway` Responses runtime and logs a warning saying so. For non-Gemini models LiteLLM drops the Gemini `thinkingConfig`, so `thinking_effort` has no effect, and its streamed answers (`:streamGenerateContent`) report zero tokens. The CLI therefore reaches the gateway through a relay Chack runs on 127.0.0.1 for each run: it forwards every request unchanged, except that a streaming request is sent as `:generateContent` and its answer, which carries the real usage, goes back to the CLI as one server-sent event, and that every call goes to the run's model: the CLI's own helper calls (chat compression, loop detection, routing, summaries) name Google models such as `gemini-3-pro-preview` that a gateway may not serve or a profile never chose. If a gateway still reports no usage, Chack logs that the run's tokens and cost are not counted. The Chack tool server gets the `LLM_GATEWAY_*` variables, so sub-agents it starts use the same gateway.

Chack reads the models the key may use from the gateway (`/models`, or `/model/info` with `LLM_GATEWAY_TYPE=litellm`). Vendor IDs such as `claude-opus-5-5` are mapped to the gateway's own names, and a model the key cannot use fails before the run with the allowed list. `list_gateway_models()` returns that list for model pickers. With `litellm`, the gateway also supplies context windows and the prices used for cost accounting. Other gateway names are priced from the OpenRouter table when they name the same model: `opus-5.5` as `anthropic/claude-opus-5.5`, `gemma-4-26b-a4b` as `google/gemma-4-26b-a4b-it`, and a few aliases such as `qwen3-coder-480b-a35b-instruct` as `qwen/qwen3-coder`. A gateway's own router (`<gateway>/auto`) is never priced as OpenRouter's. `LLM_GATEWAY_MODEL_MAP` overrides the mapping.

Claude Code marks the system prompt, tools and last message for prompt caching, and some gateway models reject the whole request for it (Vertex Gemini: "does not support explicit context caching"). Chack then runs the request again once without the markers (`DISABLE_PROMPT_CACHING`) and leaves them off for that model for the rest of the process; set `prompt_cache_enabled: false` in the config's `agent:` section (`ChackConfig.model.prompt_cache_enabled`) for such a model to skip the rejected request.

Some gateway models end a run without an answer: the provider breaks their tool calls (Kimi K2.5 tool names leaking `<|tool_call_argument_begin|>`), the model writes a tool call as text instead of calling it, or it fills the output schema with a placeholder such as `pending`. Chack does not return those as answers. The run fails with `error` `backend_failure_malformed_tool_call` or `backend_failure_no_answer`, both under the `backend_failure` prefix, so callers can retry or move to another model. Whether an agent had to use its tools is the agent's own rule: set `required_tool_names` for agents whose answer is only valid after a tool call, since a model may answer without calling any.

## Tools

`ToolsConfig` exposes local execution, browser and search tools, PDF extraction, specialist researchers, ChatGPT browser researchers, sub-Chack delegation, parallel research, a research administrator, and a shared research queue. Individual tools and source integrations are enabled with fields in [`chack_tools/config.py`](chack_tools/config.py), which is the source of truth.

The defaults matter:

- `task_steps_manager_enabled` defaults to `true`; set it to `false` for agents that should reason about workflow without maintaining a runtime plan.
- Other functional tool families default to disabled.
- `exec_enabled` (default `false`) is the one switch for command execution, on every backend. chack's `exec` tool is the only shell a run gets: the model CLIs' own shells (Codex, Claude Code `Bash`, opencode `bash`, Copilot bash/powershell, Gemini `run_shell_command`) are always turned off, so an exec-disabled agent cannot run commands at all and an exec-enabled one keeps exec's timeout, output cap and tool budget. Without `exec` a run also gets none of the CLIs' file tools (read, edit, write, patch, search): it works only through Chack's tools.
- `min_tools_used` defaults to `0`; set it when a workflow must gather evidence before answering. A run that finishes below it is resumed and asked for more calls, so a high minimum spends tokens on filler calls.
- An identical call to a read-only tool made within `tool_result_reuse_seconds` (default 60) returns the earlier result instead of running again. Shell/script tools, tools that act on a live target (`request`, `probe`, `browser`, `verify`, ...), tools whose names change or report changing state (`create`, `delete`, `start`, `poll`, ...), and results longer than `tool_result_reuse_max_chars` are never reused; a state-changing call clears the reused results. A tool can opt in or out with `reuse_tool_results(tool, enabled)`.
- A tool result longer than `tool_output_max_chars` (default 40000) is saved to a file under `tool_output_dir` (default: the system temp directory), and the agent receives `tool_output_preview_chars` of its start and end plus the file path. This applies whenever the agent has Chack's `exec` tool to read the file; otherwise the result is cut to the same preview.
- `max_tools_used: 0` means unlimited.
- `required_tool_names` can require persistence or reporting tools before a run is accepted.

```yaml
tools:
  task_steps_manager_enabled: true
  exec_enabled: true
  brave_enabled: true
  playwright_enabled: true
  scientific_enabled: true
  researcher_administrator_enabled: true
  researcher_administrator_researchers: [scientific, websearcher]
  min_tools_used: 2
  max_tools_used: 40
  required_tool_names: [save_result]
  required_tool_call_attempts: 3
```

`tools_override` replaces the configured tool list for one run, while `tools_append` adds tools. For continuation turns that must retain overridden tools and provider memory, pass `reuse_session_executor=True` on every turn. The first executor configuration remains authoritative until that session is reset.

```python
first = agent.run(
    "review-001",
    "Inspect the repository.",
    tools_override=[custom_tool],
    reuse_session_executor=True,
)
follow_up = agent.run(
    "review-001",
    "Recheck the disputed result.",
    tools_override=[custom_tool],
    reuse_session_executor=True,
)
agent.reset_session("review-001")
```

For native-planning CLI backends, Chack maps the task-manager requirement to the provider's planning tool. For transported tools, it enforces required calls and maximum calls at the runtime or MCP boundary where supported.

## Sessions, compaction, and memory

Calls with the same session ID can continue provider conversation state. `agent.idle_reset_minutes` and `agent.max_age_minutes` rotate stale sessions when nonzero. `reset_session(session_id)` clears a session and, by default, finalizes its long-term memory first.

Automatic compaction uses `agent.max_context_tokens` and `agent.compaction_threshold_ratio`. `memory_max_messages` and `memory_reset_to_messages` bound local or summary-backed history where applicable. On the LangGraph backend they count conversation turns, not tool rounds; a long tool loop compacts only near the context limit (`CHACK_LANGGRAPH_CONTEXT_TOKENS`, default 200,000, when the window is not configured) and keeps the request verbatim. Near that limit it first clears old tool outputs (the newest five stay) and condenses what they held, read whole in chunks, into the checkpoint summary, and also shrinks and retries once when the provider reports a context overflow. Its loop recovers instead of ending early: an empty or length-cut reply, or prose where a schema answer is required, gets a continue prompt (at most two per run); unparseable tool arguments are returned to the model as an error; a call repeated a third time is flagged; and the last allowed turn answers without tools instead of raising `MaxTurnsExceeded`. The model also gets a built-in `todo_write` plan tool (not counted against the tool budget; `CHACK_LANGGRAPH_PLAN_TOOL=0` removes it): open items are recalled after ten calls without an update and an answer that leaves items open is sent back once. Length-cut replies are joined to their continuation, and tool-call markup a gateway leaves inside a string argument is cut off before the tool runs. A reply stopped by the provider's content filter is asked again up to twice before the run fails, since such filters are not deterministic. File-backed long-term summaries are controlled by `long_term_memory_enabled`, `long_term_memory_dir`, and `long_term_memory_max_chars`.

To compact an existing provider conversation before a specific continuation, opt in on that call:

```python
result = agent.run(
    "review-001",
    "Continue from the verified findings.",
    compact_before_resume=True,
    resume_compaction_instructions="Preserve verified evidence and unresolved hypotheses.",
)
```

Compaction is fail-open: its status and error are recorded in `RunResult`, and the requested continuation still runs.

## Parallel tools and background research

On the LangGraph backend, `CHACK_LANGGRAPH_PARALLEL_TOOLS=4` lets up to four tool calls issued in the same model turn overlap. The default is `1` (serial); values are clamped to 1–16 and invalid values use the default. Results, tool-budget notes and repeat warnings remain in issue order. Each worker inherits the run's session, cancellation, usage, artifact and telemetry context. Runtime/cost warnings are applied once in the ordered result loop. Calls waiting for a worker do not start after cancellation; already running tools must cooperate with cancellation or finish within their own timeout.

Turns containing browser MCP calls, planning tools or researcher job-management calls stay serial to preserve shared-state dependencies. Tool authors can also set `chack_parallel_safe=False` on a function tool to keep its turn serial. Other tools, including `exec`, may overlap when enabled: use this setting only when calls are independent and concurrent access is safe for their files and targets. This setting overlaps tool execution within a turn; the model receives the next turn after all its calls finish.

For work that should continue across model turns, the researcher administrator already exposes background jobs, independently of this setting:

1. Launch with `start_researchers_async`; retain the returned `job_id` and task IDs.
2. Continue other work. Use `poll_researchers_async(wait_seconds=0)` for a quick snapshot, `get_researcher_task` for health/progress diagnostics, or `list_researcher_jobs` to recover a forgotten handle after compaction.
3. When waiting is appropriate, use one completion-aware poll at a time. Ordinary researcher waits are 30–120 seconds; browser research waits are 300–1800 seconds. Polls return early on completion. Routine polls keep `include_outputs=false`.
4. Retrieve finished evidence with `get_researcher_result`: start with `view=summary`, then page through `parsed` or `raw` using `next_offset` when needed.
5. Use `cancel_researcher_task` or `retry_researcher_task` for ordinary failed or unneeded work. Active browser researchers retain their existing cancellation protections and hard deadlines.

Background jobs remain owned by their administrator run and its lifecycle; they are not a detached service that outlives run shutdown. Long shell commands can use `exec` with explicit background log/output files and short follow-up `ps`/`tail` commands.

## Limits and results

`agent.max_runtime_minutes` and `agent.max_cost_usd` use `0` for unlimited. Warning and critical ratios can inject remaining-budget context before a hard limit. Self-critique is disabled by default and can be enabled with `self_critique_rounds` or per-run overrides.

`RunResult` includes the output, provider steps, tool counts, token counts, rounds, estimated cost, timing, task session ID, compaction status, and any terminal error. A configured JSON schema is sent through the selected backend's structured-output mechanism when available and validated by Chack's CLI adapters where required.

The synchronous `run()` and asynchronous `arun()` APIs support per-run tool limits, required tools, self-critique, callbacks, prompt variables, context, working directory, cancellation, output-schema override, compaction, and executor reuse. See [`Chack.run`](chack_agent/agent.py) for the exact signature and [`chack_agent/config.py`](chack_agent/config.py) for every configuration field.

## Telemetry (OpenTelemetry)

Every `run()` is traced with OpenTelemetry using [OpenInference](https://github.com/Arize-ai/openinference) attributes, so any OTLP backend (Arize Phoenix, Langfuse, Jaeger, Grafana Tempo, an OpenTelemetry Collector...) can show it:

- one `AGENT` span per run, named `main_action.sub_action`, with session ID, model, input, output, token counts (including cache reads/writes), cost, tool counts, rounds, and the terminal error;
- one `TOOL` child span per tool call with its input, result (truncated) and error status, including tools called by CLI backends (Codex, Claude Code, Copilot, Gemini) through the Chack MCP server;
- run events (compaction, budget warnings, limits...) as span events.

Nothing is exported unless an endpoint is set. Use the standard OpenTelemetry variables or the matching `logging` fields (variables win):

| Environment | `logging` field | Meaning |
|---|---|---|
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `otlp_endpoint` | OTLP/HTTP collector URL; `/v1/traces` is appended to a base URL |
| `OTEL_EXPORTER_OTLP_HEADERS` | `otlp_headers` | e.g. `authorization=Bearer <key>` |
| `OTEL_SERVICE_NAME` | `otel_service_name` | defaults to `chack-agent` |
| `OTEL_RESOURCE_ATTRIBUTES` (`openinference.project.name=...`) | `otel_project` | project the traces are grouped under (Phoenix project) |

```yaml
logging:
  otlp_endpoint: https://phoenix.example.com
  otel_project: my-agents
```

Keep the API key out of the file: `export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer <key>"`.

If the host application already installed a tracer provider, Chack uses it and exports nothing itself. CLI backends and their tool processes receive the exporter settings and a `TRACEPARENT`, so their spans join the same trace. Set `CHACK_OTEL_LLM_SPANS=1` and install the `otel_llm` extra to also record one `LLM` span per model request on the in-process backends. Recorded values are capped by `CHACK_TELEMETRY_MAX_VALUE_CHARS` (default 16000) and tool results by `CHACK_TELEMETRY_TOOL_RESULT_MAX_BYTES` (default 8192).

## GitHub Action

For queue requests with read-enabled curated knowledge, the client-side researcher wrapper queries the policy-bound local Qdrant corpus before submitting a Deep/Pro browser job. The bounded, provenance-tagged passages are embedded in the exact browser prompt as untrusted leads, and an audit receipt is preserved with the run; the browser cannot choose or write a corpus.

`playwright_fetch` also requires the Python `playwright` package plus installed browser binaries. Once installed, set `tools.playwright_enabled: true` to expose it. If Playwright is missing or Chromium cannot launch, the tool is not registered.

The composite Action accepts prompts, schemas, model settings, JSON overrides for tools/session/agent configuration, limits, and provider credentials. Track `master`, matching the Naxus consumers:

```yaml
- name: Run chack-agent
  id: chack
  uses: carlospolop/chack-agent@master
  with:
    provider: openai
    model_primary: gpt-5.4
    system_prompt: You are a repository reviewer.
    prompt_file: review_prompt.txt
    output_schema_file: review.schema.json
    tools_config_json: '{"task_steps_manager_enabled":false,"exec_enabled":true,"min_tools_used":0}'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Read result
  run: echo '${{ steps.chack.outputs.final-message }}'
```

[`action.yml`](action.yml) is the exact input and output contract.

## Development

The core runtime is in `chack_agent/`; tool registration and specialist agents are in `chack_tools/`.

```bash
python -m pytest -q
```

Custom tools that need application resource settings can declare their environment names in `tools.mcp_env_vars`. Codex, Claude Code, Gemini and Copilot pass the current values to their MCP subprocess; opencode already inherits the caller environment. Unset values stay unset. This does not add values to model prompts; keep credentials in the existing credential configuration.

Use `chack_agent.backends.tool_payloads.preserve_mcp_input_schema(tool)` on tools that require their complete declared input schema over MCP, including nested fields and enum values. JSON argument names work directly; safe Python aliases from earlier bridge versions remain accepted.
