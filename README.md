# chack-agent

`chack-agent` is the shared agent runtime used by Naxus. It provides one Python API over OpenAI, OpenRouter, Codex CLI, Claude Code, Gemini CLI, GitHub Copilot CLI, and LangGraph, with configurable tools, sessions, budgets, structured output, and usage accounting.

AISecurityAuditor and Dynamic-AIgent install the `master` branch directly. Keep `master` compatible and green because downstream runs receive merged changes without advancing a version pin.

## Install

```bash
pip install chack-agent
```

Python 3.10 or newer is required.

CLI backends also require their provider command on `PATH`: `codex`, `claude`, `gemini`, or `copilot`. The Python package includes Playwright, but browser-backed tools require an installed Chromium binary:

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
| `gemini` | Gemini CLI | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or an OpenRouter route |
| `copilot` | GitHub Copilot CLI | `COPILOT_GITHUB_TOKEN` or existing CLI authentication |
| `langgraph` | LangGraph with an OpenRouter `ChatOpenAI` transport | `OPENROUTER_API_KEY` |

Aliases are normalized when configuration loads. Unsupported providers and unsupported thinking-effort values for known models fail early. The accepted common effort vocabulary is `none`, `minimal`, `low`, `medium`, `high`, `xhigh`/`extra_high`, and `max`; each backend maps supported values to its native control.

See [the backend guide](chack_agent/backends/README.md) for transport, memory, compaction, and guardrail differences.

## Tools

`ToolsConfig` exposes local execution, browser and search tools, PDF extraction, specialist researchers, ChatGPT browser researchers, sub-Chack delegation, parallel research, a research administrator, and a shared research queue. Individual tools and source integrations are enabled with fields in [`chack_tools/config.py`](chack_tools/config.py), which is the source of truth.

The defaults matter:

- `task_steps_manager_enabled` defaults to `true`; set it to `false` for agents that should reason about workflow without maintaining a runtime plan.
- Other functional tool families default to disabled.
- `min_tools_used` defaults to `10`; set it explicitly when a workflow should allow fewer calls.
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

Automatic compaction uses `agent.max_context_tokens` and `agent.compaction_threshold_ratio`. `memory_max_messages` and `memory_reset_to_messages` bound local or summary-backed history where applicable. File-backed long-term summaries are controlled by `long_term_memory_enabled`, `long_term_memory_dir`, and `long_term_memory_max_chars`.

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

## Limits and results

`agent.max_runtime_minutes` and `agent.max_cost_usd` use `0` for unlimited. Warning and critical ratios can inject remaining-budget context before a hard limit. Self-critique is disabled by default and can be enabled with `self_critique_rounds` or per-run overrides.

`RunResult` includes the output, provider steps, tool counts, token counts, rounds, estimated cost, timing, task session ID, compaction status, and any terminal error. A configured JSON schema is sent through the selected backend's structured-output mechanism when available and validated by Chack's CLI adapters where required.

The synchronous `run()` and asynchronous `arun()` APIs support per-run tool limits, required tools, self-critique, callbacks, prompt variables, context, working directory, cancellation, output-schema override, compaction, and executor reuse. See [`Chack.run`](chack_agent/agent.py) for the exact signature and [`chack_agent/config.py`](chack_agent/config.py) for every configuration field.

## GitHub Action

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
