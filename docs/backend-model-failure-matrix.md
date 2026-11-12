# Bakkt model and Chack backend failure matrix

This comparison uses the Bakkt LiteLLM gateway configuration assigned to
Naxus user `carlospolop`, accessed with AWS profile `naxusai-admin` on
2026-11-12. No private key or account configuration is stored here. The ten
model IDs below were advertised by the gateway model catalog. Claude Opus 4.8
replaces the earlier Opus 5.5 smoke sample. No GPT model is included.

## Protocol

Every case runs in a new Chack session with fresh temporary local files.
Results count only when the answer is correct, the task actually calls the
required tool family, and it stays within its tool budget. A JSON answer
without inspecting the files fails. No task scans or probes an external
target. Resumed matrix cases have a 150-second wall-clock limit; the few
baseline cases completed before that limit was introduced are identified in
the raw results by their recorded durations.

| Code | Task | Check | Tool budget |
| --- | --- | --- | --- |
| `R` | `repair_allocator` | Repair a local Python cents allocator with two bugs, preserve its API, run tests, and pass hidden behavior cases | 12 calls |
| `L` | `reconcile_ledger` | Join posted invoices, payments, and refunds in three local CSVs; exclude void and unknown payments; calculate the top customer, balance, open invoice count, and ignored payment total | 12 calls |
| `J` | `join_report` | Join two local CSVs, apply adjustments and a stable tie break, and verify Python's dictionary order guarantee from official documentation | 8 calls |

The prompts state the work and desired output without naming a specific tool
or prescribing a sequence. `R` and `L` need local file access; `J` needs both
local file and web access. `R` is checked by executing hidden functional cases
against the resulting module, not by trusting the model's claim that it fixed
the bug. `L` and `J` require exact JSON fields and values. The source check
for `J` accepts official docs.python.org pages. Tool calls are counted from
Chack's run result. Times are wall clock under concurrent load and are useful
only as rough observations. The [Python data model reference](https://docs.python.org/3/reference/datamodel.html)
also documents the Python 3.7 insertion-order guarantee; it was added to the
source allowlist after a valid LangGraph answer exposed that omission. Saved
JSON answers are rescored with the corrected allowlist in the table below.

Cells list failed tests and the reason: `answer` means wrong result,
`tools` means required local or web evidence was not used, `budget` means too
many calls, `timeout` means the case exceeded its CLI limit, and `error`
means the route failed. `route` means the Bakkt gateway credential is not
supported by that backend. An em dash means all three tasks passed. `unrun`
would mean a case was not completed. These are single attempts unless a rerun
is explicitly noted; they do not establish statistical rankings.

## Failures by model and backend

| Model | openai | openrouter | gateway | codex | claude | gemini | kimi | qwen | copilot | opencode | langgraph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kimi-k3.1 | — | R:route, L:route, J:route | — | L:error, J:answer+tools | — | — | — | — | — | — | — |
| glm-5.2 | L:answer+tools, J:answer+tools | R:route, L:route, J:route | L:answer+tools, J:answer+tools | L:answer+tools, J:answer+tools | — | — | — | — | L:answer+tools, J:answer+tools | R:timeout | — |
| minimax-m2.5 | L:answer+tools, J:answer+tools | R:route, L:route, J:route | L:answer+tools, J:answer+tools | L:answer+tools, J:answer+tools | — | J:timeout | J:timeout | — | L:answer+tools, J:answer+tools | L:answer | L:answer |
| deepseek-v4.1-flash | L:answer+tools, J:answer+tools | R:route, L:route, J:route | L:answer+tools, J:answer+tools | L:answer+tools, J:answer+tools | J:error | — | — | — | R:error, L:answer+tools, J:answer+tools | — | — |
| kimi-k2.5 | L:answer+tools, J:answer+tools | R:route, L:route, J:route | L:answer+tools, J:answer+tools | L:answer+tools, J:answer+tools | R:answer+tools, J:error | R:answer, L:answer, J:answer+tools | R:timeout, L:answer, J:answer | — | L:answer+tools, J:answer+tools | — | — |
| qwen3-coder-480b-a35b-instruct | R:error, L:error, J:error | R:route, L:route, J:route | R:error, L:error, J:error | R:error, L:error, J:error | R:error, L:error, J:error | R:error, L:error, J:error | R:error, L:error, J:error | R:error, L:error, J:error | R:error, L:error, J:error | R:error, L:error, J:error | R:error, L:error, J:error |
| gemini-3.5-flash | R:error | R:route, L:route, J:route | R:error, J:error | — | — | J:budget | R:error, L:error, J:error | — | — | L:answer, J:answer+tools | — |
| gemini-3.1-pro-preview | — | R:route, L:route, J:route | — | — | — | J:budget | R:error, L:error, J:error | J:error | J:tools | R:budget, J:error | — |
| claude-opus-4.8 | — | R:route, L:route, J:route | — | — | — | — | — | — | — | — | — |
| claude-sonnet-5.5 | — | R:route, L:route, J:route | — | — | — | — | — | — | — | — | — |

## Route and cost interpretation

Across the five reachable Chinese models, Qwen Code passed **15/15** cases,
LangGraph **14/15**, and OpenCode **13/15**. Kimi K3.1 had the broadest model
coverage at **28/30** gateway-compatible cases after the Gemini relay fix;
the two remaining failures were Codex CLI tool-call/verification cases.
Claude Opus 4.8 and Sonnet 5.5 each passed **30/30** compatible cases in this
single-run sample. These counts describe these three tasks and gateway routes,
not general model quality. The Qwen3 Coder 480B model could not be assessed:
the advertised route returned an upstream Vertex publisher-model 404 across
the tested backends.

The generic LangGraph loop passed all three tasks with K3.1, GLM 5.2,
DeepSeek V4.1 Flash, and K2.5. Its MiniMax ledger failure was a wrong open
invoice count after reading the files, not a missing tool. That result did not
justify an additional loop change beyond the checkpoint and tool-guidance
work already in this branch. Gemini CLI initially failed all K3.1 cases
because the gateway rejected its generation fields. The relay now retries
only explicitly rejected fields, and the three post-fix K3.1 reruns passed.

The `openai` and `copilot` providers redirect to Chack's Gateway Responses
runtime with custom gateway credentials, so their columns are repeated
gateway-runtime trials rather than independent OpenAI or Copilot transports.
`openrouter` does not accept the Bakkt gateway credential and cannot be
compared with these Bakkt model IDs through this configuration. Kimi CLI's
current JSONL output can omit token usage; a zero recorded cost for those
cases is unknown cost. The gateway catalog can advertise a model whose
upstream publisher route is unavailable; the failed cases identify that
separately from model reasoning errors.

The reproducible harness is
[`scripts/benchmark_model_backend_matrix.py`](../scripts/benchmark_model_backend_matrix.py),
using the tasks in [`scripts/benchmark_backends.py`](../scripts/benchmark_backends.py).
[`scripts/render_backend_model_failures.py`](../scripts/render_backend_model_failures.py)
formats the JSONL case results into the table above. The private JSONL and
gateway key stay outside the repository. The sanitized
[`backend-model-matrix-results.json`](backend-model-matrix-results.json)
records every case's pass state, reason, time, and tool count without raw
model output or credentials.
