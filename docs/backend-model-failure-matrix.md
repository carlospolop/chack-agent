# Bakkt model and Chack backend comparison

## Top three backends per model

Ranked by tasks passed, then observed total time across the three tasks.
Each pairing was run once; tied pass counts and timings are provisional.
The OpenAI and Copilot columns below use Chack's Gateway runtime with this
Bakkt credential, so they are excluded from this independent-backend ranking.

| Model | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Kimi K3.1 | OpenCode 3/3 | LangGraph 3/3 | Gateway 3/3 |
| GLM 5.2 | Qwen Code 3/3 | Gemini CLI 3/3 | Kimi Code 3/3 |
| MiniMax M2.5 | Qwen Code 3/3 | Claude Code 3/3 | LangGraph 2/3 |
| DeepSeek V4.1 Flash | LangGraph 3/3 | Gemini CLI 3/3 | OpenCode 3/3 |
| Kimi K2.5 | LangGraph 3/3 | Qwen Code 3/3 | OpenCode 3/3 |
| Qwen3 Coder 480B | Unavailable: Bakkt route returned 404 | — | — |
| Gemini 3.5 Flash | Claude Code 3/3 | Qwen Code 3/3 | Codex 3/3 |
| Gemini 3.1 Pro | Claude Code 3/3 | Codex 3/3 | Gateway 3/3 |
| Claude Opus 4.8 | LangGraph 3/3 | Kimi Code 3/3 | Codex 3/3 |
| Claude Sonnet 5.5 | OpenCode 3/3 | LangGraph 3/3 | Gateway 3/3 |

Across the five reachable Chinese models, Qwen Code passed 15/15 tasks,
LangGraph 14/15, and OpenCode 13/15. The table below shows the failures behind
each result.

## How to read the tables

Each cell lists **only the tasks that failed**. **✓** means all three passed.
**404** means the model route was unavailable. **N/A** means OpenRouter cannot
use the Bakkt gateway credential; no model run was attempted there.

| Task shown in a cell | What the agent had to do |
| --- | --- |
| **Repair** | Find and fix two bugs in a local Python allocator, run its tests, and pass hidden behavior checks. |
| **Ledger** | Reconcile invoices, payments, and refunds across three CSV files, including void and unknown payments. |
| **Web** | Join two CSV files, break a tie correctly, and verify a Python fact from official documentation online. |

### CLI backends

| Model | Codex | Claude Code | Gemini CLI | Kimi Code | Qwen Code | OpenCode |
| --- | --- | --- | --- | --- | --- | --- |
| Kimi K3.1 | Ledger, Web | ✓ | ✓ | ✓ | ✓ | ✓ |
| GLM 5.2 | Ledger, Web | ✓ | ✓ | ✓ | ✓ | Repair |
| MiniMax M2.5 | Ledger, Web | ✓ | Web | Web | ✓ | Ledger |
| DeepSeek V4.1 Flash | Ledger, Web | Web | ✓ | ✓ | ✓ | ✓ |
| Kimi K2.5 | Ledger, Web | Repair, Web | Repair, Ledger, Web | Repair, Ledger, Web | ✓ | ✓ |
| Qwen3 Coder 480B | 404 | 404 | 404 | 404 | 404 | 404 |
| Gemini 3.5 Flash | ✓ | ✓ | Web | Repair, Ledger, Web | ✓ | Ledger, Web |
| Gemini 3.1 Pro | ✓ | ✓ | Web | Repair, Ledger, Web | Web | Repair, Web |
| Claude Opus 4.8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Claude Sonnet 5.5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### API and LangGraph

| Model | OpenAI* | Gateway | Copilot* | LangGraph | OpenRouter |
| --- | --- | --- | --- | --- | --- |
| Kimi K3.1 | ✓ | ✓ | ✓ | ✓ | N/A |
| GLM 5.2 | Ledger, Web | Ledger, Web | Ledger, Web | ✓ | N/A |
| MiniMax M2.5 | Ledger, Web | Ledger, Web | Ledger, Web | Ledger | N/A |
| DeepSeek V4.1 Flash | Ledger, Web | Ledger, Web | Repair, Ledger, Web | ✓ | N/A |
| Kimi K2.5 | Ledger, Web | Ledger, Web | Ledger, Web | ✓ | N/A |
| Qwen3 Coder 480B | 404 | 404 | 404 | 404 | N/A |
| Gemini 3.5 Flash | Repair | Repair, Web | ✓ | ✓ | N/A |
| Gemini 3.1 Pro | ✓ | ✓ | Web | ✓ | N/A |
| Claude Opus 4.8 | ✓ | ✓ | ✓ | ✓ | N/A |
| Claude Sonnet 5.5 | ✓ | ✓ | ✓ | ✓ | N/A |

**Note:** With Bakkt credentials, Chack routes the **OpenAI** and **Copilot** columns
through its Gateway Responses runtime. They are additional gateway trials,
not independent OpenAI or Copilot model transports.

## Important compatibility limits

- Kimi Code with Gemini models hit Bakkt's `thought_signature` rejection on
  tool continuation.
- Gemini CLI with Gemini models exceeded the eight-call budget on the Web
  task, despite correct answers.
- Qwen3 Coder 480B returned an upstream 404 before inference.

## How the comparison was run

The Bakkt LiteLLM configuration assigned to Naxus user `carlospolop` was
used on 2026-11-12. No GPT model was included. The matrix contains **330 task cases**
(10 models × 11 backends × 3 tasks): **204/300 executable cases passed**;
the other 30 cases are OpenRouter credential incompatibilities.

Every executable case got a fresh Chack session and temporary local files.
A pass required the correct answer, actual calls to the needed local or web
tools, and at most 12 tool calls for Repair/Ledger or eight for Web. The code
repair was checked against the resulting module, not the model's claim that it
was fixed. The resumed cases had a 150-second wall-clock limit; a few earlier
baseline cases ran before that limit was added. There was one baseline attempt
per case, with the three Gemini CLI K3.1 cases rerun after the relay fix.
Latency under concurrent load and pass rates are not statistical rankings.
No external target was scanned or probed.

The [sanitized per-case results](backend-model-matrix-results.json) include
pass state, failure reason, time, and tool count. The
[table renderer](../scripts/render_backend_model_failures.py) and
[benchmark runner](../scripts/benchmark_model_backend_matrix.py) reproduce the
comparison. Kimi CLI may omit usage, so recorded zero cost means unknown cost.
