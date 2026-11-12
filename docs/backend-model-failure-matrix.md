# Bakkt model and Chack backend comparison

## What the results say

**For the five reachable Chinese models, start with Qwen Code.** It passed all
15 task runs. LangGraph passed 14/15 and OpenCode passed 13/15. Kimi K3.1 was
the most portable Chinese model: 28/30 runs passed across the ten backends
that accept Bakkt credentials. Claude Opus 4.8 and Sonnet 5.5 each passed
30/30. These are results from three tasks run once per pairing, so use them
as compatibility evidence rather than a general model ranking.

**Qwen3 Coder 480B could not be evaluated.** Bakkt advertised it, but its
upstream route returned 404 before the model could attempt the tasks.

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

## What caused the important failures

- **Kimi K3.1 + Codex:** the ledger case produced a malformed tool call; the
  web case answered without completing the required tool checks. The other
  nine compatible backends passed all three tasks with K3.1 after the Gemini
  relay fix.
- **Gemini models + Kimi Code:** tool continuation hit Bakkt's
  `thought_signature` rejection. This is a gateway/CLI compatibility failure.
- **Gemini CLI + Gemini models:** the web task produced the right answer but
  exceeded its eight-call tool budget.
- **MiniMax + LangGraph:** the agent read the ledger files but counted one
  extra open invoice. This did not indicate a missing tool or a broken loop.
- **Qwen3 Coder 480B:** an upstream Vertex publisher-model 404 prevented
  inference on every compatible backend.

The Gemini CLI initially failed all three Kimi K3.1 tasks because Bakkt
rejected generation fields. Chack now removes a field only after that exact
rejection; all three post-fix reruns passed. A valid citation to the
[official Python data model reference](https://docs.python.org/3/reference/datamodel.html)
was also added to the scorer before the final table was generated.

## How the comparison was run

The Bakkt LiteLLM configuration assigned to Naxus user `carlospolop` was
accessed with AWS profile `naxusai-admin` on 2026-11-12. The ten model IDs
were in Bakkt's catalog. Claude Opus **4.8** replaces the earlier 5.5 smoke
sample; no GPT model was included. The matrix contains **330 task cases**
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

The [sanitized per-case results](backend-model-matrix-results.json) contain
pass state, failure reason, time, and tool count without raw model output or
credentials. The
[table renderer](../scripts/render_backend_model_failures.py) accepts this
JSON file with `--compact` and a comma-separated `--models` list to regenerate
the tables. The
[benchmark runner](../scripts/benchmark_model_backend_matrix.py) and
[task definitions](../scripts/benchmark_backends.py) allow new runs. Kimi
CLI may omit usage from its JSONL output, so a recorded zero cost means
unknown cost. The private key, raw result files, and benchmark session homes
were removed after testing.
