# Bakkt model and backend functional matrix

For the full ten-model, eleven-backend comparison with Claude Opus 4.8 and
harder repair, ledger, and verified web tasks, see the
[failure matrix](backend-model-failure-matrix.md). This page preserves the
earlier exploratory smoke observations, including the separate Opus 5.5 sample.

Tested on 2026-11-11 and 2026-11-12 with the Bakkt gateway configuration for Naxus user `carlospolop`. The private API key is not stored in this repository. Kimi Code 2.5.3 and Qwen Code 0.27.2 were run with Node.js 22.20. The gateway model IDs were passed through unchanged. These are small functional samples, not statistically reliable quality rankings.

Run more samples with `scripts/benchmark_backends.py` and `LLM_GATEWAY_BASE_URL`, `LLM_GATEWAY_API_KEY`, and `LLM_GATEWAY_TYPE=litellm` in the environment. Targets use `backend:gateway-model:gateway`, for example `kimi:bakkt/kimi-k3.1:gateway`. The harness writes temporary local fixtures and checks answers, actual tool calls, latency, reported usage, and an eight-call tool budget. It never scans an external target.

## Increasing task difficulty

| Task | Expected result | Tools the task requires |
| --- | --- | --- |
| `code_reasoning` | Output of a short deduplication program: `47` | None |
| `shell` | Local `printf` returns `chack-42` | Chack `exec` |
| `public_web` | Python dictionary insertion order became a language guarantee in 3.7, with an official docs URL | Web search or fetch |
| `combined_report` | From one local CSV: active score sum `55`, three distinct active users; verify Python 3.7 in official docs | Local file access and web |
| `join_report` | Join two local CSVs, apply one adjustment per user, break a tie by first active row: `iris`, score `21`, tied with `maple`; verify Python 3.7 in official docs | Local file access and web |

The combined and join prompts state the desired report and evidence, without naming tools or prescribing a sequence. A passing row needs the correct answer, observed tool families, and no more than eight counted calls. The official source used for scoring is the [Python 3.7 release note](https://docs.python.org/3/whatsnew/3.7.html) or the [built-in types documentation](https://docs.python.org/3/library/stdtypes.html).

## Chinese model routes

| Gateway model | Kimi Code CLI | Qwen Code CLI |
| --- | --- | --- |
| `kimi-k3.1` | Code, command, combined and join passed. Join: 15.6s, 3 calls. | All four passed. Join: 15.1s, 4 calls. |
| `glm-5.2` | Code, command, combined and join passed. Join: 32.4s, 5 calls. | Code, command and join passed; join: 62.9s, 4 calls. One combined run returned the expected fields; another timed out at 90s. |
| `minimax-m2.5` | Command and combined passed. Code reasoning timed out at 90s in the full harness; an earlier direct run answered `39` instead of `47`. | Code and command passed. Combined returned the correct fields but used 10 calls against the eight-call budget. |
| `deepseek-v4.1-flash` | Code, command and combined passed; combined: 7.7s, 2 calls. An earlier direct code run answered incorrectly, so repeatability needs checking. | Code, command and combined passed; combined: 45.0s, 5 calls. |
| `kimi-k2.5` | Simple arithmetic passed; harder tasks not tested. | Simple arithmetic passed; harder tasks not tested. |
| `qwen3-coder-480b-a35b-instruct` | Gateway 404. | Gateway 404. |

The Qwen coder route was advertised by `/v1/models`, but the gateway returned an upstream Vertex publisher-model unavailable error in both new backends. Kimi print-mode JSONL omitted token usage, so its reported zero cost is **unknown cost**, not a free run. Qwen reported usage and estimated cost; model and prompt size affected it substantially.

## Gemini and Claude model routes

| Gateway model | Backend observations |
| --- | --- |
| `gemini-3.5-flash` | Qwen passed command, combined and join; join took 53.3s and 7 calls. Codex, Gemini CLI and LangGraph passed the combined task. On join, Gemini CLI and LangGraph answered correctly but each recorded 9/8 calls. Kimi answered the code question, then its tool continuation failed with a gateway `thought_signature` 400. |
| `gemini-3.1-pro-preview` | Kimi, Qwen and Codex answered the code question. Multi-tool tasks not tested. |
| `claude-opus-5.5` | Kimi and Qwen passed command, combined and join. Join: Kimi 24.0s/4 calls; Qwen 26.5s/2 calls. LangGraph passed combined and, after the tool-guidance change, join at 17.6s/5 calls. Claude Code passed command and combined; its join answer was correct but it did not check the web source. Codex passed command, but the gateway's Vertex organization policy rejected its structured-output request on combined. |
| `claude-sonnet-5.5` | Kimi and Codex returned the correct code reasoning in verbose form. Qwen initially got a gateway 400 because Qwen sent an OpenAI-style reasoning field to the Claude Vertex route; omitting that field for Claude routes made the model answer. Multi-tool tasks not tested. |

A direct-executor Claude Code probe initially failed to start its MCP server because the probe lacked a full Chack session context. Full `Chack.run()` benchmark cases started the MCP server and completed tool tasks; the full-run results above supersede that probe. The Claude Code join failure in this matrix is a **source-verification failure**, not a wrong calculation: it returned the expected fields without a web call.

## Current interpretation

K3.1 is the strongest Chinese route in this small multi-tool sample: both new backends passed all tested tiers, and Kimi used fewer calls for the two-file task. GLM 5.2 is also solid in Kimi; Qwen's longer and variable web runs need repetition. Qwen Code was more consistent on MiniMax code reasoning, while Kimi was faster on the DeepSeek combined task. For Gemini 3.5 Flash with tools, Qwen, Codex, Gemini CLI and LangGraph worked; Kimi's tool continuation did not. Claude Opus worked through Kimi, Qwen, LangGraph and Claude Code, with different verification and cost behavior.

Repeat the same tasks across several runs before choosing production defaults. In particular, compare verified success, tool-budget compliance, latency and real provider billing; Kimi's missing usage prevents a fair cost comparison today.
