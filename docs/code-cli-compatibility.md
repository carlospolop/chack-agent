# Kimi Code and Qwen Code backends

Chack accepts agent.provider values kimi and qwen, with kimi-code/kimi_code
and qwen-code/qwen_code aliases. These use the current TypeScript CLIs:
@moonshot-ai/kimi-code (the kimi command) and @qwen-code/qwen-code (the qwen
command). The older Python kimi-code/kimi-cli packages have a different
command and configuration contract.

Install the CLIs separately from chack-agent. Kimi Code 2.5.3 and Qwen Code
0.27.2 were used for the initial smoke checks. Set KIMI_CLI_PATH or
QWEN_CLI_PATH if the command is not on PATH. A direct Kimi run needs
KIMI_API_KEY; a direct Qwen run needs QWEN_API_KEY and an explicit
OpenAI-compatible QWEN_BASE_URL. For a gateway, use api_key_type: gateway,
LLM_GATEWAY_BASE_URL, and LLM_GATEWAY_API_KEY. The adapters pass the gateway
model name unchanged to the CLI.

Each Chack session has a private CLI home and an empty launch directory. The
CLI session ID is saved for the next invocation; compact_for_resume() sends
Kimi /compact or Qwen /compress to that session. The configured output
schema is included in the prompt. Qwen's result event supplies usage; Kimi's
current print-mode JSONL may omit usage, in which case token cost is
unavailable for that turn.

Qwen's OpenAI-style reasoning setting is omitted for Claude model routes:
Bakkt's Claude Vertex route rejects that request field. Kimi and Qwen both
retain the configured reasoning setting for other compatible models. See
[the Bakkt smoke matrix](model-compatibility-smoke.md) for measured route and
tool-use differences.

## Execution tool compatibility

| CLI | Official execution tool | Required argument | Other documented arguments |
| --- | --- | --- | --- |
| Claude Code | Bash | command | description, timeout (ms), run_in_background |
| Kimi Code | Bash | command | cwd, timeout (ms), run_in_background, description, disable_timeout |
| Codex CLI | exec_command | cmd | workdir, tty, yield_time_ms, max_output_tokens, shell, login |
| Gemini CLI | run_shell_command | command | description, dir_path, is_background |
| Qwen Code | run_shell_command | command | description, directory, is_background |

Sources: [Claude Code hooks reference](https://code.claude.com/docs/en/hooks),
[Kimi Code built-in tools](https://github.com/MoonshotAI/kimi-code/blob/main/docs/en/reference/tools.md),
[Codex CLI tool specification](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/shell_spec.rs),
[Gemini CLI shell tool](https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/shell.md),
and [Qwen Code shell tool](https://github.com/QwenLM/qwen-code/blob/main/docs/developers/tools/shell.md).

These are distinct schemas. Chack does not translate their execution calls or
provide compatibility aliases for them. The Kimi and Qwen adapters disable
their native host execution and file tools; only Chack's MCP exec is offered
when tools.exec_enabled is true. When it is false, the model has no host
execution or file tool. The initial tool registries are checked by
tests/test_code_cli_request.py against a loopback model endpoint, so CLI
updates that reintroduce built-in tools are visible before deployment.
Qwen also receives a CLI `--max-tool-calls` cap when Chack has a tool budget,
covering native web calls that do not pass through the MCP budget counter.

The CLI's own web search follows tools.native_web_search_enabled. When it is
enabled and tools.keyless_web_search_enabled is true, a keyless Exa MCP
fallback is configured. Kimi may also offer WebSearch and FetchURL when
its host provides them.

No backend uses a native shell tool for target scanning. Functional smoke
checks use arithmetic prompts and printf in a local temporary directory.
