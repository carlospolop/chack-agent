# Agent instructions

## Branch hygiene

- **Use the configured default branch.** Check the repository setting; it may be `main`, `master`, or something else. Base ordinary work on it and merge approved changes back into it. Target release branches only when required. Respect reviews, checks, and branch protection.
- **Create only needed branches.** Check `git status` and worktrees first. Use one short-lived branch per change, reuse it for revisions, and preserve others' work. Avoid speculative or duplicate branches.
- **Clean up after a verified merge.** Confirm the changes landed, including after squash/rebase, and that no open PR or unmerged work needs the branch. Delete only your completed task's remote/local branch and worktree; prune stale refs. Never delete default, protected, shared, or unrelated branches.
- **Leave a clean environment.** Check for leftover branches, worktrees, and uncommitted changes. Clean up safely or explain what must remain; never merge or delete when authorization or safety is uncertain.

## Model CLI configuration

- **Codex config keys can break between versions.** The per-run `config.toml` turns off prompt parts a headless run never uses (`_CODEX_UNUSED_PROMPT_PARTS_OFF` in `chack_agent/backends/codex_backend.py`). Some are experimental keys whose type has already changed once (`tools.experimental_request_user_input` is a table, not a boolean); if a Codex release changes one again, Codex rejects the config and every run fails. After bumping the Codex CLI, run `tests/test_codex_model_catalog.py` with that CLI installed: it renders the real prompt with `codex debug prompt-input` and fails on a rejected key or a returning prompt part.
- **Native web search stays on by default.** The CLI's own web search is enabled unless `tools.native_web_search_enabled` is false, `CHACK_DISABLE_CODEX_NATIVE_WEB` is set, or the backend cannot run it (an LLM gateway, or Codex without a Codex/ChatGPT login).
- **Claude Code built-ins are an allowlist.** `--tools` keeps only `_CLAUDE_KEPT_NATIVE_TOOLS` plus the shell and web tools the config allows; MCP tools are unaffected (that was not true in older Claude Code releases). New built-ins appear with Claude Code updates and stay hidden; add one to the allowlist only if headless runs need it. `tests/test_claude_code_request.py` captures a real CLI request and checks the tool list and that no CLAUDE.md, auto-memory or attribution text is sent.
