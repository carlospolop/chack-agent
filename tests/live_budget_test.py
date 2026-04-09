#!/usr/bin/env python3
"""Live integration test: run each CLI backend with budget already exceeded.

This script:
1. Sets budget env vars to simulate 92% runtime consumed (CRITICAL)
2. Appends budget warning text to the user prompt (what agent.py does)
3. Invokes each CLI with --max-turns 1 (cheapest possible — one response)
4. Checks that the model's response acknowledges the budget warning

Usage:
    python tests/live_budget_test.py [claude|codex|copilot]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chack_agent.budget_warning_state import (
    budget_prompt_warning,
    export_budget_env,
    export_spent_usd_env,
)

# ── Simulated budget state: 92% runtime, 75% cost ──────────────────
START_EPOCH = time.time() - 552  # 552s of 600s = 92% of 10min
MAX_RUNTIME_SECONDS = 600.0  # 10 minutes
MAX_COST_USD = 10.0
SPENT_USD = 7.5  # 75% of $10

# The prompt we'll send — very cheap (one-turn, simple question)
BASE_PROMPT = (
    "Reply with ONLY 'BUDGET_ACK' if you see any budget or time warning in this message. "
    "Otherwise reply 'NO_WARNING'. Do NOT use any tools. Just reply with that single word."
)


def setup_budget_env():
    """Set budget env vars as agent.py would."""
    export_budget_env(
        start_epoch=START_EPOCH,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        max_cost_usd=MAX_COST_USD,
        warning_ratio=0.6,
        critical_ratio=0.9,
        injection_enabled=True,
    )
    export_spent_usd_env(SPENT_USD)


def build_prompt_with_warning() -> str:
    """Append budget warning to prompt, exactly as agent.py _invoke() does."""
    bw = budget_prompt_warning(
        start_epoch=START_EPOCH,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        elapsed_runtime_seconds=time.time() - START_EPOCH,
        spent_usd=SPENT_USD,
        max_cost_usd=MAX_COST_USD,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    prompt = BASE_PROMPT + bw
    return prompt


def test_claude():
    """Run Claude Code CLI with budget warning in prompt."""
    claude_path = shutil.which("claude")
    if not claude_path:
        print("SKIP: claude not found")
        return False

    setup_budget_env()
    prompt = build_prompt_with_warning()

    print(f"\n{'='*60}")
    print("CLAUDE CODE TEST")
    print(f"{'='*60}")
    print(f"Prompt ({len(prompt)} chars):")
    print(prompt[:500])
    print("...")
    print()

    # Claude: positional prompt, --print for non-interactive, --max-turns 1
    cmd = [
        claude_path,
        "--print",
        "--verbose",
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--max-turns", "1",
        prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ},
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        print(f"Exit code: {result.returncode}")
        print(f"Output: {output[:1000]}")
        if stderr:
            print(f"Stderr (last 500): {stderr[-500:]}")

        if "BUDGET_ACK" in output.upper():
            print("\n✅ CLAUDE: Model SAW the budget warning!")
            return True
        elif "NO_WARNING" in output.upper():
            print("\n❌ CLAUDE: Model did NOT see the budget warning")
            return False
        else:
            if any(w in output.lower() for w in ["budget", "warning", "time", "cost", "limit"]):
                print("\n✅ CLAUDE: Model acknowledged budget context (non-standard response)")
                return True
            print(f"\n⚠️  CLAUDE: Unexpected response: {output[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("\n❌ CLAUDE: Timed out after 90s")
        return False
    except Exception as e:
        print(f"\n❌ CLAUDE: Error: {e}")
        return False


def test_codex():
    """Run Codex CLI with budget warning in prompt."""
    codex_path = shutil.which("codex")
    if not codex_path:
        print("SKIP: codex not found")
        return False

    setup_budget_env()
    prompt = build_prompt_with_warning()

    print(f"\n{'='*60}")
    print("CODEX TEST")
    print(f"{'='*60}")
    print(f"Prompt ({len(prompt)} chars):")
    print(prompt[:500])
    print("...")
    print()

    # Codex: exec subcommand, --json output, prompt via stdin ("-")
    # No --max-turns flag exists; prompt tells model not to use tools.
    cmd = [
        codex_path,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-",  # read prompt from stdin
    ]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ},
        )
        raw_output = result.stdout.strip()
        stderr = result.stderr.strip()
        print(f"Exit code: {result.returncode}")

        # Codex outputs JSON lines — extract message content
        output_text = ""
        for line in raw_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    msg = event.get("message", "")
                    content = event.get("content", "")
                    text = event.get("text", "")
                    output_text += str(msg or content or text or "")
            except json.JSONDecodeError:
                output_text += line

        print(f"Output: {output_text[:1000] if output_text else raw_output[:1000]}")
        if stderr:
            print(f"Stderr (last 300): {stderr[-300:]}")

        check_text = (output_text or raw_output).upper()
        if "BUDGET_ACK" in check_text:
            print("\n✅ CODEX: Model SAW the budget warning!")
            return True
        elif "NO_WARNING" in check_text:
            print("\n❌ CODEX: Model did NOT see the budget warning")
            return False
        else:
            if any(w in check_text.lower() for w in ["budget", "warning", "time", "cost", "limit"]):
                print("\n✅ CODEX: Model acknowledged budget context")
                return True
            print(f"\n⚠️  CODEX: Unexpected response")
            return False
    except subprocess.TimeoutExpired:
        print("\n❌ CODEX: Timed out after 90s")
        return False
    except Exception as e:
        print(f"\n❌ CODEX: Error: {e}")
        return False


def test_copilot():
    """Run Copilot CLI with budget warning in prompt."""
    copilot_path = shutil.which("copilot")
    if not copilot_path:
        print("SKIP: copilot not found")
        return False

    setup_budget_env()
    prompt = build_prompt_with_warning()

    print(f"\n{'='*60}")
    print("COPILOT CLI TEST")
    print(f"{'='*60}")
    print(f"Prompt ({len(prompt)} chars):")
    print(prompt[:500])
    print("...")
    print()

    # Copilot CLI: -p for prompt, --output-format json, --allow-all-tools
    # No --max-turns flag; prompt tells model not to use tools.
    cmd = [
        copilot_path,
        "-p", prompt,
        "--allow-all-tools",
        "--output-format", "json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ},
        )
        raw_output = result.stdout.strip()
        stderr = result.stderr.strip()
        print(f"Exit code: {result.returncode}")

        # Copilot outputs JSON lines — extract message content
        output_text = ""
        for line in raw_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    content = event.get("content", "")
                    message = event.get("message", "")
                    text = event.get("text", "")
                    output_text += str(content or message or text or "")
            except json.JSONDecodeError:
                output_text += line

        print(f"Output: {output_text[:1000] if output_text else raw_output[:1000]}")
        if stderr:
            print(f"Stderr (last 300): {stderr[-300:]}")

        check_text = (output_text or raw_output).upper()
        if "BUDGET_ACK" in check_text:
            print("\n✅ COPILOT: Model SAW the budget warning!")
            return True
        elif "NO_WARNING" in check_text:
            print("\n❌ COPILOT: Model did NOT see the budget warning")
            return False
        else:
            if any(w in check_text.lower() for w in ["budget", "warning", "time", "cost", "limit"]):
                print("\n✅ COPILOT: Model acknowledged budget context")
                return True
            print(f"\n⚠️  COPILOT: Unexpected response")
            return False
    except subprocess.TimeoutExpired:
        print("\n❌ COPILOT: Timed out after 90s")
        return False
    except Exception as e:
        print(f"\n❌ COPILOT: Error: {e}")
        return False


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["claude", "codex", "copilot"]

    print("Budget Warning Live Integration Test")
    print(f"Simulated state: {(time.time() - START_EPOCH)/60:.1f}/{MAX_RUNTIME_SECONDS/60:.0f} min elapsed, ${SPENT_USD}/${MAX_COST_USD} spent")

    # Show what the warning looks like
    bw = budget_prompt_warning(
        start_epoch=START_EPOCH,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        elapsed_runtime_seconds=time.time() - START_EPOCH,
        spent_usd=SPENT_USD,
        max_cost_usd=MAX_COST_USD,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    print(f"\nBudget warning that will be appended:\n{bw}\n")

    results = {}
    if "claude" in targets:
        results["claude"] = test_claude()
    if "codex" in targets:
        results["codex"] = test_codex()
    if "copilot" in targets:
        results["copilot"] = test_copilot()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for backend, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL" if passed is False else "⚠️  SKIP"
        print(f"  {backend:>10}: {status}")

    all_passed = all(v for v in results.values() if v is not None)
    print(f"\nOverall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
