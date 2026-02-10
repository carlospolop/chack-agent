#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from typing import Optional


def _print_steps(result) -> None:
    print("\n--- steps ---")
    for idx, item in enumerate(result.new_items, start=1):
        item_type = getattr(item, "type", type(item).__name__)
        print(f"[{idx}] {item_type}")
        raw = getattr(item, "raw_item", None)
        if raw is not None:
            try:
                print(json.dumps(raw, indent=2, default=str))
            except TypeError:
                print(str(raw))
        output = getattr(item, "output", None)
        if output is not None and item_type == "tool_call_output_item":
            print(f"output: {output[:100]}")
    print("--- end steps ---\n")
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, Runner, function_tool, set_default_openai_client, set_tracing_disabled


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/google/gemini-3-flash-preview"

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _build_openrouter_client() -> AsyncOpenAI:
    api_key = _get_env("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY in env.")
        sys.exit(1)

    headers = {}
    referer = _get_env("OPENROUTER_HTTP_REFERER")
    title = _get_env("OPENROUTER_APP_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers=headers or None,
    )


@function_tool(name_override="exec_local")
def exec_local(command: str) -> str:
    """Execute a local shell command and return stdout+stderr."""
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        timeout=30,
        env=None,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return output.strip() or "(no output)"


def main() -> None:
    client = _build_openrouter_client()
    set_tracing_disabled(True)
    set_default_openai_client(client, use_for_tracing=False)

    agent = Agent(
        name="OpenRouter PoC",
        instructions=(
            "You are a helpful autonomous agent. You perform tasks without asking questions and "
            "don't stop until the work is done. You have access to the following tool: "
            "exec_local(command) that executes a local shell command and returns the output. "
            "Use it to inspect the codebase and find security issues (use rg/grep, cat, ls). "
            "Call the tool as many times as needed."
        ),
        tools=[exec_local],
        model=MODEL_NAME,
        model_settings=ModelSettings(),
    )

    prompt = (
        "Please, check the code inside the repository in /Users/carlospolop/git/wiki-infrastructure/chack-agent and try to find security issues. You must locate the code, read it and find vulnerabilities"
    )

    result = Runner.run_sync(agent, prompt)
    _print_steps(result)
    print(result.final_output)


if __name__ == "__main__":
    main()
