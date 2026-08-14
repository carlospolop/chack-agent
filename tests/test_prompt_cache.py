from __future__ import annotations

from chack_agent.backends.prompt_cache import (
    PROMPT_CACHE_BREAKPOINT,
    openai_model_requires_explicit_prompt_cache,
    split_prompt_cache_breakpoint,
)


def test_split_prompt_cache_breakpoint_preserves_exact_bytes() -> None:
    stable = "stable context\nwith spacing\n\n"
    dynamic = "\nchanging round data\n"
    parts = split_prompt_cache_breakpoint(
        f"{stable}{PROMPT_CACHE_BREAKPOINT}{dynamic}"
    )

    assert parts.has_breakpoint is True
    assert parts.stable_prefix.encode("utf-8") == stable.encode("utf-8")
    assert parts.dynamic_suffix.encode("utf-8") == dynamic.encode("utf-8")
    assert parts.prompt_without_marker == f"{stable}{dynamic}"
    assert parts.cache_key(leading_prompt="system") == parts.cache_key(
        leading_prompt="system"
    )


def test_split_prompt_cache_breakpoint_leaves_unmarked_prompt_unchanged() -> None:
    parts = split_prompt_cache_breakpoint("ordinary prompt")

    assert parts.has_breakpoint is False
    assert parts.prompt_without_marker == "ordinary prompt"


def test_split_prompt_cache_breakpoint_rejects_multiple_markers() -> None:
    prompt = f"a{PROMPT_CACHE_BREAKPOINT}b{PROMPT_CACHE_BREAKPOINT}c"

    try:
        split_prompt_cache_breakpoint(prompt)
    except ValueError as exc:
        assert "only one cache breakpoint" in str(exc)
    else:
        raise AssertionError("multiple prompt-cache markers must be rejected")


def test_openai_explicit_cache_requirement_is_version_aware() -> None:
    assert openai_model_requires_explicit_prompt_cache("gpt-5.6-sol") is True
    assert openai_model_requires_explicit_prompt_cache("gpt-5.7") is True
    assert openai_model_requires_explicit_prompt_cache("gpt-6.0") is True
    assert openai_model_requires_explicit_prompt_cache("gpt-5.4") is False
    assert openai_model_requires_explicit_prompt_cache("o3") is False
