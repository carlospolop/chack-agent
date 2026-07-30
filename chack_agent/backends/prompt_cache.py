from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


# Agent YAMLs may place this marker after a large prefix that remains identical
# across otherwise stateless invocations. Backends remove the marker before the
# model sees the prompt and map the two sides to their native prompt layers.
PROMPT_CACHE_BREAKPOINT = "<!-- CHACK_PROMPT_CACHE_BREAKPOINT -->"


@dataclass(frozen=True)
class PromptCacheParts:
    stable_prefix: str
    dynamic_suffix: str
    has_breakpoint: bool

    @property
    def prompt_without_marker(self) -> str:
        return f"{self.stable_prefix}{self.dynamic_suffix}"

    def cache_key(self, *, leading_prompt: str = "") -> str:
        cacheable_prompt = f"{leading_prompt}{self.stable_prefix}"
        return prompt_cache_key(cacheable_prompt)


def split_prompt_cache_breakpoint(prompt: str) -> PromptCacheParts:
    """Split one visible, provider-neutral prompt-cache boundary.

    The exact text before the marker is preserved byte-for-byte. A second
    marker is rejected because multiple cache lifetimes need explicit backend
    semantics rather than an ambiguous text convention.
    """
    value = str(prompt or "")
    marker_count = value.count(PROMPT_CACHE_BREAKPOINT)
    if marker_count == 0:
        return PromptCacheParts(
            stable_prefix="",
            dynamic_suffix=value,
            has_breakpoint=False,
        )
    if marker_count > 1:
        raise ValueError(
            f"Prompt contains {marker_count} {PROMPT_CACHE_BREAKPOINT!r} markers; "
            "only one cache breakpoint is supported."
        )
    stable_prefix, dynamic_suffix = value.split(PROMPT_CACHE_BREAKPOINT, 1)
    return PromptCacheParts(
        stable_prefix=stable_prefix,
        dynamic_suffix=dynamic_suffix,
        has_breakpoint=True,
    )


def prompt_cache_key(cacheable_prompt: str) -> str:
    digest = hashlib.sha256(str(cacheable_prompt or "").encode("utf-8")).hexdigest()
    return f"chack-{digest[:48]}"


def openai_model_requires_explicit_prompt_cache(model_name: str) -> bool:
    match = re.match(r"^gpt-(\d+)\.(\d+)", str(model_name or "").strip().lower())
    if not match:
        return False
    major, minor = (int(value) for value in match.groups())
    return major > 5 or (major == 5 and minor >= 6)
