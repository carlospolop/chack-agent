from __future__ import annotations

import os
import re
import logging
from typing import Iterable

from agents import Agent, ModelSettings, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI
from chack_tools.telemetry import log_event

from .config import ChackConfig, resolve_backend_type


_LOGGER = logging.getLogger("chack.long_term_memory")


def _resolve_dir(config_path: str, rel_dir: str) -> str:
    if os.path.isabs(rel_dir):
        return rel_dir
    base_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(base_dir, rel_dir))


def _sanitize_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(session_id))
    return cleaned.strip("_") or "session"


def get_long_term_memory_path(config_path: str, session_id: str, rel_dir: str) -> str:
    directory = _resolve_dir(config_path, rel_dir)
    os.makedirs(directory, exist_ok=True)
    safe_id = _sanitize_session_id(session_id)
    return os.path.join(directory, f"{safe_id}.txt")


def load_long_term_memory(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def save_long_term_memory(path: str, content: str, max_chars: int) -> None:
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars].rstrip()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def format_messages(messages: Iterable) -> str:
    lines = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role") or msg.get("type") or "message").lower()
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", msg.__class__.__name__).lower()
            content = getattr(msg, "content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


_LONG_TERM_MEMORY_SUMMARY_PROMPT = """### ROLE
You are a memory summarization assistant for Chack, an autonomous AI assistant specialized in code execution and cloud and ci/cd infrastructure management.

### TASK
Analyze the conversation and extract only the most important, durable information that will help in future interactions.
Your goal is to extract things from the conversation that would have helped Chack complete similar tasks faster or more effectively in the future.
No need to store everything, only the most relevant and useful information to be more effective in the future.
If you find fake information in the previous memory also remove it. The long term memory should be as accurate and useful as possible, so if you find something that is not useful or is wrong, just remove it.

Moreover, note that you will be given the current conversation and the previous long term memory (if any).
Your response must combine both sources of information to generate an updated long term memory.
Whatever your response is, it'll be the new long term memory for future interactions.
So your answer must not contain questions or ask for clarifications, it must be only the updated long term memory that will be used in the future.

#### WHAT TO IGNORE
- Temporary information or one-time queries
- Conversational filler
- Information already in previous memory (unless updated)
- Specific command outputs or data (keep only patterns/learnings)

### FORMAT
- Use clear, organized structure
- Be concise and factual
- Use bullet points or short paragraphs
- IMPORTANT: Return plain text under {max_chars} characters. The long term memory cannot be longer than {max_chars} characters.
"""


def build_long_term_memory(
    config: ChackConfig,
    conversation_text: str,
    previous_memory: str,
    max_chars: int,
) -> str:
    model_name = config.model.primary
    model: str | OpenAIResponsesModel = model_name

    backend_type = resolve_backend_type(config)
    uses_openrouter_routed_model = (
        backend_type in {"openrouter", "langgraph"} or "/" in str(model_name or "")
    )
    if uses_openrouter_routed_model:
        api_key = (
            str(getattr(config.credentials, "openrouter_api_key", "") or "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when long-term memory uses a routed model"
            )
        base_url = (
            str(getattr(config.credentials, "openrouter_base_url", "") or "").strip()
            or os.environ.get("OPENROUTER_BASE_URL", "").strip()
            or "https://openrouter.ai/api/v1"
        )
        headers: dict[str, str] = {}
        referer = (
            str(getattr(config.credentials, "openrouter_http_referer", "") or "").strip()
            or os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        )
        title = (
            str(getattr(config.credentials, "openrouter_app_name", "") or "").strip()
            or os.environ.get("OPENROUTER_APP_NAME", "").strip()
        )
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=headers or None)
        model = OpenAIResponsesModel(model=model_name, openai_client=client)

    system = _LONG_TERM_MEMORY_SUMMARY_PROMPT.replace("{max_chars}", str(max_chars))

    human = (
        "### Previous memory (if any):\n"
        f"{previous_memory or 'None'}\n\n"
        "### Full conversation:\n"
        f"{conversation_text}\n\n"
        "### Write the updated long-term memory now."
    )
    agent = Agent(
        name="ChackMemory",
        instructions=system,
        model=model,
        model_settings=ModelSettings(),
    )
    result = Runner.run_sync(agent, human)
    content = getattr(result, "final_output", "") or ""
    content = content.strip()
    was_truncated = False
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars].rstrip()
        was_truncated = True

    _LOGGER.info(
        "Long-term memory summarization performed: backend=%s provider=%s model=%s conversation_chars=%s previous_memory_chars=%s output_chars=%s truncated=%s",
        backend_type,
        str(getattr(config.model, "provider", "") or ""),
        str(model_name or ""),
        len(conversation_text or ""),
        len(previous_memory or ""),
        len(content or ""),
        was_truncated,
    )
    log_event(
        "long_term_memory_summarized",
        payload={
            "backend": backend_type,
            "provider": str(getattr(config.model, "provider", "") or ""),
            "model": str(model_name or ""),
            "conversation_chars": int(len(conversation_text or "")),
            "previous_memory_chars": int(len(previous_memory or "")),
            "output_chars": int(len(content or "")),
            "max_chars": int(max_chars or 0),
            "truncated": bool(was_truncated),
        },
    )
    return content
