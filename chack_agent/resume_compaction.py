from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_RESUME_COMPACTION_INSTRUCTIONS = (
    "Preserve every task-critical fact needed to continue accurately: the supplied "
    "context and functionality metadata, checks or findings already produced, prior "
    "round conclusions and notes, decisions, constraints, exact identifiers, unresolved "
    "questions, and promising areas still to inspect. Remove repetition, superseded "
    "intermediate chatter, and failed approaches."
)


@dataclass
class ResumeCompactionResult:
    backend: str
    method: str
    attempted: bool = False
    succeeded: bool = False
    duration_seconds: float = 0.0
    raw_responses: list[Any] = field(default_factory=list)
    error: str = ""

