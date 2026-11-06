from typing import Any, Dict, Optional

from .otel import record_event


def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None, **_context: Any) -> bool:
    """Record ``event_type`` on the current trace span.

    Returns whether the event was recorded: there must be an active span
    (inside ``Chack.run`` or a tool call) and a tracer configured.
    """
    event_type = str(event_type or "").strip().lower()
    if not event_type:
        return False
    return record_event(event_type, payload or {})
