"""The model, backend and thinking effort a run actually used."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Attribute a built executor carries so a run reads what that executor runs,
# not a config that may have changed since it was built.
EXECUTOR_RUN_MODEL_ATTRIBUTE = "chack_run_model"


@dataclass(frozen=True)
class RunModel:
    """What a run was sent to.

    ``model`` is the concrete model id (never a tier alias), ``alias`` the tier
    alias it was resolved from ("" when the caller named a model directly),
    ``backend`` the resolved backend (``codex``, ``claude``, ``openrouter``...)
    and ``thinking_effort`` the level the run asked for.
    """

    model: str = ""
    alias: str = ""
    backend: str = ""
    thinking_effort: str = ""


def run_model_for_config(config: Any) -> RunModel:
    """The run model of an alias-resolved config."""
    model_cfg = getattr(config, "model", None)
    primary = str(getattr(model_cfg, "primary", "") or "").strip()
    alias = ""
    if primary and primary == str(getattr(model_cfg, "primary_alias_model", "") or "").strip():
        alias = str(getattr(model_cfg, "primary_alias", "") or "").strip()
    return RunModel(
        model=primary,
        alias=alias,
        backend=str(getattr(model_cfg, "provider", "") or "").strip().lower(),
        thinking_effort=str(
            getattr(getattr(config, "agent", None), "thinking_effort", "") or ""
        ).strip(),
    )


def tag_executor(executor: Any, config: Any) -> None:
    """Record on ``executor`` the run model of the config it was built from."""
    try:
        setattr(executor, EXECUTOR_RUN_MODEL_ATTRIBUTE, run_model_for_config(config))
    except (AttributeError, TypeError):
        pass


def executor_run_model(executor: Any) -> Optional[RunModel]:
    value = getattr(executor, EXECUTOR_RUN_MODEL_ATTRIBUTE, None)
    return value if isinstance(value, RunModel) else None
