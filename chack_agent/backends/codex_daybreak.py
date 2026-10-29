"""Opt Codex agents into Daybreak only when the signed-in account can use it.

Daybreak is OpenAI's trusted-access program for cybersecurity work. Codex
exposes it as the ``daybreak`` config key (``-c daybreak=true`` on ``codex
exec``, ``/daybreak`` in the TUI) gated by the ``cli_daybreak`` feature. When
it is on, ``codex exec`` asks the model catalog which cyber access programs the
current account may select for the model and sends ``daybreak_blue`` (or
``daybreak_red``) with the turn. An account that is not verified for the
program gets no Daybreak entry in the catalog, and ``codex exec`` then refuses
the whole run ("Daybreak is unavailable for model ...").

Therefore Daybreak is enabled per launch only after the account's model
catalog shows a Daybreak program for the agent's model. ChatGPT credentials
read the same ``/backend-api/codex/models`` catalog Codex reads; other
credentials ask ``codex app-server`` (``model/list``). Neither costs inference. The answer is cached in-process and on
disk under a SHA-256 fingerprint of the credential identity; the credential is
never stored. A run that still fails with a Daybreak rejection is retried
without it and the account is remembered as unavailable.

``CHACK_CODEX_DAYBREAK`` selects the behaviour: ``auto`` (default) probes,
``off`` never enables it, ``on`` skips the probe but still falls back.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Optional

_LOGGER = logging.getLogger("chack.codex_backend")

DAYBREAK_MODE_ENV = "CHACK_CODEX_DAYBREAK"
DAYBREAK_CACHE_PATH_ENV = "CHACK_CODEX_DAYBREAK_CACHE_PATH"
DAYBREAK_CACHE_TTL_ENV = "CHACK_CODEX_DAYBREAK_CACHE_TTL_SECONDS"
DAYBREAK_PROBE_TIMEOUT_ENV = "CHACK_CODEX_DAYBREAK_PROBE_TIMEOUT_SECONDS"
CHATGPT_BASE_URL_ENV = "CHACK_CODEX_CHATGPT_BASE_URL"
DEFAULT_CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"

# `codex exec` honours `daybreak` from 0.161; 0.162 gated it behind the
# opt-in `cli_daybreak` feature, which became default-on in 0.163.
MIN_CODEX_VERSION = (0, 162, 0)
DEFAULT_CACHE_TTL_SECONDS = 6 * 3600
DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0

# Preference order matches Codex's own `ModelAccessPrograms::daybreak()`.
_DAYBREAK_PROGRAMS = ("daybreak_blue", "daybreak_red")

_STATUS_AVAILABLE = "available"
_STATUS_UNAVAILABLE = "unavailable"

_LOCK = threading.Lock()
_MEMORY_CACHE: dict[str, dict[str, Any]] = {}
_VERSION_CACHE: dict[str, Optional[tuple[int, int, int]]] = {}


def daybreak_mode(value: str) -> str:
    """Normalize ``CHACK_CODEX_DAYBREAK`` to ``auto``, ``off`` or ``on``."""
    normalized = str(value or "").strip().lower()
    if normalized in {"0", "false", "no", "off", "disabled", "disable", "never"}:
        return "off"
    if normalized in {"1", "true", "yes", "on", "force", "always", "enabled"}:
        return "on"
    return "auto"


def daybreak_cli_args() -> list[str]:
    """Arguments that enable Daybreak for one ``codex exec`` invocation."""
    return ["--enable", "cli_daybreak", "-c", "daybreak=true"]


def _normalize_program(value: Any) -> str:
    # The app-server speaks camelCase (`daybreakBlue`); the Responses API and
    # the core protocol use snake_case (`daybreak_blue`).
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value or "").strip()).lower()
    return text.replace("-", "_")


def program_from_catalog(models: list[Any], model: str) -> str:
    """Return the Daybreak program the catalog grants for ``model``, else ``""``."""
    for entry in models or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("model", "") or entry.get("slug", "") or entry.get("id", "")) != model:
            continue
        programs = entry.get("availableAccessPrograms")
        if programs is None:
            programs = entry.get("available_access_programs")
        if not isinstance(programs, dict):
            return ""
        offered = {_normalize_program(item) for item in programs.get("cyber") or []}
        for program in _DAYBREAK_PROGRAMS:
            if program in offered:
                return program
        return ""
    return ""


def _has_access_metadata(models: list[Any]) -> bool:
    return any(
        isinstance(entry, dict)
        and isinstance(
            entry.get("availableAccessPrograms", entry.get("available_access_programs")),
            dict,
        )
        for entry in models or []
    )


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    parts = str(token or "").strip().split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        value = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def credential_fingerprint(auth_mode: str, credential: str, account_id: str = "") -> str:
    """Hash the identity behind a credential; the credential itself never leaves memory.

    ChatGPT access tokens rotate, so they are keyed by workspace and user from
    the JWT claims when present, which keeps one cache entry per verified user.
    """
    mode = str(auth_mode or "").strip().lower()
    secret = str(credential or "").strip()
    if not secret:
        return ""
    identity = ""
    if mode == "chatgpt":
        claims = _decode_jwt_claims(secret)
        auth_claims = claims.get("https://api.openai.com/auth")
        if not isinstance(auth_claims, dict):
            auth_claims = {}
        user = str(
            auth_claims.get("chatgpt_user_id", "")
            or auth_claims.get("user_id", "")
            or claims.get("sub", "")
            or ""
        ).strip()
        workspace = str(account_id or auth_claims.get("chatgpt_account_id", "") or "").strip()
        if user and workspace:
            identity = f"chatgpt-user:{workspace}:{user}"
    if not identity:
        identity = f"{mode}-secret:{secret}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def codex_version(codex_path: str) -> Optional[tuple[int, int, int]]:
    """Parse ``codex --version`` once per executable path."""
    with _LOCK:
        if codex_path in _VERSION_CACHE:
            return _VERSION_CACHE[codex_path]
    version: Optional[tuple[int, int, int]] = None
    try:
        completed = subprocess.run(
            [codex_path, "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", completed.stdout or "")
        if match:
            version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except (OSError, subprocess.SubprocessError):
        version = None
    with _LOCK:
        _VERSION_CACHE[codex_path] = version
    return version


def looks_like_daybreak_rejection(text: str) -> bool:
    """Whether a failed Codex run was refused because of the Daybreak selection."""
    normalized = str(text or "").lower()
    if "access_program" in normalized or "access program" in normalized:
        return True
    if "daybreak" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "unavailable",
            "not available",
            "isn't available",
            "turn off daybreak",
            "requires",
            "not allowed",
            "not eligible",
            "ineligible",
            "not verified",
            "unverified",
            "could not be confirmed",
            "could not be determined",
            "unauthorized",
            "forbidden",
            "denied",
            "permission",
            "not supported",
            "unsupported",
            "invalid",
            "unknown",
        )
    )


def _cache_path(env_value: Callable[[str, str], str]) -> str:
    explicit = str(env_value(DAYBREAK_CACHE_PATH_ENV, "") or "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    home_base = str(
        env_value("CHACK_CODEX_HOME_BASE", os.path.expanduser("~/.codex/chack")) or ""
    ).strip() or os.path.expanduser("~/.codex/chack")
    return os.path.join(os.path.expanduser(home_base), "daybreak_support.json")


def _cache_ttl(env_value: Callable[[str, str], str]) -> float:
    try:
        return max(0.0, float(env_value(DAYBREAK_CACHE_TTL_ENV, "") or DEFAULT_CACHE_TTL_SECONDS))
    except (TypeError, ValueError):
        return float(DEFAULT_CACHE_TTL_SECONDS)


def cache_key(fingerprint: str, model: str, version: Optional[tuple[int, int, int]]) -> str:
    version_text = ".".join(str(part) for part in version) if version else "unknown"
    raw = f"{fingerprint}|{model}|{version_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_disk_cache(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_disk_cache(path: str, key: str, entry: dict[str, Any], ttl: float) -> None:
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        data = _read_disk_cache(path)
        now = time.time()
        data = {
            name: value
            for name, value in data.items()
            if isinstance(value, dict) and now - float(value.get("checked_at", 0) or 0) < ttl
        }
        data[key] = entry
        fd, temp_path = tempfile.mkstemp(prefix=".daybreak_", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    except OSError as exc:
        _LOGGER.debug("Could not persist Daybreak support cache: %s", exc)


def _cached_entry(
    key: str,
    env_value: Callable[[str, str], str],
) -> Optional[dict[str, Any]]:
    ttl = _cache_ttl(env_value)
    now = time.time()
    with _LOCK:
        entry = _MEMORY_CACHE.get(key)
    if entry and now - float(entry.get("checked_at", 0) or 0) < ttl:
        return entry
    entry = _read_disk_cache(_cache_path(env_value)).get(key)
    if isinstance(entry, dict) and now - float(entry.get("checked_at", 0) or 0) < ttl:
        with _LOCK:
            _MEMORY_CACHE[key] = entry
        return entry
    return None


def remember(
    key: str,
    *,
    program: str,
    reason: str,
    env_value: Callable[[str, str], str],
    persist: bool = True,
) -> None:
    """Record whether Daybreak works for one credential, model and Codex version."""
    entry = {
        "status": _STATUS_AVAILABLE if program else _STATUS_UNAVAILABLE,
        "program": program,
        "reason": str(reason or "")[:300],
        "checked_at": time.time(),
    }
    with _LOCK:
        _MEMORY_CACHE[key] = entry
    if persist:
        _write_disk_cache(_cache_path(env_value), key, entry, _cache_ttl(env_value))


def probe_chatgpt_catalog(
    bearer_token: str,
    account_id: str,
    model: str,
    *,
    client_version: Optional[tuple[int, int, int]],
    base_url: str = DEFAULT_CHATGPT_BASE_URL,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> tuple[Optional[str], str]:
    """Read the caller-specific catalog Codex uses for ChatGPT sign-in.

    Same contract as :func:`probe_catalog`. A direct read avoids Codex's 5s
    catalog refresh budget, which silently degrades to the bundled catalog.
    """
    import requests

    version = ".".join(str(part) for part in client_version) if client_version else "0.0.0"
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/models",
            params={"client_version": version},
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "ChatGPT-Account-ID": account_id,
                "originator": "codex_cli_rs",
                "Accept": "application/json",
            },
            timeout=max(1.0, float(timeout_seconds)),
        )
    except Exception as exc:
        return None, f"catalog request failed: {type(exc).__name__}"
    if response.status_code in {401, 403}:
        return None, f"catalog request was not authorized (HTTP {response.status_code})"
    if response.status_code != 200:
        return None, f"catalog request failed (HTTP {response.status_code})"
    try:
        body = response.json()
    except ValueError:
        return None, "catalog response was not JSON"
    models = body.get("models") if isinstance(body, dict) else body
    if not isinstance(models, list):
        return None, "catalog response had no models"
    program = program_from_catalog(models, model)
    if program:
        return program, f"catalog offers {program} for {model}"
    if not any(
        isinstance(entry, dict) and str(entry.get("slug", "") or entry.get("model", "")) == model
        for entry in models
    ):
        # An explicit model id outside the catalog keeps Codex's automatic
        # treatment; Daybreak cannot be confirmed for it.
        return "", f"{model} is not in the account catalog"
    return "", f"catalog offers no Daybreak program for {model}"


def probe_catalog(
    codex_path: str,
    model: str,
    env: dict[str, str],
    *,
    auth_file: str = "",
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> tuple[Optional[str], str]:
    """Ask ``codex app-server`` which Daybreak program the account may select.

    Returns ``(program, reason)``: a program name when available, ``""`` when
    the account or model definitely lacks it, and ``None`` when the probe could
    not decide (treat as unavailable for this run without caching it).

    The probe runs in a throwaway ``CODEX_HOME`` holding only a copy of the
    agent's ``auth.json`` so the agent's MCP servers are never started.
    """
    probe_home = tempfile.mkdtemp(prefix="chack-daybreak-")
    os.chmod(probe_home, 0o700)
    process: Optional[subprocess.Popen[str]] = None
    try:
        if auth_file and os.path.isfile(auth_file):
            target = os.path.join(probe_home, "auth.json")
            shutil.copyfile(auth_file, target)
            os.chmod(target, 0o600)
        probe_env = dict(env)
        probe_env["CODEX_HOME"] = probe_home
        process = subprocess.Popen(
            [codex_path, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=probe_env,
        )
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))

        def _send(message: dict[str, Any]) -> None:
            assert process is not None and process.stdin is not None
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        # A reader thread keeps the deadline enforceable on every platform
        # (select() does not accept pipes on Windows).
        lines: queue.Queue[Optional[str]] = queue.Queue()

        def _read_stdout() -> None:
            assert process is not None and process.stdout is not None
            for raw in process.stdout:
                lines.put(raw)
            lines.put(None)

        threading.Thread(target=_read_stdout, daemon=True).start()

        def _response(request_id: int) -> Optional[dict[str, Any]]:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    line = lines.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    continue
                if line is None:
                    return None
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict) and message.get("id") == request_id:
                    return message

        _send(
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "chack-daybreak-probe", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
            }
        )
        if _response(1) is None:
            return None, "app-server did not initialize"
        _send({"method": "initialized"})

        _send({"id": 2, "method": "account/read", "params": {"refreshToken": False}})
        account_reply = _response(2) or {}
        account = (account_reply.get("result") or {}).get("account")
        account_type = str((account or {}).get("type", "") or "")
        if account_type not in {"chatgpt", "apiKey"}:
            return "", f"account type {account_type or 'none'} cannot use Daybreak"

        models: list[Any] = []
        cursor: Optional[str] = None
        for request_id in range(3, 23):
            params: dict[str, Any] = {"includeHidden": True}
            if cursor:
                params["cursor"] = cursor
            _send({"id": request_id, "method": "model/list", "params": params})
            reply = _response(request_id)
            if reply is None or "error" in reply:
                return None, "model/list failed"
            result = reply.get("result") or {}
            models.extend(result.get("data") or [])
            if program_from_catalog(models, model):
                break
            cursor = result.get("nextCursor")
            if not cursor:
                break
        program = program_from_catalog(models, model)
        if program:
            return program, f"catalog offers {program} for {model}"
        if not _has_access_metadata(models):
            # Codex falls back to its bundled catalog, which carries no
            # access-program metadata, when the account catalog does not load
            # within its own 5s refresh budget. That says nothing about access.
            return None, "catalog has no access-program metadata"
        return "", f"catalog offers no Daybreak program for {model}"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, f"probe failed: {type(exc).__name__}"
    finally:
        if process is not None:
            try:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass
        shutil.rmtree(probe_home, ignore_errors=True)


def resolve_program(
    *,
    codex_path: str,
    model: str,
    fingerprint: str,
    env: dict[str, str],
    env_value: Callable[[str, str], str],
    auth_file: str = "",
    chatgpt_bearer: str = "",
    chatgpt_account_id: str = "",
) -> str:
    """Return the Daybreak program to enable for this launch, or ``""``."""
    mode = daybreak_mode(env_value(DAYBREAK_MODE_ENV, "auto"))
    if mode == "off" or not model or not fingerprint:
        return ""
    version = codex_version(codex_path)
    if version is None or version < MIN_CODEX_VERSION:
        return ""
    key = cache_key(fingerprint, model, version)
    cached = _cached_entry(key, env_value)
    if cached is not None:
        return str(cached.get("program", "") or "")
    if mode == "on":
        # Forced: try it and rely on the rejection fallback to remember a "no".
        return _DAYBREAK_PROGRAMS[0]
    try:
        timeout = float(
            env_value(DAYBREAK_PROBE_TIMEOUT_ENV, "") or DEFAULT_PROBE_TIMEOUT_SECONDS
        )
    except (TypeError, ValueError):
        timeout = DEFAULT_PROBE_TIMEOUT_SECONDS
    if chatgpt_bearer and chatgpt_account_id:
        program, reason = probe_chatgpt_catalog(
            chatgpt_bearer,
            chatgpt_account_id,
            model,
            client_version=version,
            base_url=str(
                env_value(CHATGPT_BASE_URL_ENV, DEFAULT_CHATGPT_BASE_URL)
                or DEFAULT_CHATGPT_BASE_URL
            ),
            timeout_seconds=timeout,
        )
    else:
        program, reason = probe_catalog(
            codex_path,
            model,
            env,
            auth_file=auth_file,
            timeout_seconds=timeout,
        )
    if program is None:
        # Undecided probes never enable Daybreak and are only remembered for
        # this process so a transient outage does not stick for hours.
        remember(key, program="", reason=reason, env_value=env_value, persist=False)
        _LOGGER.info("Codex Daybreak left off: %s", reason)
        return ""
    remember(key, program=program, reason=reason, env_value=env_value)
    _LOGGER.info(
        "Codex Daybreak %s for model=%s: %s",
        "enabled" if program else "unavailable",
        model,
        reason,
    )
    return program


def mark_unavailable(
    *,
    codex_path: str,
    model: str,
    fingerprint: str,
    reason: str,
    env_value: Callable[[str, str], str],
) -> None:
    """Remember that a run was refused while Daybreak was on."""
    if not fingerprint or not model:
        return
    key = cache_key(fingerprint, model, codex_version(codex_path))
    remember(key, program="", reason=reason, env_value=env_value)


def reset_caches_for_tests() -> None:
    with _LOCK:
        _MEMORY_CACHE.clear()
        _VERSION_CACHE.clear()
