from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests


CodexAuthUpdatedCallback = Callable[[str], None]
CodexAuthInvalidCallback = Callable[[str], None]

CODEX_CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

_codex_auth_updated_callback: Optional[CodexAuthUpdatedCallback] = None
_codex_auth_invalid_callback: Optional[CodexAuthInvalidCallback] = None


def set_codex_auth_updated_callback(callback: Optional[CodexAuthUpdatedCallback]) -> None:
    global _codex_auth_updated_callback
    _codex_auth_updated_callback = callback


def set_codex_auth_invalid_callback(callback: Optional[CodexAuthInvalidCallback]) -> None:
    global _codex_auth_invalid_callback
    _codex_auth_invalid_callback = callback


def emit_codex_auth_updated(auth_json: str) -> None:
    callback = _codex_auth_updated_callback
    if callback is None:
        return
    callback(str(auth_json or ""))


def emit_codex_auth_invalid(raw_value: str) -> None:
    callback = _codex_auth_invalid_callback
    if callback is None:
        return
    callback(str(raw_value or ""))


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) != 3:
        raise ValueError("JWT must have 3 dot-separated sections")
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise ValueError("JWT payload must be an object")
    return parsed


def is_codex_refresh_token(raw_value: Any) -> bool:
    raw_text = str(raw_value or "").strip()
    return raw_text.startswith("rt_") and "." in raw_text and "{" not in raw_text


def extract_codex_refresh_token_from_any(raw_value: Any) -> str:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return ""
    if is_codex_refresh_token(raw_text):
        return raw_text
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Codex auth payload must be a JSON object")
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError("Codex auth payload must include a tokens object")
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not is_codex_refresh_token(refresh_token):
        raise ValueError("Codex auth payload must include a valid refresh_token")
    return refresh_token


def _build_placeholder_jwt(header: dict[str, Any], payload: dict[str, Any]) -> str:
    def _b64url(value: dict[str, Any]) -> str:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("utf-8")

    return f"{_b64url(header)}.{_b64url(payload)}.sig"


def build_codex_auth_json_from_refresh_token(refresh_token: str) -> str:
    refresh_token = str(refresh_token or "").strip()
    if not is_codex_refresh_token(refresh_token):
        raise ValueError("Invalid Codex refresh token")

    id_token = _build_placeholder_jwt(
        {"alg": "RS256", "kid": "placeholder", "typ": "JWT"},
        {
            "iss": "https://auth.openai.com",
            "sub": "placeholder-sub",
            "aud": [CODEX_CHATGPT_CLIENT_ID],
            "exp": 1,
            "iat": 1,
            "email": "placeholder@example.com",
            "email_verified": True,
            "auth_provider": "google",
            "auth_time": 1,
            "jti": "placeholder-jti",
            "rat": 1,
            "sid": "placeholder-sid",
            "at_hash": "placeholder",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "placeholder-account",
                "chatgpt_plan_type": "plus",
                "chatgpt_user_id": "placeholder-user",
                "user_id": "placeholder-user",
                "groups": [],
                "organizations": [],
            },
        },
    )
    access_token = _build_placeholder_jwt(
        {"alg": "RS256", "kid": "placeholder", "typ": "JWT"},
        {
            "iss": "https://auth.openai.com",
            "sub": "placeholder-sub",
            "aud": ["https://api.openai.com/v1"],
            "exp": 1,
            "iat": 1,
            "nbf": 1,
            "jti": "placeholder-jti",
            "client_id": "placeholder-client",
            "session_id": "placeholder-session",
            "sl": "placeholder",
            "scp": ["openid", "profile", "email"],
            "pwd_auth_time": 1,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "placeholder-account",
                "chatgpt_account_user_id": "placeholder-user__placeholder-account",
                "chatgpt_compute_residency": "no_constraint",
                "chatgpt_plan_type": "plus",
                "chatgpt_user_id": "placeholder-user",
                "user_id": "placeholder-user",
            },
            "https://api.openai.com/profile": {},
        },
    )
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": access_token,
                "account_id": "placeholder-account",
                "id_token": id_token,
                "refresh_token": refresh_token,
            },
            "last_refresh": "1970-01-01T00:00:00Z",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def normalize_codex_auth_json(raw_value: Any) -> str:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return ""
    raw_text = build_codex_auth_json_from_refresh_token(
        extract_codex_refresh_token_from_any(raw_text)
    )
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Codex auth cache must be a JSON object")

    auth_mode = str(payload.get("auth_mode", "") or "").strip().lower()
    if auth_mode != "chatgpt":
        raise ValueError("Codex auth cache must use auth_mode='chatgpt'")

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError("Codex auth cache must include a tokens object")

    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    access_token = str(tokens.get("access_token", "") or "").strip()
    id_token = str(tokens.get("id_token", "") or "").strip()
    account_id = str(tokens.get("account_id", "") or "").strip()
    if not refresh_token:
        raise ValueError("Codex auth cache must include a refresh_token")
    if not access_token:
        raise ValueError("Codex auth cache must include an access_token")
    if not id_token:
        raise ValueError("Codex auth cache must include an id_token")

    normalized = {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": access_token,
            "account_id": account_id,
            "id_token": id_token,
            "refresh_token": refresh_token,
        },
        "last_refresh": str(payload.get("last_refresh", "") or ""),
    }
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _get_client_id_from_auth_json(normalized_auth_json: str) -> str:
    payload = json.loads(normalized_auth_json)
    tokens = payload.get("tokens") or {}
    id_token = str(tokens.get("id_token", "") or "").strip()
    if not id_token:
        return CODEX_CHATGPT_CLIENT_ID
    try:
        id_token_payload = _decode_jwt_payload(id_token)
    except Exception:
        return CODEX_CHATGPT_CLIENT_ID
    aud = id_token_payload.get("aud")
    if isinstance(aud, list) and aud:
        return str(aud[0] or "").strip() or CODEX_CHATGPT_CLIENT_ID
    if isinstance(aud, str) and aud.strip():
        return aud.strip()
    return CODEX_CHATGPT_CLIENT_ID


def refresh_codex_auth(raw_value: Any, *, timeout_seconds: int = 20) -> str:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        raise ValueError("Codex refresh token is required")

    normalized_auth_json = normalize_codex_auth_json(raw_text)
    payload = json.loads(normalized_auth_json)
    tokens = payload.get("tokens") or {}
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    account_id = str(tokens.get("account_id", "") or "").strip()
    if not refresh_token:
        raise ValueError("Codex auth cache must include a refresh_token")

    client_id = _get_client_id_from_auth_json(normalized_auth_json)
    response = requests.post(
        CODEX_OAUTH_TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
        timeout=timeout_seconds,
    )
    try:
        response_payload = response.json()
    except Exception:
        response_payload = {}
    if response.status_code != 200:
        error_message = (
            response_payload.get("error_description")
            or response_payload.get("error")
            or response.text
            or "Could not refresh Codex auth cache"
        )
        raise ValueError(str(error_message).strip())

    refreshed_access_token = str(response_payload.get("access_token", "") or "").strip()
    refreshed_id_token = str(response_payload.get("id_token", "") or "").strip()
    refreshed_refresh_token = str(response_payload.get("refresh_token", "") or "").strip()
    if not refreshed_access_token or not refreshed_id_token or not refreshed_refresh_token:
        raise ValueError("OAuth refresh response did not include the expected tokens")

    try:
        access_payload = _decode_jwt_payload(refreshed_access_token)
        refreshed_account_id = str(
            (
                (access_payload.get("https://api.openai.com/auth") or {}).get(
                    "chatgpt_account_id"
                )
            )
            or account_id
        ).strip()
    except Exception:
        refreshed_account_id = account_id

    refreshed_auth_json = json.dumps(
        {
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": refreshed_access_token,
                "account_id": refreshed_account_id,
                "id_token": refreshed_id_token,
                "refresh_token": refreshed_refresh_token,
            },
            "last_refresh": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return normalize_codex_auth_json(refreshed_auth_json)


def force_codex_auth_refresh(raw_value: str) -> str:
    payload = json.loads(normalize_codex_auth_json(raw_value))
    payload["last_refresh"] = "1970-01-01T00:00:00Z"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_codex_refresh_token(raw_value: str) -> str:
    payload = json.loads(normalize_codex_auth_json(raw_value))
    return str(((payload.get("tokens") or {}).get("refresh_token")) or "").strip()


def get_codex_last_refresh(raw_value: str) -> Optional[datetime]:
    try:
        payload = json.loads(normalize_codex_auth_json(raw_value))
    except Exception:
        return None
    raw_timestamp = str(payload.get("last_refresh", "") or "").strip()
    if not raw_timestamp:
        return None
    try:
        if raw_timestamp.endswith("Z"):
            raw_timestamp = raw_timestamp[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw_timestamp)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
