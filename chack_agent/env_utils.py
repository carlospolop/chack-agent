import base64
import os
from typing import Optional


def _resolve_path(base_dir: str, value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _write_aws_profiles(profiles: dict) -> Optional[str]:
    if not profiles:
        return None
    home_dir = os.path.expanduser("~")
    aws_dir = os.path.join(home_dir, ".aws")
    try:
        os.makedirs(aws_dir, exist_ok=True)
    except OSError:
        aws_dir = os.path.join("/tmp", "chack-aws")
        os.makedirs(aws_dir, exist_ok=True)
    creds_path = os.path.join(aws_dir, "credentials")
    config_path = os.path.join(aws_dir, "config")

    with open(creds_path, "w", encoding="utf-8") as handle:
        for name, values in profiles.items():
            if not isinstance(values, dict):
                continue
            access_key = values.get("aws_access_key_id", "")
            secret_key = values.get("aws_secret_access_key", "")
            if not access_key or not secret_key:
                continue
            handle.write(f"[{name}]\n")
            handle.write(f"aws_access_key_id = {access_key}\n")
            handle.write(f"aws_secret_access_key = {secret_key}\n\n")

    with open(config_path, "w", encoding="utf-8") as handle:
        for name, values in profiles.items():
            if not isinstance(values, dict):
                continue
            region = values.get("aws_region", "") or values.get("region", "")
            if not region:
                continue
            profile_name = "default" if name == "default" else f"profile {name}"
            handle.write(f"[{profile_name}]\n")
            handle.write(f"region = {region}\n\n")

    return aws_dir


def _ensure_aws_dir() -> Optional[str]:
    home_dir = os.path.expanduser("~")
    aws_dir = os.path.join(home_dir, ".aws")
    try:
        os.makedirs(aws_dir, exist_ok=True)
    except OSError:
        aws_dir = os.path.join("/tmp", "chack-aws")
        os.makedirs(aws_dir, exist_ok=True)
    return aws_dir


def export_env(config, config_path: str) -> None:
    base_dir = os.path.dirname(os.path.abspath(config_path))
    for key, value in (config.env or {}).items():
        if value is None:
            continue
        os.environ[str(key)] = str(value)

    aws_profiles_b64 = os.environ.get("CHACK_AWS_PROFILES", "").strip()
    if aws_profiles_b64:
        try:
            decoded = base64.b64decode(aws_profiles_b64).decode("utf-8")
        except Exception:
            decoded = ""
        if decoded.strip():
            aws_dir = _ensure_aws_dir()
            if aws_dir:
                creds_path = os.path.join(aws_dir, "credentials")
                with open(creds_path, "w", encoding="utf-8") as handle:
                    handle.write(decoded.strip() + "\n")
                os.environ["AWS_SHARED_CREDENTIALS_FILE"] = creds_path

    gcp_credentials_path = os.environ.get("GCP_CREDENTIALS_PATH", "").strip()
    if gcp_credentials_path and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        candidate = _resolve_path(base_dir, gcp_credentials_path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = candidate
    gcp_quota_project = os.environ.get("GCP_QUOTA_PROJECT", "").strip()
    if gcp_quota_project and not os.environ.get("GOOGLE_CLOUD_CPP_USER_PROJECT"):
        os.environ["GOOGLE_CLOUD_CPP_USER_PROJECT"] = gcp_quota_project

    os.environ["CHACK_EXEC_TIMEOUT"] = str(config.tools.exec_timeout_seconds)
    os.environ["CHACK_EXEC_MAX_OUTPUT"] = str(config.tools.exec_max_output_chars)
    exec_cwd = str(getattr(config.tools, "exec_cwd", "") or "").strip()
    if exec_cwd:
        os.environ["CHACK_EXEC_CWD"] = _resolve_path(base_dir, exec_cwd)
