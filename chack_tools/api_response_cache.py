"""Short-lived S3 cache for paid search API responses.

Relaunching an agent should not pay again for the same SerpAPI or Brave query.
When ``CHACK_API_CACHE_BUCKET`` is set, the paid search tools look a request up
in S3 before calling the provider and store every successful, complete response.

Environment (read once, when a tool is built; the Codex backend forwards these
names to the MCP server that runs the tools):
    CHACK_API_CACHE_BUCKET   S3 bucket (enables the cache)
    CHACK_API_CACHE_PREFIX   key prefix, e.g. one per tenant
    CHACK_API_CACHE_MAX_AGE  maximum age in seconds (capped at 16 hours)
    CHACK_API_CACHE_REGION   bucket region (defaults to the AWS SDK region)

The object key is the sha256 of the provider, method, URL, request body and a
hash of the credential, so neither keys nor headers are ever stored. Errors,
rate limits and quota or auth failures are never cached, and any S3 problem only
logs at debug level and falls back to the live call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Mapping, Optional


logger = logging.getLogger(__name__)

BUCKET_ENV = "CHACK_API_CACHE_BUCKET"
PREFIX_ENV = "CHACK_API_CACHE_PREFIX"
MAX_AGE_ENV = "CHACK_API_CACHE_MAX_AGE"
REGION_ENV = "CHACK_API_CACHE_REGION"
CACHE_ENV_VARS = (BUCKET_ENV, PREFIX_ENV, MAX_AGE_ENV, REGION_ENV)

MAX_AGE_CAP_SECONDS = 16 * 60 * 60
# Tolerate small clock differences between the task that stored an entry and
# the one reading it; anything further in the future is not trusted.
_FUTURE_SKEW_SECONDS = 300
_FORMAT_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _clean_prefix(prefix: str) -> str:
    return "/".join(part for part in str(prefix or "").split("/") if part.strip())


def _clean_provider(provider: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", str(provider or "").strip().lower()) or "provider"


class ApiResponseCache:
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        max_age_seconds: float = MAX_AGE_CAP_SECONDS,
        region: str = "",
        client: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        self.bucket = str(bucket or "").strip()
        self.prefix = _clean_prefix(prefix)
        self.max_age_seconds = max(0.0, min(float(max_age_seconds), float(MAX_AGE_CAP_SECONDS)))
        self.region = str(region or "").strip()
        self._client = client
        self._client_lock = threading.Lock()
        self._clock = clock

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> Optional["ApiResponseCache"]:
        env = os.environ if environ is None else environ
        bucket = str(env.get(BUCKET_ENV, "") or "").strip()
        if not bucket:
            return None
        try:
            max_age = float(env.get(MAX_AGE_ENV, "") or MAX_AGE_CAP_SECONDS)
        except (TypeError, ValueError):
            max_age = MAX_AGE_CAP_SECONDS
        return cls(
            bucket,
            prefix=str(env.get(PREFIX_ENV, "") or ""),
            max_age_seconds=max_age,
            region=str(env.get(REGION_ENV, "") or ""),
        )

    def object_key(self, provider: str, method: str, url: str, body: Any, credential: str) -> str:
        credential_hash = hashlib.sha256(str(credential or "").encode("utf-8")).hexdigest()
        material = _canonical({
            "provider": str(provider or ""),
            "method": str(method or "").upper(),
            "url": str(url or ""),
            "body": body,
            "credential": credential_hash,
        })
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        name = f"{_clean_provider(provider)}/{digest}.json"
        return f"{self.prefix}/{name}" if self.prefix else name

    def _s3(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import boto3
                    from botocore.config import Config

                    self._client = boto3.client(
                        "s3",
                        region_name=self.region or None,
                        config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2}),
                    )
        return self._client

    def get(self, key: str) -> Any:
        """Return the cached payload, or None on a miss, an expired entry or any S3 problem."""
        try:
            response = self._s3().get_object(Bucket=self.bucket, Key=key)
            entry = json.loads(response["Body"].read())
        except Exception as exc:  # misses arrive as NoSuchKey or AccessDenied
            logger.debug("API cache lookup skipped for %s: %s", key, type(exc).__name__)
            return None
        if not isinstance(entry, dict) or entry.get("v") != _FORMAT_VERSION or "payload" not in entry:
            return None
        try:
            age = self._clock() - float(entry.get("stored_at"))
        except (TypeError, ValueError):
            return None
        if age < -_FUTURE_SKEW_SECONDS or age >= self.max_age_seconds:
            return None
        return entry["payload"]

    def put(self, key: str, payload: Any) -> None:
        try:
            body = json.dumps(
                {"v": _FORMAT_VERSION, "stored_at": self._clock(), "payload": payload},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._s3().put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")
        except Exception as exc:
            logger.debug("API cache store skipped for %s: %s", key, type(exc).__name__)
