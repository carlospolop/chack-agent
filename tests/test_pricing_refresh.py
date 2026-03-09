import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("CHACK_PRICING_AUTO_REFRESH", "0")

from chack_agent import pricing


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class PricingRefreshTests(unittest.TestCase):
    def test_refresh_downloads_remote_yaml_into_cache_and_prefers_it(self) -> None:
        remote_yaml = b"models:\n  provider/model:\n    input: 1\n    cache_read: 0.1\n    output: 2\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "XDG_CACHE_HOME": tmpdir,
                "CHACK_PRICING_AUTO_REFRESH": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch.object(
                    pricing,
                    "urlopen",
                    return_value=_FakeResponse(
                        remote_yaml,
                        headers={
                            "ETag": '"abc123"',
                            "Last-Modified": "Mon, 10 Mar 2025 10:00:00 GMT",
                        },
                    ),
                ) as mocked_urlopen:
                    resolved_path = pricing.refresh_pricing_from_github_if_newer()

                self.assertEqual(mocked_urlopen.call_count, 1)
                self.assertTrue(resolved_path.endswith("pricing/pricing.yaml"))
                self.assertEqual(
                    pricing._cached_pricing_path().read_text(encoding="utf-8"),
                    remote_yaml.decode("utf-8"),
                )
                self.assertEqual(
                    pricing._load_pricing_metadata(),
                    {
                        "etag": '"abc123"',
                        "last_modified": "Mon, 10 Mar 2025 10:00:00 GMT",
                    },
                )

                loaded = pricing.load_pricing(pricing.resolve_pricing_path(refresh=False))
                self.assertIn("provider/model", loaded.models)
                self.assertEqual(loaded.models["provider/model"].input, 1.0)

    def test_resolve_pricing_path_honors_override_without_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = os.path.join(tmpdir, "custom-pricing.yaml")
            with open(override_path, "w", encoding="utf-8") as handle:
                handle.write("models: {}\n")

            with patch.dict(
                os.environ,
                {
                    "CHACK_PRICING": override_path,
                    "CHACK_PRICING_AUTO_REFRESH": "1",
                },
                clear=False,
            ):
                with patch.object(pricing, "urlopen") as mocked_urlopen:
                    resolved_path = pricing.resolve_pricing_path()

                self.assertEqual(resolved_path, override_path)
                mocked_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
