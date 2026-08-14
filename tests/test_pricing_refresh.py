import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("CHACK_PRICING_AUTO_REFRESH", "0")

from chack_agent import pricing
from scripts.update_openrouter_pricing import _build_yaml


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
    def test_build_yaml_defaults_cache_read_to_input_price(self) -> None:
        body = _build_yaml(
            [
                {
                    "id": "provider/model",
                    "pricing": {
                        "prompt": "0.000002",
                        "completion": "0.000004",
                    },
                }
            ]
        )

        self.assertIn("input: 2", body)
        self.assertIn("cache_read: 2", body)
        self.assertIn("output: 4", body)

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

    def test_estimate_cost_falls_back_to_claude_last_dash_as_dot(self) -> None:
        table = pricing.PricingTable(
            models={
                "anthropic/claude-sonnet-4.6": pricing.ModelPricing(
                    input=2.0,
                    cached_input=0.5,
                    output=6.0,
                )
            }
        )

        cost = pricing.estimate_cost(
            table,
            "claude-sonnet-4-6",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )

        self.assertEqual(cost, 2.0)

    def test_estimate_cost_falls_back_to_claude_last_dash_as_dot_with_provider(self) -> None:
        table = pricing.PricingTable(
            models={
                "anthropic/claude-sonnet-4.6": pricing.ModelPricing(
                    input=2.0,
                    cached_input=0.5,
                    output=6.0,
                )
            }
        )

        cost = pricing.estimate_cost(
            table,
            "anthropic/claude-sonnet-4-6",
            prompt_tokens=0,
            completion_tokens=1_000_000,
        )

        self.assertEqual(cost, 6.0)

    def test_estimate_cost_treats_cache_reads_and_writes_as_disjoint_input(self) -> None:
        table = pricing.PricingTable(
            models={
                "gpt-5.6": pricing.ModelPricing(
                    input=5.0,
                    cached_input=0.5,
                    cache_write=6.25,
                    output=30.0,
                )
            }
        )

        cost = pricing.estimate_cost(
            table,
            "gpt-5.6",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_prompt_tokens=600_000,
            cache_write_tokens=300_000,
        )

        self.assertEqual(cost, 2.675)

    def test_estimate_cost_supports_one_hour_cache_write_rate(self) -> None:
        table = pricing.PricingTable(
            models={
                "claude": pricing.ModelPricing(
                    input=1.0,
                    cached_input=0.1,
                    cache_write=1.25,
                    output=5.0,
                )
            }
        )

        cost = pricing.estimate_cost(
            table,
            "claude",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cache_write_tokens=1_000_000,
            cache_write_rate_multiplier=1.6,
        )

        self.assertEqual(cost, 2.0)


if __name__ == "__main__":
    unittest.main()
