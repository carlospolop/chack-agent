import pytest

from chack_agent import model_aliases
from chack_agent.pricing import (
    ModelPricing,
    PricingTable,
    _package_pricing_path,
    estimate_cost,
    estimate_costs_by_model,
    load_pricing,
)


def _table(**models):
    return PricingTable(models={name: ModelPricing(*rates) for name, rates in models.items()})


def test_openrouter_routing_names_cost_the_vendor_rate():
    table = _table(**{"openai/gpt-6-sol": (5.0, 0.5, 30.0)})

    cost = estimate_cost(table, "openrouter/openai/gpt-6-sol", prompt_tokens=1_000_000, completion_tokens=100_000)

    assert cost == pytest.approx(5.0 + 3.0)


def test_unknown_openrouter_model_stays_unpriced():
    table = _table(**{"openai/gpt-6-sol": (5.0, 0.5, 30.0)})

    assert estimate_cost(table, "openrouter/openai/gpt-9", prompt_tokens=10, completion_tokens=10) is None


@pytest.mark.parametrize(
    "alias",
    ["OPENROUTER_BEST_QUALITY", "OPENROUTER_CHEAP_BUT_QUALITY", "OPENROUTER_BEST_CHEAPEST"],
)
def test_default_openrouter_models_have_a_cost(alias):
    table = load_pricing(_package_pricing_path())
    model = model_aliases.MODEL_ALIASES[alias]

    assert estimate_cost(table, model, prompt_tokens=1000, completion_tokens=1000) > 0


def test_unlisted_model_is_priced_at_its_family_top_rate_only_when_asked():
    table = _table(**{"anthropic/claude-a": (1.0, 0.1, 5.0), "claude-b": (3.0, 0.3, 15.0), "gpt-z": (90.0, 9.0, 900.0)})

    assert estimate_cost(table, "gateway/claude-next", prompt_tokens=10, completion_tokens=10) is None
    cost = estimate_cost(table, "gateway/claude-next", prompt_tokens=1_000_000, completion_tokens=0, fallback=True)
    assert cost == pytest.approx(3.0)


def test_model_of_an_unknown_family_is_priced_at_the_table_top_rate():
    table = _table(**{"claude-b": (3.0, 0.3, 15.0), "gpt-z": (90.0, 9.0, 900.0)})

    cost = estimate_cost(table, "mystery-1", prompt_tokens=0, completion_tokens=1_000_000, fallback=True)

    assert cost == pytest.approx(900.0)


def test_fallback_pricing_sums_into_per_model_costs():
    table = _table(**{"claude-b": (3.0, 0.3, 15.0)})

    total, missing = estimate_costs_by_model(
        table, {"claude-b": (1_000_000, 0, 0, 0), "claude-x": (1_000_000, 0, 0, 0)}, fallback=True
    )

    assert total == pytest.approx(6.0)
    assert missing == []

@pytest.mark.parametrize(
    ("model", "priced_as"),
    [
        # OpenRouter names the instruction-tuned Gemma "-it".
        ("bakkt/gemma-4-26b-a4b", "google/gemma-4-26b-a4b-it"),
        # Gateways often drop the "claude-" of Claude tiers.
        ("bakkt/opus-5.5", "anthropic/claude-opus-5.5"),
        ("bakkt/opus-5", "anthropic/claude-opus-5"),
        ("bakkt/opus-4.8", "anthropic/claude-opus-4.8"),
        ("bakkt/sonnet-5", "anthropic/claude-sonnet-5"),
        ("bakkt/haiku-4.5", "anthropic/claude-haiku-4.5"),
        ("corp/opus-4-8", "anthropic/claude-opus-4.8"),
        # OpenRouter's qwen3-coder is the 480B-A35B instruct model.
        ("bakkt/qwen3-coder-480b-a35b-instruct", "qwen/qwen3-coder"),
        # Names that already matched keep their price.
        ("bakkt/claude-opus-5.5", "anthropic/claude-opus-5.5"),
        ("vertex_ai/claude-opus-4-8", "anthropic/claude-opus-4.8"),
        ("bakkt/minimax-m2.5", "minimax/minimax-m2.5"),
        ("openrouter/auto", "openrouter/auto"),
        ("openrouter/openai/gpt-6-sol", "openai/gpt-6-sol"),
    ],
)
def test_gateway_model_names_resolve_to_the_openrouter_price(model, priced_as):
    table = load_pricing(_package_pricing_path())
    expected = estimate_cost(table, priced_as, prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert expected
    assert estimate_cost(table, model, prompt_tokens=1_000_000, completion_tokens=1_000_000) == expected


@pytest.mark.parametrize(
    "model",
    [
        # A gateway's own router is not OpenRouter's.
        "bakkt/auto",
        "auto",
        "corp/free",
        # Close names of other models stay unpriced.
        "bakkt/minimax-m2",
        "bakkt/minimax-m2.1",
        "bakkt/opus-9",
        "bakkt/qwen3-coder-480b",
        "bakkt/code",
    ],
)
def test_gateway_names_without_an_openrouter_model_stay_unpriced(model):
    table = load_pricing(_package_pricing_path())

    assert estimate_cost(table, model, prompt_tokens=10, completion_tokens=10) is None
