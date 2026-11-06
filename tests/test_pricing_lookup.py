import pytest

from chack_agent import model_aliases
from chack_agent.pricing import ModelPricing, PricingTable, estimate_cost, load_pricing, _package_pricing_path


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
