"""LLM providers behind one interface.

The `Provider` protocol is the whole point of this file: the runner never knows
which model vendor it's talking to. That's a working miniature of the
"multi-provider abstraction layer" a GEO monitor needs so it can survive one
vendor repricing or deprecating a model without a rewrite.

`MockProvider` lets the whole tool run — and the tests pass — with no API key and
no network. `AnthropicProvider` / `OpenAIProvider` are the real thing behind the
same seam.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


# (input, output) USD per 1,000,000 tokens. Rough public list prices — they drift,
# which is exactly why cost tracking belongs in the tool, not a spreadsheet.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "mock": (0.0, 0.0),
}


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """USD for one call. Unknown models cost 0 rather than crashing a long run."""
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, model: str) -> Completion: ...


# --- Mock -----------------------------------------------------------------

# Weights are deliberately lopsided: the big marketing-driven brands dominate and
# the négociant / value plays barely surface. That's the interesting GEO finding
# to reproduce — AI shopping answers cluster on a handful of famous names.
_MOCK_WEIGHTS: dict[str, float] = {
    "Caymus": 10,
    "Silver Oak": 9,
    "Duckhorn": 7,
    "Josh Cellars": 6,
    "Stag's Leap Wine Cellars": 6,
    "Meiomi": 5,
    "Decoy": 4,
    "Justin": 4,
    "Ridge": 4,
    "Chateau Montelena": 3,
    "Frog's Leap": 3,
    "Charles Krug": 2,
    "Heitz": 2,
    "De Negoce": 1,
    "The Negociant Wine Company": 1,
}

_MOCK_TEMPLATES = [
    "For that I'd start with {a}, then look at {b} and {c}.",
    "A few solid picks: {a}, {b}, and {c}. {a} is the crowd favorite.",
    "You can't go wrong with {a} or {b}; {c} is a good value alternative.",
    "Popular choices here are {a}, {b}, and {c}.",
]


def _weighted_sample(
    rng: random.Random, items: list[str], weights: list[float], k: int
) -> list[str]:
    """Weighted sample WITHOUT replacement (random.sample doesn't do weights)."""
    items = list(items)
    weights = list(weights)
    picks: list[str] = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        r = rng.uniform(0, total)
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                picks.append(items.pop(i))
                weights.pop(i)
                break
    return picks


class MockProvider:
    """Deterministic-with-a-seed stand-in that mimics a stochastic model.

    Stamps the requested model onto each Completion so cost tracking shows real
    numbers even in mock mode (default model is a cheap tier).
    """

    name = "mock"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def complete(self, prompt: str, *, model: str) -> Completion:
        names = list(_MOCK_WEIGHTS)
        weights = list(_MOCK_WEIGHTS.values())
        k = self._rng.choice([2, 3, 3, 4])
        picks = _weighted_sample(self._rng, names, weights, k)
        # Pad the template if we drew fewer than 3.
        while len(picks) < 3:
            picks.append(picks[-1])
        template = self._rng.choice(_MOCK_TEMPLATES)
        text = template.format(a=picks[0], b=picks[1], c=picks[2])
        return Completion(
            text=text,
            model=model,
            input_tokens=max(1, int(len(prompt.split()) * 1.3)),
            output_tokens=max(1, int(len(text.split()) * 1.3)),
        )


# --- Real providers -------------------------------------------------------


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError("Anthropic provider needs: pip install anthropic") from e
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, prompt: str, *, model: str) -> Completion:
        # NOTE: current Anthropic models (Opus 4.8/4.7, Sonnet 5, Fable 5) reject a
        # non-default `temperature`, so we don't send one. Run-to-run variance comes
        # from the model's inherent sampling — which is precisely the thing a GEO
        # monitor exists to measure, so we want it, not a turned-up temperature.
        resp = self._client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Completion(
            text=text,
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        try:
            import openai
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError("OpenAI provider needs: pip install openai") from e
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY

    def complete(self, prompt: str, *, model: str) -> Completion:
        resp = self._client.chat.completions.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return Completion(
            text=text,
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )


def get_provider(name: str, *, seed: int | None = None) -> Provider:
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"unknown provider {name!r} (choose: mock, anthropic, openai)")
