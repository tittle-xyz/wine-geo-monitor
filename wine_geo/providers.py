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
import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class Source:
    """One retrieved web source behind a grounded answer — the retrieval evidence."""

    url: str
    title: str


@dataclass
class GroundedCompletion:
    """A completion made with web search enabled: the answer plus what it retrieved.

    `sources` is the citation trail — the pages the model actually pulled in. `searched` is
    the definitive 'did retrieval happen' signal: a declared tool is optional, so an answer
    with no sources and no searches means the model chose to answer from memory instead.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    sources: list[Source]
    num_searches: int
    stop_reason: str | None = None

    @property
    def searched(self) -> bool:
        return self.num_searches > 0 or bool(self.sources)


def parse_search_blocks(blocks) -> tuple[str, list[Source], int]:
    """Split a grounded response's content into (answer text, sources, search-call count).

    Duck-types the SDK content blocks so it's testable without the SDK: a `text` block is
    answer text, a `server_tool_use` block is one search the model issued, and a
    `web_search_tool_result` block carries a *list* of results on success (or a single
    error object, e.g. max_uses_exceeded — counted as a search with no usable sources).
    """
    text_parts: list[str] = []
    sources: list[Source] = []
    num_searches = 0
    for b in blocks:
        bt = getattr(b, "type", None)
        if bt == "text":
            text_parts.append(getattr(b, "text", "") or "")
        elif bt == "server_tool_use":
            num_searches += 1
        elif bt == "web_search_tool_result":
            content = getattr(b, "content", None)
            if isinstance(content, list):
                for r in content:
                    sources.append(Source(url=getattr(r, "url", "") or "",
                                          title=getattr(r, "title", "") or ""))
    return "".join(text_parts), sources, num_searches


def parse_openai_search(output_items, output_text) -> tuple[str, list[Source], int]:
    """Normalize OpenAI Responses API output into (text, sources, search-call count).

    Duck-typed for testability: `output_items` is resp.output (each item has `.type`, and
    message items carry `.content[].annotations[]` url_citations with `.url`/`.title`);
    `output_text` is resp.output_text. Same target shape as parse_search_blocks — each
    provider normalizes its own wire format to the common one, per ADR-0005.
    """
    sources: list[Source] = []
    num_searches = 0
    for it in output_items:
        if "search" in (getattr(it, "type", "") or ""):
            num_searches += 1
        for c in getattr(it, "content", None) or []:
            for a in getattr(c, "annotations", None) or []:
                url = getattr(a, "url", "") or ""
                if url:
                    sources.append(Source(url=url, title=getattr(a, "title", "") or ""))
    return output_text or "", sources, num_searches


@dataclass
class BatchRequest:
    """One prompt to fulfill via a provider's batch API, tagged with a custom_id.

    The custom_id is how a batch result maps back to its request (results come back
    unordered). Keep it to `^[a-zA-Z0-9_-]{1,64}$` — Anthropic enforces that shape.
    """

    custom_id: str
    prompt: str
    model: str


# (input, output) USD per 1,000,000 tokens. Rough public list prices — they drift,
# which is exactly why cost tracking belongs in the tool, not a spreadsheet.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-mini": (0.25, 2.00),   # GPT-5 mini tier; estimate — confirm on the pricing page
    "gpt-4o": (2.50, 10.0),
    "mock": (0.0, 0.0),
}

# Both the Anthropic Message Batches API and the OpenAI Batch API bill at half list
# price. That's the whole trade #9 is about: latency for cost.
BATCH_DISCOUNT = 0.5

# Answers come back in ~400 tokens; keep sync and batch paths on the same ceiling.
MAX_TOKENS = 400
# gpt-5 / o-series reasoning models spend hidden reasoning tokens before the visible answer,
# so their completion budget needs headroom above the terse ceiling or the answer comes back empty.
REASONING_MAX_TOKENS = 2000

# Web search is a paid, separately-billed server tool. The basic variant works on Haiku;
# newer models can use web_search_20260209 (dynamic filtering), but the basic one returns
# the same citations, which is all the #23 retrieval probe needs.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}
# Grounded answers carry search results + citations, so give them more room than the terse
# ungrounded ceiling.
GROUNDED_MAX_TOKENS = 1024


def estimate_cost(
    input_tokens: int, output_tokens: int, model: str, *, batch: bool = False
) -> float:
    """USD for one call. Unknown models cost 0 rather than crashing a long run.

    `batch=True` applies the batch-API discount, so the cost pipeline reflects what a
    batched run actually costs, not list price.
    """
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, model: str) -> Completion: ...


# Batch is an *optional* capability: a provider advertises it with `supports_batch =
# True` and a `complete_batch`. The collector checks the flag and falls back to the
# synchronous path when it's absent, so a new provider needs only `complete` to work.
class BatchProvider(Protocol):
    name: str
    supports_batch: bool

    def complete_batch(
        self, requests: list[BatchRequest], *, poll_interval: float, timeout: float, log=None
    ) -> dict[str, Completion]:
        """Submit all requests as one job, block until done, return {custom_id: Completion}.

        Only successful results are returned; the caller treats any missing custom_id
        as a failed sample. Raises TimeoutError / RuntimeError on a batch that never
        finishes or ends in a terminal error state.
        """
        ...


# Web search is another optional capability, advertised with `supports_search = True` and a
# `complete_grounded`. Same pattern as batch: a provider without it just can't run the
# retrieval probe. `force=True` uses tool_choice to guarantee a search (so the probe isn't
# silently answered from memory); `force=False` measures whether the model searches on its own.
class SearchProvider(Protocol):
    name: str
    supports_search: bool

    def complete_grounded(
        self, prompt: str, *, model: str, force: bool = True
    ) -> GroundedCompletion: ...


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
    """Seeded stand-in that mimics a stochastic model.

    **What a seed reproduces:** the sampled *distribution* — and therefore every metric
    derived from it (share-of-voice, confidence intervals, Jaccard), exactly. Stamps the
    requested model onto each Completion so cost tracking shows real numbers even in mock
    mode (default model is a cheap tier).

    **What it does not:** which `sample_index` gets which response. `collect()` samples
    concurrently, so the order threads take their draws is scheduling-dependent — the
    *bag* of responses is stable, its order isn't. Pass `concurrency=1` for byte-identical
    raw output.
    """

    name = "mock"
    supports_batch = True
    supports_search = True

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def complete_batch(self, requests, *, poll_interval=0.0, timeout=0.0, log=None):
        """Offline batch: fulfill each request in order, no network, no waiting.

        Exercises the whole batch code path (custom_id round-trip, result mapping,
        the batch billing tier) in CI with no API key.
        """
        if log:
            log(f"mock batch: fulfilling {len(requests)} requests offline")
        return {r.custom_id: self.complete(r.prompt, model=r.model) for r in requests}

    def complete(self, prompt: str, *, model: str) -> Completion:
        names = list(_MOCK_WEIGHTS)
        weights = list(_MOCK_WEIGHTS.values())
        # One call = one contiguous block of the RNG sequence. runner.py samples via
        # asyncio.to_thread, so complete() runs on several worker threads sharing this
        # generator; holding the lock across every draw keeps a call's draws contiguous,
        # which is what makes the sampled distribution reproducible for a seed. Without it
        # that holds only because the GIL rarely switches inside these short calls — luck,
        # not a guarantee, and 3.13's scheduling breaks it.
        with self._lock:
            k = self._rng.choice([2, 3, 3, 4])
            picks = _weighted_sample(self._rng, names, weights, k)
            template = self._rng.choice(_MOCK_TEMPLATES)
        # Pad the template if we drew fewer than 3.
        while len(picks) < 3:
            picks.append(picks[-1])
        text = template.format(a=picks[0], b=picks[1], c=picks[2])
        return Completion(
            text=text,
            model=model,
            input_tokens=max(1, int(len(prompt.split()) * 1.3)),
            output_tokens=max(1, int(len(text.split()) * 1.3)),
        )

    def complete_grounded(
        self, prompt: str, *, model: str, force: bool = True
    ) -> GroundedCompletion:
        """Offline stand-in: reuse the weighted pick for text, return canned sources.

        Lets the grounded path — and anything built on it — run in tests with no key and no
        network, the same way `complete` does for the ungrounded path.
        """
        c = self.complete(prompt, model=model)
        sources = [
            Source(url="https://example.com/napa-value-picks", title="Best value Napa Cabs"),
            Source(url="https://example.com/under-40", title="Napa Cabernet under $40"),
        ]
        return GroundedCompletion(
            text=c.text, model=c.model,
            input_tokens=c.input_tokens, output_tokens=c.output_tokens,
            sources=sources, num_searches=1, stop_reason="end_turn",
        )


# --- Real providers -------------------------------------------------------


class AnthropicProvider:
    name = "anthropic"
    supports_batch = True
    supports_search = True

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError("Anthropic provider needs: pip install anthropic") from e
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete_grounded(  # pragma: no cover - network
        self, prompt: str, *, model: str, force: bool = True
    ) -> GroundedCompletion:
        """Answer with the web search tool enabled; return the answer + retrieved sources.

        `force=True` sets tool_choice so the model must search (verified: forcing works on
        this server tool), keeping the retrieval condition clean rather than letting the
        model silently answer from memory. The citation trail comes back as
        `web_search_tool_result` blocks in the same response — one call, no loop. A long
        agentic search could stop with 'pause_turn'; that's surfaced in stop_reason, not
        handled, since the monitor's prompts are single-shot.
        """
        kwargs = {
            "model": model,
            "max_tokens": GROUNDED_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [WEB_SEARCH_TOOL],
        }
        if force:
            kwargs["tool_choice"] = {"type": "any"}
        resp = self._client.messages.create(**kwargs)
        text, sources, num = parse_search_blocks(resp.content)
        return GroundedCompletion(
            text=text, model=model,
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
            sources=sources, num_searches=num, stop_reason=resp.stop_reason,
        )

    def complete(self, prompt: str, *, model: str) -> Completion:
        # NOTE: current Anthropic models (Opus 4.8/4.7, Sonnet 5, Fable 5) reject a
        # non-default `temperature`, so we don't send one. Run-to-run variance comes
        # from the model's inherent sampling — which is precisely the thing a GEO
        # monitor exists to measure, so we want it, not a turned-up temperature.
        resp = self._client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Completion(
            text=text,
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    def complete_batch(  # pragma: no cover - env dependent
        self, requests, *, poll_interval=15.0, timeout=21600.0, log=None
    ):
        """Message Batches API: submit all prompts as one job (~50% off), poll, retrieve.

        Most batches finish in well under an hour; the daily monitoring job is tiny.
        Only succeeded results come back — the collector marks any missing custom_id
        as an errored sample.
        """
        import time

        req_model = {r.custom_id: r.model for r in requests}
        batch = self._client.messages.batches.create(
            requests=[
                {
                    "custom_id": r.custom_id,
                    "params": {
                        "model": r.model,
                        "max_tokens": MAX_TOKENS,
                        "messages": [{"role": "user", "content": r.prompt}],
                    },
                }
                for r in requests
            ]
        )
        deadline = time.monotonic() + timeout
        while True:
            current = self._client.messages.batches.retrieve(batch.id)
            if current.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Anthropic batch {batch.id} still {current.processing_status} "
                    f"after {timeout:.0f}s"
                )
            if log:
                counts = getattr(current, "request_counts", "")
                log(f"anthropic batch {batch.id}: {current.processing_status} {counts}")
            time.sleep(poll_interval)

        out: dict[str, Completion] = {}
        for result in self._client.messages.batches.results(batch.id):
            if result.result.type != "succeeded":
                if log:
                    log(f"anthropic batch {result.custom_id}: {result.result.type}")
                continue
            msg = result.result.message
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            out[result.custom_id] = Completion(
                text=text,
                model=req_model.get(result.custom_id, msg.model),
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
            )
        return out


def _openai_token_budget(model: str) -> dict:
    """The output-budget kwargs for an OpenAI chat completion, keyed by model family.

    gpt-5 / o-series reasoning models reject `max_tokens` (they take `max_completion_tokens`) and
    burn hidden reasoning tokens before the visible answer, so they need more headroom. Pure and
    isolated so the branch a live run had to correct stays regression-tested without a network call.
    """
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": REASONING_MAX_TOKENS}
    return {"max_tokens": MAX_TOKENS}


class OpenAIProvider:
    name = "openai"
    supports_batch = True
    supports_search = True

    def __init__(self) -> None:
        try:
            import openai
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError("OpenAI provider needs: pip install openai") from e
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY

    def complete(self, prompt: str, *, model: str) -> Completion:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **_openai_token_budget(model),
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return Completion(
            text=text,
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    def complete_grounded(  # pragma: no cover - network
        self, prompt: str, *, model: str, force: bool = True
    ) -> GroundedCompletion:
        """Grounded via the Responses API web_search tool; normalize to GroundedCompletion.

        A different endpoint from the ungrounded chat-completions path, but the same
        capability and the same normalized shape — the seam hides the vendor difference
        (ADR-0005). `force=True` sets tool_choice so the model must search.
        """
        kwargs = {"model": model, "tools": [{"type": "web_search"}], "input": prompt}
        if force:
            kwargs["tool_choice"] = {"type": "web_search"}
        resp = self._client.responses.create(**kwargs)
        text, sources, num = parse_openai_search(resp.output, getattr(resp, "output_text", ""))
        usage = resp.usage
        return GroundedCompletion(
            text=text, model=model,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            sources=sources, num_searches=num, stop_reason=getattr(resp, "status", None),
        )

    def complete_batch(  # pragma: no cover - env dependent
        self, requests, *, poll_interval=15.0, timeout=21600.0, log=None
    ):
        """Batch API: upload a JSONL of chat requests, create the batch (~50% off), poll, download.

        Only succeeded rows come back; the collector marks any missing custom_id as an
        errored sample.
        """
        import io
        import json
        import time

        req_model = {r.custom_id: r.model for r in requests}
        payload = "\n".join(
            json.dumps({
                "custom_id": r.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": r.model,
                    "max_tokens": MAX_TOKENS,
                    "messages": [{"role": "user", "content": r.prompt}],
                },
            })
            for r in requests
        )
        upload = self._client.files.create(
            file=("batch.jsonl", io.BytesIO(payload.encode("utf-8"))), purpose="batch"
        )
        batch = self._client.batches.create(
            input_file_id=upload.id, endpoint="/v1/chat/completions", completion_window="24h"
        )
        deadline = time.monotonic() + timeout
        while True:
            current = self._client.batches.retrieve(batch.id)
            if current.status == "completed":
                break
            if current.status in ("failed", "expired", "cancelled", "cancelling"):
                raise RuntimeError(f"OpenAI batch {batch.id} ended in status {current.status}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"OpenAI batch {batch.id} still {current.status} after {timeout:.0f}s"
                )
            if log:
                counts = getattr(current, "request_counts", "")
                log(f"openai batch {batch.id}: {current.status} {counts}")
            time.sleep(poll_interval)

        out: dict[str, Completion] = {}
        body_text = self._client.files.content(current.output_file_id).text
        for line in body_text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id")
            resp = row.get("response") or {}
            body = resp.get("body") or {}
            if resp.get("status_code") != 200 or not body.get("choices"):
                if log:
                    log(f"openai batch {cid}: status {resp.get('status_code')}")
                continue
            usage = body.get("usage") or {}
            out[cid] = Completion(
                text=body["choices"][0]["message"].get("content") or "",
                model=req_model.get(cid, ""),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        return out


class OllamaProvider:
    """A real model running locally via Ollama — no API key, no network, no cost.

    Sits between `mock` and the paid providers: the mock is deterministic canned
    text, the paid providers cost money and need keys, and this is an actual LLM
    you can run offline for free. Handy for developing against real (stochastic)
    output without spending anything, and a clean demonstration that the Provider
    seam doesn't care whether the model is remote or on your laptop.

    Talks to Ollama's native HTTP endpoint with the standard library only, so it
    adds no dependency — in keeping with the stdlib-only core (ADR-0003). Point it
    at a non-default host with OLLAMA_HOST. There's no batch API: local generation
    has no batch discount to chase, so the collector falls back to the sync path.
    """

    name = "ollama"
    supports_search = False  # local model, no web access

    def __init__(self) -> None:
        import os

        self._host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def complete(self, prompt: str, *, model: str) -> Completion:
        import json
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": MAX_TOKENS},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                f"Ollama request to {self._host} failed ({e}). Is `ollama serve` running "
                f"and is the model pulled (`ollama pull {model}`)?"
            ) from e

        # Ollama reports real token counts; cost stays $0 since it's not in PRICING.
        return Completion(
            text=data.get("message", {}).get("content", ""),
            model=model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )


def get_provider(name: str, *, seed: int | None = None) -> Provider:
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"unknown provider {name!r} (choose: mock, anthropic, openai, ollama)")
