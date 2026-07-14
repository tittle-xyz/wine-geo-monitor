"""The local Ollama provider — request shape and reply parsing, no server needed.

A real local model is network-bound and env-dependent, so these tests stub urllib
to prove the two things that are ours to get right: the request we send Ollama
(endpoint, model, messages, token ceiling) and how we read its reply (content plus
the real token counts it reports) back into a Completion. Cost stays $0 because a
local model isn't in PRICING — the same seam that prices the paid providers.
"""

import json

from wine_geo.providers import (
    MAX_TOKENS,
    Completion,
    OllamaProvider,
    estimate_cost,
    get_provider,
)


class _FakeResponse:
    """Stand-in for the urlopen context manager: `with urlopen(...) as r: r.read()`."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _patch_urlopen(monkeypatch, payload, captured):
    """Replace urllib's urlopen with a stub that records the request and returns payload."""

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_get_provider_returns_ollama():
    assert isinstance(get_provider("ollama"), OllamaProvider)


def test_ollama_parses_content_and_real_token_counts(monkeypatch):
    payload = {
        "message": {"role": "assistant", "content": "Try Silver Oak or Caymus."},
        "prompt_eval_count": 12,
        "eval_count": 34,
    }
    _patch_urlopen(monkeypatch, payload, {})

    c = OllamaProvider().complete("best napa cab?", model="llama3.1:8b")

    assert isinstance(c, Completion)
    assert c.text == "Try Silver Oak or Caymus."
    assert c.model == "llama3.1:8b"
    assert (c.input_tokens, c.output_tokens) == (12, 34)
    # A local model isn't in PRICING, so the run is genuinely free.
    assert estimate_cost(c.input_tokens, c.output_tokens, c.model) == 0.0


def test_ollama_sends_expected_request(monkeypatch):
    captured = {}
    _patch_urlopen(monkeypatch, {"message": {"content": "ok"}}, captured)

    OllamaProvider().complete("value cab under $30?", model="llama3.2:3b")

    assert captured["url"].endswith("/api/chat")
    body = captured["body"]
    assert body["model"] == "llama3.2:3b"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "value cab under $30?"}]
    assert body["options"]["num_predict"] == MAX_TOKENS


def test_ollama_missing_counts_default_to_zero(monkeypatch):
    # An older or edge Ollama reply without token counts must not crash a run.
    _patch_urlopen(monkeypatch, {"message": {"content": "hi"}}, {})

    c = OllamaProvider().complete("hi", model="llama3.1:8b")

    assert (c.input_tokens, c.output_tokens) == (0, 0)


def test_ollama_honors_host_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://box:1234/")  # trailing slash is trimmed
    captured = {}
    _patch_urlopen(monkeypatch, {"message": {"content": "ok"}}, captured)

    OllamaProvider().complete("x", model="m")

    assert captured["url"] == "http://box:1234/api/chat"
