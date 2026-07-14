"""Typed open-extraction + deterministic classification (no Ollama server needed).

The spike's lesson encoded as tests: the LLM returns candidate strings, and CODE —
not the model — decides list membership. So a returned 'Bogle Cabernet' resolves to
the tracked Bogle, while an off-list winery becomes a discovery candidate. urllib is
stubbed, so these run offline like the provider tests.
"""

import json

from wine_geo.discover import KINDS, classify, open_extract
from wine_geo.extract import build_patterns


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps({"message": {"content": json.dumps(payload)}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _patch(monkeypatch, payload, captured=None):
    def fake_urlopen(req, timeout=None):
        if captured is not None:
            captured["body"] = json.loads(req.data)
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


PATTERNS = build_patterns([
    {"name": "Bogle", "aliases": []},
    {"name": "Caymus", "aliases": ["Caymus Vineyards"]},
])


def test_classify_maps_variant_to_canonical_and_flags_novel():
    known, novel = classify(["Bogle Cabernet", "Caymus Vineyards", "Cakebread Cellars"], PATTERNS)
    assert known == {"Bogle", "Caymus"}
    assert novel == ["Cakebread Cellars"]


def test_open_extract_returns_typed_buckets(monkeypatch):
    payload = {
        "producers": ["Caymus", "Cakebread Cellars"],
        "regions": ["Napa", "Moon Mountain"],
        "publications": ["Wine Spectator"],
    }
    _patch(monkeypatch, payload)
    out = open_extract("some answer text", model="llama3.1:8b")
    assert set(out) == set(KINDS)
    assert out["producers"] == ["Caymus", "Cakebread Cellars"]
    assert out["regions"] == ["Napa", "Moon Mountain"]
    assert out["publications"] == ["Wine Spectator"]


def test_open_extract_prompt_omits_any_candidate_list(monkeypatch):
    # The whole anti-hallucination premise: we must NOT inject the tracked list to echo.
    # (The system prompt does carry a few illustrative example names — that's fine; the
    # guarantee is that the reference LISTS aren't fed in. So tracked producers that
    # aren't examples must be absent, and the user turn must be only the answer text.)
    captured = {}
    _patch(monkeypatch, {"producers": []}, captured)
    open_extract("answer", model="m")
    sent = json.dumps(captured["body"])
    assert "Duckhorn" not in sent and "Silver Oak" not in sent  # tracked, not examples
    assert captured["body"]["messages"][-1]["content"] == "ANSWER:\nanswer"
    assert captured["body"]["options"]["temperature"] == 0
    assert captured["body"]["format"] == "json"


def test_open_extract_missing_and_malformed_are_safe(monkeypatch):
    _patch(monkeypatch, {"producers": ["Caymus"]})  # regions/publications absent
    out = open_extract("x", model="m")
    assert out["producers"] == ["Caymus"]
    assert out["regions"] == [] and out["publications"] == []
