"""Tests for the grounded-search seam (#23 retrieval probe) — pure logic, no network.

The live Anthropic path is verified by the scratchpad spike; here we pin the parsing
(which turns SDK content blocks into text + sources + a definitive 'did it search' signal)
and the capability flags, all offline.
"""

from __future__ import annotations

from types import SimpleNamespace

from wine_geo.providers import (
    AnthropicProvider,
    GroundedCompletion,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    Source,
    parse_search_blocks,
)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _search():
    return SimpleNamespace(type="server_tool_use")


def _result(items):
    return SimpleNamespace(type="web_search_tool_result", content=items)


def _src(url, title):
    return SimpleNamespace(url=url, title=title)


class TestParseSearchBlocks:
    def test_extracts_text_sources_and_count(self):
        blocks = [
            _search(),
            _result([_src("https://camxwine.com", "CAM X"), _src("https://a.com", "A")]),
            _text("Here are some picks."),
        ]
        text, sources, n = parse_search_blocks(blocks)
        assert text == "Here are some picks."
        assert n == 1
        assert [s.url for s in sources] == ["https://camxwine.com", "https://a.com"]

    def test_no_search_blocks_means_answered_from_memory(self):
        text, sources, n = parse_search_blocks([_text("From memory.")])
        assert n == 0 and sources == [] and text == "From memory."

    def test_error_content_is_a_search_with_no_sources(self):
        # web_search_tool_result content is an object (not a list) on error.
        err = _result(SimpleNamespace(error_code="max_uses_exceeded"))
        text, sources, n = parse_search_blocks([_search(), err])
        assert n == 1 and sources == []

    def test_multiple_searches_counted(self):
        blocks = [_search(), _result([_src("u1", "t1")]), _search(), _result([_src("u2", "t2")])]
        _, sources, n = parse_search_blocks(blocks)
        assert n == 2 and len(sources) == 2


class TestGroundedCompletionSearched:
    def test_searched_true_with_sources(self):
        assert GroundedCompletion("t", "m", 1, 1, [Source("u", "t")], 0).searched is True

    def test_searched_true_with_search_calls_no_sources(self):
        assert GroundedCompletion("t", "m", 1, 1, [], 2).searched is True

    def test_searched_false_when_answered_from_memory(self):
        assert GroundedCompletion("t", "m", 1, 1, [], 0).searched is False


class TestCapabilityFlags:
    # Class attributes — no construction (real providers need keys/network to instantiate).
    def test_search_capable_providers(self):
        assert MockProvider.supports_search is True
        assert AnthropicProvider.supports_search is True

    def test_providers_without_search(self):
        assert OpenAIProvider.supports_search is False
        assert OllamaProvider.supports_search is False


class TestMockGrounded:
    def test_offline_grounded_completion(self):
        g = MockProvider(seed=1).complete_grounded("best value Napa cab", model="mock")
        assert g.searched is True
        assert g.sources and all(s.url for s in g.sources)
        assert g.text  # reuses the weighted mock pick
