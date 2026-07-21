"""Tests for the prominence proxy (#23) — pure logic, canned candidates, no network."""

from __future__ import annotations

from wine_geo.prominence import (
    domain_match,
    parse_enwiki_sitelink,
    parse_length,
    parse_pageviews,
    parse_wikidata_search,
    prior_with_prominence,
    prominence_level,
    resolve,
)

WINE_TERMS = ["wine", "winery", "vineyard", "négociant", "cabernet"]

# The winery we mean...
WINERY = {
    "title": "Caymus Vineyards", "description": "Napa Valley winery",
    "summary": "Caymus is a winery producing Cabernet Sauvignon in Rutherford.",
    "categories": ["Napa Valley wineries"], "summary_len": 3200, "pageviews": 45000,
}
# ...vs a same-named other thing.
NAMESAKE = {
    "title": "Cameron Hughes (footballer)", "description": "English football player",
    "summary": "A midfielder who played in the lower leagues.",
    "categories": ["Living people", "Footballers"], "summary_len": 900, "pageviews": 3000,
}


class TestDomainMatch:
    def test_in_domain_matches(self):
        assert domain_match(WINERY, WINE_TERMS) is True

    def test_out_of_domain_rejected(self):
        assert domain_match(NAMESAKE, WINE_TERMS) is False

    def test_min_hits_raises_the_bar(self):
        # Only one wine term present ("wine"); require two.
        thin = {"summary": "a wine brand", "categories": [], "description": ""}
        assert domain_match(thin, WINE_TERMS, min_hits=1) is True
        assert domain_match(thin, WINE_TERMS, min_hits=2) is False


class TestResolve:
    def test_picks_the_in_domain_entity_over_a_namesake(self):
        assert resolve([NAMESAKE, WINERY], WINE_TERMS)["title"] == "Caymus Vineyards"

    def test_none_when_no_candidate_is_in_domain(self):
        assert resolve([NAMESAKE], WINE_TERMS) is None

    def test_prefers_more_read_candidate(self):
        small = {**WINERY, "title": "Tiny Winery", "pageviews": 100, "summary_len": 400}
        big = {**WINERY, "title": "Big Winery", "pageviews": 90000, "summary_len": 5000}
        assert resolve([small, big], WINE_TERMS)["title"] == "Big Winery"


class TestProminenceLevel:
    # Thresholds from live data: Caymus 960 pv/mo=prominent, Cameron Hughes 273=established.
    def test_absent_when_unresolved(self):
        assert prominence_level(None) == "absent"

    def test_absent_when_no_article(self):
        assert prominence_level({"pageviews": 0, "summary_len": 0}) == "absent"

    def test_prominent(self):
        assert prominence_level(WINERY) == "prominent"  # 45000 pv

    def test_established_midrange(self):
        # Cameron Hughes territory: has an article, moderate traffic.
        assert prominence_level({"pageviews": 273, "summary_len": 5371}) == "established"

    def test_thin_has_article_but_no_readers(self):
        assert prominence_level({"pageviews": 50, "summary_len": 2000}) == "thin"


class TestPriorWithProminence:
    def test_too_new_stays_unknown_regardless(self):
        assert prior_with_prominence("likely-unknown", "prominent") == "likely-unknown"

    def test_predates_and_prominent_is_known(self):
        assert prior_with_prominence("likely-known", "prominent") == "likely-known"

    def test_predates_but_thin_becomes_likely_thin(self):
        # The Cameron Hughes cell date alone gets wrong.
        assert prior_with_prominence("likely-known", "thin") == "likely-thin"

    def test_predates_but_absent_footprint_is_only_likely_thin(self):
        # Wikipedia absence is weak negative evidence (De Négoce is known yet has no article),
        # so it softens to 'worth a probe', never overrides date to 'unknown'.
        assert prior_with_prominence("borderline", "absent") == "likely-thin"

    def test_unknown_inputs_pass_through(self):
        assert prior_with_prominence("unknown-cutoff", "prominent") == "unknown-cutoff"


class TestParsers:
    """Parse real API shapes, captured from live responses."""

    def test_wikidata_search(self):
        obj = {"search": [{"id": "Q1", "label": "Cameron Hughes Wine",
                           "description": "American wine négociant", "extra": "ignored"}]}
        out = parse_wikidata_search(obj)
        assert out == [{"id": "Q1", "label": "Cameron Hughes Wine",
                        "description": "American wine négociant"}]

    def test_length(self):
        assert parse_length({"query": {"pages": {"123": {"title": "X", "length": 5137}}}}) == 5137

    def test_length_missing_page_is_zero(self):
        assert parse_length({"query": {"pages": {"-1": {"title": "X", "missing": ""}}}}) == 0

    def test_pageviews_median_ignores_partial_month(self):
        # [969, 961, 24] -> median 961, so a partial current month can't drag the figure.
        obj = {"items": [{"views": 969}, {"views": 961}, {"views": 24}]}
        assert parse_pageviews(obj) == 961

    def test_pageviews_empty_is_zero(self):
        assert parse_pageviews({"items": []}) == 0

    def test_enwiki_sitelink(self):
        obj = {"entities": {"Q1": {"sitelinks": {
            "enwiki": {"title": "Caymus Vineyards"}, "frwiki": {"title": "Caymus"}}}}}
        title, count = parse_enwiki_sitelink(obj, "Q1")
        assert title == "Caymus Vineyards" and count == 2
