"""Tests for the #23 attribution diagnosis — the pure, no-network logic."""

from __future__ import annotations

from types import SimpleNamespace

from wine_geo.diagnose import (
    brand_in_sources,
    classify_probe,
    diagnose,
    diagnose_retrieval,
    knowledge_prior,
    recommend,
)

CUTOFFS = {
    "gpt-4o-mini": {"reliable_cutoff": "2023-10"},
    "claude-haiku-4-5": {"reliable_cutoff": "2025-02"},
}


class TestKnowledgePrior:
    def test_postdates_cutoff_is_likely_unknown(self):
        # CAM X (2025) vs GPT-4o-mini (Oct 2023)
        assert knowledge_prior(2025, "gpt-4o-mini", CUTOFFS) == "likely-unknown"

    def test_well_before_cutoff_is_likely_known(self):
        assert knowledge_prior(1972, "claude-haiku-4-5", CUTOFFS) == "likely-known"

    def test_same_year_is_borderline(self):
        assert knowledge_prior(2025, "claude-haiku-4-5", CUTOFFS) == "borderline"

    def test_unknown_model_and_missing_date_degrade(self):
        assert knowledge_prior(2025, "some-future-model", CUTOFFS) == "unknown-cutoff"
        assert knowledge_prior(None, "gpt-4o-mini", CUTOFFS) == "unknown-date"


class TestClassifyProbe:
    ANCHORS = ["cameron hughes", "négoc", "2020"]

    def test_disown_with_no_anchor_is_disowns(self):
        assert classify_probe("I'm not familiar with that producer.", self.ANCHORS) == "disowns"

    def test_accurate_anchor_is_knows(self):
        txt = "Founded by Cameron Hughes as a négoce project."
        assert classify_probe(txt, self.ANCHORS) == "knows"

    def test_confident_but_no_anchor_is_confabulates(self):
        # The De Négoce 'founded by Jon Bonné' failure: confident, zero real anchors.
        txt = "It was founded by Jon Bonné, a wine critic."
        assert classify_probe(txt, self.ANCHORS) == "confabulates"

    def test_no_anchors_given_is_unverified(self):
        assert classify_probe("A fine California winery.", []) == "unverified"

    def test_disown_plus_anchor_is_hedges(self):
        txt = "I'm not certain, but Cameron Hughes may be involved."
        assert classify_probe(txt, self.ANCHORS) == "hedges"


class TestDiagnose:
    def test_not_known_is_knowledge_gap(self):
        assert diagnose(known=False, ranked_rate=0.0).startswith("NOT KNOWN")

    def test_known_but_unranked_is_ranking_gap(self):
        # Cameron Hughes: known, ~0% surfaced.
        assert diagnose(known=True, ranked_rate=0.003).startswith("NOT RANKED")

    def test_known_and_ranked(self):
        assert diagnose(known=True, ranked_rate=0.4) == "known-and-ranked"

    def test_no_probe_and_not_surfaced_is_inconclusive(self):
        assert "inconclusive" in diagnose(known=None, ranked_rate=0.0)

    def test_no_probe_but_surfaced_needs_no_probe(self):
        assert "no probe needed" in diagnose(known=None, ranked_rate=0.4)


def _src(url, title=""):
    return SimpleNamespace(url=url, title=title)


class TestBrandInSources:
    def test_matches_brand_in_mangled_url(self):
        # 'CAM X' -> 'camx' should match 'camxwine.com'
        assert brand_in_sources("CAM X", ["CAMX"], [_src("https://camxwine.com/")]) is True

    def test_matches_alias_in_title(self):
        srcs = [_src("https://winebusiness.com/x", "Cameron Hughes Launches CAM X")]
        assert brand_in_sources("CAM X", ["Cameron Hughes"], srcs) is True

    def test_no_match_when_absent(self):
        assert brand_in_sources("CAM X", ["Cameron Hughes"], [_src("https://caymus.com/")]) is False

    def test_short_aliases_ignored(self):
        # a 2-char alias must not spuriously match; only forms >= 4 chars count
        assert brand_in_sources("XY", ["XY"], [_src("https://anything-xy-here.com/")]) is False


class TestDiagnoseRetrieval:
    def test_not_retrieved(self):
        v = diagnose_retrieval(retrieved_rate=0.0, recommended_rate=0.0)
        assert v.startswith("NOT RETRIEVED")

    def test_retrieved_but_not_ranked(self):
        assert diagnose_retrieval(retrieved_rate=0.8, recommended_rate=0.0).startswith(
            "RETRIEVED BUT NOT RANKED")

    def test_retrieved_and_recommended(self):
        assert diagnose_retrieval(retrieved_rate=0.8, recommended_rate=0.5).startswith(
            "RETRIEVED & RECOMMENDED")


class TestRecommend:
    def _retr(self, retrieved, recommended, sources=()):
        return {"retrieved_rate": retrieved, "recommended_rate": recommended,
                "sample_sources": list(sources)}

    def test_not_retrieved_names_target_sources(self):
        # CAM X shape: not known, not retrieved -> a not-retrieved lever citing the real targets.
        srcs = [_src("https://www.winedeals.com/x"), _src("https://totalwine.com/y")]
        recs = recommend(known=False, ranked_rate=0.0, retrieval=self._retr(0.0, 0.0, srcs))
        by_cause = {r.cause: r for r in recs}
        assert "not-retrieved" in by_cause and "not-known" in by_cause
        assert "winedeals.com" in by_cause["not-retrieved"].detail
        assert "totalwine.com" in by_cause["not-retrieved"].detail
        assert by_cause["not-retrieved"].timescale == "weeks"

    def test_retrieved_but_not_ranked(self):
        recs = recommend(known=True, ranked_rate=0.0, retrieval=self._retr(0.8, 0.0))
        assert any(r.cause == "not-ranked" for r in recs)

    def test_parametric_ranking_gap_without_grounding(self):
        # known but not surfaced and no grounded probe -> a ranking lever, hedged to confirm.
        recs = recommend(known=True, ranked_rate=0.0, retrieval=None)
        assert any(r.cause == "not-ranked" for r in recs)

    def test_no_probe_no_retrieval_is_inconclusive(self):
        recs = recommend(known=None, ranked_rate=0.0, retrieval=None)
        assert any(r.cause == "inconclusive" for r in recs)

    def test_known_and_ranked_is_maintain(self):
        recs = recommend(known=True, ranked_rate=0.5, retrieval=None)
        assert [r.cause for r in recs] == ["none"]
