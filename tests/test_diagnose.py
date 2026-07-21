"""Tests for the #23 attribution diagnosis — the pure, no-network logic."""

from __future__ import annotations

from wine_geo.diagnose import classify_probe, diagnose, knowledge_prior

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
        assert classify_probe("Founded by Cameron Hughes as a négoce project.", self.ANCHORS) == "knows"

    def test_confident_but_no_anchor_is_confabulates(self):
        # The De Négoce 'founded by Jon Bonné' failure: confident, zero real anchors.
        assert classify_probe("It was founded by Jon Bonné, a wine critic.", self.ANCHORS) == "confabulates"

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
