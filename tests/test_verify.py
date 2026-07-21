"""Tests for the intervention verifier (#23 Phase 2) — the pure difference-of-changes core.

Measurements are (share, ci_lo, ci_hi), as trend.load_series produces. The four cases map
to the four graded verdicts.
"""

from __future__ import annotations

from wine_geo.verify import verify_intervention


class TestVerifyIntervention:
    def test_likely_real(self):
        # Treated jumps 0->5% with tight bands (a real move); control stays flat.
        r = verify_intervention(
            treated_base=(0.0, 0.0, 0.01), treated_latest=(0.05, 0.04, 0.06),
            control_base=(0.02, 0.015, 0.025), control_latest=(0.02, 0.015, 0.025),
        )
        assert r.treated_moved and not r.control_moved and r.beyond_wobble
        assert round(r.diff_of_changes, 4) == 0.05
        assert r.verdict.startswith("LIKELY REAL")

    def test_confounded_when_control_moves_too(self):
        # Both brands jump ~5% — a category-wide lift, not the intervention.
        r = verify_intervention(
            treated_base=(0.0, 0.0, 0.01), treated_latest=(0.05, 0.04, 0.06),
            control_base=(0.01, 0.005, 0.015), control_latest=(0.06, 0.055, 0.065),
        )
        assert r.treated_moved and r.control_moved and not r.beyond_wobble
        assert abs(r.diff_of_changes) < 1e-9
        assert r.verdict.startswith("CONFOUNDED")

    def test_no_detectable_effect_when_treated_within_wobble(self):
        # Treated's before/after bands overlap — the "move" is within its give-or-take.
        r = verify_intervention(
            treated_base=(0.0, 0.0, 0.03), treated_latest=(0.05, 0.02, 0.08),
            control_base=(0.02, 0.015, 0.025), control_latest=(0.02, 0.015, 0.025),
        )
        assert not r.treated_moved
        assert r.verdict.startswith("NO DETECTABLE EFFECT")

    def test_suggestive_when_difference_within_its_own_wobble(self):
        # Treated moved and control didn't, but the control's bands are so wide the
        # difference-of-changes can't clear its own give-or-take.
        r = verify_intervention(
            treated_base=(0.0, 0.0, 0.005), treated_latest=(0.05, 0.045, 0.055),
            control_base=(0.02, 0.0, 0.09), control_latest=(0.02, 0.0, 0.09),
        )
        assert r.treated_moved and not r.control_moved and not r.beyond_wobble
        assert r.verdict.startswith("SUGGESTIVE")
