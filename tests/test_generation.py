"""Model-generation axis (#23 Phase 2) — trajectory ordering + the graded flip verdict.

The pure core is tested offline: knowledge_trajectory with provider=None (priors only), and
grade_trajectory over hand-built ModelKnowledge rows. The four "nobody knows / someone knows"
branches map to the four verdicts.
"""

from __future__ import annotations

from wine_geo.diagnose import _year
from wine_geo.generation import (
    ModelKnowledge,
    grade_trajectory,
    knowledge_trajectory,
)

CUTOFFS = {
    "gpt-4o-mini": {"reliable_cutoff": "2023-10"},
    "gpt-4.1-mini": {"reliable_cutoff": "2024-06"},
    "gpt-5-mini": {"reliable_cutoff": "2024-10"},
}


def _mk(model, cutoff, verdict):
    return ModelKnowledge(model, cutoff, _year(cutoff), prior="", verdict=verdict)


class TestTrajectory:
    def test_orders_by_cutoff_and_predicts(self):
        # Pass the fleet out of order; priors-only (no provider).
        traj = knowledge_trajectory(
            "Atelier Ilaria", 2024,
            ["gpt-5-mini", "gpt-4o-mini", "gpt-4.1-mini"], CUTOFFS,
        )
        assert [t.model for t in traj] == ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"]
        # 2024 brand vs Oct-2023 cutoff -> a priori unknown; vs 2024 cutoffs -> borderline.
        assert traj[0].prior == "likely-unknown"
        assert traj[1].prior == "borderline"
        assert all(t.verdict is None for t in traj)  # no probing happened


class TestGrade:
    def test_gap_closed_at_predicted_crossing(self):
        traj = [
            _mk("gpt-4o-mini", "2023-10", "disowns"),
            _mk("gpt-4.1-mini", "2024-06", "hedges"),
            _mk("gpt-5-mini", "2024-10", "knows"),
        ]
        v = grade_trajectory(traj, 2024)
        assert v.verdict == "GAP CLOSED"
        assert v.flip_model == "gpt-5-mini"
        assert "matches the predicted crossing" in v.detail

    def test_too_new_when_newest_predates_documentation(self):
        traj = [_mk("gpt-4o-mini", "2023-10", "disowns")]
        v = grade_trajectory(traj, 2025)
        assert v.verdict == "TOO NEW"

    def test_too_obscure_when_cutoff_passed_but_still_unknown(self):
        # newest cutoff (2024) already spans the 2023 documentation, yet still disowned.
        traj = [_mk("gpt-5-mini", "2024-10", "disowns")]
        v = grade_trajectory(traj, 2023)
        assert v.verdict == "TOO OBSCURE"

    def test_too_new_uses_month_precision(self):
        # A Fall-2024 brand vs a Jun-2024 newest cutoff: same YEAR, but the brand is later.
        # Year-only comparison would wrongly say TOO OBSCURE; month precision says TOO NEW.
        traj = [
            _mk("gpt-4o-mini", "2023-10", "disowns"),
            _mk("gpt-5-mini", "2024-06", "disowns"),
        ]
        v = grade_trajectory(traj, "2024-11")
        assert v.verdict == "TOO NEW"

    def test_borderline_when_same_year_but_only_year_precision(self):
        # since is year-only and equal to the newest cutoff's year -> genuinely undecidable.
        traj = [_mk("gpt-5-mini", "2024-06", "disowns")]
        v = grade_trajectory(traj, 2024)
        assert v.verdict == "BORDERLINE"

    def test_made_up_only_when_model_could_have_learned_it(self):
        # docs (2020) predate the newest cutoff (2024): the model had the data, yet it confabulates.
        traj = [
            _mk("gpt-4o-mini", "2023-10", "disowns"),
            _mk("gpt-5-mini", "2024-06", "confabulates"),
        ]
        v = grade_trajectory(traj, 2020)
        assert v.verdict == "MADE UP"

    def test_confabulation_does_not_shadow_too_new(self):
        # newest confabulates, but the brand postdates its cutoff -> TOO NEW headline + a caveat,
        # not MADE UP (the model never had the data, so the confabulation isn't the real cause).
        traj = [
            _mk("gpt-4o-mini", "2023-10", "disowns"),
            _mk("gpt-5-mini", "2024-06", "confabulates"),
        ]
        v = grade_trajectory(traj, "2024-11")
        assert v.verdict == "TOO NEW"
        assert "invents confident" in v.detail

    def test_early_flip_is_flagged(self):
        # a "known" flip earlier than the documentation date is suspicious — surface it.
        traj = [_mk("gpt-4o-mini", "2023-10", "knows")]
        v = grade_trajectory(traj, 2025)
        assert v.verdict == "GAP CLOSED"
        assert "double-check --since" in v.detail

    def test_inconclusive_when_nothing_probed(self):
        traj = [ModelKnowledge("gpt-4o-mini", "2023-10", 2023, "likely-unknown", verdict=None)]
        v = grade_trajectory(traj, 2024)
        assert v.verdict == "INCONCLUSIVE"
