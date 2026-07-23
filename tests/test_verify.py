"""Tests for the intervention verifier (#23 Phase 2) — the pure difference-of-changes core.

Measurements are (share, ci_lo, ci_hi), as trend.load_series produces. The four cases map
to the four graded verdicts.
"""

from __future__ import annotations

import json
import os

import pytest

from wine_geo.verify import verify_from_partitions, verify_intervention


def _write(root, date, rows):
    d = os.path.join(root, date)
    os.makedirs(d)
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(producer, share, half=0.01, prompt="p0"):
    return {"prompt_id": prompt, "producer": producer, "share": share,
            "ci_lo": max(0.0, share - half), "ci_hi": min(1.0, share + half)}


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


class TestVerifyFromPartitions:
    def test_likely_real_over_endpoints(self, tmp_path):
        # Treated jumps 0->5% (tight bands); control flat. Base/latest default to first/last day.
        root = str(tmp_path)
        _write(root, "2026-07-01", [_row("CAM X", 0.0), _row("Control", 0.02)])
        _write(root, "2026-07-10", [_row("CAM X", 0.05), _row("Control", 0.02)])

        pv = verify_from_partitions(root, "CAM X", "Control")

        assert (pv.base, pv.latest) == ("2026-07-01", "2026-07-10")
        assert pv.treated_seen and pv.control_seen
        assert round(pv.result.diff_of_changes, 4) == 0.05
        assert pv.result.verdict.startswith("LIKELY REAL")

    def test_absent_treated_reads_as_zero(self, tmp_path):
        # CAM X isn't named at all on the baseline day -> its baseline measurement is a real 0%,
        # not missing data. It appears (tightly) on the latest day.
        root = str(tmp_path)
        _write(root, "2026-07-01", [_row("Control", 0.02)])
        _write(root, "2026-07-10", [_row("CAM X", 0.06), _row("Control", 0.02)])

        pv = verify_from_partitions(root, "CAM X", "Control")

        assert pv.treated_base == (0.0, 0.0, 0.0)
        assert pv.treated_seen  # seen on at least one day

    def test_confounded_when_control_moves_too(self, tmp_path):
        root = str(tmp_path)
        _write(root, "2026-07-01", [_row("CAM X", 0.0), _row("Control", 0.01)])
        _write(root, "2026-07-10", [_row("CAM X", 0.05), _row("Control", 0.06)])

        pv = verify_from_partitions(root, "CAM X", "Control")

        assert pv.result.verdict.startswith("CONFOUNDED")

    def test_pin_exact_dates(self, tmp_path):
        # Three days; pin the middle-to-last window explicitly.
        root = str(tmp_path)
        _write(root, "2026-07-01", [_row("CAM X", 0.0), _row("Control", 0.02)])
        _write(root, "2026-07-05", [_row("CAM X", 0.0), _row("Control", 0.02)])
        _write(root, "2026-07-10", [_row("CAM X", 0.05), _row("Control", 0.02)])

        pv = verify_from_partitions(root, "CAM X", "Control",
                                    base="2026-07-05", latest="2026-07-10")

        assert (pv.base, pv.latest) == ("2026-07-05", "2026-07-10")

    def test_needs_two_days(self, tmp_path):
        root = str(tmp_path)
        _write(root, "2026-07-01", [_row("CAM X", 0.0), _row("Control", 0.02)])
        with pytest.raises(ValueError, match="need >= 2"):
            verify_from_partitions(root, "CAM X", "Control")

    def test_pinned_date_missing_raises(self, tmp_path):
        root = str(tmp_path)
        _write(root, "2026-07-01", [_row("CAM X", 0.0), _row("Control", 0.02)])
        _write(root, "2026-07-10", [_row("CAM X", 0.05), _row("Control", 0.02)])
        with pytest.raises(ValueError, match="not among partitions"):
            verify_from_partitions(root, "CAM X", "Control", base="2026-07-04")
