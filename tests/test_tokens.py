"""Output-token reservation analysis (wine_geo.tokens) — the pure distribution + packing math."""

from __future__ import annotations

from wine_geo.tokens import _pct, output_distribution, packing_gain


def _rec(pid, out, inp=20, model="m", error=None):
    return {"prompt_id": pid, "model": model, "input_tokens": inp,
            "output_tokens": out, "error": error}


def test_pct_nearest_rank():
    vals = list(range(1, 101))  # 1..100 sorted
    assert _pct(vals, 50) == 51    # nearest-rank on 0-index
    assert _pct(vals, 95) == 95
    assert _pct([], 95) == 0


def test_output_distribution_groups_and_saturation():
    recs = [_rec("p0", o) for o in (100, 100, 100, 400)] + [_rec("p1", o) for o in (50, 50, 50, 50)]
    rows = {r["key"][0]: r for r in output_distribution(recs, cap=400)}
    assert rows["p0"]["n"] == 4
    assert rows["p0"]["max"] == 400
    assert rows["p0"]["cap_hits"] == 0.25          # 1 of 4 reached the 400 cap
    assert rows["p1"]["mean"] == 50 and rows["p1"]["cap_hits"] == 0.0


def test_errored_samples_are_skipped():
    rows = output_distribution([_rec("p0", 100), _rec("p0", 0, error="boom")])
    assert rows[0]["n"] == 1


def test_packing_gain_math():
    # input 0: reserving at p95=100 vs cap=400 fits 4x the calls
    assert round(packing_gain(0, 100, cap=400), 2) == 4.0
    # a prompt that saturates the cap can't be packed tighter
    assert packing_gain(0, 400, cap=400) == 1.0
    # input tokens count toward the reservation: (100+400)/(100+100) = 2.5
    assert round(packing_gain(100, 100, cap=400), 2) == 2.5
