"""Trend loading and drift detection (wine_geo.trend)."""

import json
import os

from wine_geo.trend import detect_drift, load_series


def _write(root, date, rows):
    d = os.path.join(root, date)
    os.makedirs(d)
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(producer, share, prompt="p0"):
    return {"prompt_id": prompt, "producer": producer, "share": share,
            "ci_lo": max(0.0, share - 0.1), "ci_hi": min(1.0, share + 0.1),
            "n": 25, "hits": round(share * 25)}


def test_load_series_orders_dates_and_filters_prompt(tmp_path):
    root = str(tmp_path)
    _write(root, "2026-07-02", [_row("Bogle", 0.9), _row("X", 0.5, prompt="p1")])
    _write(root, "2026-07-01", [_row("Bogle", 0.8)])

    dates, series = load_series(root, "p0")

    assert dates == ["2026-07-01", "2026-07-02"]     # sorted ascending
    assert "X" not in series                          # other prompt filtered out
    assert series["Bogle"]["2026-07-01"][0] == 0.8


def test_detect_drift_flags_only_nonoverlapping(tmp_path):
    root = str(tmp_path)
    _write(root, "2026-07-01", [_row("Steady", 0.50), _row("Mover", 0.10)])
    _write(root, "2026-07-02", [_row("Steady", 0.55), _row("Mover", 0.80)])

    dates, series = load_series(root, "p0")
    rows = detect_drift(dates, series)
    by = {r["producer"]: r for r in rows}

    assert by["Mover"]["significant"] is True     # [0-0.2] vs [0.7-0.9] disjoint
    assert by["Steady"]["significant"] is False   # [0.4-0.6] vs [0.45-0.65] overlap
    assert rows[0]["producer"] == "Mover"         # sorted by |change|


def test_detect_drift_needs_two_days(tmp_path):
    root = str(tmp_path)
    _write(root, "2026-07-01", [_row("Bogle", 0.9)])
    dates, series = load_series(root, "p0")
    assert detect_drift(dates, series) == []
