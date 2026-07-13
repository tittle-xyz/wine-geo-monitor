"""Share-of-voice trend and drift detection over daily partitions. See #11.

Reads the durable JSONL layer each daily run writes (`<root>/<date>/metrics.jsonl`),
builds a per-producer time series for one prompt, and flags producers whose share has
moved *beyond the noise floor* — i.e. the latest day's 95% CI no longer overlaps the
baseline day's. The whole point is to not cry wolf: most day-to-day wiggle is sampling
noise, and only a non-overlapping move counts as real drift.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from .schema import read_jsonl


def load_series(root, prompt_id):
    """Return (dates, series): dates sorted ascending; series[producer][date] =
    (share, ci_lo, ci_hi) for the given prompt across every `<root>/<date>/metrics.jsonl`."""
    files = sorted(glob.glob(os.path.join(str(root), "*", "metrics.jsonl")))
    dates, series = [], {}
    for f in files:
        date = os.path.basename(os.path.dirname(f))
        dates.append(date)
        for r in read_jsonl(f):
            if r.get("prompt_id") != prompt_id:
                continue
            series.setdefault(r["producer"], {})[date] = (
                r.get("share", 0.0), r.get("ci_lo", 0.0), r.get("ci_hi", 0.0),
            )
    return dates, series


def _at(points, date):
    """(share, ci_lo, ci_hi) for a producer on a date; absent == not mentioned == 0."""
    return points.get(date, (0.0, 0.0, 0.0))


def detect_drift(dates, series):
    """Flag producers whose latest-day CI no longer overlaps the baseline-day CI.

    Baseline = first date, latest = last date. Returns rows sorted by |change|:
    {producer, baseline, latest, change, significant}. `significant` is True only when
    the two 95% intervals are disjoint — a move the noise floor can't explain.
    """
    if len(dates) < 2:
        return []
    base, last = dates[0], dates[-1]
    rows = []
    for producer, points in series.items():
        b_share, b_lo, b_hi = _at(points, base)
        l_share, l_lo, l_hi = _at(points, last)
        if b_share == 0 and l_share == 0:
            continue
        disjoint = l_lo > b_hi or l_hi < b_lo
        rows.append({
            "producer": producer,
            "baseline": b_share,
            "latest": l_share,
            "change": l_share - b_share,
            "significant": disjoint,
        })
    rows.sort(key=lambda r: -abs(r["change"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wine_geo.trend",
        description="Share-of-voice trend & drift over daily partitions.",
    )
    ap.add_argument("root", help="dir of daily partitions (each holds <date>/metrics.jsonl)")
    ap.add_argument("--prompt", default="p0")
    ap.add_argument("--chart", help="render a trend PNG here (needs the viz extra)")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args(argv)

    dates, series = load_series(args.root, args.prompt)
    if len(dates) < 2:
        print(f"need >= 2 daily partitions under {args.root!r} (found {len(dates)})")
        return 1

    drift = detect_drift(dates, series)
    print(f"Trend over {len(dates)} days ({dates[0]} -> {dates[-1]}), prompt {args.prompt}")
    print(f"  {'producer':<24}{'first':>7}{'last':>7}{'change':>8}   drift")
    for r in drift[:args.top]:
        flag = "** beyond noise" if r["significant"] else "within noise"
        print(f"  {r['producer']:<24}{r['baseline'] * 100:6.0f}%{r['latest'] * 100:6.0f}%"
              f"{r['change'] * 100:+7.0f}   {flag}")
    moved = [r for r in drift if r["significant"]]
    print(f"\n{len(moved)} of {len(drift)} producers moved beyond the noise floor.")

    if args.chart:
        from .viz import render_trend_chart
        flagged = {r["producer"] for r in moved}
        path = render_trend_chart(dates, series, args.chart,
                                  prompt_id=args.prompt, top=args.top, drift=flagged)
        print(f"wrote trend chart {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
