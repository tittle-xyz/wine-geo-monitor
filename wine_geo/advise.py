"""Cost-aware sample-size advisor: how many samples per prompt for a target precision,
and what that costs. See #8.

The cost-of-confidence chart shows the √n trade-off; this turns it into a
recommendation. Anchor the cost to a real run's realized spend with `--from`:

    python -m wine_geo.advise --target 0.10 --from out/claude
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .schema import read_jsonl
from .stats import ci_half_width_for, recommend_sample_size

_TARGETS = [0.05, 0.08, 0.10, 0.15]


def cost_per_sample(run_dir):
    """Realized USD per sample from a run's cost.jsonl (None if unavailable)."""
    path = Path(run_dir) / "cost.jsonl"
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    samples = sum(r.get("samples", 0) for r in rows)
    return sum(r.get("cost", 0.0) for r in rows) / samples if samples else None


def prompt_count(run_dir, default):
    """Distinct prompts in a run's raw.jsonl (falls back to `default`)."""
    raw = Path(run_dir) / "raw.jsonl"
    if raw.is_file():
        return len({r.get("prompt_id") for r in read_jsonl(raw)}) or default
    return default


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wine_geo.advise",
        description="How many samples per prompt for a target precision — and the cost.",
    )
    ap.add_argument("--target", type=float, help="target CI half-width, e.g. 0.10 for ±10 pts")
    ap.add_argument("--budget", type=float, help="total USD budget (needs a cost per sample)")
    ap.add_argument("--from", dest="from_run", help="saved run dir; reads cost/sample + prompts")
    ap.add_argument("--cost-per-sample", type=float, help="USD per sample (if not using --from)")
    ap.add_argument("--prompts", type=int, help="number of prompts (default: from run or config)")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--chart", help="render a samples-vs-precision PNG here (needs the viz extra)")
    args = ap.parse_args(argv)

    cps = cost_per_sample(args.from_run) if args.from_run else args.cost_per_sample
    n_prompts = args.prompts or (
        prompt_count(args.from_run, len(config.DEFAULT_PROMPTS))
        if args.from_run else len(config.DEFAULT_PROMPTS)
    )

    def cost_of(n):
        return None if cps is None else n * n_prompts * cps

    conf = round(args.confidence * 100)
    header = f"Sample-size advisor — {conf}% CI, worst case, {n_prompts} prompts"
    header += f", ${cps:.5f}/sample" if cps is not None else " (cost per sample unknown)"
    print(header)
    print(f"  {'target':>9}  {'samples/prompt':>14}  {'projected run cost':>18}")
    for w in _TARGETS:
        n = recommend_sample_size(w, confidence=args.confidence)
        c = cost_of(n)
        print(f"  {'±' + format(w * 100, '.0f') + ' pts':>9}  {n:>14,}  "
              f"{('$' + format(c, ',.2f')) if c is not None else '—':>18}")

    if args.target:
        n = recommend_sample_size(args.target, confidence=args.confidence)
        c = cost_of(n)
        line = f"\n→ For ±{args.target * 100:.0f} pts: {n:,} samples/prompt"
        print(line + (f"  (~${c:,.2f} per full run)" if c is not None else ""))

    if args.budget is not None:
        if cps is None:
            print("\n--budget needs a cost per sample (pass --from or --cost-per-sample).")
            return 1
        n = int(args.budget / (n_prompts * cps))
        if n < 1:
            print(f"\n→ ${args.budget:,.2f} isn't enough for even 1 sample/prompt.")
        else:
            print(f"\n→ ${args.budget:,.2f} buys {n:,} samples/prompt "
                  f"→ about ±{ci_half_width_for(n, confidence=args.confidence) * 100:.1f} pts.")

    if args.chart:
        from .viz import render_sample_size_chart
        path = render_sample_size_chart(args.chart, confidence=args.confidence)
        print(f"\nwrote sample-size chart {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
