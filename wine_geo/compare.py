"""Cross-engine comparison: a grouped share-of-voice chart from saved runs.

Cross-engine measurement is a pure function of runs already on disk — no new API
calls. Run the monitor against two providers with `--out-dir`, then point this at
those directories:

    python -m wine_geo --provider anthropic --model claude-haiku-4-5 --out-dir out/claude
    python -m wine_geo --provider openai    --model gpt-4o-mini      --out-dir out/openai
    python -m wine_geo.compare out/claude=Claude out/openai=OpenAI --prompt p0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .schema import read_jsonl


def load_run(spec):
    """Parse a `dir` or `dir=Label` spec into (label, metrics_rows, run_dir)."""
    path, _, label = spec.partition("=")
    p = Path(path)
    if p.is_dir():
        metrics, run_dir, default_label = p / "metrics.jsonl", p, p.name
    else:
        metrics, run_dir, default_label = p, p.parent, p.stem
    return (label or default_label, read_jsonl(metrics), run_dir)


def align_shares(runs, prompt_id, top=14):
    """Merge per-model metrics for one prompt into a chartable table.

    Returns (labels, producers, table): `labels` in input order; `producers` ordered
    ascending by their max share across models (leader last), truncated to `top`;
    `table[producer][label]` is `(share, ci_lo, ci_hi)`, with zeros where a model
    never named that producer.
    """
    labels = [r[0] for r in runs]
    per = {
        lbl: {row["producer"]: row for row in rows if row["prompt_id"] == prompt_id}
        for lbl, rows, *_ in runs
    }
    producers = set()
    for d in per.values():
        producers |= set(d)

    def max_share(p):
        return max(per[lbl].get(p, {}).get("share", 0.0) for lbl in labels)

    producers = sorted((p for p in producers if max_share(p) > 0), key=max_share)
    if top:
        producers = producers[-top:]

    table = {
        p: {
            lbl: (
                per[lbl].get(p, {}).get("share", 0.0),
                per[lbl].get(p, {}).get("ci_lo", 0.0),
                per[lbl].get(p, {}).get("ci_hi", 0.0),
            )
            for lbl in labels
        }
        for p in producers
    }
    return labels, producers, table


def _prompt_text(run_dir, prompt_id):
    """Best-effort: recover the human prompt text from a run's raw.jsonl."""
    raw = Path(run_dir) / "raw.jsonl"
    if raw.is_file():
        for row in read_jsonl(raw):
            if row.get("prompt_id") == prompt_id:
                return row.get("prompt_text")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wine_geo.compare",
        description="Cross-engine share-of-voice chart from saved run directories.",
    )
    ap.add_argument("runs", nargs="+", help="run dir or metrics.jsonl, optionally dir=Label")
    ap.add_argument("--prompt", default="p0", help="which prompt to chart (default p0)")
    ap.add_argument("--out", default="out/cross_engine.png")
    ap.add_argument("--top", type=int, default=14, help="max producers to show (by peak share)")
    args = ap.parse_args(argv)

    runs = [load_run(s) for s in args.runs]
    labels, producers, table = align_shares(runs, args.prompt, top=args.top)
    if not producers:
        print(f"no producers mentioned for prompt {args.prompt!r}")
        return 1

    from .viz import render_cross_engine_chart

    path = render_cross_engine_chart(
        labels, producers, table, args.out,
        prompt_id=args.prompt, prompt_text=_prompt_text(runs[0][2], args.prompt),
    )
    print(f"wrote cross-engine chart {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
