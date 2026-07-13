"""Command-line entry point: run the pipeline stages and print a report.

This is the un-orchestrated path — handy for a quick local run. The same three
stage functions are what the Dagster assets call (see wine_geo/definitions.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .extract import build_patterns, load_producers
from .pipeline import (
    aggregate_stage,
    collect,
    cost_rows,
    cost_stage,
    extract_stage,
    metrics_rows,
    sweep_cost_confidence,
)
from .providers import get_provider
from .report import render_report
from .schema import write_jsonl


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="wine_geo", description="A tiny GEO monitor for wine brands.")
    p.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai"],
                   help="mock needs no API key (default)")
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=config.DEFAULT_N, help="samples per prompt")
    p.add_argument("--concurrency", type=int, default=config.DEFAULT_CONCURRENCY)
    p.add_argument("--producers", default=str(config.PRODUCERS_PATH))
    p.add_argument("--prompts", help="path to a text file, one prompt per line")
    p.add_argument("--seed", type=int, help="reproducible runs (omit for real variance)")
    p.add_argument("--out-dir", help="also write raw.jsonl / mentions.jsonl / metrics.jsonl here")
    p.add_argument("--chart", help="render a share-of-voice PNG to this path (needs the viz extra)")
    p.add_argument("--chart-prompt", default="p0", help="which prompt to chart (default p0)")
    p.add_argument("--cost-curve", help="render the cost-vs-confidence curve PNG to this path")
    p.add_argument("--cost-curve-ns", default="10,25,50,100,200",
                   help="comma-separated sample sizes for --cost-curve")
    return p.parse_args(argv)


def _load_prompts(path):
    if not path:
        return config.DEFAULT_PROMPTS
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(argv=None):
    args = _parse_args(argv)

    config.load_dotenv()  # pick up ANTHROPIC_API_KEY / OPENAI_API_KEY from a local .env

    producers = load_producers(args.producers)
    patterns = build_patterns(producers)
    universe = [p["name"] for p in producers]
    provider = get_provider(args.provider, seed=args.seed)
    prompts = _load_prompts(args.prompts)

    # The three stages, exactly as the Dagster assets run them.
    raw = collect(prompts, provider=provider, model=args.model, n=args.n,
                  concurrency=args.concurrency, seed=args.seed)
    mentions = extract_stage(raw, patterns)
    results = aggregate_stage(raw, mentions, universe, seed=args.seed)

    cost = cost_stage(raw)
    summary = {
        "provider": args.provider,
        "model": args.model,
        "samples": len(raw),
        "in": cost["total"]["input_tokens"],
        "out": cost["total"]["output_tokens"],
        "cost": cost["total"]["cost"],
    }
    render_report(results, summary)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "raw.jsonl", raw)
        write_jsonl(out / "mentions.jsonl", mentions)
        write_jsonl(out / "metrics.jsonl", metrics_rows(results))
        write_jsonl(out / "cost.jsonl", cost_rows(cost))
        print(f"\nwrote raw / mentions / metrics / cost under {out}/")

    if args.chart:
        from .viz import render_chart
        chosen = next((r for r in results if r["prompt_id"] == args.chart_prompt), results[0])
        path = render_chart(metrics_rows(results), producers, args.chart,
                            prompt_id=chosen["prompt_id"], prompt_text=chosen["prompt"],
                            jaccard=chosen["jaccard"], n=chosen["n"])
        print(f"wrote chart {path}")

    if args.cost_curve:
        from .viz import render_cost_curve
        ns = [int(x) for x in args.cost_curve_ns.split(",") if x.strip()]
        points = sweep_cost_confidence(prompts, patterns, universe,
                                       provider_name=args.provider, model=args.model, ns=ns,
                                       concurrency=args.concurrency, seed=args.seed)
        path = render_cost_curve(points, args.cost_curve, provider=args.provider, model=args.model)
        print(f"wrote cost curve {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
