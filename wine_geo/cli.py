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
from .pipeline import aggregate_stage, collect, extract_stage, metrics_rows
from .providers import estimate_cost, get_provider
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
    return p.parse_args(argv)


def _load_prompts(path):
    if not path:
        return config.DEFAULT_PROMPTS
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(argv=None):
    args = _parse_args(argv)

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

    summary = {
        "provider": args.provider,
        "model": args.model,
        "samples": len(raw),
        "in": sum(r.input_tokens for r in raw),
        "out": sum(r.output_tokens for r in raw),
        "cost": sum(estimate_cost(r.input_tokens, r.output_tokens, r.model) for r in raw),
    }
    render_report(results, summary)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "raw.jsonl", raw)
        write_jsonl(out / "mentions.jsonl", mentions)
        write_jsonl(out / "metrics.jsonl", metrics_rows(results))
        print(f"\nwrote raw / mentions / metrics under {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
