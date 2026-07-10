"""Dagster wrapper: the GEO monitor as a scheduled, DAILY-PARTITIONED asset DAG.

    raw_samples ─► mentions ─► metrics ─► share_of_voice_chart
         │              └──────►┘   (metrics needs the full sample set for its denominators)
         └─► cost                    (token spend is a pure function of the raw layer)

Every asset is partitioned by day, so each date is its own immutable slice you can
backfill and re-run independently — the real shape of GEO measurement, where you
aggregate over a rolling multi-week window. The stage logic lives in
wine_geo.pipeline; these assets are a thin orchestration wrapper over it.

Run it:
    pip install -e ".[dagster,viz]"
    dagster dev                       # uses [tool.dagster] in pyproject.toml
    # or materialize one day, headless:
    dagster asset materialize -m wine_geo.definitions --select "*" --partition 2026-07-05
"""

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import dagster as dg

from .config import DEFAULT_CONCURRENCY, DEFAULT_MODEL, DEFAULT_N, DEFAULT_PROMPTS, PRODUCERS_PATH
from .extract import build_patterns, load_producers
from .pipeline import aggregate_stage, collect, cost_stage, extract_stage, metrics_rows
from .providers import get_provider
from .schema import Mention, RawSample

# One partition per day. Backfill a range, or let the schedule fill the latest.
daily = dg.DailyPartitionsDefinition(start_date="2026-07-01")


class SamplingConfig(dg.Config):
    """Run-time knobs for a scan. Defaults make the whole DAG materialize offline."""

    provider: str = "mock"
    model: str = DEFAULT_MODEL
    n: int = DEFAULT_N
    concurrency: int = DEFAULT_CONCURRENCY
    seed: Optional[int] = 42  # keep None in prod for real variance
    prompts: Optional[List[str]] = None


@dg.asset(partitions_def=daily, group_name="wine_geo",
          description="Raw model responses for the day — the immutable, paid layer.")
def raw_samples(context: dg.AssetExecutionContext, config: SamplingConfig) -> List[dict]:
    day = context.partition_key
    provider = get_provider(config.provider, seed=config.seed)
    prompts = config.prompts or DEFAULT_PROMPTS
    samples = collect(prompts, provider=provider, model=config.model, n=config.n,
                      concurrency=config.concurrency, seed=config.seed,
                      run_id=f"scan-{day}", ts=day)
    context.add_output_metadata({"date": day, "prompts": len(prompts), "samples": len(samples)})
    return [asdict(s) for s in samples]


@dg.asset(partitions_def=daily, group_name="wine_geo",
          description="Producer mentions extracted from the day's raw responses.")
def mentions(context: dg.AssetExecutionContext, raw_samples: List[dict]) -> List[dict]:
    patterns = build_patterns(load_producers(PRODUCERS_PATH))
    raw = [RawSample(**r) for r in raw_samples]
    ms = extract_stage(raw, patterns)
    context.add_output_metadata({"mentions": len(ms)})
    return [asdict(m) for m in ms]


@dg.asset(partitions_def=daily, group_name="wine_geo",
          description="Share-of-voice, confidence intervals, and run-to-run instability.")
def metrics(
    context: dg.AssetExecutionContext, raw_samples: List[dict], mentions: List[dict]
) -> dict:
    producers = load_producers(PRODUCERS_PATH)
    universe = [p["name"] for p in producers]
    raw = [RawSample(**r) for r in raw_samples]
    ms = [Mention(**m) for m in mentions]
    results = aggregate_stage(raw, ms, universe, seed=42)
    rows = metrics_rows(results)
    prompt_summaries = [
        {"prompt_id": r["prompt_id"], "prompt": r["prompt"], "n": r["n"], "jaccard": r["jaccard"]}
        for r in results
    ]
    # A markdown table renders right in the Dagster UI for each partition.
    md = ["| prompt | top producer | share | Jaccard |", "|---|---|---:|---:|"]
    for r in results:
        top, top_share = max(((n, s) for n, (s, h, _) in r["sov"].items() if h > 0),
                             key=lambda x: x[1], default=("—", 0.0))
        md.append(f"| {r['prompt']} | {top} | {top_share * 100:.0f}% | {r['jaccard']:.2f} |")
    context.add_output_metadata({
        "producer_rows": len(rows),
        "prompts": len(results),
        "summary": dg.MetadataValue.md("\n".join(md)),
    })
    return {"producers": rows, "prompts": prompt_summaries}


@dg.asset(partitions_def=daily, group_name="wine_geo",
          description="Token spend for the day, derived from the raw layer (list price).")
def cost(context: dg.AssetExecutionContext, raw_samples: List[dict]) -> dict:
    raw = [RawSample(**r) for r in raw_samples]
    c = cost_stage(raw)
    md = ["| provider / model | samples | in tok | out tok | cost (USD) |",
          "|---|---:|---:|---:|---:|"]
    for m in c["by_model"]:
        md.append(f"| {m['provider']} / {m['model']} | {m['samples']} | "
                  f"{m['input_tokens']:,} | {m['output_tokens']:,} | ${m['cost']:.4f} |")
    context.add_output_metadata({
        "total_cost_usd": round(c["total"]["cost"], 6),
        "samples": c["total"]["samples"],
        "by_model": dg.MetadataValue.md("\n".join(md)),
    })
    return c


@dg.asset(partitions_def=daily, group_name="wine_geo",
          description="Share-of-voice chart (PNG) for the headline prompt.")
def share_of_voice_chart(context: dg.AssetExecutionContext, metrics: dict) -> None:
    from .viz import render_chart

    prompts = metrics["prompts"]
    if not prompts:
        return
    p0 = prompts[0]
    out = Path("out") / f"share_of_voice_{context.partition_key}.png"
    path = render_chart(metrics["producers"], load_producers(PRODUCERS_PATH), out,
                        prompt_id=p0["prompt_id"], prompt_text=p0["prompt"],
                        jaccard=p0["jaccard"], n=p0["n"])
    context.add_output_metadata({"path": dg.MetadataValue.path(str(Path(path).resolve()))})


geo_scan_job = dg.define_asset_job("geo_scan_job", selection=dg.AssetSelection.all())

# A daily schedule derived from the partition set — materializes the latest day.
daily_schedule = dg.build_schedule_from_partitioned_job(geo_scan_job, hour_of_day=6)

defs = dg.Definitions(
    assets=[raw_samples, mentions, cost, metrics, share_of_voice_chart],
    jobs=[geo_scan_job],
    schedules=[daily_schedule],
)
