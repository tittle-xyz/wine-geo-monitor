"""Dagster wrapper: the GEO monitor as a scheduled asset DAG.

    raw_samples ─► mentions ─► metrics
         └───────────────────►┘   (metrics needs the full sample set for its denominators)

This is the shape that matters for a real monitoring product: a scheduled,
partitionable, backfillable data pipeline where every layer is a re-derivable
function of the immutable raw layer. The stage logic lives in wine_geo.pipeline;
these assets are a thin orchestration wrapper over it.

Run it:
    pip install -e ".[dagster]"
    dagster dev                       # uses [tool.dagster] in pyproject.toml
    # or materialize headless:
    dagster asset materialize -m wine_geo.definitions --select "*"
"""

from dataclasses import asdict
from typing import List, Optional

import dagster as dg

from .config import DEFAULT_CONCURRENCY, DEFAULT_MODEL, DEFAULT_N, DEFAULT_PROMPTS, PRODUCERS_PATH
from .extract import build_patterns, load_producers
from .pipeline import aggregate_stage, collect, extract_stage, metrics_rows
from .providers import get_provider
from .schema import Mention, RawSample


class SamplingConfig(dg.Config):
    """Run-time knobs for a scan. Defaults make the whole DAG materialize offline."""

    provider: str = "mock"
    model: str = DEFAULT_MODEL
    n: int = DEFAULT_N
    concurrency: int = DEFAULT_CONCURRENCY
    seed: Optional[int] = 42  # keep None in prod for real variance
    prompts: Optional[List[str]] = None


@dg.asset(group_name="wine_geo", description="Raw model responses — the immutable, paid layer.")
def raw_samples(context: dg.AssetExecutionContext, config: SamplingConfig) -> List[dict]:
    provider = get_provider(config.provider, seed=config.seed)
    prompts = config.prompts or DEFAULT_PROMPTS
    samples = collect(prompts, provider=provider, model=config.model, n=config.n,
                      concurrency=config.concurrency, seed=config.seed)
    context.add_output_metadata({
        "prompts": len(prompts),
        "samples": len(samples),
        "provider": config.provider,
        "model": config.model,
    })
    return [asdict(s) for s in samples]


@dg.asset(group_name="wine_geo", description="Producer mentions extracted from raw responses.")
def mentions(context: dg.AssetExecutionContext, raw_samples: List[dict]) -> List[dict]:
    patterns = build_patterns(load_producers(PRODUCERS_PATH))
    raw = [RawSample(**r) for r in raw_samples]
    ms = extract_stage(raw, patterns)
    context.add_output_metadata({"mentions": len(ms)})
    return [asdict(m) for m in ms]


@dg.asset(group_name="wine_geo",
          description="Share-of-voice, confidence intervals, and run-to-run instability.")
def metrics(context: dg.AssetExecutionContext, raw_samples: List[dict], mentions: List[dict]) -> dict:
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
    # Surface the headline in the Dagster UI: top brand + instability per prompt.
    preview = {
        r["prompt_id"]: {
            "top": max(((n, s) for n, (s, h, _) in r["sov"].items() if h > 0),
                       key=lambda x: x[1], default=("—", 0.0))[0],
            "jaccard": round(r["jaccard"], 2),
        }
        for r in results
    }
    context.add_output_metadata({
        "producer_rows": len(rows),
        "prompts": len(results),
        "preview": dg.MetadataValue.json(preview),
    })
    return {"producers": rows, "prompts": prompt_summaries}


geo_scan_job = dg.define_asset_job("geo_scan_job", selection=dg.AssetSelection.all())

daily_schedule = dg.ScheduleDefinition(
    name="daily_geo_scan",
    job=geo_scan_job,
    cron_schedule="0 6 * * *",  # 06:00 daily — a monitoring cadence
)

defs = dg.Definitions(
    assets=[raw_samples, mentions, metrics],
    jobs=[geo_scan_job],
    schedules=[daily_schedule],
)
