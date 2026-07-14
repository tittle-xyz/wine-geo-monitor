"""The three pipeline stages as plain functions — no orchestrator, no I/O policy.

Both the CLI and the Dagster assets call these. Keeping the logic here (and out of
Dagster) is deliberate: the business logic stays unit-testable and portable, and
the orchestrator is a thin wrapper you could swap.

    collect      →  raw samples   (the expensive, paid, run-once layer)
    extract      →  mentions      (re-derivable; re-run free as matching improves)
    aggregate    →  metrics       (share-of-voice + CI + instability)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from .extract import extract_mentions
from .providers import BatchRequest, estimate_cost, get_provider
from .runner import sample_prompt
from .schema import Mention, RawSample
from .sla import Mechanism, Sla, mechanism_for
from .stats import bootstrap_ci, mean_pairwise_jaccard, share_of_voice


def collect(prompts, *, provider, model, n, concurrency, seed=None, run_id=None, ts=None,
            sla=Sla.REALTIME, log=None):
    """Dumb collector: sample each prompt n times, capture raw responses. No interpretation.

    `sla` picks the fulfillment mechanism (see wine_geo.sla): `real-time` samples
    synchronously; `same-day`/`overnight` use the provider's batch API (~50% cheaper)
    when it advertises one, falling back to synchronous collection otherwise. Either
    way the output is the same `RawSample` records, so nothing downstream changes —
    only their `billing_tier`.
    """
    run_id = run_id or uuid4().hex[:12]
    ts = ts or datetime.now(timezone.utc).isoformat()

    if mechanism_for(sla) is Mechanism.BATCH:
        if getattr(provider, "supports_batch", False):
            return _collect_batch(prompts, provider=provider, model=model, n=n,
                                  run_id=run_id, ts=ts, log=log)
        if log:
            log(f"provider {provider.name!r} has no batch path; collecting synchronously")

    samples: list[RawSample] = []
    for i, prompt in enumerate(prompts):
        completions = asyncio.run(
            sample_prompt(provider, prompt, model=model, n=n, concurrency=concurrency, seed=seed)
        )
        for j, c in enumerate(completions):
            samples.append(
                RawSample(
                    run_id=run_id,
                    ts=ts,
                    provider=provider.name,
                    model=model,
                    prompt_id=f"p{i}",
                    prompt_text=prompt,
                    sample_index=j,
                    response_text=c.text,
                    input_tokens=c.input_tokens,
                    output_tokens=c.output_tokens,
                )
            )
    return samples


def _collect_batch(prompts, *, provider, model, n, run_id, ts, log=None):
    """Batch path: submit every (prompt x sample) as one job, map results back to RawSamples.

    One batch for the whole scan is the efficient shape — a single submit/poll cycle
    covers all prompts. Results carry `billing_tier="batch"` so the cost pipeline
    prices them at the batch rate. Any custom_id missing from the results (an errored
    or expired request) becomes an errored, zero-cost sample rather than a gap.
    """
    requests: list[BatchRequest] = []
    index: dict[str, tuple[str, str, int]] = {}
    for i, prompt in enumerate(prompts):
        for j in range(n):
            cid = f"p{i}-{j}"  # ^[a-zA-Z0-9_-]{1,64}$ — Anthropic's custom_id rule
            requests.append(BatchRequest(custom_id=cid, prompt=prompt, model=model))
            index[cid] = (f"p{i}", prompt, j)

    if log:
        log(f"submitting {len(requests)} requests to the {provider.name} batch API "
            f"({len(prompts)} prompts x {n} samples)")
    completions = provider.complete_batch(requests, log=log)

    samples: list[RawSample] = []
    for cid, (pid, ptext, j) in index.items():
        c = completions.get(cid)
        samples.append(
            RawSample(
                run_id=run_id,
                ts=ts,
                provider=provider.name,
                model=model,
                prompt_id=pid,
                prompt_text=ptext,
                sample_index=j,
                response_text=c.text if c else "",
                input_tokens=c.input_tokens if c else 0,
                output_tokens=c.output_tokens if c else 0,
                billing_tier="batch",
                error=None if c else "no result returned for this batch request",
            )
        )
    samples.sort(key=lambda s: (s.prompt_id, s.sample_index))
    return samples


def extract_stage(raw, patterns):
    """Raw responses → one Mention row per (sample, producer) detected."""
    out: list[Mention] = []
    for r in raw:
        for producer in sorted(extract_mentions(r.response_text, patterns)):
            out.append(Mention(r.run_id, r.prompt_id, r.sample_index, producer))
    return out


def aggregate_stage(raw, mentions, universe, *, seed=None):
    """Reconstruct per-sample mention sets (incl. empties) and compute metrics per prompt.

    Needs BOTH raw (for the full set of samples, so share denominators are right even
    when a sample mentioned nothing) and mentions (which producers per sample).
    """
    by_sample: dict[tuple[str, int], set[str]] = defaultdict(set)
    for m in mentions:
        by_sample[(m.prompt_id, m.sample_index)].add(m.producer)

    prompts: dict[str, dict] = {}
    for r in raw:
        info = prompts.setdefault(r.prompt_id, {"text": r.prompt_text, "indices": set()})
        info["indices"].add(r.sample_index)

    results = []
    for pid in sorted(prompts):
        info = prompts[pid]
        run_sets = [by_sample.get((pid, idx), set()) for idx in sorted(info["indices"])]
        sov = share_of_voice(run_sets, universe)
        ci = {
            name: bootstrap_ci([1 if name in s else 0 for s in run_sets], seed=seed)
            for name, (_, hits, _) in sov.items()
            if hits > 0
        }
        results.append({
            "prompt_id": pid,
            "prompt": info["text"],
            "n": len(run_sets),
            "sov": sov,
            "ci": ci,
            "jaccard": mean_pairwise_jaccard(run_sets),
        })
    return results


def _accumulate(bucket, r, cost):
    bucket["samples"] += 1
    bucket["input_tokens"] += r.input_tokens
    bucket["output_tokens"] += r.output_tokens
    bucket["cost"] += cost


def cost_stage(raw):
    """Aggregate token spend from the raw layer into a cost breakdown.

    A pure function of `RawSample` records: cost is re-derivable, so recomputing it
    never re-pays an API call. Returns the run total plus breakdowns by
    (provider, model) and by prompt — the structured input the cost charts and the
    Dagster `cost` asset consume (see #7).

    Batch (#9) discounts are live: a sample's `billing_tier` decides whether it's
    priced at list or batch rate, so charts reflect what a batched run actually cost.
    Cache-read (#12) discounts land once that path captures the tokens it saves.
    """
    def _bucket(**extra):
        return {**extra, "samples": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}

    total = _bucket()
    by_model: dict[tuple[str, str, str], dict] = {}
    by_prompt: dict[str, dict] = {}
    for r in raw:
        tier = getattr(r, "billing_tier", "standard")
        cost = estimate_cost(r.input_tokens, r.output_tokens, r.model, batch=(tier == "batch"))
        _accumulate(total, r, cost)
        model_bucket = by_model.setdefault(
            (r.provider, r.model, tier),
            _bucket(provider=r.provider, model=r.model, billing_tier=tier),
        )
        _accumulate(model_bucket, r, cost)
        prompt_bucket = by_prompt.setdefault(r.prompt_id, _bucket(prompt_id=r.prompt_id))
        _accumulate(prompt_bucket, r, cost)

    return {
        "total": total,
        "by_model": sorted(by_model.values(), key=lambda d: d["cost"], reverse=True),
        "by_prompt": sorted(by_prompt.values(), key=lambda d: d["prompt_id"]),
    }


def sweep_cost_confidence(prompts, patterns, universe, *, provider_name, model, ns,
                          concurrency, seed, prompt_id="p0"):
    """Run the whole pipeline at a range of sample sizes and report cost vs. precision.

    Returns one point per n: run cost (from `cost_stage`) and the mean 95% CI
    half-width across mentioned producers for `prompt_id`. Deterministic given a
    seed. This is the data behind the "cost of confidence" chart — it makes the
    core token-economics trade-off (linear cost, ~1/√n precision) measurable.
    """
    points = []
    for n in ns:
        provider = get_provider(provider_name, seed=seed)
        raw = collect(prompts, provider=provider, model=model, n=n,
                      concurrency=concurrency, seed=seed)
        mentions = extract_stage(raw, patterns)
        results = aggregate_stage(raw, mentions, universe, seed=seed)
        agg = next((r for r in results if r["prompt_id"] == prompt_id), results[0])
        halves = [(hi - lo) / 2 for lo, hi in agg["ci"].values()]
        mean_half = sum(halves) / len(halves) if halves else 0.0
        cost = cost_stage(raw)
        points.append({
            "n": n,
            "samples": cost["total"]["samples"],
            "cost": cost["total"]["cost"],
            "ci_half_width": mean_half,
        })
    return points


def cost_rows(cost):
    """Flatten the per-(provider, model) cost breakdown into tidy rows for storage."""
    rows = []
    for m in cost["by_model"]:
        avg = m["cost"] / m["samples"] if m["samples"] else 0.0
        rows.append({
            "provider": m["provider"],
            "model": m["model"],
            "billing_tier": m.get("billing_tier", "standard"),
            "samples": m["samples"],
            "input_tokens": m["input_tokens"],
            "output_tokens": m["output_tokens"],
            "cost": m["cost"],
            "cost_per_sample": avg,
        })
    return rows


def metrics_rows(results):
    """Flatten the metrics into tidy rows (one per prompt×producer) for storage."""
    rows = []
    for r in results:
        for name, (share, hits, n) in r["sov"].items():
            if hits > 0:
                lo, hi = r["ci"].get(name, (0.0, 0.0))
                rows.append({
                    "prompt_id": r["prompt_id"],
                    "producer": name,
                    "n": n,
                    "hits": hits,
                    "share": share,
                    "ci_lo": lo,
                    "ci_hi": hi,
                })
    return rows
