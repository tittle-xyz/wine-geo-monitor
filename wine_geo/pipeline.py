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
from .runner import sample_prompt
from .schema import Mention, RawSample
from .stats import bootstrap_ci, mean_pairwise_jaccard, share_of_voice


def collect(prompts, *, provider, model, n, concurrency, seed=None, run_id=None, ts=None):
    """Dumb collector: sample each prompt n times, capture raw responses. No interpretation."""
    run_id = run_id or uuid4().hex[:12]
    ts = ts or datetime.now(timezone.utc).isoformat()
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
