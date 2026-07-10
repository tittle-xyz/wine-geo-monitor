# 2. A dumb collector and a durable raw layer

Date: 2026-07-09

## Status

Accepted

## Context

A GEO monitor's expensive step is the LLM API calls; everything after (mention
extraction, share-of-voice, confidence intervals) is cheap and will change often as
the matching and metrics improve. If extraction runs inside the sampling loop, the
expensive step is welded to the cheap one: every change to analysis means re-paying
for API calls, and there is no record of what the model actually said.

## Decision

Split the work into pipeline stages with a durable layer between each (`collect →
extract → aggregate`), and keep the collector **dumb** — it only captures raw model
responses (`RawSample`) and does no interpretation. Downstream stages are pure
functions of the stored raw layer.

## Consequences

- **Collection cost is decoupled from analysis iteration.** The API calls run once;
  extraction and metrics re-run for free against the stored raw layer.
- **The raw layer is an audit trail.** Every metric is a re-derivable function of the
  actual responses, so "why did the score change?" is answerable.
- The seam is data at rest (JSONL), so stages can run on different schedules, be
  backfilled independently, and — if ever needed — be reimplemented in another language.
- Cost: an extra storage layer and the discipline of never letting the collector
  interpret. Worth it.
