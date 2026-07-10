# 3. Plain Python with a swappable provider seam

Date: 2026-07-09

## Status

Accepted

## Context

The workload is I/O-bound (waiting on rate-limited API calls), so raw language speed
barely matters. What does matter: fast iteration on the analysis, a rich stats/plotting
ecosystem, and the ability to survive a model vendor repricing or deprecating a model.
Two forces pull in different directions — the analysis half wants Python; a future
high-volume ingestion worker might want a compiled language.

## Decision

Write it in plain Python, standard-library-only in the core, and put every model vendor
behind one `Provider` interface (`mock` / `anthropic` / `openai`). Orchestrators (the
CLI, the Dagster assets) are thin wrappers over the stage functions in `pipeline.py`.

## Consequences

- **Vendor independence.** Swapping providers, or adding one, is a new class behind the
  same interface — the runner never knows which vendor it's talking to.
- **Offline by default.** The `mock` provider means tests and CI run with no API key.
- **The seam protects a future language change.** Because stages talk through data at
  rest, the collector could be reimplemented as a compiled binary later without touching
  the analysis — a decision deferred, not foreclosed.
- Cost: an interface and a mock to maintain. Cheap relative to the flexibility.
