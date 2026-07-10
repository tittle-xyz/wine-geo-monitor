# 1. Record architecture decisions

Date: 2026-07-09

## Status

Accepted

## Context

This is a small project, but the interesting choices (a dumb collector, a durable
raw layer, a swappable provider seam) are exactly the ones a reader — human or agent —
will want the *why* for, not just the *what*. Comments explain a line; they don't
explain a decision.

## Decision

Keep short Architecture Decision Records under `docs/adr/`, one per significant
choice, in the Nygard format (Context / Decision / Consequences). Number them
sequentially; never rewrite an accepted one — supersede it with a new record.

## Consequences

The *why* is recoverable from the repo itself, so onboarding doesn't depend on the
author's memory. The cost is discipline: a genuinely load-bearing decision gets a
record; routine changes do not.
