# 5. Provider parity: table stakes, comparable metrics, and optional enrichment

Date: 2026-07-21

## Status

Accepted

## Context

Every model vendor sits behind one `Provider` interface (ADR-0003), but the vendors do
not offer identical capabilities. Anthropic has a server-side web-search tool; OpenAI
exposes token logprobs that Anthropic has none of; a local Ollama model has neither and
no batch discount to chase. Left unstated, that asymmetry invites two opposite mistakes:
building a number we compare *across* vendors on a capability one vendor lacks (so the
comparison is meaningless), or forgoing genuinely useful vendor-specific richness out of
a misplaced insistence that everything be uniform.

The measuring instrument must not be provider-dependent — the same reason it must not be
stochastic (ADR-0002/0004). But "uniform metric" and "no vendor-specific features" are
different rules, and only the first is one we want.

## Decision

Sort every provider capability into one of three tiers.

1. **Table stakes — required to be a provider at all.** Sampling a prompt (`complete`)
   and token accounting for cost. A vendor that cannot meet this contract isn't a viable
   provider; we can't ship the product on it. This is the floor, not a choice.

2. **Comparable metrics — strict parity.** Anything that produces a number compared
   *across* providers (share-of-voice, the attribution verdict) is computed the same way
   for every vendor, and therefore may depend only on table-stakes capabilities. A
   cross-provider number must never rest on a capability not every provider has.

3. **Optional enrichment — vendor-specific allowed, behind the seam.** Extra signals
   (grounded retrieval, and proprietary ones) are advertised with a `supports_X` flag and
   added whenever they deliver value and richness that is actually useful — **even if only
   one vendor can offer them.** The constraints: an enrichment must never contaminate a
   comparable metric, and its scope is stated honestly in reporting ("Anthropic-only",
   "retrieval is per-engine"). Commitment: for a capability every vendor *can* support,
   we reach full parity across providers — parity is a promise, not an aspiration; for a
   capability only one vendor offers, we still implement it where it earns its keep.

The test for any new capability is one question: **is this a number I compare across
providers, or an enrichment of one provider's behavior?** Comparable → table stakes only.
Enrichment → optional, behind the seam.

Enforce it structurally, not by memory: metric code depends only on `complete`;
enrichment paths gate on `supports_X`. A comparable metric then *cannot* reach a
vendor-specific capability without someone deliberately routing it there.

## Consequences

- **Comparability is guaranteed by construction**, not by discipline — the cross-provider
  numbers physically can't depend on a non-universal capability.
- **Vendor richness isn't forfeited.** Proprietary and not-yet-universal capabilities are
  welcome; they just live in the enrichment tier, clearly scoped.
- Two worked cases fix the boundary: **web search + citations** is *capability-symmetric,
  format-asymmetric* — both vendors can ground and cite, the seam normalizes the shape, so
  it carries a parity obligation (Anthropic built; OpenAI owed). **Logprobs** are
  *capability-asymmetric* — Anthropic has none — so they can enrich OpenAI reporting but
  may never back a cross-provider metric.
- Cost: each optional capability carries a `supports_X` flag and an honest label, and each
  symmetric one carries a standing parity to-do until every provider implements it.
