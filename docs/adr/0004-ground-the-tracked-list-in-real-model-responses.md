# 4. Ground the tracked producer list in real model responses

Date: 2026-07-13

## Status

Accepted

## Context

The tracked producer list (`wine_geo/data/producers.json`) began as an *assumption* —
the famous Napa names the `mock` provider was built around. The mock's whole premise
is that AI shopping answers cluster on a handful of marketing-heavy brands, so the
list was the cast for that story.

A first real run (Claude `claude-haiku-4-5`, n=25 across the six default prompts,
saved under the durable raw layer) showed two things:

1. **The branding-bias thesis holds.** Asked for the *best value* Napa Cabernet
   *under $40*, the model still leads with famous, over-budget names — Stag's Leap
   (76%), Caymus (56%), Silver Oak (20%) — none of which fit the price constraint.
2. **But the list was measuring the wrong universe.** A large share of the model's
   actual recommendations were producers the list did not track — Bogle-tier value
   labels like Barefoot, Blackstone, Columbia Crest, Franciscan, 14 Hands, Rutherford
   Hill, Rodney Strong. So share-of-voice was answering *"how often does the model
   name one of our assumed brands?"* rather than *"who does the model actually
   recommend?"*

An assumption-based list quietly bakes the answer in. If we only track the brands we
expect, we can only ever confirm our expectations.

## Decision

**Derive the tracked list from real model output, not from intuition.** Harvest the
producers models actually name across a real run, add the recurring real ones, and
deliberately record the négociant hidden-labels we want to watch (de Négoce, and
Cameron Hughes' current label **CAM X** — replacing the barely-attested early name
"The Negociant Wine Company"). List-changing decisions are made from run data held in
the raw layer, which makes them auditable and repeatable.

The `mock` provider stays synthetic and unchanged: it demonstrates the *machinery*;
**real runs define the universe.**

## Consequences

- **Share-of-voice reflects the real recommendation landscape**, not our priors — the
  headline finding gets more honest and more interesting.
- **The list will drift as models change.** That is expected, not a defect: refresh
  it from periodic real runs. The durable raw layer makes each refresh cheap and
  auditable (re-derive, don't re-pay).
- **Coverage is never complete** — this is an open-world problem, and a model can
  always name a brand we don't track. A follow-up should *measure* that coverage gap
  (what fraction of a model's recommendations fall inside the list) and consider an
  open-world discovery mode. Tracked in the roadmap.
- **The mock now shows newly-added real brands as "never mentioned"** (they aren't in
  its synthetic weights). Acceptable: mock output is illustrative, not empirical.
