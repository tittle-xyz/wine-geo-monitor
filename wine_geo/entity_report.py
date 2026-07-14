"""Multi-entity angles on a saved run — regions, specificity, and rating surface.

All three are *re-derived from the durable raw layer* (no new model calls) and all
three are computed with the same deterministic regex matcher that backs the producer
metric — so they're reproducible, not stochastic. This is the "free metrics" pass:
point it at any raw.jsonl and it answers three questions the producer view can't.

    python -m wine_geo.entity_report out/claude/raw.jsonl

1. Region share-of-voice + prestige-vs-value tilt — does value-seeking still get
   steered to expensive AVAs? (the brand-bias finding, one level up)
2. Specificity — does the model name a producer, or dodge to "a nice Napa Cab"?
3. Rating surface — does the AI channel cite scores/publications at all?
"""

from __future__ import annotations

import re
import sys

from .config import DATA_DIR, PRODUCERS_PATH
from .extract import (
    build_forms,
    build_patterns,
    extract_disambiguated,
    extract_mentions,
    has_score,
    load_producers,
    mask_matches,
)
from .schema import RawSample, read_jsonl
from .stats import share_of_voice

REGIONS_PATH = DATA_DIR / "regions.json"
PUBLICATIONS_PATH = DATA_DIR / "publications.json"

# A "generic gesture": a place + a Cabernet/blend word with no producer attached, e.g.
# "a Napa Cab", "Sonoma Cabernet", "Bordeaux blend". Built from the region vocabulary.
_CATEGORY_WORDS = r"(?:cabs?|cabernets?|red blends?|blends?|reds?)"


def _generic_pattern(region_forms: list[str]) -> re.Pattern:
    alts = "|".join(re.escape(f) for f in sorted(region_forms, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alts})\s+{_CATEGORY_WORDS}\b", re.IGNORECASE)


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m wine_geo.entity_report <raw.jsonl>")
        return 1
    path = argv[0]

    producers = load_producers(PRODUCERS_PATH)
    regions = load_producers(REGIONS_PATH)
    publications = load_producers(PUBLICATIONS_PATH)
    prod_pat = build_patterns(producers)
    region_forms = build_forms(regions)  # longest-form-first, for disambiguation
    pub_pat = build_patterns(publications)
    tier_of = {r["name"]: r.get("tier", "?") for r in regions}
    region_strs = [r["name"] for r in regions] + [
        a for r in regions for a in r.get("aliases", [])
    ]
    generic_re = _generic_pattern(region_strs)

    samples = [s for s in read_jsonl(path, RawSample) if s.response_text and not s.error]
    if not samples:
        print(f"no usable responses in {path}")
        return 1
    n = len(samples)
    # Regions are matched on producer-MASKED text so 'Rutherford Hill' the producer
    # can't be miscounted as 'Rutherford' the AVA; and disambiguated so 'Sonoma County'
    # isn't also credited to Sonoma Valley via the bare 'Sonoma' alias.
    region_sets = [
        extract_disambiguated(mask_matches(s.response_text, prod_pat), region_forms)
        for s in samples
    ]
    prod_sets = [extract_mentions(s.response_text, prod_pat) for s in samples]

    print("\n" + "=" * 68)
    print(f"  ENTITY REPORT — {path}")
    print(f"  {n} responses ({samples[0].provider} / {samples[0].model})")
    print("=" * 68)

    # 1. Region share-of-voice + prestige/value tilt --------------------------
    sov = share_of_voice(region_sets, [r["name"] for r in regions])
    ranked = sorted(sov.items(), key=lambda kv: -kv[1][0])
    print("\n[1] REGION share-of-voice (top 10)")
    print(f"    {'region':<22} {'tier':<9} share")
    for name, (share, hits, _) in ranked[:10]:
        if hits:
            bar = "█" * round(share * 20)
            print(f"    {name:<22} {tier_of[name]:<9} {share:5.0%}  {bar}")
    prestige_hits = sum(
        1 for s in region_sets if any(tier_of[r] == "prestige" for r in s)
    )
    value_hits = sum(1 for s in region_sets if any(tier_of[r] == "value" for r in s))
    print(f"\n    responses naming a PRESTIGE region: {_pct(prestige_hits, n)}")
    print(f"    responses naming a VALUE region:    {_pct(value_hits, n)}")
    print("    (every default prompt is value-seeking — so a prestige tilt here is")
    print("     the brand-bias finding one level up: asked cheap, points expensive.)")

    # 2. Specificity ----------------------------------------------------------
    named_producer = sum(1 for s in prod_sets if s)
    generic = sum(1 for s in samples if generic_re.search(s.response_text))
    print("\n[2] SPECIFICITY — does it commit to a brand or dodge to a category?")
    print(f"    responses naming >=1 producer:      {_pct(named_producer, n)}")
    print(f"    responses using a generic gesture:  {_pct(generic, n)}  (e.g. 'a Napa Cab')")

    # 3. Rating surface -------------------------------------------------------
    pub_sets = [extract_mentions(s.response_text, pub_pat) for s in samples]
    with_pub = sum(1 for s in pub_sets if s)
    with_score = sum(1 for s in samples if has_score(s.response_text))
    pub_counter: dict[str, int] = {}
    for s in pub_sets:
        for p in s:
            pub_counter[p] = pub_counter.get(p, 0) + 1
    print("\n[3] RATING SURFACE — does the AI channel use scores/publications?")
    print(f"    responses citing a publication:     {_pct(with_pub, n)}")
    print(f"    responses citing a numeric score:   {_pct(with_score, n)}  (90-100 pts)")
    if pub_counter:
        top = ", ".join(
            f"{k} ({v})" for k, v in sorted(pub_counter.items(), key=lambda kv: -kv[1])
        )
        print(f"    publications named: {top}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
