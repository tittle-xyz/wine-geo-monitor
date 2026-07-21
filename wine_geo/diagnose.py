"""Attribution: diagnose *why* a brand's share-of-voice is what it is (issue #23, Phase 1).

Two brands can share the same 0% symptom for opposite reasons — the model has never
heard of one (a **knowledge gap**), and simply never picks the other (a **ranking gap**).
They need opposite fixes, so telling them apart is the whole job here.

The design the CAM X spike pointed to: **cheap lookups first, paid probes only to confirm.**

    LOOKUP (free)                      PROBE (paid, optional)
    ─────────────────                  ──────────────────────
    knowledge_prior  date vs cutoff    knowledge_probe  does the model actually know it?
    ranking_signal   mine saved runs   ─
    prominence       web footprint     ─   (TODO)
    retrieval_signal search presence   ─   (TODO — needs a grounded surface)

`knowledge_prior` and `ranking_signal` need no network and no key: they set a prior that
tells you whether a paid probe is even worth running, and for which (brand × model) cells.

    python -m wine_geo.diagnose "CAM X" --since 2025 --model claude-haiku-4-5
    python -m wine_geo.diagnose "CAM X" --since 2025 --probe --provider anthropic
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from . import config
from .extract import build_patterns, extract_mentions, load_producers
from .prominence import (
    load_snapshot,
    prior_with_prominence,
    prominence_level,
)
from .schema import read_jsonl

CUTOFFS_PATH = config.DATA_DIR / "model_cutoffs.json"


# --- ground-truth lookups ----------------------------------------------------

def load_cutoffs(path=CUTOFFS_PATH) -> dict:
    """{model: {reliable_cutoff, training_cutoff, source}} from the data file."""
    return json.loads(Path(path).read_text())["models"]


def _year(date_str: str | int | None) -> int | None:
    """Pull a year out of an int year or a 'YYYY' / 'YYYY-MM' string."""
    if date_str is None:
        return None
    if isinstance(date_str, int):
        return date_str
    try:
        return int(str(date_str)[:4])
    except ValueError:
        return None


def knowledge_prior(since, model: str, cutoffs: dict) -> str:
    """A priori guess at whether `model` could know a brand documented since `since`.

    Compares the brand's documentation year to the model's *reliable* cutoff. A brand
    that postdates the cutoff can't be in the model's parametric knowledge — that's a
    knowledge gap you can predict before spending a token.
    """
    entry = cutoffs.get(model)
    if not entry:
        return "unknown-cutoff"
    cutoff_year = _year(entry.get("reliable_cutoff"))
    since_year = _year(since)
    if since_year is None or cutoff_year is None:
        return "unknown-date"
    if since_year > cutoff_year:
        return "likely-unknown"       # postdates the cutoff → knowledge gap a priori
    if since_year <= cutoff_year - 1:
        return "likely-known"         # comfortably before — but prominence still matters
    return "borderline"               # same year as the cutoff


# --- ranking signal (mine the already-saved raw responses; free) -------------

def ranking_signal(brand: str, producers: list[dict], run_dirs) -> dict:
    """How often `brand` is actually named in saved responses, per model.

    Uses the repo's own regex matcher — the official instrument — so these counts line
    up with share-of-voice. A near-zero rate means 'not surfaced'; crossed with the
    knowledge probe it separates not-known from not-ranked.
    """
    entry = next((p for p in producers if p["name"] == brand), None)
    if entry is None:
        raise ValueError(f"{brand!r} not in the producer list")
    patterns = build_patterns([entry])

    hits: Counter = Counter()
    totals: Counter = Counter()
    for run_dir in run_dirs:
        for fp in glob.glob(str(Path(run_dir) / "**" / "raw.jsonl"), recursive=True):
            for rec in read_jsonl(fp):
                model = rec.get("model", "?")
                totals[model] += 1
                if brand in extract_mentions(rec.get("response_text", ""), patterns):
                    hits[model] += 1
    return {
        m: {"hits": hits[m], "total": totals[m], "rate": hits[m] / totals[m] if totals[m] else 0.0}
        for m in sorted(totals)
    }


# --- knowledge probe (paid; confirms the prior) ------------------------------

PROBE = (
    "Tell me about the wine producer '{brand}'. Who is behind it, roughly when did it "
    "start, and what kind of wine do they make? If you're not certain it exists or you "
    "don't recognize it, say so plainly."
)

DISOWN_MARKERS = [
    "not familiar", "not aware", "no information", "couldn't find", "could not find",
    "don't have", "do not have", "not certain", "not sure", "unable to find",
    "not recognize", "doesn't appear", "does not appear", "cannot verify", "can't verify",
    "may not exist", "no record", "not a wine producer i", "not well-known enough",
]


def classify_probe(text: str, anchors: list[str]) -> str:
    """Label one probe response: knows / hedges / disowns / confabulates / unverified.

    `anchors` are falsifiable facts the *real* brand would let the model state (founder,
    era, style). They're what makes this honest: a confident answer with zero accurate
    anchors is a confabulation, not knowledge — the De Négoce 'founded by Jon Bonné' case.
    """
    t = text.lower()
    disowned = any(m in t for m in DISOWN_MARKERS)
    anchor_hits = sum(1 for a in anchors if a.lower() in t)
    if disowned and anchor_hits == 0:
        return "disowns"
    if disowned and anchor_hits:
        return "hedges"
    if not anchors:
        return "unverified"           # answered confidently, but we gave it nothing to check
    if anchor_hits:
        return "knows"
    return "confabulates"             # confident, no accurate anchor


def knowledge_probe(brand, provider, model, *, n, anchors) -> dict:
    """Ask the model directly about `brand` n times; return the verdict tally + a sample.

    Imported lazily so the free lookups never require a provider or a key.
    """
    from .providers import estimate_cost

    verdicts: list[str] = []
    sample = ""
    cost = 0.0
    for i in range(n):
        c = provider.complete(PROBE.format(brand=brand), model=model)
        cost += estimate_cost(c.input_tokens, c.output_tokens, model)
        verdicts.append(classify_probe(c.text, anchors))
        if i == 0:
            sample = c.text
    return {"tally": Counter(verdicts), "sample": sample, "cost": cost, "n": n}


# --- classification ----------------------------------------------------------

def diagnose(known: bool | None, ranked_rate: float, *, ranked_floor: float = 0.02) -> str:
    """Cross known? × ranked? into a cause. `known=None` means the probe wasn't run."""
    surfaced = ranked_rate > ranked_floor
    if known is None:
        if surfaced:
            return "known-and-ranked (surfaced; no probe needed)"
        return "inconclusive — not surfaced, but is that not-known or not-ranked? run --probe"
    if not known:
        return "NOT KNOWN (knowledge gap) — optimization can't help; needs coverage/retrieval/paid"
    if not surfaced:
        return "NOT RANKED (ranking gap) — known but not picked; fixable via attributes/positioning"
    return "known-and-ranked"


# --- TODO(#23) seams — flesh these out as we test ----------------------------

def prominence(brand: str) -> dict:
    """TODO(#23): web-footprint proxy (Wikipedia/Wikidata presence, result count).

    Separates 'too new' (postdates cutoff) from 'too obscure' (predates it but thinly
    documented) — the Cameron Hughes case, where date said 'known' but the model hedged.
    Knowledge ≈ date × prominence × model size.
    """
    raise NotImplementedError("prominence proxy not built yet — see #23")


def retrieval_signal(brand: str, query: str) -> dict:
    """TODO(#23): the not-retrieved probe. Needs a grounded surface (Perplexity / Google
    AI mode) — base-model APIs don't retrieve, so this mode is invisible to `knowledge_probe`.
    Cheap web checks that steer it: does the brand rank for `query`, is it in the favored
    sources, does robots.txt block GPTBot/ClaudeBot/PerplexityBot.
    """
    raise NotImplementedError("retrieval probe not built yet — see #23")


# --- CLI ---------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wine_geo.diagnose",
        description="Diagnose why a brand's share-of-voice is what it is (#23).",
    )
    ap.add_argument("brand", help="canonical producer name, e.g. 'CAM X'")
    ap.add_argument("--since", help="year the brand became documented, e.g. 2025")
    ap.add_argument("--runs", nargs="+", default=["out"], help="dirs holding saved raw.jsonl")
    ap.add_argument("--producers", default=str(config.PRODUCERS_PATH))
    ap.add_argument("--model", default=config.DEFAULT_MODEL, help="model to reason/probe about")
    ap.add_argument("--anchor", action="append", default=[],
                    help="a falsifiable fact the real brand would let a model state (repeatable)")
    ap.add_argument("--probe", action="store_true", help="also run the paid knowledge probe")
    ap.add_argument("--provider", default="anthropic",
                    choices=["mock", "anthropic", "openai", "ollama"])
    ap.add_argument("--n", type=int, default=5, help="probe samples")
    ap.add_argument("--prominence-store", default=config.PROMINENCE_PATH,
                    help="durable prominence snapshot (JSONL)")
    ap.add_argument("--refresh-prominence", action="store_true",
                    help="fetch prominence from Wikipedia/Wikidata now and snapshot it")
    args = ap.parse_args(argv)

    producers = load_producers(args.producers)
    cutoffs = load_cutoffs()
    entry = next((p for p in producers if p["name"] == args.brand), None)

    print(f"# Diagnosis: {args.brand}\n")

    # 1. knowledge prior — date only (free)
    print("## Knowledge prior (documentation date vs reliable cutoff)")
    prior_for_model = None
    for model, cut in cutoffs.items():
        p = knowledge_prior(args.since, model, cutoffs)
        marker = "  <- target model" if model == args.model else ""
        print(f"   {model:22s} cutoff {cut.get('reliable_cutoff','?')}  -> {p}{marker}")
        if model == args.model:
            prior_for_model = p
    print()

    # 2. prominence — refine the prior with how much the brand was written about (free lookup,
    #    snapshotted so a diagnosis stays re-derivable even as Wikipedia changes; ADR-0002).
    print(f"## Prominence (Wikipedia/Wikidata footprint, snapshot: {args.prominence_store})")
    if args.refresh_prominence:
        from .prominence import fetch_prominence, save_snapshot
        aliases = entry.get("aliases", []) if entry else []
        rec = fetch_prominence(args.brand, aliases, config.DOMAIN_TERMS)
        save_snapshot(args.prominence_store, rec)
        print(f"   fetched: {rec['resolved_title']} — {rec['pageviews']} pv/mo, "
              f"{rec['summary_len']} bytes -> {rec['level']}")
    snap = load_snapshot(args.prominence_store).get(args.brand)
    if snap:
        level = prominence_level(snap)
        prior_for_model = prior_with_prominence(prior_for_model, level)
        print(f"   {args.brand}: {snap.get('resolved_title')} -> {level}  "
              f"(refined prior: {prior_for_model})")
    else:
        print("   (no snapshot — run with --refresh-prominence)")
    print()

    # 3. ranking signal (free)
    print(f"## Ranking signal (saved responses in {args.runs})")
    signal = ranking_signal(args.brand, producers, args.runs)
    ranked_rate = signal.get(args.model, {}).get("rate", 0.0)
    for model, s in signal.items():
        print(f"   {model:22s} {s['hits']:4d}/{s['total']:<5d} {100*s['rate']:5.1f}%")
    if not signal:
        print("   (no saved runs found)")
    print()

    # 4. knowledge probe (paid, optional)
    known = None
    if args.probe:
        config.load_dotenv()
        from .providers import get_provider
        provider = get_provider(args.provider)
        print(f"## Knowledge probe ({args.model}, n={args.n})")
        if not args.anchor:
            print("   ! no --anchor given: can detect disown vs confident, but not verify accuracy")
        res = knowledge_probe(args.brand, provider, args.model, n=args.n, anchors=args.anchor)
        tally = ", ".join(f"{k} x{v}" for k, v in res["tally"].most_common())
        print(f"   verdicts: {tally}")
        print(f"   sample:   {res['sample'][:200].replace(chr(10), ' ')}...")
        print(f"   spend:    ${res['cost']:.4f}")
        # "known" if the plurality verdict is knows/hedges/unverified (i.e. not disown/confab)
        top = res["tally"].most_common(1)[0][0]
        known = top in ("knows", "hedges", "unverified")
        print()

    # 5. diagnosis
    print("## Diagnosis")
    print(f"   {diagnose(known, ranked_rate)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
