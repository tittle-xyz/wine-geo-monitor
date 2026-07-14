"""Ratings A/B: does asking for "90+ point" wines change what the model recommends?

Matched prompt pairs (see data/rating_ab_prompts.json) — same request, one plain and
one rating-primed. Generate both conditions per model with the normal CLI, then this
module compares them. The design dodges the fame confound: we EXPLICITLY ask for
highly-rated wines, so if the recommendation set moves no more than the model's own
run-to-run noise, ratings don't steer it — no external score dataset needed.

Two numbers carry it:
- rec gap = (within-condition pairwise Jaccard) − (across-condition pairwise Jaccard).
  The within value is the stochastic noise floor; a gap near 0 means priming changed
  nothing real. Comparing gaps across models is the point — run several and eyeball.
- rating surface = how often an answer cites a numeric score / a publication, neutral
  vs primed. This moves even when the recommendations don't — the "rating theater".

    # generate first (normal CLI), then:
    python -m wine_geo.ab_experiment llama:out/ab_n:out/ab_r claude:out/c_n:out/c_r
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

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
from .stats import mean_pairwise_jaccard

AB_PROMPTS_PATH = DATA_DIR / "rating_ab_prompts.json"


def write_prompt_files(out_dir) -> tuple[str, str]:
    """Materialize the base/primed prompt pairs as two files for the CLI's --prompts.

    Same order in both, so prompt p_i in one condition pairs with p_i in the other.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(AB_PROMPTS_PATH.read_text())
    base = out_dir / "prompts_base.txt"
    primed = out_dir / "prompts_primed.txt"
    base.write_text("".join(p["base"] + "\n" for p in pairs))
    primed.write_text("".join(p["primed"] + "\n" for p in pairs))
    return str(base), str(primed)


def _by_prompt(path):
    grouped = defaultdict(list)
    for s in read_jsonl(path, RawSample):
        if s.response_text and not s.error:
            grouped[s.prompt_id].append(s)
    return grouped


def _jaccard(a: set, b: set) -> float:
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def ab_metrics(neutral_dir, rated_dir) -> dict:
    """Compute the A/B metrics for one model from its two run directories."""
    producers = load_producers(PRODUCERS_PATH)
    pp = build_patterns(producers)
    pubs = build_patterns(load_producers(DATA_DIR / "publications.json"))
    regions = load_producers(DATA_DIR / "regions.json")
    rforms = build_forms(regions)
    tier = {r["name"]: r.get("tier", "?") for r in regions}

    neutral = _by_prompt(f"{neutral_dir}/raw.jsonl")
    rated = _by_prompt(f"{rated_dir}/raw.jsonl")

    def psets(samples):
        return [extract_mentions(x.response_text, pp) for x in samples]

    within, across = [], []
    for pid in neutral:
        ns, rs = psets(neutral[pid]), psets(rated.get(pid, []))
        within += [mean_pairwise_jaccard(ns), mean_pairwise_jaccard(rs)]
        pairs = [_jaccard(a, b) for a in ns for b in rs]
        if pairs:
            across.append(sum(pairs) / len(pairs))
    floor = sum(within) / len(within) if within else 1.0
    acr = sum(across) / len(across) if across else 1.0

    alln = [x for v in neutral.values() for x in v]
    allr = [x for v in rated.values() for x in v]

    def rate(samples, fn):
        return sum(1 for x in samples if fn(x.response_text)) / len(samples) if samples else 0.0

    def prestige(t):
        hits = extract_disambiguated(mask_matches(t, pp), rforms)
        return any(tier[r] == "prestige" for r in hits)

    def cites_pub(t):
        return bool(extract_mentions(t, pubs))

    return {
        "n": len(alln),
        "floor": floor,
        "across": acr,
        "gap": floor - acr,
        "score_n": rate(alln, has_score),
        "score_r": rate(allr, has_score),
        "pub_n": rate(alln, cites_pub),
        "pub_r": rate(allr, cites_pub),
        "prestige_n": rate(alln, prestige),
        "prestige_r": rate(allr, prestige),
    }


def format_table(results: list[tuple[str, dict]]) -> str:
    """Render the cross-model comparison as a fixed-width text table."""
    w = 18
    line = "-" * (36 + w * len(results))

    def pct(x):
        return f"{x:.0%}"

    def row(label, fn):
        cells = "".join(f"{fn(m):>{w}}" for _, m in results)
        return f"  {label:<34}" + cells

    header = f"  {'metric':<34}" + "".join(f"{lbl:>{w}}" for lbl, _ in results)
    out = [
        "  A/B ACROSS MODELS  (neutral → rating-primed)", "  " + line, header, "  " + line,
        row("samples/condition", lambda m: str(m["n"])),
        row("rec noise floor (within-Jaccard)", lambda m: f"{m['floor']:.2f}"),
        row("rec across-condition Jaccard", lambda m: f"{m['across']:.2f}"),
        row("  gap (≈0 = priming moved nothing)", lambda m: f"{m['gap']:+.2f}"),
        row("cites a numeric score", lambda m: f"{pct(m['score_n'])}→{pct(m['score_r'])}"),
        row("cites a publication", lambda m: f"{pct(m['pub_n'])}→{pct(m['pub_r'])}"),
        row("prestige-region tilt", lambda m: f"{pct(m['prestige_n'])}→{pct(m['prestige_r'])}"),
    ]
    return "\n".join(out)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m wine_geo.ab_experiment <label>:<neutral_dir>:<rated_dir> ...")
        return 1
    results = []
    for spec in argv:
        label, ndir, rdir = spec.split(":")
        results.append((label, ab_metrics(ndir, rdir)))
    print("\n" + format_table(results) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
