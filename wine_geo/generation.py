"""#23 Phase 2, the model-generation axis: does a brand's KNOWLEDGE gap close over generations?

The daily-partition verifier (verify.py) moves along *time* and checks the retrieval/ranking
levers — the things you change on the web this week. The KNOWLEDGE lever is different in kind:
you cannot make a model *know* a brand by editing a page. It only closes when the provider
retrains and ships a newer generation whose reliable cutoff now spans the brand's documentation
date. So the axis here isn't time — it's **model generation, ordered by reliable cutoff**, and
the clock is quarters-to-years, not days.

Done right, this separates the two sub-causes a knowledge gap can have — which need *opposite*
fixes and look identical until you cross generations:

  - TOO NEW (temporal)        documentation postdates the cutoff -> closes on its own next
                              generation. Lever: get documented now, before the next training run.
  - TOO OBSCURE (prominence)  the cutoff already passed the documentation date, yet the model
                              still doesn't know it -> retraining won't help. Lever: coverage.

The discriminator is predict-then-measure (the CAM X spike method): does the brand flip
disown->knows at the generation whose cutoff first crosses its documentation date? The table of
cutoffs sets the prior; the anchor-verified probe is the ground truth.

    python -m wine_geo.generation "Atelier Ilaria" --provider openai \\
        --models gpt-4o-mini gpt-4.1-mini gpt-5-mini --control Caymus
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from . import config
from .diagnose import _year, knowledge_prior, knowledge_probe, load_cutoffs
from .extract import load_producers

# A measured verdict counts as real knowledge only when it's "knows" — the probe's anchor check
# is what separates that from "confabulates" (confident, but every anchor fact wrong).
_KNOWS = "knows"


@dataclass
class ModelKnowledge:
    """One generation on the axis: its predicted prior and, if probed, its measured verdict."""
    model: str
    cutoff: str | None            # reliable_cutoff string, e.g. "2024-06"
    cutoff_year: int | None
    prior: str                    # knowledge_prior verdict (free, predicted)
    verdict: str | None = None    # measured plurality: knows/hedges/disowns/confabulates/unverified
    tally: dict = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.verdict == _KNOWS


def _plurality(tally):
    return tally.most_common(1)[0][0] if tally else None


def _doc_vs_cutoff(since, cutoff):
    """Is the brand's documentation date "after" / "before" / "same-year" as a model's cutoff?

    Both sides may be month-precise ("2024-11", "2024-06") or year-only ("2024"). When both carry
    a month, compare at month precision; within a shared year but only year precision we genuinely
    can't tell, so return "same-year" and let the caller flag it rather than guess. None if either
    date is missing/unparseable.
    """
    if since is None or not cutoff:
        return None
    s, c = str(since), str(cutoff)
    if len(s) >= 7 and len(c) >= 7:                 # both YYYY-MM -> lexicographic order works
        return "after" if s > c else "before" if s < c else "same-year"
    sy, cy = _year(s), _year(c)
    if sy is None or cy is None:
        return None
    return "after" if sy > cy else "before" if sy < cy else "same-year"


def knowledge_trajectory(brand, since, model_ids, cutoffs, *, provider=None, anchors=(), n=5):
    """Per model, ordered by reliable cutoff ascending: the predicted prior, and — when a provider
    is given — the measured, anchor-verified knowledge verdict. `provider=None` is the free,
    priors-only pass. One provider drives every model id (keep the fleet single-vendor so the only
    thing changing across the axis is the cutoff, not the vendor)."""
    # Order by the full "YYYY-MM" string so months break same-year ties (2024-06 before 2024-10);
    # cutoff_year stays year-granular for the crossing test, which only needs the year.
    ordered = sorted(model_ids, key=lambda m: cutoffs.get(m, {}).get("reliable_cutoff") or "")
    out = []
    for m in ordered:
        cut = cutoffs.get(m, {}).get("reliable_cutoff")
        row = ModelKnowledge(m, cut, _year(cut), knowledge_prior(since, m, cutoffs))
        if provider is not None:
            res = knowledge_probe(brand, provider, m, n=n, anchors=list(anchors))
            row.tally = dict(res["tally"])
            row.verdict = _plurality(res["tally"])
        out.append(row)
    return out


@dataclass
class GenerationVerdict:
    # GAP CLOSED / TOO NEW / TOO OBSCURE / MADE UP / INCONCLUSIVE
    verdict: str
    flip_model: str | None
    detail: str


def grade_trajectory(traj, since) -> GenerationVerdict:
    """Grade the disown->knows flip against the predicted cutoff crossing.

    `traj` is expected in ascending-cutoff order (as knowledge_trajectory returns it), so the last
    probed row is the newest generation and the first `known` row is where the lever fired.
    """
    since_year = _year(since)
    probed = [t for t in traj if t.verdict is not None]
    if not probed:
        return GenerationVerdict("INCONCLUSIVE", None,
                                 "priors only — no models were probed; add a --provider to measure")

    newest = probed[-1]
    first_known = next((t for t in probed if t.known), None)

    if first_known is not None:
        on_schedule = since_year is None or (first_known.cutoff_year or 0) >= since_year
        why = (f"which matches the predicted crossing of the {since} documentation date"
               if on_schedule else
               f"but that predates the {since} documentation date — double-check --since")
        return GenerationVerdict(
            "GAP CLOSED", first_known.model,
            f"flipped disown→knows at {first_known.model} (cutoff {first_known.cutoff}), {why}. "
            "The knowledge gap closed because the generation caught up — no web change could.")

    # No model truly knows it. The date relationship is the primary cause; a newest-model
    # confabulation is only the headline (MADE UP) when the model *could* have learned the brand
    # (its cutoff spans the docs). Otherwise it's a caveat on the real, date-driven verdict.
    rel = _doc_vs_cutoff(since, newest.cutoff)
    made_up = newest.verdict == "confabulates"
    caveat = ("" if not made_up else
              f" And the newest model ({newest.model}) invents confident details rather than "
              "admitting it doesn't know the brand — a reputational risk of its own.")

    if rel == "before":
        # the newest cutoff already spans the documentation — the model had the chance to learn it.
        if made_up:
            return GenerationVerdict(
                "MADE UP", None,
                f"the {since} docs predate the newest model's cutoff ({newest.cutoff}), so it had "
                "the data — yet it recognizes the name and fails every anchor fact. Confident "
                "hallucination, not knowledge; treat as NOT KNOWN.")
        return GenerationVerdict(
            "TOO OBSCURE", None,
            f"the newest model's cutoff ({newest.cutoff}) already spans the {since} documentation, "
            f"yet it still {newest.verdict} the brand — retraining won't fix this. "
            "The lever is coverage/prominence, not time.")

    if rel == "same-year":
        return GenerationVerdict(
            "BORDERLINE", None,
            f"the newest model ({newest.model}, cutoff {newest.cutoff}) and the {since} docs share "
            "a year — can't separate too-new from too-obscure without a month-precise --since."
            + caveat)

    # rel == "after" (documentation postdates the newest cutoff), or no usable date to compare.
    return GenerationVerdict(
        "TOO NEW", None,
        f"the {since} documentation postdates even the newest model ({newest.model}, cutoff "
        f"{newest.cutoff}) — too new for any generation. Re-run when a newer model ships."
        + caveat)


def _defaults_from_producer(brand, producers_path):
    """Pull since / anchors / aliases stored with the brand in producers.json, if present."""
    for p in load_producers(producers_path):
        if p["name"].lower() == brand.lower():
            return p.get("since"), list(p.get("anchors", [])), list(p.get("aliases", []))
    return None, [], []


def _fmt_tally(tally) -> str:
    from collections import Counter
    return ", ".join(f"{k} x{v}" for k, v in Counter(tally).most_common()) or "—"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wine_geo.generation",
        description="Does a brand's knowledge gap close across model generations? (#23 Phase 2)")
    ap.add_argument("brand", help="canonical producer name, e.g. 'Atelier Ilaria'")
    ap.add_argument("--since", help="year the brand became documented (default: producers.json)")
    ap.add_argument("--models", nargs="+", help="model ids on the axis (default: all in cutoffs)")
    ap.add_argument("--anchor", action="append", default=[],
                    help="a falsifiable fact the real brand lets a model state (repeatable; "
                         "default: from producers.json)")
    ap.add_argument("--provider", default="openai",
                    choices=["mock", "anthropic", "openai", "ollama"])
    ap.add_argument("--n", type=int, default=5, help="probe samples per model")
    ap.add_argument("--priors-only", action="store_true", help="free predict-only pass; no probing")
    ap.add_argument("--control", help="a long-known brand to calibrate the probe (reads 'knows')")
    ap.add_argument("--producers", default=str(config.PRODUCERS_PATH))
    args = ap.parse_args(argv)

    cutoffs = load_cutoffs()
    models = args.models or list(cutoffs.keys())
    since_default, anchors_default, _ = _defaults_from_producer(args.brand, args.producers)
    since = args.since or since_default
    anchors = args.anchor or anchors_default

    provider = None
    if not args.priors_only:
        config.load_dotenv()
        from .providers import get_provider
        provider = get_provider(args.provider)

    print(f"# Knowledge over model generations: {args.brand}"
          f"{f' (documented {since})' if since else ''}")
    if provider is not None and not anchors:
        print("  ! no anchors given — can detect disown vs confident, but not verify accuracy")
    src = args.provider if provider else "none"
    print(f"  fleet ordered by reliable cutoff (provider: {src}):")

    traj = knowledge_trajectory(args.brand, since, models, cutoffs,
                                provider=provider, anchors=anchors, n=args.n)
    for t in traj:
        measured = "measured: —"
        if t.verdict:
            measured = f"measured: {t.verdict:12s} ({_fmt_tally(t.tally)})"
        print(f"   {t.model:16s} cutoff {str(t.cutoff or '?'):8s} predict {t.prior:16s} {measured}")
    print()

    grade = grade_trajectory(traj, since)
    print("## Verdict")
    print(f"   [{grade.verdict}] {grade.detail}")

    if args.control and provider is not None:
        ctrl = knowledge_trajectory(args.control, None, models, cutoffs,
                                    provider=provider, anchors=[args.control.split()[0]], n=args.n)
        all_known = all(c.known for c in ctrl)
        mark = "✓" if all_known else "✗ (probe/anchor may be miscalibrated)"
        knew = ", ".join(f"{c.model}:{c.verdict}" for c in ctrl)
        print(f"\n   calibration — {args.control} (long-known) should read 'knows' on every model: "
              f"{mark}\n     {knew}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
