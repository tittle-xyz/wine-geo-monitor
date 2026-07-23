"""Did an intervention actually move a brand's share-of-voice? (#23 Phase 2, deliverable 3.)

"It went up" isn't proof. Share drifts on its own — the model updates, the category shifts,
or it's plain sampling noise. So the honest question is: did the treated brand move *more than
a comparable untouched brand (the control)*, by more than the measurement wobble? The control
absorbs the shocks that hit both brands, so the leftover — how much MORE the treated brand
changed than the control — is what's specific to the intervention.

Everything here is stated in plain terms on purpose: a "give-or-take" is the wobble on a
measured percent from limited sampling (a 95% confidence interval); "moved beyond the noise
floor" means a change bigger than that wobble (the #11 drift test); the "difference of the
changes" is the treated brand's change minus the control's. The core is pure and testable;
wiring it to the daily partitions (via trend.load_series) is the next step.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

Z = 1.96  # 95% — matches the stored bootstrap CIs

# A measurement is a (share, ci_lo, ci_hi) tuple — exactly trend.load_series's format.


def _se(lo: float, hi: float) -> float:
    """The give-or-take (standard error) implied by a 95% CI half-width.

    Approximate: bootstrap CIs aren't perfectly symmetric, so this is a working estimate,
    not a guarantee — which is why the verdict is graded, not a p-value.
    """
    return max(hi - lo, 0.0) / (2 * Z)


def _disjoint(a, b) -> bool:
    """Do two measurements' give-or-take bands NOT overlap? (a real move, not just wobble)"""
    _, a_lo, a_hi = a
    _, b_lo, b_hi = b
    return a_lo > b_hi or a_hi < b_lo


def _pts(x: float) -> str:
    return f"{x * 100:+.0f} pts"


@dataclass
class VerifyResult:
    treated_change: float       # how much the treated brand moved
    control_change: float       # how much the untouched comparison brand moved
    diff_of_changes: float      # treated_change - control_change (what's specific to treated)
    give_or_take: float         # the wobble on that difference (~1.96 x its standard error)
    treated_moved: bool         # treated's own bands are disjoint (moved beyond its noise)
    control_moved: bool
    beyond_wobble: bool         # |diff_of_changes| exceeds its own give-or-take
    verdict: str                # graded, plain-language


def verify_intervention(treated_base, treated_latest, control_base, control_latest) -> VerifyResult:
    """Compare a treated brand's before/after to an untouched control's, honestly.

    Each argument is a (share, ci_lo, ci_hi) measurement. Returns the two changes, the
    difference between them, the wobble on that difference, and a graded verdict.
    """
    dt = treated_latest[0] - treated_base[0]
    dc = control_latest[0] - control_base[0]
    did = dt - dc

    # Wobble on each change (two wobbly points → their difference is wobblier), then on the
    # difference of the two changes. math.hypot adds them in quadrature (variances add).
    se_t = math.hypot(_se(*treated_base[1:]), _se(*treated_latest[1:]))
    se_c = math.hypot(_se(*control_base[1:]), _se(*control_latest[1:]))
    give = Z * math.hypot(se_t, se_c)

    treated_moved = _disjoint(treated_base, treated_latest)
    control_moved = _disjoint(control_base, control_latest)
    beyond = give > 0 and abs(did) > give

    return VerifyResult(
        treated_change=dt, control_change=dc, diff_of_changes=did, give_or_take=give,
        treated_moved=treated_moved, control_moved=control_moved, beyond_wobble=beyond,
        verdict=_verdict(dt, dc, did, treated_moved, control_moved, beyond),
    )


def _verdict(dt, dc, did, treated_moved, control_moved, beyond) -> str:
    if not treated_moved:
        return (f"NO DETECTABLE EFFECT — the treated brand's move ({_pts(dt)}) is within its "
                f"give-or-take; nothing to attribute yet (is it too soon for this lever?)")
    if beyond:
        return (
            f"LIKELY REAL — treated moved {_pts(dt)} while the control moved {_pts(dc)}; the "
            f"difference ({_pts(did)}) clears the wobble, so the move is specific to the "
            f"brand. Caveat: a confound timed exactly with the intervention can't be ruled out"
        )
    if control_moved:
        return (f"CONFOUNDED — treated moved {_pts(dt)} but the control moved {_pts(dc)} too; "
                f"the category shifted, not attributable to the intervention")
    return (f"SUGGESTIVE — treated moved {_pts(dt)} and the control barely did ({_pts(dc)}), but "
            f"the difference ({_pts(did)}) is within its own give-or-take; needs more samples")


# --- wiring to the durable daily partitions (trend.load_series's (share, ci_lo, ci_hi)) ---


def _measure(series, brand, date):
    """(share, ci_lo, ci_hi) for a brand on a date; absent == not recommended == 0.

    Mirrors trend._at: a brand that simply wasn't named on a day reads as a real 0%, not as
    missing data — which is exactly the before-state an intervention is trying to move off of.
    """
    return series.get(brand, {}).get(date, (0.0, 0.0, 0.0))


@dataclass
class PartitionVerify:
    """A verdict wired to real days: which two dates, the two brands, their four measurements,
    and the pure VerifyResult over them. `*_seen` flags a brand that never appears at all."""
    base: str
    latest: str
    treated: str
    control: str
    treated_base: tuple
    treated_latest: tuple
    control_base: tuple
    control_latest: tuple
    result: VerifyResult
    treated_seen: bool
    control_seen: bool


def verify_from_partitions(root, treated, control, *, prompt_id="p0", base=None, latest=None):
    """Read the durable daily metrics under `root` and verify the treated brand's before/after
    move against an untouched control's.

    Baseline/latest default to the first/last day with data (like the #11 drift test); pass
    `base`/`latest` to pin exact dates — e.g. the day before you intervened vs. two weeks after.
    Raises ValueError if fewer than two days exist or a pinned date is missing.
    """
    from .trend import load_series

    dates, series = load_series(root, prompt_id)
    if len(dates) < 2:
        raise ValueError(f"need >= 2 daily partitions under {root!r} (found {len(dates)})")
    b = base or dates[0]
    la = latest or dates[-1]
    for d in (b, la):
        if d not in dates:
            raise ValueError(f"date {d!r} not among partitions {dates}")

    tb, tl = _measure(series, treated, b), _measure(series, treated, la)
    cb, cl = _measure(series, control, b), _measure(series, control, la)
    return PartitionVerify(
        base=b, latest=la, treated=treated, control=control,
        treated_base=tb, treated_latest=tl, control_base=cb, control_latest=cl,
        result=verify_intervention(tb, tl, cb, cl),
        treated_seen=treated in series, control_seen=control in series,
    )


def _fmt_move(label, name, base_m, latest_m, moved) -> str:
    band = "moved beyond its give-or-take" if moved else "within its give-or-take"
    return (f"   {label:8s}{name:22.22}{base_m[0] * 100:4.0f}% -> {latest_m[0] * 100:4.0f}%"
            f"   ({_pts(latest_m[0] - base_m[0])})   {band}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wine_geo.verify",
        description="Did an intervention move a brand's share-of-voice? "
                    "Difference-of-changes vs. an untouched control (#23 Phase 2).",
    )
    ap.add_argument("root", help="dir of daily partitions (each holds <date>/metrics.jsonl)")
    ap.add_argument("brand", help="the treated brand — the one you intervened on")
    ap.add_argument("--control", required=True,
                    help="an untouched comparison brand; it absorbs shocks that hit both")
    ap.add_argument("--prompt", default="p0")
    ap.add_argument("--base", help="baseline date (default: first partition)")
    ap.add_argument("--latest", help="after date (default: last partition)")
    ap.add_argument("--lever", help="what you changed, printed in the report header")
    args = ap.parse_args(argv)

    try:
        pv = verify_from_partitions(args.root, args.brand, args.control,
                                    prompt_id=args.prompt, base=args.base, latest=args.latest)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    print(f"# Intervention check: {args.brand}  (prompt {args.prompt}, {pv.base} -> {pv.latest})")
    if args.lever:
        print(f"  lever: {args.lever}")
    print()
    if not pv.treated_seen:
        print(f"   ! '{args.brand}' never appears in these partitions — reads as 0% throughout")
    if not pv.control_seen:
        print(f"   ! control '{args.control}' never appears — a control stuck at 0 can't "
              f"absorb a category-wide shock, so CONFOUNDED can't be ruled out")
    r = pv.result
    print(_fmt_move("treated", args.brand, pv.treated_base, pv.treated_latest, r.treated_moved))
    print(_fmt_move("control", args.control, pv.control_base, pv.control_latest, r.control_moved))
    print()
    print(f"   difference of the changes: {_pts(r.diff_of_changes)}, "
          f"give-or-take {r.give_or_take * 100:.0f} pts")
    print()
    print("## Verdict")
    print(f"   {r.verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
