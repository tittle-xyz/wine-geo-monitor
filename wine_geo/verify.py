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

import math
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
