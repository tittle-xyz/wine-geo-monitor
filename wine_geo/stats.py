"""The statistics that make a GEO number trustworthy.

Point estimates lie when the thing you're measuring is stochastic. So alongside
share-of-voice we compute a bootstrap confidence interval (how much the number
would wobble if you sampled again) and the run-to-run Jaccard overlap (how much
the *set* of recommended brands changes between identical prompts). When two
brands' CIs overlap, their apparent ranking difference is inside the noise floor.
"""

from __future__ import annotations

import random
from itertools import combinations


def share_of_voice(
    run_mentions: list[set[str]], universe: list[str]
) -> dict[str, tuple[float, int, int]]:
    """For each producer: (share, hits, n) where share = fraction of runs mentioning it."""
    n = len(run_mentions)
    out: dict[str, tuple[float, int, int]] = {}
    for name in universe:
        hits = sum(1 for s in run_mentions if name in s)
        out[name] = (hits / n if n else 0.0, hits, n)
    return out


def bootstrap_ci(
    outcomes: list[int], *, iters: int = 1000, alpha: float = 0.05, seed: int | None = None
) -> tuple[float, float]:
    """Percentile bootstrap CI for a proportion. `outcomes` is a list of 0/1 hits."""
    n = len(outcomes)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        sum(outcomes[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    """Mean Jaccard similarity over all pairs of runs — 1.0 = identical, 0.0 = disjoint.

    Two empty sets count as identical (nothing recommended, consistently).
    """
    pairs = list(combinations(range(len(sets)), 2))
    if not pairs:
        return 1.0
    total = 0.0
    for i, j in pairs:
        union = sets[i] | sets[j]
        total += 1.0 if not union else len(sets[i] & sets[j]) / len(union)
    return total / len(pairs)
