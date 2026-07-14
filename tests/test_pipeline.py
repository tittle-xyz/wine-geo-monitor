from wine_geo.extract import build_patterns
from wine_geo.pipeline import aggregate_stage, collect, extract_stage, metrics_rows
from wine_geo.providers import MockProvider

PRODUCERS = [
    {"name": "Caymus", "aliases": []},
    {"name": "Silver Oak", "aliases": []},
    {"name": "De Negoce", "aliases": ["DeNegoce"]},
]
UNIVERSE = [p["name"] for p in PRODUCERS]


def _collect(seed=1, n=20, concurrency=4):
    return collect(["best napa cab?"], provider=MockProvider(seed=seed),
                   model="claude-haiku-4-5", n=n, concurrency=concurrency, seed=seed)


def test_collector_produces_raw_records():
    raw = _collect(n=20)
    assert len(raw) == 20
    assert all(r.prompt_id == "p0" for r in raw)
    assert all(r.response_text for r in raw)
    assert sorted(r.sample_index for r in raw) == list(range(20))


def test_collector_samples_the_same_distribution_for_a_seed():
    """A seed reproduces the *bag* of responses, not the per-index assignment.

    collect() samples concurrently, so which task wins which RNG draws is
    scheduling-dependent — asserting list equality passes on 3.9/3.11 by GIL luck and
    fails on 3.13. The bag is what the mock actually guarantees.
    """
    a = sorted(r.response_text for r in _collect(seed=1))
    b = sorted(r.response_text for r in _collect(seed=1))
    assert a == b


def test_collector_is_byte_identical_at_concurrency_1():
    # With no concurrency there's no interleaving, so even per-index order is stable.
    a = [r.response_text for r in _collect(seed=1, concurrency=1)]
    b = [r.response_text for r in _collect(seed=1, concurrency=1)]
    assert a == b


def test_metrics_are_reproducible_for_a_seed():
    # The point of the fix: even though per-index order may differ under concurrency,
    # the derived share-of-voice is a function of the bag, so it's stable for a seed.
    def sov(seed):
        raw = _collect(seed=seed, n=30)
        mentions = extract_stage(raw, build_patterns(PRODUCERS))
        r = aggregate_stage(raw, mentions, UNIVERSE, seed=7)[0]
        return {name: hits for name, (_, hits, _) in r["sov"].items()}

    assert sov(3) == sov(3)


def test_extract_only_returns_universe_producers():
    raw = _collect()
    mentions = extract_stage(raw, build_patterns(PRODUCERS))
    assert all(m.producer in set(UNIVERSE) for m in mentions)


def test_aggregate_shares_and_denominator():
    raw = _collect(n=20)
    mentions = extract_stage(raw, build_patterns(PRODUCERS))
    results = aggregate_stage(raw, mentions, UNIVERSE, seed=1)
    assert len(results) == 1
    r = results[0]
    assert r["n"] == 20  # denominator is all samples, not just ones with mentions
    assert set(r["sov"]) == set(UNIVERSE)
    for _, (share, hits, n) in r["sov"].items():
        assert 0.0 <= share <= 1.0 and n == 20
    rows = metrics_rows(results)
    for row in rows:
        assert {"prompt_id", "producer", "share", "ci_lo", "ci_hi"} <= set(row)
        assert row["ci_lo"] <= row["share"] + 1e-9  # CI brackets the estimate (lower side)
