from wine_geo.extract import build_patterns
from wine_geo.pipeline import aggregate_stage, collect, extract_stage, metrics_rows
from wine_geo.providers import MockProvider

PRODUCERS = [
    {"name": "Caymus", "aliases": []},
    {"name": "Silver Oak", "aliases": []},
    {"name": "De Negoce", "aliases": ["DeNegoce"]},
]
UNIVERSE = [p["name"] for p in PRODUCERS]


def _collect(seed=1, n=20):
    return collect(["best napa cab?"], provider=MockProvider(seed=seed),
                   model="claude-haiku-4-5", n=n, concurrency=4, seed=seed)


def test_collector_produces_raw_records():
    raw = _collect(n=20)
    assert len(raw) == 20
    assert all(r.prompt_id == "p0" for r in raw)
    assert all(r.response_text for r in raw)
    assert sorted(r.sample_index for r in raw) == list(range(20))


def test_collector_is_deterministic_with_seed():
    a = [r.response_text for r in _collect(seed=1)]
    b = [r.response_text for r in _collect(seed=1)]
    assert a == b


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
