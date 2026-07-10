from wine_geo.stats import bootstrap_ci, mean_pairwise_jaccard, share_of_voice


def test_share_of_voice_counts():
    sov = share_of_voice([{"A"}, {"A", "B"}, set()], ["A", "B", "C"])
    assert sov["A"] == (2 / 3, 2, 3)
    assert sov["B"] == (1 / 3, 1, 3)
    assert sov["C"] == (0.0, 0, 3)


def test_bootstrap_ci_bounds_and_determinism():
    outcomes = [1, 1, 1, 0, 1, 0, 1, 1]
    lo, hi = bootstrap_ci(outcomes, seed=7)
    assert 0.0 <= lo <= hi <= 1.0
    # same seed -> same interval
    assert bootstrap_ci(outcomes, seed=7) == (lo, hi)


def test_bootstrap_ci_empty():
    assert bootstrap_ci([], seed=1) == (0.0, 0.0)


def test_jaccard_identical_and_disjoint():
    assert mean_pairwise_jaccard([{"A"}, {"A"}]) == 1.0
    assert mean_pairwise_jaccard([{"A"}, {"B"}]) == 0.0
    assert mean_pairwise_jaccard([set(), set()]) == 1.0  # consistently nothing


def test_jaccard_single_run_defaults_to_one():
    assert mean_pairwise_jaccard([{"A", "B"}]) == 1.0
