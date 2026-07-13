"""Sample-size math behind the advisor (wine_geo.stats)."""

import pytest

from wine_geo.stats import ci_half_width_for, recommend_sample_size


def test_recommend_sample_size_worst_case():
    # 95% CI, p=0.5: n = (1.96/w)^2 * 0.25.
    assert recommend_sample_size(0.10) == 97    # (z/0.10)^2*0.25 ≈ 96.04 -> 97
    assert recommend_sample_size(0.05) == 385   # halving the target ~quadruples n


def test_meets_target_without_wild_oversampling():
    n = recommend_sample_size(0.10)
    assert ci_half_width_for(n) <= 0.10 + 1e-9  # actually hits the target
    assert ci_half_width_for(n) > 0.09          # and isn't hugely over the mark


def test_more_samples_are_more_precise():
    assert ci_half_width_for(400) < ci_half_width_for(100)


def test_tighter_confidence_needs_more_samples():
    at99 = recommend_sample_size(0.10, confidence=0.99)
    at95 = recommend_sample_size(0.10, confidence=0.95)
    assert at99 > at95


def test_rejects_out_of_range_targets():
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            recommend_sample_size(bad)
