"""Cost aggregation over the raw layer (see #7)."""

from pytest import approx

from wine_geo import config
from wine_geo.extract import build_patterns, load_producers
from wine_geo.pipeline import cost_rows, cost_stage, sweep_cost_confidence
from wine_geo.schema import RawSample


def _sample(provider, model, prompt_id, sample_index, tin, tout):
    return RawSample(
        run_id="r1",
        ts="2026-07-10T00:00:00+00:00",
        provider=provider,
        model=model,
        prompt_id=prompt_id,
        prompt_text="best napa cab?",
        sample_index=sample_index,
        response_text="Caymus, Silver Oak.",
        input_tokens=tin,
        output_tokens=tout,
    )


def test_cost_stage_totals_and_breakdowns():
    # claude-haiku-4-5 is (1.0, 5.0) $/1M -> (1000*1 + 1000*5)/1e6 = 0.006 per sample.
    raw = [
        _sample("anthropic", "claude-haiku-4-5", "p0", 0, 1000, 1000),
        _sample("anthropic", "claude-haiku-4-5", "p0", 1, 1000, 1000),
        _sample("openai", "gpt-4o-mini", "p1", 0, 1000, 1000),  # (0.15,0.60)->0.00075
    ]
    cost = cost_stage(raw)

    assert cost["total"]["samples"] == 3
    assert cost["total"]["input_tokens"] == 3000
    assert cost["total"]["cost"] == approx(0.01275)

    # Total is exactly the sum of the per-model buckets.
    assert cost["total"]["cost"] == approx(sum(m["cost"] for m in cost["by_model"]))

    # by_model is sorted most-expensive first.
    assert cost["by_model"][0]["model"] == "claude-haiku-4-5"
    assert cost["by_model"][0]["samples"] == 2
    assert cost["by_model"][0]["cost"] == approx(0.012)

    by_prompt = {p["prompt_id"]: p for p in cost["by_prompt"]}
    assert by_prompt["p0"]["cost"] == approx(0.012)
    assert by_prompt["p1"]["cost"] == approx(0.00075)


def test_cost_stage_empty():
    cost = cost_stage([])
    assert cost["total"] == {
        "samples": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
    }
    assert cost["by_model"] == []
    assert cost["by_prompt"] == []


def test_cost_rows_shape_and_per_sample():
    raw = [
        _sample("anthropic", "claude-haiku-4-5", "p0", 0, 1000, 1000),
        _sample("anthropic", "claude-haiku-4-5", "p0", 1, 1000, 1000),
    ]
    rows = cost_rows(cost_stage(raw))
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "anthropic"
    assert row["model"] == "claude-haiku-4-5"
    assert row["samples"] == 2
    assert row["cost"] == approx(0.012)
    assert row["cost_per_sample"] == approx(0.006)


def test_sweep_cost_rises_and_ci_tightens():
    producers = load_producers(config.PRODUCERS_PATH)
    patterns = build_patterns(producers)
    universe = [p["name"] for p in producers]

    points = sweep_cost_confidence(
        config.DEFAULT_PROMPTS[:1], patterns, universe,
        provider_name="mock", model="claude-haiku-4-5",
        ns=[10, 50, 200], concurrency=4, seed=7, prompt_id="p0",
    )

    assert [p["n"] for p in points] == [10, 50, 200]
    # Cost grows with sample count...
    costs = [p["cost"] for p in points]
    assert costs[0] < costs[1] < costs[2]
    # ...while the confidence interval tightens (more samples -> more precise).
    widths = [p["ci_half_width"] for p in points]
    assert widths[-1] < widths[0]
