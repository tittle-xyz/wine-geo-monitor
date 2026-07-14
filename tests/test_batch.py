"""Batch-API collection path and the operation SLA seam (see #9).

The real provider batch calls are network-only (and env-gated), so these tests
exercise the mechanism end-to-end through MockProvider: the SLA→mechanism mapping,
the custom_id round-trip, the batch billing tier, and the realized cost discount.
"""

from pytest import approx

from wine_geo.pipeline import collect, cost_stage
from wine_geo.providers import BatchRequest, MockProvider, estimate_cost
from wine_geo.sla import Mechanism, Sla, mechanism_for

PROMPTS = ["best napa cab?", "value cab under $30?"]


def test_mechanism_mapping():
    assert mechanism_for(Sla.REALTIME) is Mechanism.SYNC
    assert mechanism_for(Sla.SAME_DAY) is Mechanism.BATCH
    assert mechanism_for(Sla.OVERNIGHT) is Mechanism.BATCH
    # accepts the raw string value too (what the CLI/Dagster pass)
    assert mechanism_for("overnight") is Mechanism.BATCH


def test_mock_batch_round_trips_every_custom_id():
    reqs = [
        BatchRequest(custom_id=f"p0-{j}", prompt="best napa cab?", model="mock")
        for j in range(5)
    ]
    out = MockProvider(seed=1).complete_batch(reqs)
    assert set(out) == {r.custom_id for r in reqs}
    assert all(c.text for c in out.values())


def test_overnight_collect_tags_batch_tier_and_covers_all_samples():
    raw = collect(PROMPTS, provider=MockProvider(seed=3), model="gpt-4o-mini",
                  n=4, concurrency=4, sla=Sla.OVERNIGHT)
    assert len(raw) == len(PROMPTS) * 4
    assert all(r.billing_tier == "batch" for r in raw)
    # every (prompt, sample) slot is present exactly once
    assert sorted((r.prompt_id, r.sample_index) for r in raw) == \
        sorted((f"p{i}", j) for i in range(len(PROMPTS)) for j in range(4))


def test_realtime_collect_stays_standard_tier():
    raw = collect(PROMPTS, provider=MockProvider(seed=3), model="gpt-4o-mini",
                  n=4, concurrency=4, sla=Sla.REALTIME)
    assert all(r.billing_tier == "standard" for r in raw)


def test_batch_costs_half_of_list_price():
    # Same tokens, same model — batch tier must price at exactly half.
    assert estimate_cost(1000, 1000, "gpt-4o-mini", batch=True) == approx(
        estimate_cost(1000, 1000, "gpt-4o-mini") / 2
    )


def test_cost_stage_applies_batch_discount():
    std = collect(PROMPTS, provider=MockProvider(seed=5), model="gpt-4o-mini",
                  n=6, concurrency=4, sla=Sla.REALTIME)
    bat = collect(PROMPTS, provider=MockProvider(seed=5), model="gpt-4o-mini",
                  n=6, concurrency=4, sla=Sla.OVERNIGHT)
    # Mock is deterministic given a seed, so the token totals match; only price differs.
    c_std = cost_stage(std)
    c_bat = cost_stage(bat)
    assert c_std["total"]["input_tokens"] == c_bat["total"]["input_tokens"]
    assert c_bat["total"]["cost"] == approx(c_std["total"]["cost"] / 2)
    # the batch run's by_model bucket is tagged as batch
    assert all(m["billing_tier"] == "batch" for m in c_bat["by_model"])


class _SyncOnlyProvider:
    """A provider with no batch capability — collect must fall back to synchronous."""

    name = "synconly"

    def complete(self, prompt, *, model):
        return MockProvider(seed=0).complete(prompt, model=model)


def test_batch_sla_falls_back_when_provider_lacks_batch():
    raw = collect(PROMPTS, provider=_SyncOnlyProvider(), model="mock",
                  n=3, concurrency=2, sla=Sla.OVERNIGHT)
    assert len(raw) == len(PROMPTS) * 3
    # fell back to the sync path, so these are standard tier, not batch
    assert all(r.billing_tier == "standard" for r in raw)
