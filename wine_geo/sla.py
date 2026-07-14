"""Operation-level SLA → fulfillment mechanism.

The collector asks for an outcome at a *service level*; the provider layer decides
*how* to meet it — a synchronous call, or the batch API at ~half the cost in
exchange for latency. This is the seam suggested on issue #9 (from a community
comment): you change the SLA, never the collector. A finer mechanism later — a
flex/priority tier, a cheaper model, a cache — slots in here without touching a
single line upstream.

The durable raw layer + daily partitions already make collection latency-tolerant,
so `overnight` (batch) is the natural default for scheduled monitoring.
"""

from __future__ import annotations

from enum import Enum


class Sla(str, Enum):
    REALTIME = "real-time"    # answer in seconds — a person is waiting
    SAME_DAY = "same-day"     # hours are fine
    OVERNIGHT = "overnight"   # done by morning — the scheduled-monitoring default


class Mechanism(str, Enum):
    SYNC = "sync"    # synchronous per-call (providers.Provider.complete)
    BATCH = "batch"  # the provider's batch API — ~50% cheaper, multi-hour window


# real-time must be synchronous. `same-day` and `overnight` both tolerate the batch
# API's multi-hour window today, so both resolve to BATCH — the seam lives right here,
# so a distinct `same-day` mechanism (a flex/priority tier) could slot in later without
# the collector ever knowing.
_MECHANISM: dict[Sla, Mechanism] = {
    Sla.REALTIME: Mechanism.SYNC,
    Sla.SAME_DAY: Mechanism.BATCH,
    Sla.OVERNIGHT: Mechanism.BATCH,
}


def mechanism_for(sla: Sla | str) -> Mechanism:
    """Which fulfillment mechanism does this SLA use? Accepts the enum or its value."""
    return _MECHANISM[Sla(sla)]
