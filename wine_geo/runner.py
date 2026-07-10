"""Concurrent sampling with rate limiting and retry — the resilience layer.

Every sample is a paid, rate-limited, sometimes-failing network call, and a GEO
monitor makes a lot of them. So the harness caps concurrency (a semaphore), and
retries transient failures with exponential backoff + jitter. Providers are
synchronous SDK calls, so we run each in a worker thread via `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import random

from .providers import Completion, Provider


async def _sample_one(
    provider: Provider,
    prompt: str,
    model: str,
    sem: asyncio.Semaphore,
    *,
    retries: int,
    base_delay: float,
    rng: random.Random,
) -> Completion:
    async with sem:
        attempt = 0
        while True:
            try:
                return await asyncio.to_thread(provider.complete, prompt, model=model)
            except Exception:
                if attempt >= retries:
                    raise
                delay = base_delay * (2**attempt) + rng.uniform(0, base_delay)
                await asyncio.sleep(delay)
                attempt += 1


async def sample_prompt(
    provider: Provider,
    prompt: str,
    *,
    model: str,
    n: int,
    concurrency: int,
    retries: int = 3,
    base_delay: float = 0.5,
    seed: int | None = None,
) -> list[Completion]:
    """Fire `n` samples of one prompt, at most `concurrency` in flight at once."""
    sem = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)
    tasks = [
        _sample_one(provider, prompt, model, sem, retries=retries, base_delay=base_delay, rng=rng)
        for _ in range(n)
    ]
    return await asyncio.gather(*tasks)
