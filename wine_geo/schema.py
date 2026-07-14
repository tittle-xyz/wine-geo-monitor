"""Data contracts between pipeline stages, plus JSONL read/write.

The seam between stages is data at rest (these records serialized to JSONL), not
function calls — that's what lets the stages run on different schedules, scale
independently, and (later) be reimplemented in another language. `RawSample` is
the source of truth: everything downstream is a re-derivable function of it, so
you can reprocess without re-paying for API calls.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass


@dataclass
class RawSample:
    """One model response to one prompt. The immutable, expensive-to-produce layer."""

    run_id: str
    ts: str
    provider: str
    model: str
    prompt_id: str
    prompt_text: str
    sample_index: int
    response_text: str
    input_tokens: int
    output_tokens: int
    billing_tier: str = "standard"  # "batch" when fulfilled via a batch API (~50% off)
    error: str | None = None


@dataclass
class Mention:
    """One producer detected in one sample. Tidy/long format, cheap to recompute."""

    run_id: str
    prompt_id: str
    sample_index: int
    producer: str


def _as_row(rec):
    return asdict(rec) if is_dataclass(rec) else rec


def write_jsonl(path, records) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(_as_row(rec)) + "\n")


def read_jsonl(path, cls=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if cls is None:
        return rows
    keep = {fld.name for fld in fields(cls)}
    return [cls(**{k: v for k, v in row.items() if k in keep}) for row in rows]
