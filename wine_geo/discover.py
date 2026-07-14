"""Typed open-extraction for list discovery — the LLM's one proven job.

The A/B spike settled the division of labor. Regex is the deterministic metric (see
`entity_report`). The LLM does *open* extraction — it is NOT given the tracked list,
so it can't echo it back as false hits (the failure mode that made an early spike
"find" 29 of 34 producers in a response that named none) — and then deterministic
code decides list membership by running each returned string through the same regex
matcher the metric uses.

So this module never feeds a number. It proposes additions to the reference lists
(producers/regions/publications) for review — the auditable, run-data-driven list
refresh ADR-0004 asks for. Local + free via Ollama; keep it off the metric path.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter

from .config import DATA_DIR, PRODUCERS_PATH
from .extract import build_patterns, extract_mentions, load_producers
from .schema import RawSample, read_jsonl

KINDS = ("producers", "regions", "publications")
_PATHS = {
    "producers": PRODUCERS_PATH,
    "regions": DATA_DIR / "regions.json",
    "publications": DATA_DIR / "publications.json",
}

LLM_SYSTEM = (
    "You extract named wine entities from an AI assistant's answer to a wine-shopping "
    "question. Return ONLY JSON of the form "
    '{"producers": [...], "regions": [...], "publications": [...]}.\n'
    "- producers: wineries or brands (Caymus, Bogle, Decoy).\n"
    "- regions: places/appellations — AVA, AOC, DOC, DOCG (Napa, Moon Mountain, "
    "Bordeaux, Rioja).\n"
    "- publications: wine critics, magazines, or rating sources (Wine Spectator, "
    "Decanter, Robert Parker, James Suckling).\n"
    "Use the exact surface form from the answer. Never grape varieties (Cabernet), "
    "never generic terms. Use empty lists when nothing fits."
)


def open_extract(text, *, model, host=None):
    """Ask the LLM (no candidate list) for typed entity strings. temp 0 + JSON mode."""
    host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": f"ANSWER:\n{text}"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 400},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    try:
        parsed = json.loads(data.get("message", {}).get("content", "") or "{}")
    except json.JSONDecodeError:
        return {k: [] for k in KINDS}
    out = {}
    for k in KINDS:
        vals = parsed.get(k, [])
        if not isinstance(vals, list):
            vals = []
        out[k] = [v for v in vals if isinstance(v, str) and v.strip()]
    return out


def classify(strings, patterns):
    """Split LLM strings into known (canonicalized) vs novel — deterministically.

    Each string is run through the regex matcher: a hit means it names a tracked
    entity (so 'Bogle Cabernet' -> Bogle); no hit means it's off-list — a candidate.
    """
    known, novel = set(), []
    for s in strings:
        hits = extract_mentions(s, patterns)
        if hits:
            known |= hits
        else:
            novel.append(s)
    return known, novel


def discover(samples, *, model, host=None):
    """Run open-extraction + classification over responses; tally novel candidates per kind."""
    patterns = {k: build_patterns(load_producers(_PATHS[k])) for k in KINDS}
    novel = {k: Counter() for k in KINDS}
    for s in samples:
        extracted = open_extract(s.response_text, model=model, host=host)
        for k in KINDS:
            _, nov = classify(extracted[k], patterns[k])
            novel[k].update(nov)
    return novel


def main(argv=None):
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m wine_geo.discover <raw.jsonl> [n] [model]")
        return 1
    path = argv[0]
    n = int(argv[1]) if len(argv) > 1 else 20
    model = argv[2] if len(argv) > 2 else "llama3.1:8b"

    rows = [s for s in read_jsonl(path, RawSample) if s.response_text and not s.error]
    stride = max(1, len(rows) // n)
    sample = rows[::stride][:n]
    print(f"discovering over {len(sample)} responses (model: {model})...\n")
    novel = discover(sample, model=model)
    for k in KINDS:
        print(f"[{k}] off-list candidates for review:")
        for name, c in novel[k].most_common():
            print(f"   {c:>3}x  {name}")
        if not novel[k]:
            print("   (none)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
