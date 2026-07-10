"""Detect which producers a model answer mentions.

This is deliberately a naive alias-matcher, not an LLM extractor. It's cheap,
deterministic, and testable — a good default for the bulk pass. The honest
limitation (and a great thing to talk about): string matching can't handle
paraphrase or spelling drift, so a production monitor escalates ambiguous cases
to a cheap model doing structured extraction. That's the "model tiering" lever.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _normalize(text: str) -> str:
    """Unify curly apostrophes so 'Frog's' matches 'Frog’s'."""
    return text.replace("’", "'")


def load_producers(path: str | Path) -> list[dict]:
    """Each entry: {"name": canonical, "aliases": [...], "note": optional}."""
    with open(path) as f:
        return json.load(f)


def build_patterns(producers: list[dict]) -> list[tuple[str, re.Pattern]]:
    """One compiled pattern per producer, matching the canonical name or any alias.

    Boundaries are non-alphanumeric lookarounds (not \\b) so apostrophes and
    spaces inside a name work, while 'Ridge' still won't match 'Ridgecrest'.
    """
    compiled: list[tuple[str, re.Pattern]] = []
    for p in producers:
        forms = [p["name"], *p.get("aliases", [])]
        alts = "|".join(re.escape(_normalize(f)) for f in forms)
        pattern = re.compile(rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])", re.IGNORECASE)
        compiled.append((p["name"], pattern))
    return compiled


def extract_mentions(text: str, patterns: list[tuple[str, re.Pattern]]) -> set[str]:
    """Return the set of canonical producer names mentioned in `text`."""
    norm = _normalize(text)
    return {canonical for canonical, pattern in patterns if pattern.search(norm)}
