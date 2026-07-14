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


def build_forms(producers: list[dict]) -> list[tuple[str, re.Pattern, int]]:
    """Per-*form* patterns (name and each alias separately), sorted longest-form-first.

    `build_patterns` collapses a producer's forms into one alternation, which is fine
    when entries don't overlap. Regions do: 'Sonoma' (an alias of Sonoma Valley) sits
    inside 'Sonoma County'. Resolving that needs form-level granularity + length, so
    `extract_disambiguated` can let the longest match claim the span first.
    """
    forms: list[tuple[str, re.Pattern, int]] = []
    for p in producers:
        for f in [p["name"], *p.get("aliases", [])]:
            nf = _normalize(f)
            pat = re.compile(
                rf"(?<![A-Za-z0-9])(?:{re.escape(nf)})(?![A-Za-z0-9])", re.IGNORECASE
            )
            forms.append((p["name"], pat, len(nf)))
    forms.sort(key=lambda t: -t[2])  # longest form first
    return forms


def extract_disambiguated(text: str, forms: list[tuple[str, re.Pattern, int]]) -> set[str]:
    """Like `extract_mentions`, but overlapping matches resolve longest-form-wins.

    So 'Sonoma County' is credited to Sonoma County, not also to Sonoma Valley via the
    bare 'Sonoma' alias. `forms` must be `build_forms(...)` output (already length-sorted).
    """
    norm = _normalize(text)
    claimed = bytearray(len(norm))
    found: set[str] = set()
    for canonical, pattern, _ in forms:
        for m in pattern.finditer(norm):
            if any(claimed[m.start():m.end()]):
                continue
            claimed[m.start():m.end()] = b"\x01" * (m.end() - m.start())
            found.add(canonical)
    return found


def mask_matches(text: str, patterns: list[tuple[str, re.Pattern]]) -> str:
    """Blank out every span matched by `patterns`, preserving length/offsets.

    The point is cross-entity precedence: region names are often *inside* producer
    names ('Rutherford' in 'Rutherford Hill'; the 'Stags Leap District' AVA vs the
    'Stag's Leap Wine Cellars' producer). Mask producers first, then match regions on
    the residual, and a producer never gets miscounted as its namesake region.
    """
    chars = list(_normalize(text))
    for _, pattern in patterns:
        for m in pattern.finditer("".join(chars)):
            for i in range(m.start(), m.end()):
                chars[i] = " "
    return "".join(chars)


# A model citing a real point score — "94 points", "95+", "rated 92" — is the wine
# world's marketing currency. Match the 90–100 band (scores below ~88 are rarely
# name-dropped) so the rating-surface metric can ask: does the AI channel use them?
SCORE_RE = re.compile(
    r"(?<!\d)(?:9\d|100)\s*(?:\+|-)?\s*(?:point|pt)s?\b"
    r"|(?<!\d)(?:9\d|100)\+"
    r"|\b(?:rated|scored|score of)\s+(?:9\d|100)\b",
    re.IGNORECASE,
)


def has_score(text: str) -> bool:
    """True if the text cites a numeric critic score in the 90–100 band."""
    return SCORE_RE.search(text) is not None
