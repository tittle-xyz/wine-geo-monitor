"""How much has a brand been written about? A free proxy for training-data presence (#23).

The knowledge prior in `diagnose.py` starts date-only: a brand postdating a model's
cutoff can't be known. But a brand that *predates* the cutoff still might not be known if
almost nothing was written about it. Prominence is a missing variable: knowledge ≈ date ×
prominence × model size. (The live data was humbling here — Cameron Hughes turned out to
have a full Wikipedia article, so its Haiku hedge is the *model-size* term, not low
prominence. Prominence explains the clear cases; it isn't a silver bullet.)

The proxy is **Wikipedia/Wikidata presence**, and that's a principled choice, not just a
free one: Wikipedia is itself a core training corpus, so "has a developed Wikipedia article"
is close to a direct read on "is in the training data" — not a distant correlate.

But the live data added a caveat: presence is **strong positive** evidence, absence only
**weak negative** evidence — a brand can be in the corpus via news/retail/DTC with no
Wikipedia page (De Négoce, known to the model, has none). So a prominent article confirms
'known', but an absent one only drops a pre-cutoff brand to 'likely-thin' (worth a probe),
never to 'unknown'. The *date* signal is what earns an 'unknown' verdict, not prominence.

Design for reuse across the `-monitor` family:
  - The **domain check is pluggable** (`domain_terms`), so the same resolver works for wines,
    libraries, companies — swap the terms, not the logic.
  - The **network fetch is separated from the scoring**, so everything here is pure and
    unit-testable offline with canned candidates (the fetch lives in `fetch_candidates`).

Naive and deterministic on purpose, exactly like `extract.py`: keyword-based domain matching
can't do semantics, and that honest limit is documented rather than hidden. Structured
Wikidata type (P31) constraints are the future hardening — `domain_match` is a predicate, so
swapping it in later disturbs nothing around it.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Iterable

from .schema import read_jsonl, write_jsonl

# One candidate entity, normalized from whatever source fetched it. `fetch_candidates`
# (increment 2) fills this from Wikidata search + the Wikipedia summary/pageviews APIs.
# Keeping it a plain dict keeps the pure functions trivially testable with literals.
#   {"title": str, "description": str, "summary": str, "categories": list[str],
#    "summary_len": int, "pageviews": int, "sitelinks": int}


def domain_match(candidate: dict, domain_terms: Iterable[str], *, min_hits: int = 1) -> bool:
    """Does this candidate actually belong to our domain (vs. a same-named other thing)?

    Deterministic keyword check over the candidate's description / summary / categories.
    `domain_terms` is the generic knob: pass wine terms here, library terms in a code
    monitor. This is what keeps prominence from crediting the footballer named the same
    as the winery — the entity-resolution safeguard.
    """
    hay = " ".join([
        candidate.get("description", ""),
        candidate.get("summary", ""),
        " ".join(candidate.get("categories", [])),
    ]).lower()
    return sum(1 for t in domain_terms if t.lower() in hay) >= min_hits


def resolve(candidates: list[dict], domain_terms: Iterable[str]) -> dict | None:
    """Pick the best in-domain candidate, or None if none match the domain.

    'Best' = most-read, then most-developed — pageviews first (popularity is the stronger
    prominence signal), article length as the tiebreak.
    """
    matches = [c for c in candidates if domain_match(c, domain_terms)]
    if not matches:
        return None
    return max(matches, key=lambda c: (c.get("pageviews", 0), c.get("summary_len", 0)))


# Thresholds tuned on the live validation set (Caymus 960 pv/mo = prominent, Cameron Hughes
# 273 = established). Pageviews, not article length, is the discriminating signal — the two
# wineries had near-identical ~5 KB articles, so length is near-flat across the domain and
# doesn't separate them. Calibrated on n=2 wineries; revisit as the snapshot grows.
def prominence_level(candidate: dict | None) -> str:
    """Bin a resolved candidate into absent / thin / established / prominent (pageviews-driven)."""
    if candidate is None:
        return "absent"
    if not (candidate.get("title") or candidate.get("summary_len", 0) > 0):
        return "absent"  # no article at all
    pv = candidate.get("pageviews", 0)
    if pv >= 500:
        return "prominent"
    if pv >= 150:
        return "established"
    return "thin"  # has an article, but almost no one reads it


def prior_with_prominence(date_prior: str, level: str) -> str:
    """Fold prominence into the date-only knowledge prior — the legible combining rule.

    Too new (postdates the cutoff) stays unknown no matter how prominent: prominence can't
    save a brand the model was never trained on. But for a brand that predates the cutoff,
    a thin/absent footprint downgrades 'likely-known' toward unknown — which is the whole
    point, catching the predates-but-obscure case that date alone gets wrong.
    """
    if date_prior == "likely-unknown":
        return "likely-unknown"
    if date_prior in ("likely-known", "borderline"):
        if level in ("prominent", "established"):
            return "likely-known"
        # thin OR absent -> likely-thin. Wikipedia absence is weak evidence (the brand may be
        # in the corpus via non-Wikipedia sources, like De Négoce), so it only softens the
        # prior to 'worth a probe' — it never overrides date to claim 'unknown'.
        return "likely-thin"
    return date_prior  # unknown-cutoff / unknown-date pass through unchanged


# --- pure parsers (unit-tested offline with canned API payloads) --------------

def parse_wikidata_search(obj: dict) -> list[dict]:
    """`wbsearchentities` JSON -> candidate stubs [{id, label, description}]."""
    return [
        {"id": r.get("id", ""), "label": r.get("label", ""),
         "description": r.get("description", "")}
        for r in obj.get("search", [])
    ]


def parse_length(obj: dict) -> int:
    """`prop=info` JSON -> article length in bytes (0 if the page is missing)."""
    pages = obj.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return 0
        return int(page.get("length", 0))
    return 0


def parse_pageviews(obj: dict) -> int:
    """Pageviews JSON -> a stable monthly figure (median, so a partial month can't drag it)."""
    views = [it.get("views", 0) for it in obj.get("items", [])]
    return int(statistics.median(views)) if views else 0


def parse_enwiki_sitelink(obj: dict, entity_id: str) -> tuple[str | None, int]:
    """`wbgetentities props=sitelinks` -> (enwiki title, total sitelink count)."""
    entity = obj.get("entities", {}).get(entity_id, {})
    links = entity.get("sitelinks", {})
    title = links.get("enwiki", {}).get("title")
    return title, len(links)


# --- network fetch (thin; the one impure part) -------------------------------

_UA = "wine-geo-monitor/0.1 (https://github.com/tittle-xyz/wine-geo-monitor)"
_WIKIDATA = "https://www.wikidata.org/w/api.php"
_ENWIKI = "https://en.wikipedia.org/w/api.php"
_PAGEVIEWS = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
              "en.wikipedia/all-access/all-agents/{title}/monthly/{start}/{end}")


def _get_json(  # pragma: no cover - network
    url: str, params: dict | None = None, *, timeout: float = 20.0
) -> dict:
    import json
    import urllib.parse
    import urllib.request

    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_candidates(name: str, aliases: Iterable[str] = ()) -> list[dict]:  # pragma: no cover
    """Wikidata search for the name (+aliases) -> candidate stubs with descriptions."""
    seen: dict[str, dict] = {}
    for term in [name, *aliases]:
        obj = _get_json(_WIKIDATA, {"action": "wbsearchentities", "search": term,
                                    "language": "en", "format": "json", "limit": 5})
        for c in parse_wikidata_search(obj):
            seen.setdefault(c["id"], c)
    return list(seen.values())


def enrich(candidate: dict, *, pv_window=("2026040100", "2026070100")) -> dict:  # pragma: no cover
    """Add enwiki title, article length, pageviews, sitelink count to a candidate."""
    import urllib.parse

    ids = candidate["id"]
    sl = _get_json(_WIKIDATA, {"action": "wbgetentities", "ids": ids,
                               "props": "sitelinks", "format": "json"})
    title, sitelinks = parse_enwiki_sitelink(sl, ids)
    out = {**candidate, "title": title, "sitelinks": sitelinks, "summary_len": 0, "pageviews": 0}
    if not title:
        return out
    # Missing article length or no pageviews data means "obscure", not "crash" — a 404 from
    # the pageviews API is itself the signal (zero traffic), so degrade to 0 rather than raise.
    try:
        info = _get_json(_ENWIKI, {"action": "query", "prop": "info",
                                   "titles": title, "format": "json"})
        out["summary_len"] = parse_length(info)
    except Exception:
        pass
    try:
        quoted = urllib.parse.quote(title.replace(" ", "_"), safe="")
        pv = _get_json(_PAGEVIEWS.format(title=quoted, start=pv_window[0], end=pv_window[1]))
        out["pageviews"] = parse_pageviews(pv)
    except Exception:
        pass
    return out


def fetch_prominence(  # pragma: no cover - network
    name: str, aliases: Iterable[str], domain_terms: Iterable[str]
) -> dict:
    """Full lookup for one brand: resolve the in-domain entity, enrich it, bin it.

    Returns a snapshot record — the durable, re-derivable read of what the web said.
    """
    from datetime import datetime, timezone

    candidates = fetch_candidates(name, aliases)
    in_domain = [enrich(c) for c in candidates if domain_match(c, domain_terms)]
    best = resolve(in_domain, domain_terms)
    return {
        "name": name,
        "resolved_title": best.get("title") if best else None,
        "pageviews": best.get("pageviews", 0) if best else 0,
        "summary_len": best.get("summary_len", 0) if best else 0,
        "sitelinks": best.get("sitelinks", 0) if best else 0,
        "level": prominence_level(best),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "wikidata+enwiki+pageviews",
    }


# --- snapshot store (durable, upsert-latest-by-name) -------------------------

def load_snapshot(path: str | Path) -> dict:
    """{name: record} from the snapshot JSONL (latest record per name wins)."""
    p = Path(path)
    if not p.is_file():
        return {}
    return {r["name"]: r for r in read_jsonl(p)}


def save_snapshot(path: str | Path, record: dict) -> None:
    """Upsert one record by name and rewrite the snapshot."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    snap = load_snapshot(p)
    snap[record["name"]] = record
    write_jsonl(p, list(snap.values()))
