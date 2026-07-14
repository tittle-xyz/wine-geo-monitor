from wine_geo.extract import (
    build_forms,
    build_patterns,
    extract_disambiguated,
    extract_mentions,
    has_score,
    mask_matches,
)

PRODUCERS = [
    {"name": "Caymus", "aliases": ["Caymus Vineyards"]},
    {"name": "De Negoce", "aliases": ["DeNegoce", "De Négoce"]},
    {"name": "Stag's Leap Wine Cellars", "aliases": ["Stags Leap"]},
    {"name": "Ridge", "aliases": []},
]
PATTERNS = build_patterns(PRODUCERS)

# Producers whose names contain a region word — the cross-entity collision.
COLLIDING = build_patterns([
    {"name": "Rutherford Hill", "aliases": []},
    {"name": "Rutherford Ranch", "aliases": []},
])
REGIONS = build_patterns([{"name": "Rutherford", "aliases": []}])


def test_matches_canonical_and_alias():
    assert extract_mentions("I love Caymus and DeNegoce.", PATTERNS) == {"Caymus", "De Negoce"}


def test_alias_maps_to_canonical():
    assert extract_mentions("Stags Leap is great", PATTERNS) == {"Stag's Leap Wine Cellars"}


def test_curly_apostrophe_normalized():
    # canonical has a straight apostrophe; text has a curly one
    assert "Stag's Leap Wine Cellars" in extract_mentions(
        "Stag’s Leap Wine Cellars rocks", PATTERNS
    )


def test_accented_alias():
    assert extract_mentions("Try De Négoce lot 12", PATTERNS) == {"De Negoce"}


def test_no_substring_false_positive():
    assert extract_mentions("Great Ridgecrest views", PATTERNS) == set()


def test_empty_on_no_match():
    assert extract_mentions("nothing relevant here", PATTERNS) == set()


def test_mask_prevents_region_matching_a_producer_namesake():
    # "Rutherford Hill" the producer must not be counted as "Rutherford" the AVA.
    text = "I recommend Rutherford Hill and Rutherford Ranch."
    assert extract_mentions(text, REGIONS) == {"Rutherford"}  # naive: false positive
    masked = mask_matches(text, COLLIDING)
    assert extract_mentions(masked, REGIONS) == set()  # producers masked -> no false AVA


def test_mask_keeps_a_real_region_mention():
    text = "A classic Rutherford Cabernet, plus Rutherford Hill winery."
    masked = mask_matches(text, COLLIDING)
    # the bare-AVA 'Rutherford' survives; only the producer span is blanked
    assert extract_mentions(masked, REGIONS) == {"Rutherford"}


def test_has_score_detects_point_forms():
    assert has_score("A 94-point Cab")
    assert has_score("rated 92 by the critics")
    assert has_score("scores 95+ across the board")
    assert has_score("96 points from Suckling")


def test_has_score_ignores_non_scores():
    assert not has_score("about $30 a bottle")
    assert not has_score("aged 18 months in oak")
    assert not has_score("a blend of 80% Cabernet")


# --- longest-match disambiguation (regions) --------------------------------
REGION_FORMS = build_forms([
    {"name": "Sonoma Valley", "aliases": ["Sonoma"]},
    {"name": "Sonoma County", "aliases": []},
    {"name": "Napa Valley", "aliases": ["Napa"]},
])


def test_longer_name_wins_over_bare_alias():
    # "Sonoma County" must not also be credited to Sonoma Valley via the "Sonoma" alias
    assert extract_disambiguated("A value Sonoma County Cab", REGION_FORMS) == {"Sonoma County"}


def test_bare_alias_still_matches_when_alone():
    assert extract_disambiguated("classic Sonoma fruit", REGION_FORMS) == {"Sonoma Valley"}


def test_disambiguation_finds_multiple_distinct_regions():
    got = extract_disambiguated("Napa Valley vs Sonoma County", REGION_FORMS)
    assert got == {"Napa Valley", "Sonoma County"}
