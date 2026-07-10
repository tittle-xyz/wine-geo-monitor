from wine_geo.extract import build_patterns, extract_mentions

PRODUCERS = [
    {"name": "Caymus", "aliases": ["Caymus Vineyards"]},
    {"name": "De Negoce", "aliases": ["DeNegoce", "De Négoce"]},
    {"name": "Stag's Leap Wine Cellars", "aliases": ["Stags Leap"]},
    {"name": "Ridge", "aliases": []},
]
PATTERNS = build_patterns(PRODUCERS)


def test_matches_canonical_and_alias():
    assert extract_mentions("I love Caymus and DeNegoce.", PATTERNS) == {"Caymus", "De Negoce"}


def test_alias_maps_to_canonical():
    assert extract_mentions("Stags Leap is great", PATTERNS) == {"Stag's Leap Wine Cellars"}


def test_curly_apostrophe_normalized():
    # canonical has a straight apostrophe; text has a curly one
    assert "Stag's Leap Wine Cellars" in extract_mentions("Stag’s Leap Wine Cellars rocks", PATTERNS)


def test_accented_alias():
    assert extract_mentions("Try De Négoce lot 12", PATTERNS) == {"De Negoce"}


def test_no_substring_false_positive():
    assert extract_mentions("Great Ridgecrest views", PATTERNS) == set()


def test_empty_on_no_match():
    assert extract_mentions("nothing relevant here", PATTERNS) == set()
