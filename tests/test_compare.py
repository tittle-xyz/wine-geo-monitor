"""Cross-engine data alignment (wine_geo.compare.align_shares)."""

from wine_geo.compare import align_shares


def _rows(shares, prompt="p0"):
    return [
        {"prompt_id": prompt, "producer": name, "share": s,
         "ci_lo": max(0.0, s - 0.1), "ci_hi": min(1.0, s + 0.1),
         "n": 25, "hits": round(s * 25)}
        for name, s in shares.items()
    ]


def test_align_merges_models_and_orders_by_peak_share():
    runs = [
        ("Claude", _rows({"Bogle": 0.76, "Caymus": 0.56}), "d1"),
        ("OpenAI", _rows({"Bogle": 0.92, "Robert Mondavi": 0.76}), "d2"),
    ]
    labels, producers, table = align_shares(runs, "p0", top=None)

    assert labels == ["Claude", "OpenAI"]
    assert set(producers) == {"Bogle", "Caymus", "Robert Mondavi"}
    assert producers[-1] == "Bogle"                      # highest peak share on top
    assert table["Bogle"]["Claude"][0] == 0.76
    assert table["Caymus"]["OpenAI"] == (0.0, 0.0, 0.0)  # unnamed by a model -> zeros


def test_align_respects_top_n_and_prompt_filter():
    runs = [("A", _rows({"X": 0.5, "Y": 0.3, "Z": 0.1}), "d")]

    _, producers, _ = align_shares(runs, "p0", top=2)
    assert producers == ["Y", "X"]                       # top 2 by share, ascending

    _, none_here, _ = align_shares(runs, "p9", top=None)
    assert none_here == []                               # different prompt -> nothing
