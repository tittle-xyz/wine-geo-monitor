"""A/B metric computation on tiny synthetic runs with known answers."""

from wine_geo.ab_experiment import ab_metrics, format_table, write_prompt_files
from wine_geo.schema import RawSample, write_jsonl


def _raw(pid, i, text):
    return RawSample(
        run_id="r", ts="t", provider="mock", model="m", prompt_id=pid,
        prompt_text="q", sample_index=i, response_text=text,
        input_tokens=1, output_tokens=1,
    )


def _write(dir_path, rows):
    dir_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(dir_path / "raw.jsonl", rows)


def test_identical_conditions_give_zero_gap_and_score_jump(tmp_path):
    # Same producers in both conditions (so gap≈0), but only the rated answers cite a
    # score — the exact shape the real experiment found.
    neutral = [
        _raw("p0", 0, "I recommend Caymus and Silver Oak."),
        _raw("p0", 1, "I recommend Caymus and Silver Oak."),
        _raw("p1", 0, "Try Duckhorn."),
        _raw("p1", 1, "Try Duckhorn."),
    ]
    rated = [
        _raw("p0", 0, "I recommend Caymus and Silver Oak, both 92 points."),
        _raw("p0", 1, "I recommend Caymus and Silver Oak, both 92 points."),
        _raw("p1", 0, "Try Duckhorn, rated 94 points."),
        _raw("p1", 1, "Try Duckhorn, rated 94 points."),
    ]
    _write(tmp_path / "n", neutral)
    _write(tmp_path / "r", rated)

    m = ab_metrics(str(tmp_path / "n"), str(tmp_path / "r"))
    assert m["n"] == 4
    assert m["floor"] == 1.0 and m["across"] == 1.0  # identical sets everywhere
    assert m["gap"] == 0.0
    assert m["score_n"] == 0.0 and m["score_r"] == 1.0  # scores only under priming


def test_gap_is_positive_when_priming_shifts_picks(tmp_path):
    # Rated condition recommends a different producer -> across-Jaccard drops below the
    # within-condition floor -> positive gap.
    neutral = [_raw("p0", i, "Caymus and Silver Oak") for i in range(2)]
    rated = [_raw("p0", i, "Bogle and Ridge") for i in range(2)]
    _write(tmp_path / "n", neutral)
    _write(tmp_path / "r", rated)

    m = ab_metrics(str(tmp_path / "n"), str(tmp_path / "r"))
    assert m["floor"] == 1.0  # each condition is internally consistent
    assert m["across"] == 0.0  # disjoint producer sets across conditions
    assert m["gap"] == 1.0


def test_write_prompt_files_pairs_line_up(tmp_path):
    base, primed = write_prompt_files(tmp_path)
    b = [x for x in open(base).read().splitlines() if x]
    p = [x for x in open(primed).read().splitlines() if x]
    assert len(b) == len(p) == 5
    assert "90+" in " ".join(p) and "90+" not in " ".join(b)  # priming only on primed


def test_format_table_includes_each_model(tmp_path):
    _write(tmp_path / "n", [_raw("p0", 0, "Caymus")])
    _write(tmp_path / "r", [_raw("p0", 0, "Caymus, 92 points")])
    m = ab_metrics(str(tmp_path / "n"), str(tmp_path / "r"))
    table = format_table([("llama", m), ("claude", m)])
    assert "llama" in table and "claude" in table and "gap" in table
