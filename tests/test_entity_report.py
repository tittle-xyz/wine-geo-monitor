"""Smoke test the entity-report wiring against the real reference lists."""

from wine_geo.entity_report import main
from wine_geo.schema import RawSample, write_jsonl


def _raw(i, text):
    return RawSample(
        run_id="r", ts="t", provider="anthropic", model="claude-haiku-4-5",
        prompt_id="p0", prompt_text="best value napa cab?", sample_index=i,
        response_text=text, input_tokens=1, output_tokens=1,
    )


def test_report_runs_and_reports_on_a_tiny_run(tmp_path, capsys):
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, [
        _raw(0, "For a value Napa Cab, try Bogle or a Sonoma Cabernet."),
        _raw(1, "Caymus is a classic Napa Valley pick, 95 points."),
    ])
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "REGION share-of-voice" in out
    assert "SPECIFICITY" in out
    assert "RATING SURFACE" in out


def test_report_handles_empty_input(tmp_path):
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, [])
    assert main([str(path)]) == 1
