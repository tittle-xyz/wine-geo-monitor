"""The tiny dependency-free .env loader (wine_geo.config.load_dotenv)."""

import os

import pytest

from wine_geo.config import load_dotenv


@pytest.fixture(autouse=True)
def _preserve_environ():
    # load_dotenv mutates os.environ directly, so snapshot and restore it to keep
    # these tests hermetic (no key leaks into later tests).
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_load_dotenv_sets_missing_keys(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "FOO_WGEO=bar\n"
        "export BAZ_WGEO='qux'\n"
        'QUOTED_WGEO="spa ced"\n'
        "EMPTY_WGEO=\n"
        "noequals\n"
    )
    monkeypatch.chdir(tmp_path)
    for k in ("FOO_WGEO", "BAZ_WGEO", "QUOTED_WGEO", "EMPTY_WGEO"):
        monkeypatch.delenv(k, raising=False)

    load_dotenv()

    assert os.environ["FOO_WGEO"] == "bar"
    assert os.environ["BAZ_WGEO"] == "qux"          # export prefix + single quotes
    assert os.environ["QUOTED_WGEO"] == "spa ced"   # double quotes stripped
    assert "EMPTY_WGEO" not in os.environ           # blank value skipped, can't shadow


def test_load_dotenv_never_overrides_real_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FOO_WGEO=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOO_WGEO", "from_env")

    load_dotenv()

    assert os.environ["FOO_WGEO"] == "from_env"     # existing env wins


def test_load_dotenv_no_file_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here or (in the temp tree) above
    load_dotenv()  # must not raise
