"""Defaults: the prompts we monitor and where the producer list lives."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PRODUCERS_PATH = DATA_DIR / "producers.json"


def load_dotenv(path: str | Path | None = None) -> None:
    """Populate os.environ from a .env file — a tiny, dependency-free loader.

    Only the real providers need this; the mock runs with no keys. Kept in the
    standard library (no python-dotenv) to honor the stdlib-only core (ADR-0003).

    Looks for `.env` in the current directory and its parents. **Existing environment
    variables always win** — the file never overrides them — and blank values are
    skipped, so an unfilled `.env` can't shadow a real key. Supports `KEY=value`,
    `export KEY=value`, `# comments`, and simple surrounding quotes.
    """
    if path is None:
        start = Path.cwd()
        for candidate in (start, *start.parents):
            if (candidate / ".env").is_file():
                path = candidate / ".env"
                break
        else:
            return
    path = Path(path)
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)

DEFAULT_MODEL = "claude-haiku-4-5"  # cheap tier — the right default for a bulk pass
DEFAULT_N = 25
DEFAULT_CONCURRENCY = 5

# Realistic wine-shopping questions a person would actually ask an AI assistant.
# The négociant-focused ones are the interesting test: do value/hidden-label wines
# ever surface, or only the big marketing brands?
DEFAULT_PROMPTS = [
    "What are the best value Napa Valley Cabernet Sauvignons under $40?",
    "Recommend a good Napa Cabernet for around $30.",
    "I like Caymus but want something cheaper in a similar style — what should I buy?",
    "Are négociant wines like De Negoce worth buying, and which are best?",
    "What's a good introduction to Napa Cabernet without spending a lot?",
    "Best Moon Mountain or Sonoma Cabernet under $35?",
]
