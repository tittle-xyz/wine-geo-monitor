"""Defaults: the prompts we monitor and where the producer list lives."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PRODUCERS_PATH = DATA_DIR / "producers.json"

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
