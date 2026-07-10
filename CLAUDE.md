# CLAUDE.md

Guidance for AI agents and new contributors. Keep this file short and true to the code.

## What this is

A small Generative Engine Optimization (GEO) monitor: it samples an LLM many times
per wine-shopping prompt and measures each producer's share-of-voice, with bootstrap
confidence intervals and run-to-run instability. Full picture: `README.md`.

## Architecture — where things live

Pipeline stages are plain functions in `wine_geo/pipeline.py`:

- **collect** → raw samples (`wine_geo/runner.py` + `wine_geo/providers.py`)
- **extract** → mentions (`wine_geo/extract.py`)
- **aggregate** → metrics (`wine_geo/stats.py`)

Data contracts: `wine_geo/schema.py` (`RawSample`, `Mention`). Terminal report:
`report.py`. Chart: `viz.py`. Dagster wrapper (thin): `wine_geo/definitions.py`.
Prompts/config: `wine_geo/config.py`. Tracked producers: `wine_geo/data/producers.json`.

**Design rule:** business logic stays in plain functions; the CLI and the Dagster
assets are thin wrappers over `pipeline.py`. Don't put logic in `definitions.py`.

## Run, test, lint

- `make install` — editable install with dev + viz extras
- `make run` — run the monitor with the mock provider (no API key)
- `make test` — pytest with coverage
- `make lint` — ruff
- `make chart` — render the example chart

Real providers: set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (see `.env.example`), then
`python -m wine_geo --provider anthropic --model claude-haiku-4-5`.

## Conventions

- Python ≥ 3.9. The core runs on the standard library; anthropic/openai/dagster/
  matplotlib are optional extras.
- **Keep the mock provider working and offline** — tests and CI must pass with no API key.
- Line length 100; ruff (`E,F,I`) is the linter. Run `make lint` before committing.
- Add a test in `tests/` for new stage logic.

## Gotchas

- `wine_geo/definitions.py` must **not** use `from __future__ import annotations` (it
  breaks Dagster's context-type introspection), and its asset config parameter must be
  named `config` (Dagster convention), not `cfg`.
