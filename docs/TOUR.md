# A tour of the codebase

A plain-language walk through wine-geo-monitor for a human getting oriented. It
follows **one full run — from the command you type to the chart it draws** —
touching each file in the order the data moves.

(`CLAUDE.md` is the compact version of this for AI agents; the ADRs in `docs/adr/`
explain *why* each load-bearing choice was made.)

## The 30-second model

It asks an AI the same shopping questions many times, **saves the raw answers**,
then derives everything else — who got mentioned, share-of-voice, confidence, cost —
from those saved answers. Three stages, with saved data between each:

```
collect   →   extract   →   aggregate
save raw      who's named    share + confidence + cost
```

The real work is plain functions in `wine_geo/pipeline.py`. The CLI and the Dagster
assets are just two doors that call those same functions. That separation is the
core of the design (see ADR-0002 and ADR-0003).

## Follow one run

**0 · `python -m wine_geo`** → `wine_geo/__main__.py` (a one-line shim) → `main()` in
`wine_geo/cli.py`.

**1 · Setup** (`cli.py`, `main`):
- load the tracked producers and compile a regex per name + aliases
  (`extract.load_producers`, `extract.build_patterns`)
- pick a provider (`providers.get_provider` — defaults to the offline `mock`)
- load the questions (`config.DEFAULT_PROMPTS`)

**2 · collect** (`pipeline.collect`) — sample each prompt N times.
- The real work is in `runner.sample_prompt`: a **semaphore** caps how many calls are
  in flight at once (respects rate limits, caps spend); `asyncio.to_thread` runs each
  synchronous SDK call off the event loop; transient failures **retry with
  exponential backoff + jitter**.
- Each answer becomes a `RawSample` (`schema.py`) — the immutable, paid layer,
  carrying the response text and its token counts. **This is the durable raw layer
  everything else is derived from.**

**3 · extract** (`pipeline.extract_stage`) — run `extract.extract_mentions` over each
saved answer: a regex alias-matcher with non-alphanumeric boundaries (so `Ridge`
doesn't match `Ridgecrest`). Produces one `Mention` per (sample, producer).

**4 · aggregate** (`pipeline.aggregate_stage`) — rebuild the per-sample mention sets
and compute, in `stats.py`:
- `share_of_voice` — fraction of answers naming each producer
- `bootstrap_ci` — the "probably between X and Y" range around that fraction
- `mean_pairwise_jaccard` — how much the *set* of recommendations changes run-to-run
  (1.0 = identical every time)

**5 · cost** (`pipeline.cost_stage`) — turn the token counts already on each
`RawSample` into dollars via `providers.PRICING` / `estimate_cost`. No API calls —
just math on saved data.

**6 · output** — `report.render_report` prints the terminal report; `--out-dir`
writes each layer as JSONL (`schema.write_jsonl`); `--chart` / `--cost-curve` render
PNGs via `viz.py`.

## The glue: two data shapes

Everything rides on `wine_geo/schema.py`:

| Shape | What it is |
|---|---|
| `RawSample` | one model response — expensive, immutable, **saved**. The source of truth. |
| `Mention` | one producer found in one sample — cheap, thrown away and recomputed freely. |

The seam between stages is these records written to JSONL. Because every downstream
number is a pure function of `RawSample`, you can reprocess without re-paying for API
calls — and any metric is auditable back to what the model actually said.

## The two front doors

- **`wine_geo/cli.py`** — the un-orchestrated path, for a quick local run.
- **`wine_geo/definitions.py`** — the same stage functions wrapped as **Dagster**
  assets, one partition per day, on a schedule. It holds no business logic; it just
  calls `pipeline.py`. You could remove Dagster without touching the analysis.

## Where to look for what

| File | Role |
|---|---|
| `pipeline.py` | the stages as plain functions — **start here; it's the table of contents** |
| `schema.py` | the two data shapes (`RawSample`, `Mention`) + JSONL I/O |
| `providers.py` | `Provider` interface + mock / Anthropic / OpenAI + the pricing table |
| `runner.py` | concurrent sampling, rate limiting, retry/backoff |
| `extract.py` | producer mention detection (alias matching) |
| `stats.py` | share-of-voice, bootstrap CI, pairwise Jaccard |
| `report.py` / `viz.py` | terminal report / charts |
| `cli.py` / `definitions.py` | the two front doors (CLI, Dagster) |
| `config.py` | default prompts, model, and paths |
| `data/producers.json` | the tracked producers + aliases |

## Suggested reading order

1. `pipeline.py` — every real step is a function here.
2. `schema.py` — learn the two data shapes and the rest falls into place.
3. Branch out as needed: `providers.py`, `stats.py`, `extract.py`, `runner.py`.
4. `docs/adr/` for the *why* behind the load-bearing choices.
5. Then `make run` and watch it — reading and running together is the fastest way in.
