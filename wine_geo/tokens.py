"""How much of the token budget do we actually use? Output-length distributions per prompt.

At scale the binding rate limit on an LLM provider is **tokens-per-minute**, not requests/minute.
Output length is unknown before the call, so you reserve `input + max_tokens` against the TPM budget
up front — but if a prompt's real output runs far below the cap, most of that reservation is wasted
headroom, and you could pack many more calls into the same budget by reserving at (say) the p95 of
what the prompt actually produces, then reconciling to the real usage after the call.

This measures that, per prompt: the output-token distribution, how often it *saturates* the cap
(truncated → the cap is too low, not too high), and the packing gain a p95 reservation would buy
over a blunt static cap. Because the same prompts run every day, the durable raw layer already holds
the history to learn from — no new spend.

    python -m wine_geo.tokens out/openai
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

from .providers import MAX_TOKENS
from .schema import read_jsonl


def _pct(sorted_vals, q):
    """The q-th percentile (0-100) of a pre-sorted list, nearest-rank. 0 on empty."""
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, round(q / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def output_distribution(records, *, cap=MAX_TOKENS,
                        key=lambda r: (r["prompt_id"], r.get("model", "?"))):
    """Per group (default prompt_id × model): the output-token distribution + cap saturation.

    Skips errored samples. Returns rows sorted by key:
    {key, n, in_mean, mean, p50, p95, p99, max, cap_hits} — `cap_hits` is the fraction of samples
    whose output reached the reservation ceiling (i.e. got truncated at `cap`).
    """
    groups = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        groups[key(r)].append(r)
    rows = []
    for k, rs in groups.items():
        outs = sorted(r["output_tokens"] for r in rs)
        n = len(outs)
        rows.append({
            "key": k,
            "n": n,
            "in_mean": sum(r["input_tokens"] for r in rs) / n,
            "mean": sum(outs) / n,
            "p50": _pct(outs, 50),
            "p95": _pct(outs, 95),
            "p99": _pct(outs, 99),
            "max": outs[-1],
            "cap_hits": sum(1 for o in outs if o >= cap) / n,
        })
    rows.sort(key=lambda d: d["key"])
    return rows


def packing_gain(in_mean, p95_out, *, cap=MAX_TOKENS):
    """Extra calls that fit in the same tokens-per-minute budget when you reserve at p95 of real
    output instead of the static cap. Reservation per call = input + (cap | p95); input counts
    because it's reserved too. 1.0x = no gain (the prompt already saturates the cap)."""
    learned = in_mean + p95_out
    return (in_mean + cap) / learned if learned else 1.0


def _load(root):
    if os.path.isfile(os.path.join(root, "raw.jsonl")):
        files = [os.path.join(root, "raw.jsonl")]
    else:
        files = sorted(glob.glob(os.path.join(root, "**", "raw.jsonl"), recursive=True))
    records = []
    for f in files:
        records.extend(read_jsonl(f))
    return records, files


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wine_geo.tokens",
        description="Output-token distribution vs. reservation — how tight could the budget be?")
    ap.add_argument("root", help="dir holding raw.jsonl (or day-partition subdirs)")
    ap.add_argument("--cap", type=int, default=MAX_TOKENS,
                    help=f"reservation ceiling / max_tokens (default {MAX_TOKENS})")
    args = ap.parse_args(argv)

    records, files = _load(args.root)
    if not records:
        print(f"no raw.jsonl under {args.root!r}", file=sys.stderr)
        return 1
    rows = output_distribution(records, cap=args.cap)

    print(f"# Output-token reservation ({args.root}, {len(records)} samples, cap={args.cap})")
    print("  reservation/call = input + cap; 'gain@p95' = extra calls per token-minute if you "
          "reserve at p95 instead")
    print(f"  {'prompt/model':24}{'n':>4}{'mean':>6}{'p50':>5}{'p95':>5}{'p99':>5}{'max':>5}"
          f"{'cap%':>6}{'gain@p95':>10}")
    tot_reserved = tot_used = 0.0
    for r in rows:
        pid, model = r["key"]
        gain = packing_gain(r["in_mean"], r["p95"], cap=args.cap)
        flag = "  <- saturates cap" if r["cap_hits"] >= 0.5 else ""
        print(f"  {pid + '/' + model:24.24}{r['n']:>4}{r['mean']:>6.0f}{r['p50']:>5.0f}"
              f"{r['p95']:>5.0f}{r['p99']:>5.0f}{r['max']:>5.0f}{100 * r['cap_hits']:>5.0f}%"
              f"{gain:>9.1f}x{flag}")
        tot_reserved += r["n"] * (r["in_mean"] + args.cap)
        tot_used += r["n"] * (r["in_mean"] + r["mean"])

    eff = tot_used / tot_reserved if tot_reserved else 0.0
    print()
    print(f"  overall: {100 * eff:.0f}% of the reserved token budget is actually used "
          f"(reserving max_tokens={args.cap} per call).")
    print("  reserving each prompt at its p95 packs more calls into the same TPM — biggest wins "
          "where cap% is low; saturating prompts need a HIGHER cap, not a lower reservation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
