"""Render results as a readable terminal report."""

from __future__ import annotations


def _bar(share: float, width: int = 12) -> str:
    filled = round(share * width)
    return "█" * filled + "·" * (width - filled)


def render_report(results: list[dict], summary: dict) -> None:
    print()
    print("=" * 72)
    print(f"  WINE GEO MONITOR — {summary['provider']} / {summary['model']}")
    print(f"  {summary['samples']} samples across {len(results)} prompts")
    print("=" * 72)

    for r in results:
        print(f"\n▸ {r['prompt']}")
        print(f"  (n={r['n']} samples · run-to-run Jaccard {r['jaccard']:.2f} — "
              f"1.0 = identical every time, lower = more unstable)")
        ranked = sorted(
            ((name, *r["sov"][name]) for name in r["sov"] if r["sov"][name][1] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        if not ranked:
            print("    (no tracked producers mentioned)")
            continue
        print(f"    {'producer':<28}{'share':>7}  {'95% CI':>14}   distribution")
        for name, share, hits, n in ranked:
            lo, hi = r["ci"].get(name, (0.0, 0.0))
            ci = f"[{lo*100:4.0f}-{hi*100:4.0f}%]"
            print(f"    {name:<28}{share*100:6.0f}%  {ci:>14}   {_bar(share)}")
        missed = sum(1 for name in r["sov"] if r["sov"][name][1] == 0)
        if missed:
            print(f"    … {missed} tracked producers never mentioned")

    print("\n" + "-" * 72)
    print(f"  tokens: {summary['in']:,} in / {summary['out']:,} out"
          f"   est. cost: ${summary['cost']:.4f}")
    print(f"  (this run was cheap; multiply by brands × prompts × surfaces × daily "
          f"to see the real bill)")
    print("-" * 72)
