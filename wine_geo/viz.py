"""Render share-of-voice as a static chart.

Color does a job: it encodes the story — big marketing brands vs. value labels vs.
négociant hidden-label wines — so the "value plays are invisible" finding reads at
a glance. Palette is Okabe-Ito (colorblind-safe by construction). matplotlib is
imported lazily so the rest of the package doesn't depend on it.
"""

from __future__ import annotations

from pathlib import Path

# Okabe-Ito — chosen so the négociant category (the interesting one) gets the
# vivid accent while the expected big brands stay calm.
_COLORS = {"big": "#0072B2", "value": "#E69F00", "negociant": "#D55E00"}
_LEGEND = {"big": "Big brand", "value": "Value label", "negociant": "Négociant (hidden-label)"}


def _category(note: str | None) -> str:
    n = (note or "").lower()
    if "négociant" in n or "negociant" in n:
        return "negociant"
    if "value" in n:
        return "value"
    return "big"


def render_chart(producer_rows, producers_meta, out_path, *, prompt_id,
                 prompt_text=None, jaccard=None, n=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cat_by_name = {p["name"]: _category(p.get("note")) for p in producers_meta}

    rows = sorted((r for r in producer_rows if r["prompt_id"] == prompt_id),
                  key=lambda r: r["share"])
    names = [r["producer"] for r in rows]
    shares = [r["share"] * 100 for r in rows]
    err_lo = [(r["share"] - r["ci_lo"]) * 100 for r in rows]
    err_hi = [(r["ci_hi"] - r["share"]) * 100 for r in rows]
    cats = [cat_by_name.get(nm, "big") for nm in names]
    colors = [_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(names) + 1.6)))
    ax.barh(names, shares, xerr=[err_lo, err_hi], color=colors, height=0.66,
            error_kw=dict(ecolor="#9A9A9A", elinewidth=1, capsize=3), zorder=3)

    for y, s in enumerate(shares):
        ax.text(s + err_hi[y] + 1.4, y, f"{s:.0f}%", va="center", ha="left",
                fontsize=9, color="#333333")

    ax.set_xlabel("Share of voice — % of answers that mention the producer", fontsize=9)
    ax.set_xlim(0, max(shares + [10]) * 1.28)

    ax.set_title(f"Who the model recommends — {prompt_text or prompt_id}",
                 fontsize=13, fontweight="bold", loc="left", pad=26)
    sub = []
    if n is not None:
        sub.append(f"n={n} samples")
    if jaccard is not None:
        sub.append(f"run-to-run Jaccard {jaccard:.2f}")
    sub.append("bars = share of voice · whiskers = 95% CI")
    ax.text(0.0, 1.02, "   ·   ".join(sub), transform=ax.transAxes,
            fontsize=9, color="#666666")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color="#EAEAEA", zorder=0)
    ax.set_axisbelow(True)

    present = [c for c in ("big", "value", "negociant") if c in cats]
    handles = [Patch(color=_COLORS[c], label=_LEGEND[c]) for c in present]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
