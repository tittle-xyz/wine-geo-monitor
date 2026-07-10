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


def render_cost_curve(points, out_path, *, provider=None, model=None):
    """Plot the cost of confidence: run cost (x) vs the 95% CI half-width (y).

    One series, so the title names it — no legend. The point is the shape: cost
    climbs ~linearly with samples while precision improves ~1/√n, so the curve dives
    then flattens — you pay a lot for the last bit of confidence.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    pts = sorted(points, key=lambda p: p["n"])
    xs = [p["cost"] for p in pts]
    ys = [p["ci_half_width"] * 100 for p in pts]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(xs, ys, "-o", color="#0072B2", linewidth=2, markersize=7,
            markerfacecolor="#0072B2", markeredgecolor="white", markeredgewidth=1.2,
            zorder=3)

    for p, x, y in zip(pts, xs, ys):
        ax.annotate(f"n={p['n']}", (x, y), textcoords="offset points", xytext=(7, 8),
                    fontsize=9, color="#555555")

    ax.set_xlabel("Run cost (USD, list price)", fontsize=9)
    ax.set_ylabel("95% CI half-width (pts) — lower is more precise", fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.4f}"))

    ax.set_title("The cost of confidence", fontsize=13, fontweight="bold", loc="left", pad=26)
    who = " / ".join(x for x in (provider, model) if x)
    sub = "cost grows ~linearly with samples; precision improves only ~1/√n"
    ax.text(0.0, 1.02, f"{who}   ·   {sub}" if who else sub,
            transform=ax.transAxes, fontsize=9, color="#666666")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, color="#EAEAEA", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(ys) * 1.18 if ys else 1)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
