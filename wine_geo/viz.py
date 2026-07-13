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


# Okabe-Ito, assigned to engines in fixed order (never cycled) — color = which model.
_ENGINE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]


def render_cross_engine_chart(labels, producers, table, out_path, *,
                              prompt_id=None, prompt_text=None):
    """Grouped horizontal bars: share-of-voice per producer, one bar per model.

    `labels` are the models in order; `producers` are ordered ascending by max share
    (so the leader sits on top); `table[producer][label]` is `(share, ci_lo, ci_hi)`.
    Color encodes the model — the same producer's two bars sit together so the
    cross-engine gap reads at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_eng = len(labels)
    bar_h = 0.82 / n_eng
    ys = list(range(len(producers)))

    fig, ax = plt.subplots(figsize=(9, max(3.6, 0.62 * len(producers) + 1.5)))
    for ei, label in enumerate(labels):
        color = _ENGINE_COLORS[ei % len(_ENGINE_COLORS)]
        offs = [y + (ei - (n_eng - 1) / 2) * bar_h for y in ys]
        shares = [table[p][label][0] * 100 for p in producers]
        lo = [(table[p][label][0] - table[p][label][1]) * 100 for p in producers]
        hi = [(table[p][label][2] - table[p][label][0]) * 100 for p in producers]
        ax.barh(offs, shares, height=bar_h * 0.9, color=color, zorder=3,
                xerr=[lo, hi], error_kw=dict(ecolor="#9A9A9A", elinewidth=0.9, capsize=2))
        for y, s, h in zip(offs, shares, hi):
            if s > 0:
                ax.text(s + h + 1.6, y, f"{s:.0f}%", va="center", ha="left",
                        fontsize=7.5, color="#555555")

    ax.set_yticks(ys)
    ax.set_yticklabels(producers, fontsize=9)
    ax.set_xlabel("Share of voice — % of answers that mention the producer", fontsize=9)
    ax.set_xlim(0, 108)

    ax.set_title(f"Who each model recommends — {prompt_text or prompt_id or ''}",
                 fontsize=13, fontweight="bold", loc="left", pad=24)
    ax.text(0.0, 1.02, "bars = share of voice · whiskers = 95% CI · one color per model",
            transform=ax.transAxes, fontsize=9, color="#666666")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color="#EAEAEA", zorder=0)
    ax.set_axisbelow(True)

    handles = [Patch(color=_ENGINE_COLORS[ei % len(_ENGINE_COLORS)], label=lbl)
               for ei, lbl in enumerate(labels)]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, title="model")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def render_sample_size_chart(out_path, *, confidence=0.95, marks=(0.05, 0.08, 0.10, 0.15)):
    """The planning view: samples per prompt needed for a target CI half-width.

    Purely the √n relationship (worst case p=0.5), so it renders with no API run —
    the deterministic companion to the empirical cost-of-confidence curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .stats import recommend_sample_size

    widths = [w / 1000 for w in range(30, 205, 5)]  # ±3.0 → ±20.0 pts
    xs = [w * 100 for w in widths]
    ns = [recommend_sample_size(w, confidence=confidence) for w in widths]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.plot(xs, ns, "-", color="#0072B2", linewidth=2, zorder=3)
    for w in marks:
        n = recommend_sample_size(w, confidence=confidence)
        ax.plot(w * 100, n, "o", color="#0072B2", markersize=7,
                markerfacecolor="#0072B2", markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.annotate(f"±{w * 100:.0f} pts → {n}", (w * 100, n), textcoords="offset points",
                    xytext=(9, 6), fontsize=9, color="#555555")

    ax.set_xlabel("Target 95% CI half-width (± percentage points)", fontsize=9)
    ax.set_ylabel("Samples per prompt needed", fontsize=9)
    ax.set_title("How many samples for a target precision",
                 fontsize=13, fontweight="bold", loc="left", pad=24)
    ax.text(0.0, 1.02, "worst case (p=0.5) — tightening the target by half costs ~4× the samples",
            transform=ax.transAxes, fontsize=9, color="#666666")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, color="#EAEAEA", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
