"""Shared figure style: Nature Genetics sizing, PNG + PDF output.

Widths W1/W15/W2 = 89/120/183 mm, and figsize is the final printed size.
Color encodes the model class; BIGFAM is the only saturated hue.
save(fig, "figN") writes figN.png at 400 dpi.

Local copy -- every analysis/ folder is self-contained.
"""
import matplotlib as mpl

mpl.use("Agg")

MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM

C = {
    "decay": "#C51F3F",     # BIGFAM.v2 (crimson, the one saturated hue)
    "decay_v1": "#DF7786",  # BIGFAM.v1 (tint of the same hue)
    "const": "#2E5793",     # constant-C methods
    "step": "#A97E2C",      # step-C methods
    "zero": "#B4B4B4",      # AE (no common environment modelled)
}
GENETIC = "#C6C6C6"         # additive-genetic bar segment
INK = "#1a1a1a"             # axes and text
MUT = "#555555"             # secondary text
FAINT = "#999999"           # individual points
HERO_TINT = dict(fc=C["decay"], alpha=0.05, ec="none")  # BIGFAM row background
ACC_BETWEEN = "#C65321"     # fig3: between-assumption
ACC_WITHIN = "#4E8988"      # fig3: within-assumption

LW_HERO, LW_SEC, LW_REF = 1.6, 0.9, 0.5


def rc():
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "sans-serif",   # falls back through font.sans-serif below
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        # TrueType (42), not Type 3 outlines: journals require editable text
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 6, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "legend.frameon": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "xtick.major.width": 0.5,
        "ytick.major.size": 2.5, "ytick.major.width": 0.5,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.labelcolor": INK,
        # math in the body font, STIX Sans only for glyphs it lacks
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "mathtext.fallback": "stixsans",
        "lines.solid_capstyle": "round",
    })


def panel_label(ax, s, dx=-0.12, dy=0.02):
    """Panel label: bold 8pt, just outside the top-left corner."""
    ax.text(dx, 1 + dy, s, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", color=INK)


def line_labels(ax, ends, name, color, x0, x1, xtext, gap,
                hero=(), fontsize=6, hero_fontsize=6.5):
    """Direct line labels instead of a legend, spread by `gap` where ends collide.

    ends/name/color: {key: y at line end} / {key: label} / {key: color}.
    A short leader runs x0->x1, the text sits at xtext, keys in `hero` are bold.
    """
    order = sorted(ends, key=lambda m: -ends[m])
    pos, prev = {}, None
    for m in order:
        y = ends[m] if prev is None else min(ends[m], prev - gap)
        pos[m], prev = y, y
    for m, y in pos.items():
        ax.plot([x0, x1], [ends[m], y], color=color[m], lw=0.6,
                zorder=2, clip_on=False)
        ax.text(xtext, y, name[m],
                fontsize=hero_fontsize if m in hero else fontsize,
                color=color[m], ha="left", va="center", clip_on=False,
                fontweight="bold" if m in hero else "normal")
    return pos


def save(fig, stem, tight=True):
    """Write <stem>.png next to this file.

    tight=False keeps the canvas at exactly figsize -- use it when the layout is
    set by gridspec margins, since a tight bbox would crop them and change the
    printed width.
    """
    from pathlib import Path
    here = Path(__file__).resolve().parent
    # bbox_inches=None means "use the rc default", so tight is turned off via rc_context
    ctx = {} if tight else {"savefig.bbox": None}
    with mpl.rc_context(ctx):
        fig.savefig(here / f"{stem}.png")
    print(f"wrote {stem}.png")
