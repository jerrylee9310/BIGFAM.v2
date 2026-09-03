"""Shared figure style: Nature Genetics sizing, PNG output.

Trimmed to what figS7 uses (`C`, `rc`, `panel_label`, `save`).
Local copy -- every analysis/ folder is self-contained.
"""
import matplotlib as mpl

mpl.use("Agg")

MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM

C = {
    "decay": "#C51F3F",     # BIGFAM.v2 -- the one high-saturation hero (crimson)
    "decay_v1": "#DF7786",  # BIGFAM.v1 -- tint of the hero (deep rose)
    "const": "#2E5793",     # deep steel blue
    "step": "#A97E2C",      # muted ochre
    "zero": "#B4B4B4",      # AE -- neutral grey (no shared-environment term)
}
GENETIC = "#C6C6C6"
INK = "#1a1a1a"
MUT = "#555555"
FAINT = "#999999"
HERO_TINT = dict(fc=C["decay"], alpha=0.05, ec="none")
ACC_BETWEEN = "#C65321"
ACC_WITHIN = "#4E8988"

LW_HERO, LW_SEC, LW_REF = 1.6, 0.9, 0.5


def rc():
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "sans-serif",   # falls back through font.sans-serif below
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 6, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "legend.frameon": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "xtick.major.width": 0.5,
        "ytick.major.size": 2.5, "ytick.major.width": 0.5,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.labelcolor": INK,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "mathtext.fallback": "stixsans",
        "lines.solid_capstyle": "round",
    })


def panel_label(ax, s, dx=-0.12, dy=0.02):
    ax.text(dx, 1 + dy, s, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, stem, tight=True):
    """PNG (preview) + PDF (vector) next to this file. stem has no extension."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    ctx = {} if tight else {"savefig.bbox": None}
    with mpl.rc_context(ctx):
        fig.savefig(here / f"{stem}.png")
    print(f"wrote {stem}.png")
