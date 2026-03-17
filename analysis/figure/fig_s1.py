import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np


def set_bioinfo_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,

        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",

        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "legend.frameon": False,
        "legend.fontsize": 7.0,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    })


def load_data(filepath: Path) -> pd.DataFrame:
    # Auto-detect delimiter to support both CSV and TSV inputs
    return pd.read_csv(filepath, sep=None, engine="python")


SIGNIFICANCE_ALIASES = {
    "high": "fast",
    "fast": "fast",
    "low": "slow",
    "slow": "slow",
    "similar": "similar",
}


def canonical_significance(value: str) -> str:
    return SIGNIFICANCE_ALIASES.get(str(value).lower(), str(value).lower())


def calculate_rejection_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rejection rate while accepting both legacy high/low and
    current fast/slow significance labels.
    """
    df = df.copy()
    df["ws"] = pd.to_numeric(df["ws"], errors="coerce")
    df = df.dropna(subset=["ws"])
    df["significance_norm"] = df["significance"].map(canonical_significance)

    def is_correct_rejection(row):
        ws = row["ws"]
        sig = row["significance_norm"]
        if ws < 2:
            return sig == "slow"
        elif ws > 2:
            return sig == "fast"
        else:  # ws == 2
            return sig != "similar"  # For T1E: rejection when should not reject
    
    df["is_rejected"] = df.apply(is_correct_rejection, axis=1)
    agg_df = df.groupby(["ws", "se_pattern", "method"])["is_rejected"].mean().reset_index()
    agg_df["inv_ws"] = 1.0 / agg_df["ws"]
    agg_df.rename(columns={"is_rejected": "rejection_rate"}, inplace=True)
    return agg_df


# Intuitive titles with SE values
PATTERN_TITLES = {
    'uniform_low': 'Low Uncertainty (Homo)\n($\\mathrm{SE}_{d} = 0.005$)',
    'uniform_mid': 'Moderate Uncertainty (Homo)\n($\\mathrm{SE}_{d} = 0.01$)',
    'uniform_high': 'High Uncertainty (Homo)\n($\\mathrm{SE}_{d} = 0.05$)',
    'realistic_mild': 'Mild Uncertainty (Heter)\n($\\mathrm{SE}_{1,2,3} = 0.005, 0.01, 0.02$)',
    'realistic_strong': 'Strong Uncertainty (Heter)\n($\\mathrm{SE}_{1,2,3} = 0.005, 0.02, 0.05$)',
    'extreme': 'Extreme Uncertainty (Heter)\n($\\mathrm{SE}_{1,2,3} = 0.001, 0.01, 0.1$)',
}

# Short labels for T1E bar chart x-axis
PATTERN_SHORT_LABELS = {
    'uniform_low': 'Low\nHomo',
    'uniform_mid': 'Mid\nHomo',
    'uniform_high': 'High\nHomo',
    'realistic_mild': 'Mild\nHeter',
    'realistic_strong': 'Strong\nHeter',
    'extreme': 'Extreme\nHeter',
}


def plot_all_patterns(df: pd.DataFrame, output_path: Path):
    set_bioinfo_style()

    # ---------------------------
    # Map + filter methods
    # ---------------------------
    method_map = {
        "Jensen_WLS": "BIGFAM.v2",
        "resample": "BIGFAM.v1",
    }
    df = df[df["method"].isin(method_map.keys())].copy()
    df["Method"] = df["method"].map(method_map)

    method_order = ["BIGFAM.v1", "BIGFAM.v2"]  # Plot BIGFAM.v1 first, BIGFAM.v2 on top
    df["Method"] = pd.Categorical(df["Method"], categories=method_order, ordered=True)

    # ---------------------------
    # Colors & markers
    # ---------------------------
    proposed_color = "#F84838"
    original_color = "#00B0D0"
    colors = {
        "BIGFAM.v2": proposed_color,
        "BIGFAM.v1": original_color,
    }
    markers = {
        "BIGFAM.v2": "o",
        "BIGFAM.v1": "s",
    }
    marker_size = 2.0

    # All 6 uncertainty scenarios
    all_patterns = [
        'uniform_low', 
        'uniform_mid', 
        'uniform_high', 
        'realistic_mild', 
        'realistic_strong', 
        'extreme'
    ]
    
    xticks = np.round(np.arange(0.1, 1.0, 0.1), 1)

    # ---------------------------
    # Figure: Panel A (2x3 power plots) on top, Panel B (bar chart) below
    # ---------------------------
    fig_w_in = 7.0
    fig_h_in = 6.0
    
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    
    # GridSpec: Panel A (2 rows x 3 cols) on top, Panel B below
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1, 1, 0.8],
        hspace=0.45,
        wspace=0.25,
        left=0.08,
        right=0.96,
        bottom=0.10,
        top=0.92
    )
    
    # Panel A: 2x3 grid of power plots
    axA = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            axA.append(ax)
    
    # Panel B: spans all 3 columns
    axB = fig.add_subplot(gs[2, :])

    # ---------------------------
    # Panel A: power curves (6 subplots)
    # ---------------------------
    for i, pattern in enumerate(all_patterns):
        ax = axA[i]
        sub_all = df[df["se_pattern"] == pattern].copy()

        for m in method_order:
            sub = sub_all[sub_all["Method"] == m].sort_values("inv_ws")
            ax.plot(
                sub["inv_ws"], sub["rejection_rate"],
                color=colors[m],
                marker=markers[m],
                markersize=marker_size,
                linewidth=1.2,
                zorder=3
            )

        ax.axhline(0.05, color="0.5", linestyle="--", linewidth=0.7, zorder=1)
        ax.axvline(0.5, color="0.7", linestyle=":", linewidth=0.7, zorder=1)

        ax.set_title(PATTERN_TITLES[pattern], fontsize=7, pad=4)
        ax.set_xlabel(r"Inverse decay rate ($1/w_S$)", fontsize=7)
        
        # Y-label only on leftmost plots
        if i % 3 == 0:
            ax.set_ylabel("Power", fontsize=7)
        
        ax.set_xlim(0.08, 0.92)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.set_ylim(-0.02, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        
        if i % 3 != 0:
            ax.set_yticklabels([])

    # Panel A label
    axA[0].text(
        -0.25, 1.25, "A",
        transform=axA[0].transAxes,
        fontsize=10, fontweight="bold",
        va="top", ha="left"
    )

    # ---------------------------
    # Panel B: Type I error bar plot for all patterns
    # ---------------------------
    t1e_df = df[(df["ws"] == 2.0) & (df["se_pattern"].isin(all_patterns))].copy()
    t1e_df["pattern_cat"] = pd.Categorical(t1e_df["se_pattern"], categories=all_patterns, ordered=True)

    x = np.arange(len(all_patterns))
    width = 0.35
    offsets = {
        "BIGFAM.v2": -width/2,
        "BIGFAM.v1": +width/2,
    }

    for m in method_order:
        d = t1e_df[t1e_df["Method"] == m].sort_values("pattern_cat")
        if not d.empty:
            heights = d["rejection_rate"].to_numpy()
            label = m
            axB.bar(
                x + offsets[m], heights,
                width=width,
                color=colors[m],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
                label=label
            )

    axB.set_xlabel("Uncertainty Scenario", fontsize=7)
    axB.set_ylabel("Type I Error", fontsize=7)
    axB.set_xticks(x)
    pattern_labels = [PATTERN_SHORT_LABELS[p] for p in all_patterns]
    axB.set_xticklabels(pattern_labels, fontsize=6.5)

    # Reference line at 0.05
    axB.axhline(0.05, color="0.5", linestyle="--", linewidth=0.7, zorder=1)

    ymax = float(t1e_df["rejection_rate"].max()) if not t1e_df.empty else 0.1
    axB.set_ylim(0, max(0.12, ymax * 1.2))
    axB.set_yticks([0, 0.05, 0.10])

    # Panel B label
    axB.text(
        -0.05, 1.15, "B",
        transform=axB.transAxes,
        fontsize=10, fontweight="bold",
        va="top", ha="left"
    )
    
    # Panel B legend
    legend_handles_B = [
        Patch(facecolor=proposed_color, edgecolor='black', linewidth=0.5, label="BIGFAM.v2"),
        Patch(facecolor=original_color, edgecolor='black', linewidth=0.5, label="BIGFAM.v1"),
    ]
    axB.legend(
        handles=legend_handles_B,
        loc="upper right",
        fontsize=7,
        frameon=False,
    )

    # ---------------------------
    # Save
    # ---------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SAVE] {output_path.resolve()}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    try:
        root_dir = Path(__file__).resolve().parents[2]
        fig_dir = Path(__file__).resolve().parent
    except NameError:
        root_dir = Path.cwd()
        fig_dir = Path("analysis/figure")

    input_file = root_dir / "data/simulation/power_wls_inv.csv"
    output_png = fig_dir / "fig_s1.png"
    output_pdf = fig_dir / "fig_s1.pdf"

    if not input_file.exists():
        input_file = Path("data/simulation/power_wls_inv.csv")
        output_png = Path("analysis/figure/fig_s1.png")
        output_pdf = Path("analysis/figure/fig_s1.pdf")

    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    df_raw = load_data(input_file)
    
    print("Calculating rejection rates...")
    df_processed = calculate_rejection_rate(df_raw)
    
    print("Generating supplementary figure (Panel A: 2x3 Power, Panel B: T1E)...")
    plot_all_patterns(df_processed, output_png)
    plot_all_patterns(df_processed, output_pdf)
    print("Done.")


if __name__ == "__main__":
    main()
