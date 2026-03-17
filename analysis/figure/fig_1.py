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


# Intuitive short titles for main figure
PATTERN_TITLES_SHORT = {
    'uniform_low': 'Low Uncertainty (Homo)',
    'uniform_mid': 'Moderate Uncertainty (Homo)',
    'uniform_high': 'High Uncertainty (Homo)',
    'realistic_mild': 'Mild Uncertainty (Heter)',
    'realistic_strong': 'Strong Uncertainty (Heter)',
    'extreme': 'Extreme Uncertainty (Heter)',
}


def plot_power_wls(df: pd.DataFrame, output_path: Path):
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

    # ---------------------------
    # Panel configs: 3 representative uncertainty scenarios
    # ---------------------------
    # power_patterns = ['uniform_low', 'realistic_mild', 'extreme']
    power_patterns = ['uniform_low', 'realistic_mild', 'uniform_high']
    # Reduce tick density to avoid overlapping labels in narrow subpanels
    xticks = np.round(np.arange(0.1, 1.0, 0.2), 1)

    # ---------------------------
    # Figure: Panel A (3 power plots) | Panel B (T1E bar)
    # ---------------------------
    fig_w_in = 7.0
    fig_h_in = 2.4
    
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    
    # Outer GridSpec: Panel A (3 plots) | Panel B (bar chart)
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[3.2, 1.0],
        wspace=0.25,
        left=0.08,
        right=0.96,
        bottom=0.30,
        top=0.82
    )
    
    # Inner GridSpec for Panel A: 3 subplots
    inner_A = outer[0, 0].subgridspec(1, 3, wspace=0.22)
    axA = [fig.add_subplot(inner_A[0, i]) for i in range(3)]
    
    # Panel B
    axB = fig.add_subplot(outer[0, 1])

    # ---------------------------
    # Panel A: power curves (3 subplots)
    # ---------------------------
    for i, pattern in enumerate(power_patterns):
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

        ax.set_title(PATTERN_TITLES_SHORT[pattern], fontsize=8, pad=4)
        ax.set_xlabel(r"Inverse decay rate ($1/w_S$)", fontsize=7)
        ax.set_ylabel("Power" if i == 0 else "", fontsize=7)

        ax.set_xlim(0.08, 0.92)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.set_ylim(-0.02, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        
        if i > 0:
            ax.set_yticklabels([])

    # Panel A label
    axA[0].text(
        -0.25, 1.18, "A",
        transform=axA[0].transAxes,
        fontsize=10, fontweight="bold",
        va="top", ha="left"
    )
    
    # Panel A legend
    legend_handles_A = [
        Line2D([0], [0], color=proposed_color, marker="o", linestyle="-",
               markersize=4, linewidth=1.2, label="BIGFAM.v2"),
        Line2D([0], [0], color=original_color, marker="s", linestyle="-",
               markersize=4, linewidth=1.2, label="BIGFAM.v1"),
    ]
    pos_A1 = axA[1].get_position()
    x_center_A = pos_A1.x0 + pos_A1.width / 2
    fig.legend(
        handles=legend_handles_A,
        loc="upper center",
        bbox_to_anchor=(x_center_A, 0.14),
        ncol=2,
        fontsize=7,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.5
    )

    # ---------------------------
    # Panel B: Type I error bar plot
    # ---------------------------
    t1e_df = df[(df["ws"] == 2.0) & (df["se_pattern"].isin(power_patterns))].copy()
    t1e_df["pattern_cat"] = pd.Categorical(t1e_df["se_pattern"], categories=power_patterns, ordered=True)

    x = np.arange(len(power_patterns))
    width = 0.35
    offsets = {
        "BIGFAM.v2": -width/2,
        "BIGFAM.v1": +width/2,
    }

    for m in method_order:
        d = t1e_df[t1e_df["Method"] == m].sort_values("pattern_cat")
        if not d.empty:
            heights = d["rejection_rate"].to_numpy()
            axB.bar(
                x + offsets[m], heights,
                width=width,
                color=colors[m],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

    axB.set_xlabel("Uncertainty scenario", fontsize=7)
    axB.set_ylabel("Type I error", fontsize=7)
    axB.set_xticks(x)
    # Short labels for x-axis
    pattern_labels = ['Low\nHomo', 'Mild\nHeter', 'High\nHomo']
    axB.set_xticklabels(pattern_labels, fontsize=6)

    # Reference line at 0.05
    axB.axhline(0.05, color="0.5", linestyle="--", linewidth=0.7, zorder=1)

    ymax = float(t1e_df["rejection_rate"].max()) if not t1e_df.empty else 0.1
    axB.set_ylim(0, max(0.12, ymax * 1.2))
    axB.set_yticks([0, 0.05, 0.10])

    # Panel B label
    axB.text(
        -0.25, 1.18, "B",
        transform=axB.transAxes,
        fontsize=10, fontweight="bold",
        va="top", ha="left"
    )
    
    # Panel B legend
    legend_handles_B = [
        Patch(facecolor=proposed_color, edgecolor='black', linewidth=0.5, label="BIGFAM.v2"),
        Patch(facecolor=original_color, edgecolor='black', linewidth=0.5, label="BIGFAM.v1"),
    ]
    pos_B = axB.get_position()
    x_center_B = pos_B.x0 + pos_B.width / 2
    fig.legend(
        handles=legend_handles_B,
        loc="upper center",
        bbox_to_anchor=(x_center_B, 0.14),
        ncol=2,
        fontsize=7,
        frameon=False,
        handlelength=1.2,
        handleheight=0.8,
        handletextpad=0.4,
        columnspacing=1.5
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
    output_png = fig_dir / "fig_1.png"
    output_pdf = fig_dir / "fig_1.pdf"

    if not input_file.exists():
        input_file = Path("data/simulation/power_wls_inv.csv")
        output_png = Path("analysis/figure/fig_1.png")
        output_pdf = Path("analysis/figure/fig_1.pdf")

    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    df_raw = load_data(input_file)
    
    print("Calculating rejection rates...")
    df_processed = calculate_rejection_rate(df_raw)
    
    print("Generating main figure (Panel A: Power, Panel B: Type I Error)...")
    plot_power_wls(df_processed, output_png)
    plot_power_wls(df_processed, output_pdf)
    print("Done.")


if __name__ == "__main__":
    main()
