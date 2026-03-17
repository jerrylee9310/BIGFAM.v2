import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
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
    return pd.read_csv(filepath, sep="\t")


def prepare_paired_se(df_corr: pd.DataFrame) -> pd.DataFrame:
    df_use = df_corr[df_corr["corr_method"].isin(["bootstrap", "robust"])].copy()
    df_use["DOR"] = df_use["DOR"].astype(int)

    df_paired = (
        df_use
        .pivot_table(
            index=["pheno", "ptype", "DOR"],
            columns="corr_method",
            values="se",
            aggfunc="first",
        )
        .reset_index()
        .dropna(subset=["bootstrap", "robust"])
    )
    return df_paired


def format_se_axis(
    ax: Axes,
    subset: pd.DataFrame,
    show_xlabel: bool,
):
    ymax = float(subset[["bootstrap", "robust"]].to_numpy().max()) * 1.22
    ax.set_facecolor("white")
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([1, 2, 3])
    if show_xlabel:
        ax.set_xticklabels(["1", "2", "3"])
        ax.set_xlabel("Degree of relatedness (DOR)", fontsize=7)
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)

    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False)


def add_se_subplot_box(
    ax: Axes,
    df_se: pd.DataFrame,
    ptype: str,
    method_colors: dict[str, str],
    show_xlabel: bool,
):
    subset = df_se[df_se["ptype"] == ptype].copy()
    rng = np.random.default_rng(20260309 if ptype == "binary" else 20260310)
    offset = 0.18
    width = 0.26

    for dor in [1, 2, 3]:
        dor_df = subset[subset["DOR"] == dor]
        if dor_df.empty:
            continue

        for method, xpos in [("bootstrap", dor - offset), ("robust", dor + offset)]:
            values = dor_df[method].to_numpy(dtype=float)
            box = ax.boxplot(
                [values],
                positions=[xpos],
                widths=width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.8},
                boxprops={"linewidth": 0.7, "edgecolor": "black"},
                whiskerprops={"linewidth": 0.7, "color": "black"},
                capprops={"linewidth": 0.7, "color": "black"},
            )
            box["boxes"][0].set_facecolor(method_colors[method])
            box["boxes"][0].set_alpha(0.70)

            jitter = rng.uniform(-0.045, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), xpos, dtype=float) + jitter,
                values,
                s=10 if ptype == "continuous" else 13,
                color=method_colors[method],
                alpha=0.35,
                edgecolors="none",
                zorder=3,
            )

    format_se_axis(ax, subset, show_xlabel)


def add_se_subplot_paired_scatter(
    ax: Axes,
    df_se: pd.DataFrame,
    ptype: str,
    method_colors: dict[str, str],
    show_xlabel: bool,
):
    subset = df_se[df_se["ptype"] == ptype].copy()
    rng = np.random.default_rng(20260311 if ptype == "binary" else 20260312)
    offset = 0.16

    for dor in [1, 2, 3]:
        dor_df = subset[subset["DOR"] == dor].reset_index(drop=True)
        if dor_df.empty:
            continue

        jitter = rng.uniform(-0.035, 0.035, size=len(dor_df))
        x_boot = np.full(len(dor_df), dor - offset, dtype=float) + jitter
        x_robust = np.full(len(dor_df), dor + offset, dtype=float) + jitter

        for idx, row in dor_df.iterrows():
            ax.plot(
                [x_boot[idx], x_robust[idx]],
                [row["bootstrap"], row["robust"]],
                color="0.75",
                linewidth=0.6,
                alpha=0.30,
                zorder=1,
            )

        ax.scatter(
            x_boot,
            dor_df["bootstrap"],
            s=12 if ptype == "continuous" else 15,
            color=method_colors["bootstrap"],
            alpha=0.65,
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            x_robust,
            dor_df["robust"],
            s=12 if ptype == "continuous" else 15,
            color=method_colors["robust"],
            alpha=0.65,
            edgecolors="none",
            zorder=3,
        )

        ax.scatter(
            [dor - offset, dor + offset],
            [dor_df["bootstrap"].mean(), dor_df["robust"].mean()],
            s=28,
            color=[method_colors["bootstrap"], method_colors["robust"]],
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )

    format_se_axis(ax, subset, show_xlabel)
def add_se_subplot_slopegraph(
    ax: Axes,
    df_se: pd.DataFrame,
    ptype: str,
    dor_colors: dict[int, str],
    show_xlabel: bool,
):
    subset = df_se[df_se["ptype"] == ptype].copy()
    rng = np.random.default_rng(20260313 if ptype == "binary" else 20260314)
    dor_offsets = {1: -0.18, 2: 0.0, 3: 0.18}
    ax.set_facecolor("white")

    for dor in [1, 2, 3]:
        dor_df = subset[subset["DOR"] == dor].reset_index(drop=True)
        if dor_df.empty:
            continue

        jitter = rng.uniform(-0.03, 0.03, size=len(dor_df))
        x_left = np.full(len(dor_df), 1.0 + dor_offsets[dor], dtype=float) + jitter
        x_right = np.full(len(dor_df), 2.0 + dor_offsets[dor], dtype=float) + jitter

        for idx, row in dor_df.iterrows():
            ax.plot(
                [x_left[idx], x_right[idx]],
                [row["bootstrap"], row["robust"]],
                color=dor_colors[dor],
                linewidth=0.7,
                alpha=0.20 if ptype == "continuous" else 0.30,
                zorder=1,
            )

        mean_boot = float(dor_df["bootstrap"].mean())
        mean_robust = float(dor_df["robust"].mean())
        ax.plot(
            [1.0 + dor_offsets[dor], 2.0 + dor_offsets[dor]],
            [mean_boot, mean_robust],
            color=dor_colors[dor],
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
        )
        ax.scatter(
            [1.0 + dor_offsets[dor], 2.0 + dor_offsets[dor]],
            [mean_boot, mean_robust],
            s=26,
            color=dor_colors[dor],
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )

    ymax = float(subset[["bootstrap", "robust"]].to_numpy().max()) * 1.22
    ax.set_xlim(0.6, 2.4)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([1.0, 2.0])
    if show_xlabel:
        ax.set_xticklabels(["Original", "Proposed"])
        ax.set_xlabel("Correlation-estimation method", fontsize=7)
    else:
        ax.set_xticklabels([])

    ax.set_ylabel("")
    ax.grid(False)


def add_variance_subplot(
    ax: Axes,
    df_pred: pd.DataFrame,
    ptype: str,
    decay_colors: dict[str, str],
    show_xlabel: bool,
    show_xticklabels: bool,
):
    df_plot = df_pred[df_pred["method"] == "proposed"].copy()
    df_plot["decay_class"] = df_plot["slope_test_significance"].str.lower()
    df_plot = df_plot[df_plot["decay_class"].isin(decay_colors)].copy()
    ax.set_facecolor("white")

    stable_mask = df_plot["V_G_lower"] >= 1e-4
    df_vc = df_plot[
        stable_mask &
        ((df_plot["V_G"] + df_plot["V_S"]) <= 1.0) &
        (df_plot["ptype"] == ptype)
    ].copy()

    marker = "s"

    for decay in ["slow", "similar"]:
        sub = df_vc[df_vc["decay_class"] == decay]
        if sub.empty:
            continue
        ax.scatter(
            sub["V_G"],
            sub["V_S"],
            c=decay_colors[decay],
            marker=marker,
            s=28 if ptype == "continuous" else 32,
            edgecolors="none",
            alpha=0.65,
            zorder=2,
        )

    max_vg = max(0.1, float(df_vc["V_G"].max())) * 1.08
    max_vs = max(0.1, float(df_vc["V_S"].max())) * 1.08
    ax.set_xlim(-0.02, max_vg + 0.02)
    if ptype == "binary":
        ax.set_ylim(0.0, 0.45)
        ax.set_yticks([0.00, 0.15, 0.30, 0.45])
    else:
        ax.set_ylim(0.0, max_vs + 0.02)
        ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if show_xlabel:
        ax.set_xlabel("Genetic variance", fontsize=7)
    else:
        ax.set_xlabel("")
    if not show_xticklabels:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.tick_params(axis="x", labelbottom=True)
    ax.grid(False)


def plot_figure_2(
    df_corr: pd.DataFrame,
    df_pred: pd.DataFrame,
    output_path: Path,
    ptype: str,
    panel_a_style: str = "box",
):
    """Figure 2-style panel for a single phenotype type."""
    set_bioinfo_style()

    method_colors = {
        "bootstrap": "#00B0D0",  # original
        "robust": "#F84838",     # proposed
    }
    dor_colors = {
        1: "#0E7490",
        2: "#D97706",
        3: "#7C3AED",
    }
    decay_colors = {
        "slow": "#D97706",
        "similar": "#7D7D7D",
    }

    df_se = prepare_paired_se(df_corr)

    fig = plt.figure(figsize=(6.6, 2.3))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.0],
        wspace=0.28,
        left=0.08,
        right=0.98,
        bottom=0.24,
        top=0.90,
    )
    axA = fig.add_subplot(outer[0, 0])
    axB = fig.add_subplot(outer[0, 1])

    if panel_a_style == "box":
        add_se_subplot_box(axA, df_se, ptype, method_colors, show_xlabel=True)
    elif panel_a_style == "paired_scatter":
        add_se_subplot_paired_scatter(axA, df_se, ptype, method_colors, show_xlabel=True)
    elif panel_a_style == "slopegraph":
        add_se_subplot_slopegraph(axA, df_se, ptype, dor_colors, show_xlabel=True)
    else:
        raise ValueError(f"Unknown panel_a_style: {panel_a_style}")
    add_variance_subplot(
        axB, df_pred, ptype, decay_colors, show_xlabel=True, show_xticklabels=True
    )

    axA.set_ylabel("SE of estimated correlation", fontsize=7, labelpad=8)
    axB.set_ylabel("Shared-env. variance", fontsize=7, labelpad=8)

    axA.text(
        -0.18,
        1.18,
        "A",
        transform=axA.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )
    axB.text(
        -0.20,
        1.18,
        "B",
        transform=axB.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    if panel_a_style == "slopegraph":
        panel_a_handles = [
            Line2D([0], [0], color=dor_colors[1], linewidth=2.0, label="DOR 1"),
            Line2D([0], [0], color=dor_colors[2], linewidth=2.0, label="DOR 2"),
            Line2D([0], [0], color=dor_colors[3], linewidth=2.0, label="DOR 3"),
        ]
        legend_ncol = 3
    else:
        panel_a_handles = [
            Line2D(
                [0], [0], color=method_colors["robust"], marker="s", linestyle="None",
                markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="With robust SE",
            ),
            Line2D(
                [0], [0], color=method_colors["bootstrap"], marker="s", linestyle="None",
                markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Without robust SE",
            ),
        ]
        legend_ncol = 1
    axA.legend(
        handles=panel_a_handles,
        loc="upper left",
        fontsize=6,
        frameon=False,
        handletextpad=0.3,
        ncol=legend_ncol,
        columnspacing=0.8,
    )

    panel_b_handles = [
        Line2D(
            [0], [0], color=decay_colors["slow"], marker="s", linestyle="None",
            markersize=5, label="Slow",
        ),
        Line2D(
            [0], [0], color=decay_colors["similar"], marker="s", linestyle="None",
            markersize=5, label="Similar",
        ),
    ]
    axB.legend(
        handles=panel_b_handles,
        loc="upper right",
        fontsize=6,
        frameon=False,
        handletextpad=0.3,
        ncol=1,
        columnspacing=0.8,
    )

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

    corr_file = root_dir / "data/real/corr_results.tsv"
    pred_file = root_dir / "data/real/pred_results.tsv"

    df_corr = load_data(corr_file)
    df_pred = load_data(pred_file)

    output_png = fig_dir / "fig_2.png"
    output_pdf = fig_dir / "fig_2.pdf"
    plot_figure_2(df_corr, df_pred, output_png, ptype="binary", panel_a_style="paired_scatter")
    plot_figure_2(df_corr, df_pred, output_pdf, ptype="binary", panel_a_style="paired_scatter")


if __name__ == "__main__":
    main()
