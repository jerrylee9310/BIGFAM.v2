from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


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


def prepare_corr_concordance(df_corr: pd.DataFrame) -> pd.DataFrame:
    df_use = df_corr[df_corr["corr_method"].isin(["bootstrap", "robust"])].copy()
    df_use["DOR"] = df_use["DOR"].astype(int)
    df_plot = (
        df_use
        .pivot_table(
            index=["pheno", "ptype", "DOR"],
            columns="corr_method",
            values="rho",
            aggfunc="first",
        )
        .reset_index()
        .dropna(subset=["bootstrap", "robust"])
    )
    return df_plot


def prepare_slope_ci_width(df_pred: pd.DataFrame) -> pd.DataFrame:
    df_use = df_pred[df_pred["method"].isin(["original", "proposed"])].copy()
    df_use["ci_width"] = df_use["slope_test_slope_upper"] - df_use["slope_test_slope_lower"]
    df_plot = (
        df_use
        .pivot_table(
            index=["pheno", "ptype"],
            columns="method",
            values="ci_width",
            aggfunc="first",
        )
        .reset_index()
        .dropna(subset=["original", "proposed"])
    )
    return df_plot


def add_corr_concordance_panel(ax, df_plot: pd.DataFrame, ptype: str, dor_colors: dict[int, str]):
    sub = df_plot[df_plot["ptype"] == ptype].copy()
    ax.set_facecolor("white")

    lower = float(sub[["bootstrap", "robust"]].to_numpy().min())
    upper = float(sub[["bootstrap", "robust"]].to_numpy().max())
    pad = (upper - lower) * 0.08 if upper > lower else 0.02
    xmin = max(0.0, lower - pad)
    xmax = upper + pad

    ax.plot([xmin, xmax], [xmin, xmax], color="0.65", linewidth=0.8, linestyle="--", zorder=1)

    for dor in [1, 2, 3]:
        dor_sub = sub[sub["DOR"] == dor]
        if dor_sub.empty:
            continue
        ax.scatter(
            dor_sub["bootstrap"],
            dor_sub["robust"],
            s=18 if ptype == "binary" else 14,
            color=dor_colors[dor],
            alpha=0.70,
            edgecolors="none",
            zorder=2,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_title("Binary traits" if ptype == "binary" else "Continuous traits", pad=4, fontsize=7, fontweight="normal")
    ax.set_xlabel("Estimated correlation (BIGFAM.v1)", fontsize=7)
    ax.set_ylabel("Estimated correlation (BIGFAM.v2)", fontsize=7)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False)


def add_slope_ci_panel(ax, df_plot: pd.DataFrame, ptype: str, method_colors: dict[str, str]):
    sub = df_plot[df_plot["ptype"] == ptype].copy().reset_index(drop=True)
    rng = np.random.default_rng(20260310 if ptype == "binary" else 20260311)

    jitter = rng.uniform(-0.035, 0.035, size=len(sub))
    x_v1 = np.full(len(sub), 1.0, dtype=float) + jitter
    x_v2 = np.full(len(sub), 2.0, dtype=float) + jitter

    for idx, row in sub.iterrows():
        ax.plot(
            [x_v1[idx], x_v2[idx]],
            [row["original"], row["proposed"]],
            color="0.75",
            linewidth=0.6,
            alpha=0.30,
            zorder=1,
        )

    ax.scatter(
        x_v1,
        sub["original"],
        s=15 if ptype == "binary" else 12,
        color=method_colors["original"],
        alpha=0.65,
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        x_v2,
        sub["proposed"],
        s=15 if ptype == "binary" else 12,
        color=method_colors["proposed"],
        alpha=0.65,
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        [1.0, 2.0],
        [sub["original"].mean(), sub["proposed"].mean()],
        s=30,
        color=[method_colors["original"], method_colors["proposed"]],
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    ymax = float(sub[["original", "proposed"]].to_numpy().max()) * 1.10
    ax.set_xlim(0.6, 2.4)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(["BIGFAM.v1", "BIGFAM.v2"])
    ax.set_title("Binary traits" if ptype == "binary" else "Continuous traits", pad=4, fontsize=7, fontweight="normal")
    ax.set_ylabel("Slope 95% CI width", fontsize=7)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False)


def plot_corr_concordance_figure(df_corr: pd.DataFrame, output_path: Path):
    set_bioinfo_style()
    dor_colors = {
        1: "#0E7490",
        2: "#D97706",
        3: "#7C3AED",
    }
    df_plot = prepare_corr_concordance(df_corr)

    fig = plt.figure(figsize=(6.6, 2.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.98, bottom=0.22, top=0.90, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    add_corr_concordance_panel(ax1, df_plot, "binary", dor_colors)
    add_corr_concordance_panel(ax2, df_plot, "continuous", dor_colors)

    ax1.text(-0.18, 1.17, "A", transform=ax1.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")
    ax2.text(-0.18, 1.17, "B", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")

    legend_handles = [
        Line2D([0], [0], color=dor_colors[1], marker="o", linestyle="None", markersize=4.5, label="DOR 1"),
        Line2D([0], [0], color=dor_colors[2], marker="o", linestyle="None", markersize=4.5, label="DOR 2"),
        Line2D([0], [0], color=dor_colors[3], marker="o", linestyle="None", markersize=4.5, label="DOR 3"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=6, handletextpad=0.3, ncol=1, columnspacing=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] {output_path.resolve()}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_slope_ci_width_figure(df_pred: pd.DataFrame, output_path: Path):
    set_bioinfo_style()
    method_colors = {
        "original": "#00B0D0",
        "proposed": "#F84838",
    }
    df_plot = prepare_slope_ci_width(df_pred)

    fig = plt.figure(figsize=(6.6, 2.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.98, bottom=0.22, top=0.90, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    add_slope_ci_panel(ax1, df_plot, "binary", method_colors)
    add_slope_ci_panel(ax2, df_plot, "continuous", method_colors)

    ax1.text(-0.18, 1.17, "A", transform=ax1.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")
    ax2.text(-0.18, 1.17, "B", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] {output_path.resolve()}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    root_dir = Path(__file__).resolve().parents[2]
    fig_dir = Path(__file__).resolve().parent

    corr_file = root_dir / "data/real/corr_results.tsv"
    pred_file = root_dir / "data/real/pred_results.tsv"
    df_corr = load_data(corr_file)
    for ext in [".png", ".pdf"]:
        plot_corr_concordance_figure(df_corr, fig_dir / f"fig_s2{ext}")


if __name__ == "__main__":
    main()
