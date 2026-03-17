from pathlib import Path

from fig_2 import load_data, plot_figure_2


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

    output_png = fig_dir / "fig_s4.png"
    output_pdf = fig_dir / "fig_s4.pdf"
    plot_figure_2(df_corr, df_pred, output_png, ptype="continuous", panel_a_style="paired_scatter")
    plot_figure_2(df_corr, df_pred, output_pdf, ptype="continuous", panel_a_style="paired_scatter")


if __name__ == "__main__":
    main()
