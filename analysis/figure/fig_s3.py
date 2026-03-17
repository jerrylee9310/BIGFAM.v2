from pathlib import Path

from fig_s2 import load_data, plot_slope_ci_width_figure


def main():
    root_dir = Path(__file__).resolve().parents[2]
    fig_dir = Path(__file__).resolve().parent

    pred_file = root_dir / "data/real/pred_results.tsv"
    df_pred = load_data(pred_file)

    for ext in [".png", ".pdf"]:
        plot_slope_ci_width_figure(df_pred, fig_dir / f"fig_s3{ext}")


if __name__ == "__main__":
    main()
