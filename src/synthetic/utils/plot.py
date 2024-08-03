from pathlib import PosixPath

import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns


PELETTE = {
    "SIPS": "tab:red",
    "MSIPS": "tab:red",
    "IIPS": "tab:blue",
    "MIIPS": "tab:blue",
    "RIPS": "tab:purple",
    "MRIPS": "tab:purple",
    "AIPS": "tab:green",
}
LINESTYLE = {
    "SIPS": "",
    "MSIPS": (4, 2),
    "IIPS": "",
    "MIIPS": (4, 2),
    "RIPS": "",
    "MRIPS": (4, 2),
    "AIPS": "",
}


def visualize_mean_squared_error(result_df: DataFrame, xlabel: str, img_path: PosixPath, yscale: str) -> None:
    estimators = result_df["estimator"].unique()
    palettes = {estimator: PELETTE[estimator] for estimator in estimators}
    linestyles = {estimator: LINESTYLE[estimator] for estimator in estimators}

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    y = ["se", "bias", "variance"]
    title = ["mean squared error (MSE)", "Squared Bias", "Variance"]

    ylims = []

    for ax_, y_, title_ in zip(axes, y, title):
        sns.lineplot(
            data=result_df,
            x="x",
            y=y_,
            hue="estimator",
            style="estimator",
            marker="o",
            ci=None,
            markersize=20,
            ax=ax_,
            palette=palettes,
            dashes=linestyles,
        )

        # title
        ax_.set_title(title_, fontsize=25)
        # yaxis
        # xaxis
        ax_.set_xlabel(xlabel, fontsize=18)
        ax_.set_xticks(result_df["x"].unique())
        ax_.set_xticklabels(result_df["x"].unique(), fontsize=18)
        # log scale
        ax_.set_yscale(yscale)

        if y_ == "se":
            ylims = ax_.get_ylim()
            ylims = (0.0, ylims[1])

    if yscale == "linear":
        for ax_ in axes:
            ax_.set_ylim(ylims)

    plt.tight_layout()
    plt.show()
    plt.savefig(img_path)
    plt.close()
