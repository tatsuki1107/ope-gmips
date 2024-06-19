from pathlib import PosixPath

import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns


def visualize_mean_squared_error(result_df: DataFrame, xlabel: str, img_path: PosixPath) -> None:
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
            marker="o",
            ci=None,
            markersize=20,
            ax=ax_,
        )

        # title
        ax_.set_title(title_, fontsize=25)
        # yaxis
        # xaxis
        ax_.set_xlabel(xlabel, fontsize=18)
        ax_.set_xticks(result_df["x"].unique())
        ax_.set_xticklabels(result_df["x"].unique(), fontsize=18)

        if y_ == "se":
            ylims = ax_.get_ylim()
            ylims = (0.0, ylims[1])

    # 最初のプロットのY軸範囲を他のすべてのサブプロットに適用
    for ax_ in axes:
        ax_.set_ylim(ylims)

    plt.tight_layout()
    plt.show()
    plt.savefig(img_path)
    plt.close()
