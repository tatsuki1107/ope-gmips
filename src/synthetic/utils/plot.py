from pathlib import PosixPath

import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns


PELETTE = {
    # SIPS
    "SIPS": "tab:red",
    "snSIPS": "tab:red",
    "MSIPS (true)": "tab:red",
    "MSIPS": "tab:red",
    # IIPS
    "IIPS": "tab:blue",
    "snIIPS": "tab:blue",
    "MIIPS (true)": "tab:blue",
    "MIIPS": "tab:blue",
    # RIPS
    "RIPS": "tab:purple",
    "snRIPS": "tab:purple",
    "MRIPS (true)": "tab:purple",
    "MRIPS": "tab:purple",
    # AIPS
    "AIPS (true)": "tab:green",
    "snAIPS (true)": "tab:green",
}
LINESTYLE = {
    # SIPS
    "SIPS": "",
    "snSIPS": "",
    "MSIPS (true)": (7, 2),
    "MSIPS": (1, 1),
    # IIPS
    "IIPS": "",
    "snIIPS": "",
    "MIIPS (true)": (7, 2),
    "MIIPS": (1, 1),
    # RIPS
    "RIPS": "",
    "snRIPS": "",
    "MRIPS (true)": (7, 2),
    "MRIPS": (1, 1),
    # AIPS
    "AIPS (true)": "",
    "snAIPS (true)": "",
}
TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 20
LINEWIDTH = 5
MARKERSIZE = 20


def visualize_mean_squared_error(
    result_df: DataFrame, xlabel: str, xscale: str, yscale: str, is_only_mse: bool, img_path: PosixPath
) -> None:
    estimators = result_df["estimator"].unique()
    palettes = {estimator: PELETTE[estimator] for estimator in estimators}
    linestyles = {estimator: LINESTYLE[estimator] for estimator in estimators}

    xvalue = result_df["x"].unique()
    xvalue_labels = list(map(str, xvalue))

    plt.style.use("seaborn-v0_8")

    if is_only_mse:
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        sns.lineplot(
            data=result_df,
            x="x",
            y="se",
            hue="estimator",
            style="estimator",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            markers=True,
            palette=palettes,
            dashes=linestyles,
            legend=False,
            ax=ax,
        )
        # title
        ax.set_title(f"MSE ({yscale}-scale)", fontsize=TITLE_FONTSIZE)
        # y axixs
        ax.set_ylabel("")
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        # x axis
        ax.set_xscale(xscale)
        ax.set_xticks(xvalue, xvalue_labels, fontsize=LABEL_FONTSIZE)
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

    else:
        fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)

        title = ["MSE", "Squared Bias", "Variance"]
        y = ["se", "bias", "variance"]

        ylims = []
        for i, (ax_, title_, y_) in enumerate(zip(axes, title, y)):
            sns.lineplot(
                data=result_df,
                x="x",
                y=y_,
                hue="estimator",
                style="estimator",
                ci=95 if i == 0 else None,
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                ax=ax_,
                palette=palettes,
                dashes=linestyles,
                markers=True,
                legend="full" if i == 0 else False,
            )
            if i == 0:
                handles, labels = ax_.get_legend_handles_labels()
                ax_.legend_.remove()
                for handle in handles:
                    handle.set_linewidth(LINEWIDTH)
                    handle.set_markersize(MARKERSIZE)

            # title
            ax_.set_title(f"{title_} ({yscale}-scale)", fontsize=TITLE_FONTSIZE)
            # xaxis
            if i == 1:
                ax_.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
            else:
                ax_.set_xlabel("")

            ax_.set_xscale(xscale)
            ax_.set_xticks(xvalue, xvalue_labels, fontsize=LABEL_FONTSIZE)
            ax_.get_xaxis().set_minor_formatter(plt.NullFormatter())

            # yaxis
            ax_.set_yscale(yscale)
            ax_.set_ylabel("")

            if i == 0:
                ylims = ax_.get_ylim()
                ylims = (0.0, ylims[1])

        if yscale == "linear":
            for ax_ in axes:
                ax_.set_ylim(ylims)

        fig.legend(handles, labels, fontsize=LABEL_FONTSIZE, ncol=6, loc="center", bbox_to_anchor=(0.5, 1.5))

    plt.show()
    plt.savefig(img_path)
    plt.close()
