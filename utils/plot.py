from pathlib import PosixPath
from typing import Optional

import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

from utils import LABEL_FONTSIZE
from utils import LINESTYLE
from utils import LINEWIDTH
from utils import MARKERSIZE
from utils import PELETTE
from utils import TICK_FONTSIZE
from utils import TITLE_FONTSIZE


def visualize_mean_squared_error(result_df: DataFrame, xlabel: str, xscale: str, img_path: PosixPath) -> None:
    """Visualize mean squared error, squared bias, and variance.

    Args:
        result_df: DataFrame
            simulation results.

        xlabel: str
            x axis label.

        xscale: str
            x axis scale. "linear" or "log".

        yscale: str
            y axis scale. "linear" or "log".

        is_only_mse: bool
            whether to visualize only mean squared error or not.

        img_path: PosixPath
            image path to save the plot
    """

    estimators = result_df["estimator"].unique()
    palettes = {estimator: PELETTE[estimator] for estimator in estimators}
    linestyles = {estimator: LINESTYLE[estimator] for estimator in estimators}

    xvalue = result_df["x"].unique()
    xvalue_labels = list(map(str, xvalue))

    for yscale in ["linear", "log"]:
        _visualize_only_mse(
            result_df=result_df,
            xlabel=xlabel,
            xscale=xscale,
            xvalue=xvalue,
            xvalue_labels=xvalue_labels,
            yscale=yscale,
            palettes=palettes,
            linestyles=linestyles,
            img_path=img_path / f"{yscale}_mse_only=True.png",
        )

        _visualize_mse_bias_variance(
            result_df=result_df,
            xlabel=xlabel,
            xscale=xscale,
            xvalue=xvalue,
            xvalue_labels=xvalue_labels,
            yscale=yscale,
            palettes=palettes,
            linestyles=linestyles,
            img_path=img_path / f"{yscale}_mse_only=False.png",
        )


def _visualize_only_mse(
    result_df: DataFrame,
    xlabel: str,
    xscale: str,
    xvalue: list,
    xvalue_labels: list,
    yscale: str,
    palettes: dict,
    linestyles: dict,
    img_path: PosixPath,
) -> None:
    plt.style.use("seaborn-v0_8")
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
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    # x axis
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(xvalue, xvalue_labels, fontsize=TICK_FONTSIZE)
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()


def _visualize_mse_bias_variance(
    result_df: DataFrame,
    xlabel: str,
    xscale: str,
    xvalue: list,
    xvalue_labels: list,
    yscale: str,
    palettes: dict,
    linestyles: dict,
    img_path: PosixPath,
) -> None:
    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

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
        ax_.set_xticks(xvalue, xvalue_labels, fontsize=TICK_FONTSIZE)
        ax_.get_xaxis().set_minor_formatter(plt.NullFormatter())

        # yaxis
        ax_.set_yscale(yscale)
        ax_.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax_.set_ylabel("")

        if i == 0:
            ylims = ax_.get_ylim()
            ylims = (0.0, ylims[1])

    if yscale == "linear":
        for ax_ in axes:
            ax_.set_ylim(ylims)

    fig.legend(
        handles,
        labels,
        fontsize=LABEL_FONTSIZE,
        ncol=len(palettes),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()


def visualize_train_curve_of_abstraction_model(
    train_loss: list, img_path: PosixPath, val_loss: Optional[list] = None
) -> None:
    """Visualize training curve of abstraction model

    Args:
        train_loss: list
            training loss of abstraction model.

        val_loss: list
            validation loss of abstraction model.

        img_path: PosixPath
            image path to save the plot
    """

    plt.style.use("seaborn-v0_8")
    plt.title("loss of abstraction model", fontsize=TITLE_FONTSIZE)
    plt.xlabel("epochs", fontsize=LABEL_FONTSIZE)
    plt.ylabel("loss", fontsize=LABEL_FONTSIZE)
    plt.plot(train_loss, label="train")
    if val_loss is not None:
        plt.plot(val_loss, label="val")
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig(img_path)
    plt.close()
