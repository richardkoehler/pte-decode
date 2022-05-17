"""Module for plotting decoding results."""
import os
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, cm, collections, figure, patheffects
from matplotlib import pyplot as plt
import scipy.stats
from statannotations import Annotator
from statannotations.stats import StatTest

import pte_decode
import pte_stats


def violinplot_results(
    data: pd.DataFrame,
    outpath: Union[str, Path],
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Union[Sequence, np.ndarray]] = None,
    hue_order: Optional[Union[Sequence, np.ndarray]] = None,
    stat_test: Optional[Union[Callable, str]] = "Permutation",
    alpha: float = 0.05,
    add_lines: Optional[str] = None,
    title: Optional[str] = "Classification Performance",
    figsize: Union[tuple, str] = "auto",
) -> None:
    """Plot performance as violinplot."""
    if order is None:
        order = data[x].unique()

    if hue and hue_order is None:
        hue_order = data[hue].unique()

    if figsize == "auto":
        hue_factor = 1 if not hue else len(hue_order)
        figsize = (1.1 * len(order) * hue_factor, 4)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax = sns.violinplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        inner="box",
        width=0.9,
        alpha=0.8,
        ax=ax,
    )

    ax = sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        color="white",
        alpha=0.5,
        dodge=True,
        s=6,
        ax=ax,
    )

    if stat_test:
        _add_stats(
            ax=ax,
            data=data,
            x=x,
            y=y,
            order=order,
            hue=hue,
            hue_order=hue_order,
            stat_test=stat_test,
            alpha=alpha,
            location="outside",
        )

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [
            label.replace(" ", "\n") for label in labels[: len(labels) // 2]
        ]
        _ = plt.legend(
            handles[: len(handles) // 2],
            new_labels,
            bbox_to_anchor=(1.02, 1),
            loc=2,
            borderaxespad=0.0,
            title=hue,
            labelspacing=0.7,
        )

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_xlabels = [xtick.replace(" ", "\n") for xtick in xlabels]
    ax.set_xticklabels(new_xlabels)
    ax.set_title(title, fontsize="medium", y=1.02)

    if add_lines:
        _add_lines(
            ax=ax, data=data, x=x, y=y, order=order, add_lines=add_lines
        )

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=450)
    plt.show(block=True)


def boxplot_results(
    data: pd.DataFrame,
    outpath: Union[str, Path],
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Iterable] = None,
    hue_order: Optional[Iterable] = None,
    stat_test: Optional[Union[Callable, str]] = "Permutation",
    alpha: Optional[float] = 0.05,
    add_lines: Optional[str] = None,
    add_median_labels: bool = False,
    title: Optional[str] = "Classification Performance",
    figsize: Union[tuple, str] = "auto",
) -> None:
    """Plot performance as combined boxplot and stripplot."""
    # data = data[["Channels", "Balanced Accuracy", "Subject"]]

    color = "black"
    alpha_box = 0.5

    if not order:
        order = data[x].unique()

    if hue and not hue_order:
        hue_order = data[hue].unique()

    if figsize == "auto":
        hue_factor = 1 if not hue else len(hue_order)
        figsize = (1.1 * len(order) * hue_factor, 4)

    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        boxprops=dict(alpha=alpha_box),
        showcaps=True,
        showbox=True,
        showfliers=False,
        notch=False,
        width=0.9,
        whiskerprops={
            "linewidth": 2,
            "zorder": 10,
            "alpha": alpha_box,
            "color": color,
        },
        capprops={"alpha": alpha_box, "color": color},
        medianprops=dict(
            linestyle="-", linewidth=5, color=color, alpha=alpha_box
        ),
    )

    if add_median_labels:
        _add_median_labels(ax)

    sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        dodge=True,
        s=6,
        ax=ax,
    )

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [
            label.replace(" ", "\n") for label in labels[: len(labels) // 2]
        ]
        _ = plt.legend(
            handles[: len(handles) // 2],
            new_labels,
            bbox_to_anchor=(1.02, 1),
            loc=2,
            borderaxespad=0.0,
            title=hue,
            labelspacing=0.7,
        )

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_xlabels = [xtick.replace(" ", "\n") for xtick in xlabels]
    ax.set_xticklabels(new_xlabels)
    ax.set_title(title, fontsize="medium", y=1.02)

    if add_lines:
        _add_lines(
            ax=ax, data=data, x=x, y=y, order=order, add_lines=add_lines
        )

    if stat_test:
        _add_stats(
            ax,
            data,
            x,
            y,
            order,
            hue,
            hue_order,
            stat_test=stat_test,
            alpha=alpha,
        )
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=450)
    plt.show(block=True)


def _add_lines(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: list[str],
    add_lines: str,
):
    """Add lines connecting single dots"""
    data = data.sort_values(  # type: ignore
        by=x, key=lambda k: k.map({item: i for i, item in enumerate(order)})
    )
    lines = (
        [[i, n] for i, n in enumerate(group)]
        for _, group in data.groupby([add_lines], sort=False)[y]
    )
    ax.add_collection(
        collections.LineCollection(lines, colors="grey", linewidths=1)
    )


def _add_stats(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Iterable,
    hue: Optional[str],
    hue_order: Optional[Iterable],
    stat_test: Union[str, StatTest.StatTest],
    alpha: float,
    location: str = "inside",
):
    """Perform statistical test and annotate graph."""
    if not hue:
        pairs = list(combinations(order, 2))
    else:
        pairs = [
            list(combinations(list(product([item], hue_order)), 2))
            for item in order
        ]
        pairs = [item for sublist in pairs for item in sublist]

    if stat_test == "Permutation":
        stat_test = StatTest.StatTest(
            func=_permutation_wrapper,
            n_perm=10000,
            alpha=alpha,
            test_long_name="Permutation Test",
            test_short_name="Perm.",
            stat_name="Effect Size",
        )
    annotator = Annotator.Annotator(
        ax=ax,
        pairs=pairs,
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        order=order,
    )
    annotator.configure(
        alpha=alpha,
        test=stat_test,
        text_format="simple",
        show_test_name=False,
        loc=location,
        color="grey",
    )
    annotator.apply_and_annotate()


def lineplot_prediction(
    x_1: np.ndarray,
    subplot_titles: Sequence,
    sfreq: Union[int, float],
    x_lims: tuple,
    x_2: Optional[np.ndarray] = None,
    outpath: Optional[Union[Path, str]] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    x_label: str = "Time (s)",
    y_label: Optional[str] = None,
    threshold: Union[
        int, float, tuple[Union[int, float], Union[int, float]]
    ] = (0.0, 1.0),
    alpha: float = 0.05,
    n_perm: int = 1000,
    correction_method: str = "cluster",
    two_tailed: bool = False,
    y_lims: Optional[Sequence] = None,
    compare_x1x2: bool = False,
    paired_x1x2: bool = False,
    show_plot: bool = True,
) -> figure.Figure:
    """Plot averaged time-locked predictions including statistical tests."""
    viridis = cm.get_cmap("viridis", 8)
    colors = viridis(4), viridis(2)

    if x_2 is not None:
        if compare_x1x2:
            nrows = 3
        else:
            nrows = 2
    else:
        nrows = 1

    fig, axs = plt.subplots(
        ncols=1, nrows=nrows, figsize=(6, 2.0 * nrows), sharey=True
    )
    if x_2 is None:
        # for compatibility
        axs = [axs]

    n_samples = x_1.shape[0]

    for i, data in enumerate((x_1, x_2)):
        if data is not None:
            lineplot_prediction_single(
                data=data,
                ax=axs[i],
                threshold=threshold,
                sfreq=sfreq,
                x_lims=x_lims,
                color=colors[i],
                subplot_title=subplot_titles[i],
                label=label,
                alpha=alpha,
                n_perm=n_perm,
                correction_method=correction_method,
                two_tailed=two_tailed,
            )

    if compare_x1x2:
        if x_2 is None:
            raise ValueError(
                "If `compare_x1x2` is `True`, data for `x2` must be provided."
            )
        for i, data in enumerate([x_1, x_2]):
            single_label = (
                " - ".join((subplot_titles[i], label))
                if label
                else subplot_titles[i]
            )
            axs[2].plot(
                data.mean(axis=1),
                color=colors[i],
                label=single_label,
            )
            axs[2].fill_between(
                np.arange(data.shape[0]),
                data.mean(axis=1) - scipy.stats.sem(data, axis=1),
                data.mean(axis=1) + scipy.stats.sem(data, axis=1),
                alpha=0.5,
                color=colors[i],
            )
        axs[2].set_title(
            f"{subplot_titles[0]} vs. {subplot_titles[1]}", fontsize="medium"
        )
        axs[2].set_xlabel("Time [s]")

        _pval_correction_lineplot(
            ax=axs[2],
            x=x_1,
            y=x_2,
            x_lims=x_lims,
            alpha=alpha,
            n_perm=n_perm,
            correction_method=correction_method,
            two_tailed=two_tailed,
            min_cluster_size=2,
            onesample_xy=paired_x1x2,
        )
    xticks = np.arange(0, n_samples, sfreq)
    xticklabels = np.linspace(x_lims[0], x_lims[1], len(xticks)).round(1)
    for ax in axs:
        ax.legend(loc="upper left", fontsize="small")
        if y_lims:
            ax.set_ylim(y_lims[0], y_lims[1])

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

    fig.suptitle(title, fontsize="large", y=1.01)
    fig.tight_layout()
    if outpath:
        fig.savefig(os.path.normpath(outpath), bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show(block=True)
    return fig


def _permutation_wrapper(x, y, n_perm) -> tuple:
    """Wrapper for statannotations to convert pandas series to numpy array."""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    return pte_stats.permutation_twosample(data_a=x, data_b=y, n_perm=n_perm)


def _pval_correction_lineplot(
    ax: axes.Axes,
    x: np.ndarray,
    y: Union[int, float, np.ndarray],
    x_lims: tuple,
    alpha: float,
    correction_method: str,
    n_perm: int,
    two_tailed: bool,
    onesample_xy: bool,
    min_cluster_size: int = 2,
) -> None:
    """Perform p-value correction for singe lineplot."""
    viridis = cm.get_cmap("viridis", 8)

    if onesample_xy:
        data_a = x - y
        data_b = 0.0
    else:
        data_a = x
        data_b = y

    if correction_method == "cluster":
        _, clusters_ind = pte_stats.cluster_analysis_1d(
            data_a=data_a.T,
            data_b=data_b,
            alpha=alpha,
            n_perm=n_perm,
            only_max_cluster=False,
            two_tailed=two_tailed,
            min_cluster_size=min_cluster_size,
        )
        if len(clusters_ind) == 0:
            return
        cluster_count = len(clusters_ind)
        clusters = np.zeros(data_a.shape[0], dtype=np.int32)
        for ind in clusters_ind:
            clusters[ind] = 1
    elif correction_method in ["cluster_pvals", "fdr"]:
        p_vals = pte_stats.timeseries_pvals(
            x=data_a, y=data_b, n_perm=n_perm, two_tailed=two_tailed
        )
        # plt.plot(p_vals)
        # plt.show(block=True)
        clusters, cluster_count = pte_stats.clusters_from_pvals(
            p_vals=p_vals,
            alpha=alpha,
            correction_method=correction_method,
            n_perm=n_perm,
            min_cluster_size=min_cluster_size,
        )
    else:
        raise ValueError(
            f"Unknown cluster correction method: {correction_method}."
        )

    if cluster_count > 0:
        if isinstance(y, (int, float)) and not onesample_xy:
            y_arr = np.ones((x.shape[0], 1))
            y_arr[:, 0] = y
        else:
            y_arr = y
        if onesample_xy:
            x_arr = x
        else:
            x_arr = data_a
        label = f"p-value ≤ {alpha}"
        x_labels = np.linspace(x_lims[0], x_lims[1], x_arr.shape[0]).round(2)
        for cluster_idx in range(1, cluster_count + 1):
            index = np.where(clusters == cluster_idx)[0]
            lims = np.arange(index[0], index[-1] + 1)
            y_lims = y_arr.mean(axis=1)[lims]
            if y_lims.size > 0:
                ax.fill_between(
                    x=lims,
                    y1=x_arr.mean(axis=1)[lims],
                    y2=y_lims,
                    alpha=0.5,
                    color=viridis(7),
                    label=label,
                )
                label = None  # Avoid printing label multiple times
                for i in [0, -1]:
                    ax.annotate(
                        str(x_labels[lims[i]]) + "s",
                        (lims[i], y_lims[i]),
                        xytext=(0.0, 15),
                        textcoords="offset points",
                        verticalalignment="center",
                        horizontalalignment="center",
                        arrowprops=dict(facecolor="black", arrowstyle="-"),
                    )


def lineplot_prediction_single(
    data: np.ndarray,
    ax: axes.Axes,
    threshold: Union[Iterable, int, float],
    sfreq: Union[int, float],
    x_lims: tuple,
    color: tuple,
    label: Optional[str],
    subplot_title: str,
    alpha: float,
    n_perm: int,
    correction_method: str,
    two_tailed: bool,
) -> None:
    """Plot prediction line for single model."""
    (
        threshold_value,
        threshold_arr,
    ) = pte_decode.results.timepoint.transform_threshold(
        threshold=threshold, sfreq=sfreq, data=data
    )

    # x = np.arange(data.shape[0])
    # lines = collections.LineCollection(
    #     [np.column_stack([x, dat_]) for dat_ in data.T],
    #     color=color,
    #     linewidth=1,
    #     alpha=0.3,
    # )
    # ax.add_collection(lines)

    ax.plot(data.mean(axis=1), color=color, label=label)
    ax.fill_between(
        np.arange(data.shape[0]),
        data.mean(axis=1) - scipy.stats.sem(data, axis=1),
        data.mean(axis=1) + scipy.stats.sem(data, axis=1),
        alpha=0.5,
        color=color,
        label=None,
    )

    ax.plot(
        threshold_arr,
        color="r",
        label="Threshold",
        alpha=0.5,
        linestyle="dashed",
    )

    _pval_correction_lineplot(
        ax=ax,
        x=data,
        y=threshold_value,
        x_lims=x_lims,
        alpha=alpha,
        n_perm=n_perm,
        correction_method=correction_method,
        two_tailed=two_tailed,
        min_cluster_size=2,
        onesample_xy=False,
    )

    ax.set_title(subplot_title, fontsize="medium")


def _add_median_labels(ax: axes.Axes, add_borders: bool = False) -> None:
    """Add median labels to boxplot."""
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[4 : len(lines) : lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = (
            x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        )
        text = ax.text(
            x=x,
            y=y,
            s=f"{value:.3f}",
            ha="center",
            va="center",
            fontweight="light",
            color="white",
            fontsize="medium",
        )
        if add_borders:
            # create median-colored border around white text for contrast
            text.set_path_effects(
                [
                    patheffects.Stroke(
                        linewidth=0.0, foreground=median.get_color()
                    ),
                    patheffects.Normal(),
                ]
            )
