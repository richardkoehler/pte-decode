"""Module for plotting decoding results."""
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import axes, cm, collections, figure, patheffects
from matplotlib import pyplot as plt
import scipy.stats
from statannotations import Annotator
from statannotations.stats import StatTest

import pte_decode
import pte_stats


def violinplot_results(
    data: pd.DataFrame,
    x: str,
    y: str,
    outpath: str | Path | None = None,
    hue: str | None = None,
    order: Sequence | None = None,
    hue_order: Sequence | None = None,
    stat_test: str | Callable | None = "Permutation",
    alpha: float = 0.05,
    add_lines: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | Literal["auto"] = "auto",
    show: bool = True,
) -> figure.Figure:
    """Plot performance as violinplot."""
    if order is None:
        order = list(data[x].unique())

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsize == "auto":
        if hue:
            if hue_order is None:
                hue_order = list(data[hue].unique())
            hue_factor = len(hue_order)
        else:
            hue_factor = 1
        figsize = (1 + (1.1 * len(order) * hue_factor), 4.8)  # 0.9 for paper


    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    ax = sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        # color="white", # for violinplot
        # alpha=0.6, # for violinplot
        color="black", # for boxplot
        alpha=0.9, # for boxplot
        dodge=True,
        # s=8,
        ax=ax,
    )

    # ax = sns.violinplot(
    #     x=x,
    #     y=y,
    #     hue=hue,
    #     order=order,
    #     hue_order=hue_order,
    #     data=data,
    #     inner="box",
    #     width=0.9,
    #     alpha=1.0,
    #     cut=0.2,
    #     ax=ax,
    # )
    ax = sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        width=0.5,
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
    if title is not None:
        if stat_test:
            y_coord = 1.15
        else:
            y_coord = 1.02
        ax.set_title(title, y=y_coord)

    if add_lines:
        _add_lines(
            ax=ax, data=data, x=x, y=y, order=order, add_lines=add_lines
        )

    fig.tight_layout()
    if outpath is not None:
        fig.savefig(str(outpath), bbox_inches="tight")
    if show:
        plt.show(block=True)
    return fig


def _add_lines(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Iterable[str],
    add_lines: str,
) -> None:
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
    hue: str | None,
    hue_order: Iterable | None,
    stat_test: str | StatTest.StatTest,
    alpha: float,
    location: str = "inside",
) -> None:
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
        plot="violinplot",
    )
    annotator.configure(
        alpha=alpha,
        test=stat_test,
        text_format="simple",
        loc=location,
        color="grey",
        pvalue_format={
            "pvalue_thresholds": [
                [1e-6, "0.000001"],
                [1e-5, "0.00001"],
                [1e-4, "0.0001"],
                [1e-3, "0.001"],
                [1e-2, "0.01"],
                [5e-2, "0.05"],
            ],
            "fontsize": 13,
            "text_format": "simple",
            "pvalue_format_string": "{:.3f}",
            "show_test_name": False,
        },
    )
    annotator.apply_and_annotate()


def lineplot_prediction(
    x_1: np.ndarray,
    times: np.ndarray,
    data_labels: Sequence,
    x_2: np.ndarray | None = None,
    outpath: Path | str | None = None,
    x_label: str = "Time (s)",
    y_label: str | None = None,
    threshold: int | float | tuple[int | float, int | float] = (0.0, 1.0),
    alpha: float = 0.05,
    n_perm: int = 1000,
    correction_method: str = "cluster",
    two_tailed: bool = False,
    y_lims: Sequence | None = None,
    compare_x1x2: bool = False,
    paired_x1x2: bool = False,
    show_plot: bool = True,
) -> figure.Figure:
    """Plot averaged time-locked predictions including statistical tests."""
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    if x_2 is not None:
        if compare_x1x2:
            nrows = 3
        else:
            nrows = 2
    else:
        nrows = 1

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(5.6, 4.8),
        sharex=True,
        sharey=True,
    )
    if x_2 is None:
        axs = [axs]

    for i, data in enumerate((x_1, x_2)):
        if data is not None:
            lineplot_prediction_single(
                data=data,
                times=times,
                ax=axs[i],
                label=data_labels[i],
                threshold=threshold,
                color=colors[i],
                subplot_title=data_labels[i],
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
            axs[2].plot(
                times,
                data.mean(axis=1),
                color=colors[i],
                label=data_labels[i],
            )
            axs[2].fill_between(
                times,
                data.mean(axis=1) - scipy.stats.sem(data, axis=1),
                data.mean(axis=1) + scipy.stats.sem(data, axis=1),
                alpha=0.5,
                color=colors[i],
            )
        axs[2].set_title(f"{data_labels[0]} vs. {data_labels[1]}")

        _pval_correction_lineplot(
            ax=axs[2],
            x=x_1,
            y=x_2,
            times=times,
            alpha=alpha,
            n_perm=n_perm,
            correction_method=correction_method,
            two_tailed=two_tailed,
            min_cluster_size=2,
            onesample_xy=paired_x1x2,
        )

    axs[-1].set_xlabel(x_label)

    for ax in axs:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.05))
        if y_lims:
            ax.set_ylim(y_lims[0], y_lims[1])

        ax.set_ylabel(y_label)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
    if show_plot:
        plt.show(block=True)
    return fig


def lineplot_prediction_compare(
    x_1: np.ndarray,
    x_2: np.ndarray,
    times: np.ndarray,
    data_labels: Sequence,
    x_label: str = "Time (s)",
    y_label: str | None = None,
    alpha: float = 0.05,
    n_perm: int = 1000,
    correction_method: str = "cluster",
    y_lims: Sequence | None = None,
    two_tailed: bool = False,
    paired_x1x2: bool = False,
    outpath: Path | str | None = None,
) -> figure.Figure:
    """Plot comparison of continuous prediction arrays."""
    fig, ax = plt.subplots(1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, data in enumerate([x_1, x_2]):
        ax.plot(
            times,
            data.mean(axis=1),
            label=data_labels[i],
        )
        ax.fill_between(
            times,
            data.mean(axis=1) - scipy.stats.sem(data, axis=1),
            data.mean(axis=1) + scipy.stats.sem(data, axis=1),
            alpha=0.5,
        )
    _pval_correction_lineplot(
        ax=ax,
        x=x_1,
        y=x_2,
        times=times,
        alpha=alpha,
        n_perm=n_perm,
        correction_method=correction_method,
        two_tailed=two_tailed,
        min_cluster_size=2,
        onesample_xy=paired_x1x2,
    )
    ax.set_title(f"{data_labels[0]} vs. {data_labels[1]}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if y_lims:
        ax.set_ylim(y_lims[0], y_lims[1])
    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")
    return fig


def _permutation_wrapper(x, y, n_perm) -> tuple:
    """Wrapper for statannotations to convert pandas series to numpy array."""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    return pte_stats.permutation_twosample(x=x, y=y, n_perm=n_perm)


def _pval_correction_lineplot(
    ax: axes.Axes,
    x: np.ndarray,
    y: int | float | np.ndarray,
    times: np.ndarray,
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

    if cluster_count <= 0:
        print("No clusters found.")
        return
    if isinstance(y, (int, float)):
        y_arr = np.ones((x.shape[0], 1))
        y_arr[:, 0] = y
    else:
        y_arr = y
    if onesample_xy:
        x_arr = x
    else:
        x_arr = data_a
    label = f"p ≤ {alpha}"
    x_labels = times.round(2)
    for cluster_idx in range(1, cluster_count + 1):
        index = np.where(clusters == cluster_idx)[0]
        if index.size == 0:
            print("No clusters found.")
            continue
        lims = np.arange(index[0], index[-1] + 1)
        y1 = x_arr.mean(axis=1)[lims]
        y2 = y_arr.mean(axis=1)[lims]
        ax.fill_between(
            x=times[lims],
            y1=y1,
            y2=y2,
            alpha=0.5,
            color=viridis(7),
            label=label,
        )
        label = None  # Avoid printing label multiple times
        for i in [0, -1]:
            x_label = x_labels[lims[i]]
            ax.annotate(
                str(x_label),
                (x_label, y1[i]),
                xytext=(0, 10),
                textcoords="offset points",
                verticalalignment="center",
                horizontalalignment="center",
                fontsize="x-large",
                arrowprops=dict(
                    facecolor="black", arrowstyle="-", shrinkA=0.01
                ),
            )


def lineplot_prediction_single(
    data: np.ndarray,
    times: np.ndarray,
    ax: axes.Axes,
    label: str,
    threshold: Iterable | int | float,
    color: tuple,
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
        threshold=threshold,
        data=data,
        times=times,
    )
    ax.plot(
        times,
        data.mean(axis=1),
        color=color,
        label=label,
    )
    # lines = collections.LineCollection(
    #         [np.column_stack([times, single_data]) for single_data in data.T],
    #         # color=color,
    #         linewidth=1,
    #         alpha=0.3,
    #     )
    # ax.add_collection(lines)

    ax.fill_between(
        times,
        data.mean(axis=1) - scipy.stats.sem(data, axis=1),
        data.mean(axis=1) + scipy.stats.sem(data, axis=1),
        alpha=0.3,
        color=color,
        label=None,
    )
    _pval_correction_lineplot(
        ax=ax,
        x=data,
        y=threshold_value,
        times=times,
        alpha=alpha,
        n_perm=n_perm,
        correction_method=correction_method,
        two_tailed=two_tailed,
        min_cluster_size=2,
        onesample_xy=False,
    )

    ax.plot(
        times,
        threshold_arr,
        color="black",
        label="Threshold",
        alpha=1.0,
        linestyle="--",
    )

    ax.set_title(subplot_title)


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
            # fontsize="medium",
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
