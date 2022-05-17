"""Module for calculating earliest significant time point of prediction."""
from typing import Optional, Sequence, Union

import numpy as np

import pte_stats


def get_earliest_timepoint(
    data: np.ndarray,
    x_lims: tuple,
    sfreq: Optional[Union[int, float]] = None,
    threshold: Union[
        int, float, tuple[Union[int, float], Union[int, float]]
    ] = (0.0, 1.0),
    n_perm: int = 1000,
    alpha: float = 0.05,
    correction_method: str = "cluster",
    min_cluster_size: int = 1,
) -> Optional[Union[int, float]]:
    """Get earliest timepoint of motor onset prediction."""
    print("Calculating earliest significant timepoint...")
    if correction_method not in ["cluster", "cluster_pvals", "fdr"]:
        raise ValueError(
                f"`correction_method` must be one of either `cluster`,"
                f" `cluster_pvals`, `fdr`. Got: {correction_method}."
            )

    data_t = data.T

    threshold_value, _ = transform_threshold(
        threshold=threshold, sfreq=sfreq, data=data_t
    )
    if correction_method == "cluster":
        _, clusters_ind = pte_stats.cluster_analysis_1d(
            data_a=data,
            data_b=threshold_value,
            alpha=alpha,
            n_perm=n_perm,
            only_max_cluster=False,
            two_tailed=False,
            min_cluster_size=min_cluster_size,
        )
        if len(clusters_ind) == 0:
            return
        cluster_count = len(clusters_ind)
        clusters = np.zeros(data.shape[1], dtype=np.int32)
        for ind in clusters_ind:
            clusters[ind] = 1
    elif correction_method in ["cluster_pvals", "fdr"]:
        p_vals = pte_stats.timeseries_pvals(
            x=data_t, y=threshold_value, n_perm=n_perm, two_tailed=False
        )
        clusters, cluster_count = pte_stats.clusters_from_pvals(
            p_vals=p_vals,
            alpha=alpha,
            correction_method=correction_method,
            n_perm=n_perm,
            min_cluster_size=min_cluster_size,
        )

    if cluster_count == 0:
        print("No significant clusters found.")
        return

    x_labels = np.linspace(x_lims[0], x_lims[1], data.shape[1]).round(2)
    index = np.where(clusters != 0)[0][0]
    return x_labels[index]


def transform_threshold(
    threshold: Union[int, float, Sequence],
    data: np.ndarray,
    sfreq: Optional[Union[int, float]] = None,
):
    """Take threshold input and return threshold value and array."""
    if isinstance(threshold, (int, float)):
        threshold_value = threshold
    else:
        threshold_value = np.mean(
            data[int(threshold[0] * sfreq) : int(threshold[1] * sfreq)]
        )
    threshold_arr = np.ones(data.shape[0]) * threshold_value
    return threshold_value, threshold_arr
