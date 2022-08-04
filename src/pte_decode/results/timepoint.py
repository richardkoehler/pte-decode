"""Module for calculating earliest significant time point of prediction."""
import numpy as np

import pte_stats


def get_earliest_timepoint(
    data: np.ndarray,
    times: np.ndarray,
    threshold: int | float | tuple[int | float, int | float] = 0,
    n_perm: int = 1000,
    alpha: float = 0.05,
    correction_method: str = "cluster",
    min_cluster_size: int = 1,
    resample_trials: int | None = None,
    verbose: bool = False,
) -> int | float | None:
    """Get earliest timepoint of motor onset prediction."""
    if verbose:
        print("Calculating earliest significant timepoint...")
    if correction_method not in ["cluster", "cluster_pvals", "fdr"]:
        raise ValueError(
            f"`correction_method` must be one of either `cluster`,"
            f" `cluster_pvals`, `fdr`. Got: {correction_method}."
        )

    if resample_trials is not None:
        orig_trials = data.shape[0]
        if orig_trials > resample_trials:
            data = downsample_trials(data=data, n_samples=resample_trials)
        elif orig_trials < resample_trials:
            data = upsample_trials(data=data, n_samples=resample_trials)

    # For compatibility of different methods
    data_t = data.T

    threshold_value, _ = transform_threshold(
        threshold=threshold, times=times, data=data_t
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
    else:
        raise ValueError(
            f"Invalid `correction_method`. Got:"
            f" {correction_method}. Options: ['cluster_pvals', 'fdr', 'cluster']."
        )

    if cluster_count == 0:
        if verbose:
            print("No significant clusters found.")
        return

    index = np.where(clusters != 0)[0][0]
    return times[index]


def downsample_trials(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Choose `n_samples` prediction trials."""
    n_orig = data.shape[0]
    if n_orig <= n_samples:
        return data
    new_data = np.empty((n_samples, data.shape[1]))
    random_choice = np.random.choice(
        n_orig, size=n_samples, replace=True, p=None
    )
    for i, ind in enumerate(random_choice):
        new_data[i] = data[ind]
    return new_data


def upsample_trials(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Upsample to `n_samples` prediction trials."""
    n_orig = data.shape[0]
    if n_orig >= n_samples:
        return data
    new_data = np.empty((n_samples, data.shape[1]))
    new_data[:n_orig] = data
    random_choice = np.random.choice(
        n_orig, size=(n_samples - n_orig), replace=True, p=None
    )
    for i, ind in enumerate(random_choice):
        new_data[n_orig + i] = data[ind]
    return new_data


def transform_threshold(
    threshold: int | float | tuple[int | float, int | float],
    data: np.ndarray,
    times: np.ndarray | None = None,
):
    """Take threshold input and return threshold value and array."""
    if isinstance(threshold, (int, float)):
        threshold_value = threshold
    else:
        base_start, base_end = pte_stats.handle_baseline_bytimes(
            baseline=threshold, times=times
        )
        threshold_value = np.mean(data[base_start:base_end])
    threshold_array = np.ones(data.shape[0]) * threshold_value
    return threshold_value, threshold_array
