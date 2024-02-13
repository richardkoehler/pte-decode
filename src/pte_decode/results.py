"""Module for loading results from decoding experiments."""
from collections.abc import Sequence
import json
from pathlib import Path

import mne_bids
import numpy as np
from numba import njit
import pandas as pd

import pte
import pte_stats


@njit
def threshold_events(data: np.ndarray, threshold: float | int) -> np.ndarray:
    """Apply threshold to find start and end of events.

    Arguments
    ---------
    data : np.ndarray
        Input data to apply thresholding to.
    threshold : float | int
        Threshold value.

    Returns
    -------
    np.ndarray
        Event array.
    """

    onoff = np.where(data > threshold, 1, 0)
    onoff_diff = np.zeros_like(onoff)
    onoff_diff[1:] = np.diff(onoff)
    index_start = np.where(onoff_diff == 1)[0]
    index_stop = np.where(onoff_diff == -1)[0]
    arr_start = np.stack(
        (index_start, np.zeros_like(index_start), np.ones_like(index_start)),
        axis=1,
    )
    arr_stop = np.stack(
        (index_stop, np.zeros_like(index_stop), np.ones_like(index_stop) * -1),
        axis=1,
    )
    return np.vstack((arr_start, arr_stop))


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
) -> tuple[int | float | None, int]:
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
    trials_used = data.shape[0]
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
            return None, trials_used
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
            f"Invalid `correction_method`. Got: {correction_method}."
            " Options: 'cluster_pvals', 'fdr', 'cluster'."
        )

    if cluster_count == 0:
        if verbose:
            print("No significant clusters found.")
        return None, trials_used

    index = np.where(clusters != 0)[0][0]
    return times[index], trials_used


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


def load_results_singlechannel(
    files: list[Path | str],
    scoring_key: str = "balanced_accuracy",
    average_runs: bool = False,
) -> pd.DataFrame:
    """Load results from *results.csv"""
    results = []
    for file in files:
        subject = mne_bids.get_entities_from_fname(file, on_error="ignore")[
            "subject"
        ]
        data: pd.DataFrame = pd.read_csv(  # type: ignore
            file,
            index_col="channel_name",
            header=0,
            usecols=["channel_name", scoring_key],
        )
        for ch_name in data.index.unique():
            score = (
                data.loc[ch_name]
                .mean(numeric_only=True)
                .values[0]  # type: ignore
            )
            results.append([subject, ch_name, score])
    columns = [
        "Subject",
        "Channel",
        scoring_key,
    ]
    columns = _normalize_columns(columns)
    data_out = pd.DataFrame(results, columns=columns)
    if average_runs:
        data_out = data_out.set_index(["Subject", "Channel"]).sort_index()
        results_average = []
        for ind in data_out.index.unique():
            result_av = data_out.loc[ind].mean(numeric_only=True).values[0]
            results_average.append([*ind, result_av])
        data_out = pd.DataFrame(results_average, columns=columns)
    return data_out


def load_scores(
    files: str | Path | list,
    average_runs: bool = False,
) -> pd.DataFrame:
    """Load prediction scores from *Scores.csv"""
    if not isinstance(files, list):
        files = [files]
    results = []
    for file in files:
        parent_dir = Path(file).parent.name
        sub, med, stim = pte.filetools.sub_med_stim_from_fname(parent_dir)
        data = (
            pd.read_csv(file)
            .drop(columns=["fold", "trial_ids"])
            .groupby(["channel"])
            .mean()
            .reset_index()
        )
        data["Subject"] = sub
        data["Medication"] = med
        data["Stimulation"] = stim
        data["Filename"] = parent_dir
        results.append(data)
    final = pd.concat(results)
    if average_runs:
        final = (
            final.groupby(["Subject", "Medication", "Stimulation", "channel"])
            .drop(columns=["Filename"])
            .mean()
            .reset_index()
        )
    return final


def _normalize_columns(columns: list[str]) -> list[str]:
    """Normalize column names."""
    new_columns = [
        " ".join([substr.capitalize() for substr in col.split("_")])
        for col in columns
    ]
    return new_columns


def _load_labels_single(
    fpath: str | Path,
    baseline: tuple[int | float | None, int | float | None] | None,
    baseline_mode: str | None,
    base_start: int | None,
    base_end: int | None,
) -> pd.DataFrame:
    """Load time-locked predictions from single file."""
    sub, med, stim = pte.filetools.sub_med_stim_from_fname(fpath)

    with open(fpath, "r", encoding="utf-8") as file:
        data = json.load(file)

    label_name = data["TargetName"]
    label_arr = np.stack(data["Target"], axis=0)

    if baseline is not None:
        label_arr = pte_stats.baseline_correct(
            label_arr, baseline_mode, base_start, base_end
        )

    labels = pd.DataFrame(
        data=[
            [
                sub,
                med,
                stim,
                label_name,
                label_arr,
            ]
        ],
        columns=[
            "Subject",
            "Medication",
            "Stimulation",
            "Channel",
            "Data",
        ],
    )
    return labels


def load_predictions(
    files: Sequence[Path | str],
    baseline: tuple[int | float | None, int | float | None] | None = None,
    baseline_mode: str = "zscore",
    baseline_trialwise: bool = False,
    tmin: int | float | None = None,
    tmax: int | float | None = None,
    average_predictions: bool = False,
) -> pd.DataFrame:
    """Load time-locked predictions."""
    df_list = []
    for file in files:
        data_single = load_predictions_singlefile(
            file=file,
            baseline=baseline,
            baseline_mode=baseline_mode,
            baseline_trialwise=baseline_trialwise,
            tmin=tmin,
            tmax=tmax,
        )
        if average_predictions:
            data_single["Predictions"] = (
                data_single["Predictions"]
                .apply(np.mean, axis=0)
                .apply(np.expand_dims, axis=0)
            )
        df_list.append(data_single)
    data_all: pd.DataFrame = pd.concat(df_list)  # type: ignore
    return data_all


def load_predictions_singlefile(
    file: str | Path,
    baseline: tuple[int | float | None, int | float | None]
    | None = (None, None),
    baseline_mode: str = "zscore",
    baseline_trialwise: bool = False,
    tmin: int | float | None = None,
    tmax: int | float | None = None,
) -> pd.DataFrame:
    """Load time-locked predictions from single file."""
    filename = Path(file).name
    items = filename.split("_")
    items_kept = []
    for item in items:
        if item in ("eeg", "ieeg"):
            break
        items_kept.append(item)
    filename = "_".join(items_kept)
    sub, med, stim = pte.filetools.sub_med_stim_from_fname(filename)
    with open(file, "r", encoding="utf-8") as in_file:
        pred_data = json.load(in_file)

    times = np.array(pred_data.pop("times"))
    trial_ids = list(set(pred_data.pop("trial_ids")))
    data_all = []
    for channel, pred_single in pred_data.items():
        if not pred_single:
            continue
        if channel == "predictions":
            channel = "all"
        predictions = np.stack(pred_single, axis=0)

        base_start, base_end = pte_stats.handle_baseline_bytimes(
            baseline=baseline, times=times
        )
        if baseline:
            predictions = pte_stats.baseline_correct(
                data=predictions,
                baseline_mode=baseline_mode,
                base_start=base_start,
                base_end=base_end,
                baseline_trialwise=baseline_trialwise,
            )

        if any((tmin is not None, tmax is not None)):
            predictions, times = _crop_predictions(
                preds=predictions, times=times, tmin=tmin, tmax=tmax
            )
        data_all.append(
            (
                sub,
                med,
                stim,
                channel,
                trial_ids,
                times,
                predictions,
                filename,
            ),
        )
    data_final = pd.DataFrame(
        data=data_all,
        columns=[
            "Subject",
            "Medication",
            "Stimulation",
            "Channel",
            "trial_ids",
            "times",
            "Predictions",
            "filename",
        ],
    )
    return data_final


def _crop_predictions(
    preds: np.ndarray,
    times: np.ndarray,
    tmin: int | float | None = None,
    tmax: int | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if tmin is not None:
        idx_tmin = tmin <= times
        times = times[idx_tmin]
        preds = preds[..., idx_tmin]
    if tmax is not None:
        idx_tmax = times <= tmax
        times = times[idx_tmax]
        preds = preds[..., idx_tmax]
    return preds, times
