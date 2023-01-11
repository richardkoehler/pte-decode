"""Module for loading results from decoding experiments."""
import json
from pathlib import Path
import pickle

import mne_bids
import numpy as np
import pandas as pd

import pte
import pte_stats


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
    files: list[Path | str],
    baseline: tuple[int | float | None, int | float | None] | None = None,
    baseline_mode: str = "zscore",
    baseline_trialwise: bool = False,
    tmin: int | float | None = None,
    tmax: int | float | None = None,
    average_predictions: bool = False,
    concatenate_runs: bool = True,
    average_runs: bool = False,
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
    if concatenate_runs:
        data_all = _concatenate_runs(data=data_all)
    if average_runs:
        data_all["Predictions"] = (
            data_all["Predictions"]
            .apply(np.mean, axis=0)
            .apply(np.expand_dims, axis=0)
        )
    return data_all


def _concatenate_runs(data: pd.DataFrame) -> pd.DataFrame:
    """Concatenate predictions from different runs in a single patient."""
    data_list = []
    for subject in data["Subject"].unique():
        dat_sub = data[data["Subject"] == subject]
        dat_concat = np.vstack(dat_sub["Predictions"].to_numpy())
        data_list.append([subject, dat_concat])
    return pd.DataFrame(data_list, columns=["Subject", "Predictions"])


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
    # parent_dir = Path(fpath).parent.name
    parent_dir = Path(file).parent.name
    items = parent_dir.split("_")
    items_kept = []
    for item in items:
        if item in ("eeg", "ieeg"):
            break
        items_kept.append(item)
    filename = "_".join(items_kept)
    sub, med, stim = pte.filetools.sub_med_stim_from_fname(filename)
    with open(file, "rb") as file:
        pred_data = pickle.load(file)
    predictions = np.stack(pred_data["predictions"], axis=0)
    times = np.array(pred_data["times"])

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

    data_final = pd.DataFrame(
        data=(
            (
                sub,
                med,
                stim,
                pred_data["trial_ids"],
                times,
                predictions,
                filename,
            ),
        ),
        columns=[
            "Subject",
            "Medication",
            "Stimulation",
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
