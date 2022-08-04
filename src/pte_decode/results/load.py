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
    files_or_dir: str | Path | list,
    scoring_key: str = "balanced_accuracy",
    average_runs: bool = False,
) -> pd.DataFrame:
    """Load results from *results.csv"""
    # Create Dataframes from Files
    files_or_dir = _handle_files_or_dir(
        files_or_dir=files_or_dir, extensions="results.csv"
    )
    results = []
    for file in files_or_dir:
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
    average_runs: bool = True,
) -> pd.DataFrame:
    """Load prediction scores from *Scores.csv"""
    if not isinstance(files, list):
        files = [files]
    results = []
    for file in files:
        sub, med, stim = pte.filetools.sub_med_stim_from_fname(file)
        data = (
            pd.read_csv(file)
            .drop(columns=["fold", "trial_ids"])
            .groupby(["channel"])
            .mean()
            .reset_index()
        )
        data["Filename"] = file
        data["Subject"] = sub
        data["Medication"] = med
        data["Stimulation"] = stim
        results.append(data)
    final = pd.concat(results)
    if average_runs:
        final = (
            final.groupby(["Subject", "Medication", "Stimulation", "channel"])
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


def _handle_files_or_dir(
    files_or_dir: str | Path | list,
    extensions: str | list | None = None,
) -> list:
    """Handle different cases of files_or_dir."""
    if isinstance(files_or_dir, list):
        return files_or_dir
    file_finder = pte.filetools.get_filefinder(datatype="any")
    file_finder.find_files(
        directory=files_or_dir,
        extensions=extensions,
        verbose=True,
    )
    return file_finder.files


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
    files_or_dir: str | Path | list,
    baseline: tuple[int | float | None, int | float | None] | None = None,
    baseline_mode: str = "zscore",
    baseline_trialwise: bool = False,
    average_predictions: bool = False,
    concatenate_runs: bool = True,
    average_runs: bool = False,
) -> pd.DataFrame:
    """Load time-locked predictions."""
    files_or_dir = _handle_files_or_dir(
        files_or_dir=files_or_dir, extensions="PredTimelocked.pickle"
    )

    df_list = []
    for fpath in files_or_dir:
        data_single = load_predictions_singlefile(
            fpath=fpath,
            baseline=baseline,
            baseline_mode=baseline_mode,
            baseline_trialwise=baseline_trialwise,
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


def _concatenate_runs(data: pd.DataFrame):
    """Concatenate predictions from different runs in a single patient."""
    data_list = []
    for subject in data["Subject"].unique():
        dat_sub = data[data["Subject"] == subject]
        dat_concat = np.vstack(dat_sub["Predictions"].to_numpy())
        data_list.append([subject, dat_concat])
    return pd.DataFrame(data_list, columns=["Subject", "Predictions"])


def load_predictions_singlefile(
    fpath: str | Path,
    baseline: tuple[int | float | None, int | float | None]
    | None = (None, None),
    baseline_mode: str = "zscore",
    baseline_trialwise: bool = False,
) -> pd.DataFrame:
    """Load time-locked predictions from single file."""
    sub, med, stim = pte.filetools.sub_med_stim_from_fname(fpath)
    with open(fpath, "rb") as file:
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

    data_final = pd.DataFrame(
        data=(
            (
                sub,
                med,
                stim,
                pred_data["trial_ids"],
                pred_data["times"],
                predictions,
            ),
        ),
        columns=[
            "Subject",
            "Medication",
            "Stimulation",
            "trial_ids",
            "times",
            "Predictions",
        ],
    )
    return data_final
