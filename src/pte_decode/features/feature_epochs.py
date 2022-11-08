"""Module for extracting event-based features for decoding."""
import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class FeatureEpochs:
    """Class for extracting event-based features for decoding."""

    target_begin: int | float | str = "trial_onset"
    target_end: int | float | str = "trial_end"
    rest_begin: int | float = -5.0
    rest_end: int | float = -2.0
    offset_rest_begin: int | float = 2.0
    epoch_begin: int | float = -3.0
    epoch_end: int | float = 3.0
    verbose: bool = False

    def __post_init__(self) -> None:
        # Calculate events from label
        self.target_begin = _convert_target_begin(self.target_begin)
        self.target_end = _convert_target_end(self.target_end)

    def run(
        self,
        features: pd.DataFrame,
        label: pd.Series,
        plotting_target: pd.Series,
        sfreq: int,
        bad_epochs: np.ndarray | None = None,
        out_path: Path | str | None = None,
    ) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
        trial_onsets, trial_ends = _events_from_label(label.to_numpy())

        # Construct epoched array of features and labels using events
        (
            features_concatenated,
            trials_used,
            trials_discarded,
        ) = _get_features_concatenated(
            data=features,
            trial_onsets=trial_onsets,
            trial_ends=trial_ends,
            sfreq=sfreq,
            target_begin=self.target_begin,
            target_end=self.target_end,
            rest_begin=self.rest_begin,
            rest_end=self.rest_end,
            offset_rest_begin=self.offset_rest_begin,
            bad_epochs=bad_epochs,
        )
        if self.verbose:
            print(f"Number of trials used:      {len(trials_used)}")
            print(f"Number of trials discarded: {len(trials_discarded)}")

        features_timelocked = self._get_features_timelocked(
            features=features,
            label=label,
            plotting_target=plotting_target,
            sfreq=sfreq,
            trial_onsets=trial_onsets,
            trial_ids_used=trials_used,
        )
        if out_path:
            out_path = str(Path(out_path).resolve())
            features_concatenated.to_csv(
                out_path + r"/FeaturesConcatenated.csv.gz", index=False
            )
            with gzip.open(
                out_path + r"/FeaturesTimelocked.pickle.gz", "wb"
            ) as file:
                pickle.dump(features_timelocked, file)

        return (
            features_concatenated,
            features_timelocked,
            trials_used,
            trials_discarded,
        )

    def _get_features_timelocked(
        self,
        features: pd.DataFrame,
        label: pd.Series,
        plotting_target: pd.Series,
        sfreq: int,
        trial_onsets: np.ndarray,
        trial_ids_used: np.ndarray,
    ) -> dict:
        """Get timelocked features."""
        ind_begin = int(self.epoch_begin * sfreq)
        ind_end = int(self.epoch_end * sfreq)

        data_timelocked = {
            "trial_ids": [],
            "time": np.linspace(
                self.epoch_begin,
                self.epoch_end,
                ind_end - ind_begin + 1,
            ).tolist(),
            "label": [],
            "plotting_target": [],
            "features": {column: [] for column in features.columns},
        }
        for trial_id in trial_ids_used:
            features_epoch = _get_prediction_epochs(
                data=features.values,
                trial_onsets=trial_onsets,
                trial_id=trial_id,
                ind_begin=ind_begin,
                ind_end=ind_end,
                verbose=self.verbose,
            )
            if features_epoch.size > 0:
                data_timelocked["trial_ids"].append(trial_id)
                feat_timelocked = data_timelocked["features"]
                for col, data in zip(
                    features.columns, features_epoch.T, strict=True
                ):
                    feat_timelocked[col].append(data.tolist())

                for data, name in (
                    (label.to_numpy(), "label"),
                    (plotting_target.to_numpy(), "plotting_target"),
                ):
                    data_epoch = _get_prediction_epochs(
                        data=data,
                        trial_onsets=trial_onsets,
                        trial_id=trial_id,
                        ind_begin=ind_begin,
                        ind_end=ind_end,
                        verbose=self.verbose,
                    )
                    data_timelocked[name].append(data_epoch.tolist())

        return data_timelocked


def _get_prediction_epochs(
    data: np.ndarray,
    trial_onsets: np.ndarray,
    trial_id: int,
    ind_begin: int,
    ind_end: int,
    dtype: npt.DTypeLike = np.float32,
    verbose: bool = False,
) -> np.ndarray:
    """Get epochs of data for making predictions."""
    trial_onset = trial_onsets[trial_id]
    epoch = data[trial_onset + ind_begin : trial_onset + ind_end + 1]
    if len(epoch) == ind_end - ind_begin + 1:
        return epoch.squeeze().astype(dtype)
    if verbose:
        print(
            f"Mismatch of epoch samples. Got: {len(epoch)} samples."
            f" Expected: {ind_end - ind_begin + 1} samples."
            f" Discarding epoch: No. {trial_id + 1} of {len(trial_onsets)}."
        )
    return np.atleast_1d([])


def _convert_target_end(target_end) -> int | float | str:
    if isinstance(target_end, str):
        if target_end == "trial_onset":
            return "trial_onset"
        if target_end == "trial_onset":
            return 0.0
        raise ValueError(
            "If target_end is a string, it must be "
            f"`trial_onset` or `trial_end`. Got: {target_end}."
        )
    return target_end


def _convert_target_begin(target_begin) -> int | float:
    if isinstance(target_begin, str):
        if target_begin == "trial_onset":
            return 0.0
        raise ValueError(
            "If target_begin is a string, it must be "
            f"`trial_onset`. Got: {target_begin}."
        )
    return target_begin


def _discard_trial(
    baseline: int | float,
    index_epoch: int,
    bad_epochs: np.ndarray | None = None,
) -> bool:
    """Decide if trial should be discarded."""
    if baseline <= 0.0:
        return True
    if bad_epochs is not None and index_epoch in bad_epochs:
        return True
    return False


def _events_from_label(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create array of events from given label data."""
    label_diff = np.diff(data, prepend=[0])
    if data[-1] != 0:
        label_diff[-1] = -1
    trials = np.nonzero(label_diff)[0]
    # Check for plausability of events
    if len(trials) % 2:
        raise ValueError(
            "Number of events found is odd. Please check your label data."
        )
    trial_onsets = trials[::2]
    trial_ends = trials[1::2]
    return trial_onsets, trial_ends


def _get_baseline_period(
    trial_onset: int,
    trial_end_previous: int,
    trial_id: int,
    ind_rest_end: int,
    ind_offset_begin: int,
) -> int:
    """Return index where baseline period starts."""
    baseline_end: int = trial_onset + ind_rest_end

    if trial_id != 0:
        baseline_begin: int = trial_end_previous + ind_offset_begin
    else:
        baseline_begin: int = 0

    if baseline_end <= 0:
        return 0
    return baseline_end - baseline_begin


def _get_trial_data(
    data: np.ndarray,
    sfreq: int | float,
    trial_onset: int,
    trial_end: int,
    target_begin: int,
    target_end: int | str,
    rest_begin: int,
    rest_end: int,
    dtype: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ind_rest_begin: int = trial_onset + rest_begin
    ind_rest_end: int = trial_onset + rest_end

    ind_target_begin: int = trial_onset + target_begin

    ind_target_end: int = _handle_target_end(
        target_end, trial_onset=trial_onset, trial_end=trial_end
    )

    data_rest = data[ind_rest_begin:ind_rest_end].astype(dtype)
    data_target = data[ind_target_begin:ind_target_end].astype(dtype)
    times = (
        np.hstack(
            (
                np.arange(ind_rest_begin, ind_rest_end),
                np.arange(ind_target_begin, ind_target_end),
            )
        )
        / sfreq
    )
    times_relative = (
        np.hstack(
            (
                np.arange(rest_begin, rest_end),
                np.arange(len(data_target)) + target_begin,
            )
        )
        / sfreq
    )
    return data_rest, data_target, times, times_relative


def _handle_target_end(
    target_end: int | str, trial_onset: int, trial_end: int
) -> int:
    """Handle different cases of target_end"""
    if isinstance(target_end, int):
        return trial_onset + target_end
    if target_end == "trial_onset":
        return trial_onset
    if target_end == "trial_end":
        return trial_end
    raise ValueError(
        "`target_end` must be either an int, a float,"
        f" `trial_onset` or `trial_end`. Got: {target_end}."
    )


def _get_features_concatenated(
    data: pd.DataFrame,
    trial_onsets: np.ndarray,
    trial_ends: np.ndarray,
    sfreq: int | float,
    target_begin: int | float,
    target_end: int | float | str,
    rest_begin: int | float,
    rest_end: int | float,
    offset_rest_begin: int | float,
    bad_epochs: np.ndarray | None,
    dtype: Any = np.float32,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray,]:
    """Get data by trials."""
    data_arr = data.values

    offset_begin = int(offset_rest_begin * sfreq)
    rest_begin = int(rest_begin * sfreq)
    rest_end = int(rest_end * sfreq)
    target_begin = int(target_begin * sfreq)

    if isinstance(target_end, (int, float)):
        target_end = int(target_end * sfreq)

    (
        features,
        labels,
        trial_ids,
        times,
        times_relative,
        trial_ids_used,
        trial_ids_discarded,
    ) = ([], [], [], [], [], [], [])

    for trial_id, (trial_onset, trial_end) in enumerate(
        zip(trial_onsets, trial_ends, strict=True)
    ):
        baseline_period = _get_baseline_period(
            trial_onset=trial_onset,
            trial_end_previous=trial_ends[trial_id - 1],
            trial_id=trial_id,
            ind_rest_end=rest_end,
            ind_offset_begin=offset_begin,
        )
        discard_trial = _discard_trial(
            baseline=baseline_period,
            index_epoch=trial_id,
            bad_epochs=bad_epochs,
        )
        if discard_trial:
            trial_ids_discarded.append(trial_id)
        else:
            data_rest, data_target, time_abs, time_rel = _get_trial_data(
                data=data_arr,
                sfreq=sfreq,
                trial_onset=trial_onset,
                trial_end=trial_end,
                target_begin=target_begin,
                target_end=target_end,
                rest_begin=max(rest_end - baseline_period, rest_begin),
                rest_end=rest_end,
                dtype=dtype,
            )

            trial_ids_used.append(trial_id)
            features.extend((data_rest, data_target))
            labels.extend(
                (np.zeros(len(data_rest)), np.ones(len(data_target)))
            )
            trial_ids.append(
                np.full((len(data_rest) + len(data_target)), trial_id)
            )
            times.append(time_abs)
            times_relative.append(time_rel)

    data_final = pd.DataFrame(
        np.concatenate(features, axis=0),
        columns=data.columns,
    )
    data_final["time"] = np.hstack(times)
    data_final["time relative"] = np.hstack(times_relative)
    data_final["labels"] = np.hstack(labels).astype(int)
    data_final["trial_ids"] = np.hstack(trial_ids)

    return (
        data_final,
        np.array(trial_ids_used),
        np.array(trial_ids_discarded),
    )
