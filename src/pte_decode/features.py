"""Module for extracting event-based features for decoding."""
import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class FeatureSelector:
    """Select features based on keywords."""

    feature_keywords: Sequence[str] | None
    verbose: bool = False

    def run(
        self,
        features: pd.DataFrame,
        out_path: Path | str | None = None,
    ) -> pd.DataFrame:
        feature_picks = self._pick_by_keywords(features)

        if out_path:
            out_path = Path(out_path).resolve()
            feature_picks.to_csv(
                out_path / f"{out_path.stem}_FeaturesSelected.csv.gz",
                index=False,
            )

        return feature_picks

    def _pick_by_keywords(self, features: pd.DataFrame) -> pd.DataFrame:
        """Process time points used."""
        keywords = self.feature_keywords
        if keywords is None:
            return features
        col_picks = []
        for column in features.columns:
            for keyword in keywords:
                if keyword in column:
                    col_picks.append(column)
                    break
        return features[col_picks]


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
            out_path = Path(out_path).resolve()
            features_concatenated.to_csv(
                out_path / f"{out_path.stem}_FeaturesConcatenated.gz",
                index=False,
            )
            with gzip.open(
                out_path / f"{out_path.stem}_FeaturesTimelocked.pickle.gz",
                "wb",
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


@dataclass
class FeatureEngineer:
    """Engineer features to use from given features."""

    use_times: int = 1
    normalization_mode: str | None = None
    verbose: bool = False

    def run(
        self,
        features: pd.DataFrame,
        out_path: Path | str | None = None,
    ) -> pd.DataFrame:
        feat_names = list(features.columns)
        features = self._process_use_times(features)

        if self.normalization_mode is not None:
            self._normalize(features, feat_names)

        if out_path:
            out_path = Path(out_path).resolve()
            features.to_csv(
                out_path / f"{out_path.stem}_FeaturesEngineered.csv.gz",
                index=False,
            )

        return features

    def _normalize(
        self,
        features: pd.DataFrame,
        feat_basenames: list[str],
    ) -> pd.DataFrame:
        feat_names = features.columns.str
        if self.normalization_mode == "by_earliest_sample":
            feature_norms = [
                f"{feat}_{(self.use_times - 1) * 100}ms"
                for feat in feat_basenames
            ]
        elif self.normalization_mode == "by_latest_sample":
            feature_norms = [f"{feat}_0ms" for feat in feat_basenames]
        else:
            raise ValueError(
                "`normalization_mode` must be one of either "
                "`by_earliest_sample` or `by_latest_sample`. Got: "
                f"{self.normalization_mode}."
            )
        for basename, feat_norm in zip(
            feat_basenames, feature_norms, strict=True
        ):
            picks = feat_names.startswith(basename)
            features.loc[:, picks] = features.loc[:, picks].subtract(
                features.loc[:, feat_norm].to_numpy(), axis="index"
            )
        features = features.drop(columns=feature_norms)
        return features

    def _process_use_times(self, features: pd.DataFrame) -> pd.DataFrame:
        """Process time points used."""
        if self.use_times <= 1:
            return features
        feat_processed = [
            features.rename(
                columns={col: f"{col}_0ms" for col in features.columns}
            )
        ]
        # Use additional features from previous time points
        # ``use_times = 1`` means no features from previous time points are used
        for use_time in np.arange(1, self.use_times):
            feat_new = features.shift(use_time, axis=0).rename(
                columns={
                    col: f"{col}_{(use_time) * 100}ms"
                    for col in features.columns
                }
            )
            feat_processed.append(feat_new)

        return pd.concat(feat_processed, axis=1).fillna(0.0)


@dataclass
class FeatureCleaner:
    """Clean features."""

    data_channel_names: list[str]
    data_channel_types: Sequence[str]
    label_channels: Sequence[str]
    plotting_target_channels: Sequence[str]
    side: str | None = None
    types_used: str | Sequence[str] = "all"
    hemispheres_used: str = "both"
    verbose: bool = False

    def run(
        self,
        features: pd.DataFrame,
        out_path: Path | str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        channel_picks = self.data_channel_names
        channel_picks = self._channels_by_types(channel_picks)
        channel_picks = self._channels_by_hemisphere(channel_picks)

        feature_picks = self._pick_by_channels(features, channel_picks)

        # Get label for classification
        label = pd.Series(
            _get_column_picks(
                column_picks=self.label_channels,
                features=features,
            ),
            name="label",
        )

        # Pick target for plotting predictions
        plotting_target = pd.Series(
            _get_column_picks(
                column_picks=self.plotting_target_channels,
                features=features,
            ),
            name="plotting_target",
        )

        if out_path:
            out_path = Path(out_path).resolve()

            feature_picks.to_csv(
                out_path / f"{out_path.stem}_FeaturesCleaned.csv.gz",
                index=False,
            )
            label.to_csv(
                out_path / f"{out_path.stem}_Label.csv.gz", index=False
            )
            label.to_csv(
                out_path / f"{out_path.stem}_PlottingTarget.csv.gz",
                index=False,
            )

        return feature_picks, label, plotting_target

    def _pick_by_channels(
        self, features: pd.DataFrame, channel_picks: list[str]
    ) -> pd.DataFrame:
        """Pick features by list of channels."""
        col_picks = []
        for column in features.columns:
            for ch_name in channel_picks:
                if ch_name in column:
                    col_picks.append(column)
                    break
        return features[col_picks]

    def _channels_by_types(self, ch_names: list[str]) -> list[str]:
        """Process time points used."""
        if self.types_used == "all":
            return ch_names

        if isinstance(self.types_used, str):
            self.types_used = [self.types_used]

        ch_picks = []
        for ch_name, ch_type in zip(
            self.data_channel_names, self.data_channel_types, strict=True
        ):
            if ch_type in self.types_used:
                ch_picks.append(ch_name)
        return ch_picks

    def _channels_by_hemisphere(self, ch_names: list[str]) -> list[str]:
        """Process time points used."""
        if self.hemispheres_used == "both":
            return ch_names
        if self.hemispheres_used not in ("contralat", "ipsilat"):
            raise ValueError(
                "`hemispheres_used` must be `both`, `ipsilat` or `contralat`."
                f" Got: {self.hemispheres_used}."
            )
        side = self.side
        if side is None or side not in ("right", "left"):
            raise ValueError(
                "If hemispheres_used is not `both`, `trial_side`"
                f" must be either `right` or `left`. Got: {side}."
            )
        ipsilat = {"right": "_R_", "left": "_L_"}
        contralat = {"right": "_L_", "left": "_R_"}
        hem = (
            ipsilat[side]
            if self.hemispheres_used == "ipsilat"
            else contralat[side]
        )

        ch_picks = [ch for ch in ch_names if hem in ch]
        return ch_picks


def _transform_side(side: Literal["right", "left"]) -> str:
    """Transform given keyword (eg 'right') to search string (eg 'R_')."""
    if side == "right":
        return "R_"
    if side == "left":
        return "L_"
    raise ValueError(
        f"Invalid argument for `side`. Must be right " f"or left. Got: {side}."
    )


def _init_channel_names(
    ch_names: list, use_channels: str, side: str | None = None
) -> list:
    """Initialize channels to be used."""
    case_all = ["single", "single_best", "all"]
    case_contralateral = [
        "single_contralat",
        "single_best_contralat",
        "all_contralat",
    ]
    case_ipsilateral = [
        "single_ipsilat",
        "single_best_ipsilat",
        "all_ipsilat",
    ]
    if use_channels in case_all:
        return ch_names
    # If side is none but ipsi- or contralateral channels are selected
    if side is None:
        raise ValueError(
            f"`use_channels`: {use_channels} defines a hemisphere, but "
            f"`side` is not specified. Please pass `right` or `left` "
            f"side or set use_channels to any of: {(*case_all,)}."
        )
    side = _transform_side(side)
    if use_channels in case_contralateral:
        return [ch for ch in ch_names if side not in ch]
    if use_channels in case_ipsilateral:
        return [ch for ch in ch_names if side in ch]
    raise ValueError(
        f"Invalid argument for `use_channels`. Must be one of "
        f"{case_all+case_contralateral+case_ipsilateral}. Got: "
        f"{use_channels}."
    )


def _get_column_picks(
    column_picks: Sequence[str],
    features: pd.DataFrame,
) -> pd.Series:
    """Return first found column pick from features DataFrame."""
    for pick in column_picks:
        for col in features.columns:
            if pick.lower() in col.lower():
                return pd.Series(data=features[col], name=col)
    raise ValueError(
        f"No valid column found. `column_picks` given: {column_picks}."
    )
