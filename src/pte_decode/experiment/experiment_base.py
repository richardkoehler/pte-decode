"""Module for running a decoding experiment."""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pte_decode
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import BaseCrossValidator, GroupKFold


@dataclass
class _Results:
    """Class for storing results of a single experiment."""

    use_channels: str
    save_importances: bool
    times_epochs: list[int] | list[float]
    ch_names: list[str] | None = None
    scores: list = field(init=False, default_factory=list)
    predictions_epochs: dict = field(init=False, default_factory=dict)
    predictions_concat: dict = field(init=False, default_factory=dict)
    feature_importances: list = field(init=False, default_factory=list)
    path: str = field(init=False)

    def __post_init__(self) -> None:
        self.predictions_concat = self._init_pred_concat()
        self.predictions_epochs = self._init_pred_epochs()
        if self.save_importances:
            self.feature_importances = []

    def _init_pred_concat(self) -> dict:
        """Initialize concatenated predictions."""
        return {
            "trial_ids": [],
            "labels": [],
            "predictions": [],
            "channel": [],
        }

    def _init_pred_epochs(
        self,
    ) -> dict:
        """Initialize dictionary of timelocked predictions."""
        epochs = {
            "times": self.times_epochs,
            "trial_ids": [],
        }
        if self.use_channels == "single":
            if self.ch_names is None:
                raise ValueError(
                    "If use_channels is 'single', ch_names must not be"
                    " provided."
                )
            epochs.update({ch: [] for ch in self.ch_names})
        else:
            epochs.update({"predictions": []})
        return epochs

    def _append_epoch_data(
        self, epoch_dict: dict, data: np.ndarray | list, ch_pick: str
    ) -> dict:
        """Append new results to existing results."""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        # Add prediction results to dictionary
        if self.use_channels == "single":
            epoch_dict[ch_pick].extend(data)
        else:
            epoch_dict["predictions"].extend(data)
        return epoch_dict

    def update_scores(
        self,
        fold: int,
        ch_pick: str,
        score: int | float,
        trial_ids_used: np.ndarray,
    ) -> None:
        """Update results."""
        if len(trial_ids_used) == 1:
            trial_ids_used = trial_ids_used[0]
        self.scores.append(
            [
                fold,
                ch_pick,
                score,
                trial_ids_used,
            ]
        )

    def update_predictions_epochs(
        self,
        data: list[list],
        ch_pick: str,
        trial_ids: np.ndarray,
    ) -> None:
        """Update prediction epochs."""
        if self.use_channels == "single":
            self.predictions_epochs[ch_pick].extend(data)
        else:
            self.predictions_epochs["predictions"].extend(data)
        self.predictions_epochs["trial_ids"].extend(trial_ids)

    def update_predictions_concat(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        trial_ids: np.ndarray,
        ch_pick: str,
    ) -> None:
        """Update predictions and features."""
        for item, value in (
            ("predictions", predictions),
            ("labels", labels),
            ("trial_ids", trial_ids),
            ("channel", [ch_pick] * len(predictions)),
        ):
            self.predictions_concat[item].extend(value)

    def update_feature_importances(
        self,
        fold: int,
        ch_pick: str,
        feature_names: list[str],
        feature_importances: Sequence,
    ) -> None:
        """Update feature importances."""
        self.feature_importances.extend(
            (
                [fold, ch_pick, name, importance]
                for name, importance in zip(
                    feature_names, feature_importances, strict=True
                )
            )
        )

    def set_path(self, path: Path | str, verbose: bool) -> str:
        """Set path attribute and make corresponding folder."""
        path = Path(path).resolve()
        out_dir = path.parent
        self.path = str(path)

        # Save results, check if directory exists
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)
        if verbose:
            print(f"Creating folder: \n{path}")
        return self.path

    def save_scores(
        self,
        scoring: str,
    ) -> None:
        """Save prediction scores."""
        scores_df = pd.DataFrame(
            self.scores,
            columns=[
                "fold",
                "channel",
                scoring,
                "trial_ids",
            ],
            index=None,
        )
        scores_df.to_csv(self.path + r"/Scores.csv", index=False)

    def save_predictions_concatenated(self) -> None:
        """Save concatenated predictions"""
        pd.DataFrame(self.predictions_concat).to_csv(
            self.path + r"/PredConcatenated.csv", index=False
        )

    def save_predictions_timelocked(self) -> None:
        """Save predictions time-locked to trial onset"""
        with open(self.path + r"/PredTimelocked.pickle", "wb") as file:
            pickle.dump(
                self.predictions_epochs, file, protocol=pickle.HIGHEST_PROTOCOL
            )

    def save_feature_importances(self) -> None:
        """Save feature importances"""
        if not self.feature_importances:
            print(
                "WARNING: No feature importances found. Skipping saving"
                " feature importances to file."
            )
        pd.DataFrame(
            self.feature_importances,
            columns=[
                "fold",
                "channel",
                "feature_name",
                "feature_importance",
            ],
            index=None,
        ).to_csv(self.path + r"/FeatImportances.csv", index=False)


@dataclass
class DecodingExperiment:
    """Class for running decoding experiments."""

    features: pd.DataFrame
    features_timelocked: dict[str, list[list[float]]]
    trial_ids_timelocked: Sequence[int]
    times_timelocked: list[int] | list[float]
    labels: pd.Series
    trial_ids: pd.Series
    decoder: pte_decode.Decoder
    ch_names: list[str] | None = None
    scoring: str = "balanced_accuracy"
    feature_importance: Any = False
    channels_used: str = "single"
    prediction_mode: str = "classify"
    cv_outer: BaseCrossValidator = GroupKFold(n_splits=5)
    cv_inner: BaseCrossValidator = GroupKFold(n_splits=5)
    verbose: bool = False
    data_epochs: np.ndarray = field(init=False)
    fold: int = field(init=False)
    results: _Results = field(init=False)

    def __post_init__(self) -> None:
        _assert_channels_used_is_valid(self.channels_used)
        self.results = self._init_results()
        self.data_epochs = self.features.values
        self.fold = 0

    def run(self) -> None:
        """Calculate classification performance and out results."""
        # Outer cross-validation
        split = tuple(self.cv_outer.split(
            self.features, self.labels, self.trial_ids
        ))
        if self.verbose:
            no_splits = len(tuple(split))
        for train_ind, test_ind in self.cv_outer.split(
            self.features, self.labels, self.trial_ids
        ):
            if self.verbose:
                print(f"Fold no.: {self.fold + 1}/{no_splits}")
            self._run_outer_cv(train_ind=train_ind, test_ind=test_ind)
            self.fold += 1

    def save(
        self,
        path: Path | str,
        scores: bool = True,
        predictions_concatenated: bool = True,
        predictions_timelocked: bool = True,
        feature_importances: bool = True,
        final_models: bool = True,
    ) -> None:
        """Save results to given path."""

        # self.features.loc[:, "trial_ids"] = self.trial_ids
        basename = self.results.set_path(path=path, verbose=self.verbose)
        if scores:
            self.results.save_scores(
                scoring=self.scoring,
            )
        if predictions_concatenated:
            self.results.save_predictions_concatenated()
        if predictions_timelocked:
            self.results.save_predictions_timelocked()
        if feature_importances:
            self.results.save_feature_importances()
        if final_models:
            self._save_final_model(basename)

    def _init_results(self) -> _Results:
        """Initialize results container."""

        save_importances = False if not self.feature_importance else True
        return _Results(
            ch_names=self.ch_names,
            use_channels=self.channels_used,
            save_importances=save_importances,
            times_epochs=self.times_timelocked,
        )

    def _run_outer_cv(
        self, train_ind: np.ndarray, test_ind: np.ndarray
    ) -> None:
        """Run single outer cross-validation fold."""
        # Get training and testing data and labels
        features_train, features_test = (
            pd.DataFrame(self.features.iloc[train_ind]),
            pd.DataFrame(self.features.iloc[test_ind]),
        )
        y_train = self.labels.iloc[train_ind]
        groups_train = self.trial_ids.iloc[train_ind]
        groups_test = self.trial_ids.iloc[test_ind]

        trial_ids_test = np.unique(groups_test)

        # Handle which channels are used
        ch_picks = self._get_picks_and_types(
            features=features_train,
            labels=y_train,
            groups=groups_train,
        )

        # Perform classification for each selected model
        for ch_pick in ch_picks:
            self._run_channel_pick(
                ch_pick=ch_pick,
                features_train=features_train,
                features_test=features_test,
                labels_train=y_train,
                labels_test=self.labels.iloc[test_ind],
                groups_train=groups_train,
                groups_test=groups_test,
                trial_ids_test=trial_ids_test,
            )

    def _save_final_model(self, basename: str) -> None:
        # Handle which channels are used
        ch_picks = self._get_picks_and_types(
            features=self.features,
            labels=self.labels,
            groups=self.trial_ids,
        )
        for ch_pick in ch_picks:
            cols = self._pick_features(ch_pick)
            if not cols:
                continue
            data_train = self.features[cols]
            self.decoder.fit(
                data_train=data_train,
                labels=self.labels,
                groups=self.trial_ids,
            )
            if ch_pick == "all":
                ch_pick = ch_pick.capitalize()
            filename = str(Path(basename).parent / f"FinalModel{ch_pick}")
            self.decoder.save_model(filename)

    def _run_channel_pick(
        self,
        ch_pick: str,
        features_train: pd.DataFrame,
        features_test: pd.DataFrame,
        labels_train: pd.Series,
        labels_test: pd.Series,
        groups_train: pd.Series,
        groups_test: pd.Series,
        trial_ids_test: np.ndarray,
    ) -> None:
        """Train model and save results for given channel picks"""
        cols = self._pick_features(ch_pick)
        if not cols:
            return
        # if self.verbose:
        #     print("No. of features used:", len(cols))

        data_train, data_test = features_train[cols], features_test[cols]

        self.decoder.fit(
            data_train=data_train,
            labels=labels_train,
            groups=groups_train,
        )
        predictions = self.decoder.predict(data=data_test)

        score = self.decoder.get_score(data_test, labels_test)

        prediction_epochs = self._get_prediction_epochs(
            columns=cols, trial_ids=trial_ids_test
        )

        self._update_results(
            ch_pick=ch_pick,
            trial_ids=trial_ids_test,
            score=score,
            predictions=predictions,
            labels=labels_test.to_numpy(),
            groups=groups_test.to_numpy(),
            prediction_epochs=prediction_epochs,
        )

        if self.feature_importance is not None:
            feature_importances = _get_importances(
                feature_importance=self.feature_importance,
                decoder=self.decoder,
                data=data_test,
                label=labels_test.to_numpy(),
                scoring=self.scoring,
            )
            self.results.update_feature_importances(
                fold=self.fold,
                ch_pick=ch_pick,
                feature_names=cols,
                feature_importances=feature_importances,
            )

    def _get_prediction_epochs(self, columns, trial_ids) -> list[list[float]] | None:
        """Get feature and prediction epochs."""
        feature_epochs = self._get_feature_epochs(
            features_used=columns,
            trial_ids=trial_ids,
        )
        if feature_epochs.size == 0:
            return None
        prediction_epochs = self._predict_epochs(
            data=feature_epochs,
            columns=columns,
        )
        return prediction_epochs

    def _get_feature_epochs(
        self,
        features_used: list[str],
        trial_ids: np.ndarray,
    ) -> np.ndarray:
        """Get epochs of data for making predictions."""
        feat_timelock = self.features_timelocked
        trials_timelock = self.trial_ids_timelocked
        epochs = []
        for trial in trial_ids:
            if trial in trials_timelock:
                ind = trials_timelock.index(trial)
                epoch = [feat_timelock[feat][ind] for feat in features_used]
                epochs.append(np.stack(epoch, axis=1))
        if not epochs:
            return np.atleast_1d([])
        epochs = np.stack(epochs, axis=0)
        return epochs

    def _pick_features(
        self,
        channel_pick: str,
    ) -> list:
        """Return column picks given channel picks from features DataFrame."""
        if channel_pick == "all":
            return self.features.columns.tolist()
        col_picks = []
        for column in self.features.columns:
            if channel_pick in column:
                if any(ch_name in column for ch_name in self.ch_names):
                    col_picks.append(column)
        return col_picks

    def _update_results(
        self,
        ch_pick: str,
        trial_ids: np.ndarray,
        score: int | float,
        predictions: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        prediction_epochs: list[list[float]] | None,
    ) -> None:
        """Update results."""
        self.results.update_scores(
            fold=self.fold,
            ch_pick=ch_pick,
            score=score,
            trial_ids_used=trial_ids,
        )

        self.results.update_predictions_concat(
            predictions=predictions,
            labels=labels,
            trial_ids=groups,
            ch_pick=ch_pick,
        )

        if prediction_epochs is not None:
            self.results.update_predictions_epochs(
                data=prediction_epochs,
                ch_pick=ch_pick,
                trial_ids=trial_ids,
            )

    def _get_picks_and_types(
        self, features: pd.DataFrame, labels: pd.Series, groups: pd.Series
    ) -> list[str]:
        """Return channel picks and types."""
        if "single_best" in self.channels_used:
            ch_names = self._run_inner_cv(
                features=features, labels=labels, groups=groups
            )
        elif "all" in self.channels_used:
            ch_names = ["all"]
        else:
            ch_names = self.ch_names
        return ch_names

    def _run_inner_cv(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        groups: pd.Series,
    ) -> list[str]:
        """Run inner cross-validation and return best ECOG and LFP channel."""
        results = {ch_name: [] for ch_name in self.ch_names}
        for train_ind, test_ind in self.cv_inner.split(
            features, labels, groups
        ):
            features_train, features_test = (
                features.iloc[train_ind, :],
                features.iloc[test_ind, :],
            )
            y_train, y_test = labels.iloc[train_ind], labels.iloc[test_ind]
            groups_train = groups.iloc[train_ind]
            for ch_name in self.ch_names:
                cols = [
                    col for col in features_train.columns if ch_name in col
                ]
                if not cols:
                    continue
                data_train = features_train.loc[:, cols]
                data_test = features_test.loc[:, cols]
                self.decoder.fit(
                    data_train=data_train,
                    labels=y_train,
                    groups=groups_train,
                )
                y_pred = self.decoder.predict(
                    data=data_test,
                )
                accuracy = balanced_accuracy_score(y_test, y_pred)
                results[ch_name].append(accuracy)
        results = {key: value for key, value in results.items() if value}
        results = {
            ch_name: np.mean(scores) for ch_name, scores in results.items()
        }
        best_channel: str = sorted(
            results.items(), key=lambda x: x[1], reverse=True
        )[0][0]
        return [best_channel]

    def _predict_epochs(
        self,
        data: np.ndarray,
        columns: list | None,
    ) -> list[list[float]]:
        """Make predictions for given feature epochs."""
        mode = self.prediction_mode
        predictions = []
        if data.ndim < 3:
            np.expand_dims(data, axis=0)
        for trial in data:
            trial = pd.DataFrame(trial, columns=columns)
            if mode == "prediction":
                pred = self.decoder.predict(trial).tolist()
            elif mode == "probability":
                pred = self.decoder.predict_proba(trial)[:, 1].tolist()
            elif mode == "decision_function":
                pred = self.decoder.decision_function(trial).tolist()
            else:
                raise ValueError(
                    f"Only `classification`, `probability` or "
                    f"`decision_function` are valid options for "
                    f"`mode`. Got {mode}."
                )
            predictions.append(pred)
        return predictions


def _get_importances(
    feature_importance: int | bool,
    decoder: pte_decode.Decoder,
    data: pd.DataFrame,
    label: np.ndarray,
    scoring: str,
) -> Sequence:
    """Calculate feature importances."""
    if not feature_importance:
        return []
    if feature_importance is True:
        return np.squeeze(decoder.model.coef_)
    if isinstance(feature_importance, int):
        imp_scores = permutation_importance(
            decoder.model,
            data,
            label,
            scoring=scoring,
            n_repeats=feature_importance,
            n_jobs=-1,
        ).importances_mean
        return imp_scores
    raise ValueError(
        f"`feature_importances` must be an integer or `False`. Got: "
        f"{feature_importance}."
    )


def _assert_channels_used_is_valid(channels_used: str) -> None:
    allowed_values = ("all", "single", "single_best")
    if channels_used not in allowed_values:
        raise ValueError(
            "Got invalid value for `channels_used`."
            f" Got: {channels_used}, allowed: {allowed_values}."
        )
