"""Define abstract base classes to construct Model classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import typing
from typing import Any, Callable, Literal, Tuple
import pickle
from typing import Optional, Union

from bayes_opt import BayesianOptimization
import catboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import sklearn.covariance
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import xgboost as xgb


BALANCING_METHODS = Literal[
    None,
    "oversample",
    "undersample",
    "balance_weights",
    "smote",
    "borderline_smote",
    "adasyn",
]
VALID_BALANCING_METHODS: Tuple[BALANCING_METHODS, ...] = typing.get_args(
    BALANCING_METHODS
)


@dataclass
class Decoder(ABC):
    """Basic representation of class of machine learning decoders."""

    scoring: Callable
    balancing: BALANCING_METHODS = None
    optimize: bool = False
    model: Any = field(init=False)
    data_train: pd.DataFrame = field(init=False)
    labels_train: pd.Series = field(init=False)
    groups_train: pd.Series = field(init=False)

    @abstractmethod
    def fit(
        self,
        data_train: pd.DataFrame,
        labels: pd.Series,
        groups: pd.Series,
    ) -> None:
        """Fit model to given training data and training labels."""

    @abstractmethod
    def save_model(self, filename: Path | str) -> None:
        """Save model to file"""

    def get_score(
        self,
        data_test: np.ndarray | pd.DataFrame,
        label_test: np.ndarray | pd.Series,
    ):
        """Calculate score."""
        return self.scoring(self.model, data_test, label_test)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict."""
        return self.model.predict(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict probability."""
        return self.model.predict_proba(data)

    def decision_function(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate decision function."""
        return self.model.decision_function(data)

    @staticmethod
    def _get_validation_split(
        data: pd.DataFrame,
        labels: pd.Series,
        groups: pd.Series,
        train_size: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.Series, list]:
        """Split data into single training and validation set."""
        val_split = GroupShuffleSplit(n_splits=1, train_size=train_size)
        train_ind, val_ind = next(val_split.split(data, labels, groups))
        data_train, data_val = (
            data.iloc[train_ind, :],
            data.iloc[val_ind, :],
        )
        labels_train, labels_val = (
            labels[train_ind, :],
            labels[val_ind, :],
        )
        eval_set = [(data_val, labels_val)]
        return data_train, labels_train, eval_set

    def balance_samples(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> tuple:
        """Balance class sizes to create equal class distributions.

        Parameters
        ----------
        data : numpy.ndarray of shape (n_features, n_samples)
            Data or features.
        labels : numpy.ndarray of shape (n_samples, )
            Array of class disribution
        method : {'oversample', 'undersample', 'weight'}
            Method to be used for rebalancing classes. 'oversample' will
            upsample the class with less samples. 'undersample' will
            downsample the class with more samples. 'weight' will generate
            balanced class weights. Default: 'oversample'

        Returns
        -------
        data : numpy.ndarray
            Rebalanced feature array of shape (n_features, n_samples)
        labels : numpy.ndarray
            Corresponding class distributions. Class sizes are now evenly
            balanced.
        sample_weight: numpy.ndarray of shape (n_samples, ) | None
            Sample weights if method = 'weight' else None
        """
        sample_weight = None
        if self.balancing is not None and np.mean(labels) != 0.5:
            if self.balancing == "oversample":
                resampler = RandomOverSampler(sampling_strategy="auto")
                data, labels = resampler.fit_resample(data, labels)
            elif self.balancing == "undersample":
                resampler = RandomUnderSampler(sampling_strategy="auto")
                data, labels = resampler.fit_resample(data, labels)
            elif self.balancing == "balance_weights":
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=labels
                )
            elif self.balancing == "smote":
                resampler = SMOTE(sampling_strategy="auto", k_neighbors=5)
                data, labels = resampler.fit_resample(data, labels)
            elif self.balancing == "borderline_smote":
                resampler = BorderlineSMOTE(
                    sampling_strategy="auto",
                    k_neighbors=5,
                    kind="borderline-1",
                )
                data, labels = resampler.fit_resample(data, labels)
            elif self.balancing == "adasyn":
                try:
                    resampler = ADASYN(sampling_strategy="auto", n_neighbors=5)
                    data, labels = resampler.fit_resample(data, labels)
                except ValueError as error:
                    if len(error.args) > 0 and error.args[0] == (
                        "No samples will be generated with the provided "
                        "ratio settings."
                    ):
                        pass
                    else:
                        raise error
            else:
                raise BalancingMethodNotFoundError(
                    self.balancing, VALID_BALANCING_METHODS
                )
        return data, labels, sample_weight


class BalancingMethodNotFoundError(Exception):
    """Exception raised when invalid balancing method is passed.

    Attributes:
        input_value -- input value which caused the error
        allowed -- allowed input values
        message -- explanation of the error
    """

    def __init__(
        self,
        input_value,
        allowed,
        message="Input balancing method is not an allowed value.",
    ) -> None:
        self.input_value = input_value
        self.allowed = allowed
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return (
            f"{{self.message}} Allowed values: {self.allowed}."
            f" Got: {self.input_value}."
        )


def get_decoder(
    classifier: str = "lda",
    scoring: str = "balanced_accuracy",
    balancing: str | None = None,
    optimize: bool = False,
) -> Decoder:
    """Create and return Decoder of desired type.

    Parameters
    ----------
    classifier : str
        Allowed values for `classifier`: ["catboost", "lda", "lin_svm", "lr",
        "svm_lin", "svm_poly", "svm_rbf", "xgb"].
    scoring : str | None, default="balanced_accuracy"
        Score to be calculated. Possible values:
        ["oversample", "undersample", "balance_weights"].
    balancing : str | None, default=None
        Method for balancing skewed datasets. Possible values:
        ["oversample", "undersample", "balance_weights"].

    Returns
    -------
    Decoder
        Instance of Decoder given `classifer` and `balancing` method.
    """
    classifiers = {
        "catboost": CATB,
        "dummy": Dummy,
        "lda": LDA,
        "lr": LR,
        "qda": QDA,
        # "svm_lin": SVC_Lin,
        # "svm_poly": SVC_Poly,
        # "svm_rbf": SVC_RBF,
        "xgb": XGB,
    }
    scoring_methods = {
        "balanced_accuracy": _get_balanced_accuracy,
        "log_loss": _get_log_loss,
    }

    classifier = classifier.lower()
    balancing = balancing.lower() if isinstance(balancing, str) else balancing
    scoring = scoring.lower()

    if classifier not in classifiers:
        raise DecoderNotFoundError(classifier, classifiers.keys())
    if scoring not in scoring_methods:
        raise ScoringMethodNotFoundError(scoring, scoring_methods.keys())
    return classifiers[classifier](
        balancing=balancing,
        optimize=optimize,
        scoring=scoring_methods[scoring],
    )


def _get_balanced_accuracy(model, data_test, label_test) -> Any:
    """Calculated balanced accuracy score."""
    return balanced_accuracy_score(label_test, model.predict(data_test))


def _get_log_loss(model, data_test, label_test) -> Any:
    """Calculate Log Loss score."""
    return log_loss(label_test, model.predict_proba(data_test))


class ScoringMethodNotFoundError(Exception):
    """Exception raised when invalid balancing method is passed.

    Attributes:
        input_value -- input value which caused the error
        allowed -- allowed input values
        message -- explanation of the error
    """

    def __init__(
        self,
        input_value,
        allowed,
        message="Input scoring method is not an allowed value.",
    ) -> None:
        self.input_value = input_value
        self.allowed = allowed
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return (
            f"{{self.message}} Allowed values: {self.allowed}. Got:"
            f" {self.input_value}."
        )


class DecoderNotFoundError(Exception):
    """Exception raised when invalid Decoder is passed.

    Attributes:
        input_value -- input which caused the error
        allowed -- allowed input types
        message -- explanation of the error
    """

    def __init__(
        self,
        input_value,
        allowed,
        message="Input decoding model is not an allowed value.",
    ) -> None:
        self.input_value = input_value
        self.allowed = allowed.values
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return (
            f"{{self.message}} Allowed values: {self.allowed}."
            " Got: {self.input_value}."
        )


@dataclass
class CATB(Decoder):
    """Class for CatBoostClassifier implementation."""

    def fit_and_predict(
        self,
        data_train: Union[pd.DataFrame, pd.Series],
        data_test: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> pd.Series:
        """Fit model to given training data and training labels."""
        self.data_train = data_train
        self.labels_train = labels
        self.groups_train = groups

        if self.optimize:
            self.model = self._bayesian_optimization()
        else:
            self.model = catboost.CatBoostClassifier(
                loss_function="MultiClass",
                verbose=False,
                use_best_model=True,
                eval_metric="MultiClass",
            )

        # Train outer model
        (
            self.data_train,
            self.labels_train,
            eval_set,
        ) = self._get_validation_split(
            self.data_train,
            self.labels_train,
            self.groups_train,
            train_size=0.8,
        )

        (
            self.data_train,
            self.labels_train,
            sample_weight,
        ) = self.balance_samples(self.data_train, self.labels_train)

        self.model.fit(
            self.data_train,
            self.labels_train,
            eval_set=eval_set,
            early_stopping_rounds=25,
            sample_weight=sample_weight,
            verbose=False,
        )
        return self.model.predict(data_test)

    def _bayesian_optimization(self):
        """Estimate optimal model parameters using bayesian optimization."""
        optimizer = BayesianOptimization(
            self._bo_tune,
            {
                "max_depth": (4, 10),
                "learning_rate": (0.003, 0.3),
                "bagging_temperature": (0.0, 1.0),
                "l2_leaf_reg": (1, 30),
                "random_strength": (0.01, 1.0),
            },
        )
        optimizer.maximize(init_points=10, n_iter=20, acq="ei")
        params = optimizer.max["params"]
        params["max_depth"] = round(params["max_depth"])
        return catboost.CatBoostClassifier(
            iterations=200,
            loss_function="MultiClass",
            verbose=False,
            use_best_model=True,
            eval_metric="MultiClass",
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_strength=params["random_strength"],
            bagging_temperature=params["bagging_temperature"],
            l2_leaf_reg=params["l2_leaf_reg"],
        )

    def _bo_tune(
        self,
        max_depth,
        learning_rate,
        bagging_temperature,
        l2_leaf_reg,
        random_strength,
    ):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        for train_index, test_index in cv_inner.split(
            self.data_train, self.labels_train, self.groups_train
        ):
            data_train_, data_test_ = (
                self.data_train[train_index],
                self.data_train[test_index],
            )
            y_tr, y_te = (
                self.labels_train[train_index],
                self.labels_train[test_index],
            )
            groups_tr = self.groups_train[train_index]

            (
                data_train_,
                y_tr,
                eval_set_inner,
            ) = self._get_validation_split(
                data=data_train_,
                labels=y_tr,
                groups=groups_tr,
                train_size=0.8,
            )
            data_train_, y_tr, sample_weight = self.balance_samples(
                data_train_, y_tr
            )
            inner_model = catboost.CatBoostClassifier(
                iterations=100,
                loss_function="MultiClass",
                verbose=False,
                eval_metric="MultiClass",
                max_depth=round(max_depth),
                learning_rate=learning_rate,
                bagging_temperature=bagging_temperature,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
            )
            inner_model.fit(
                data_train_,
                y_tr,
                eval_set=eval_set_inner,
                early_stopping_rounds=25,
                sample_weight=sample_weight,
                verbose=False,
            )
            y_probs = inner_model.predict_proba(data_test_)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)


@dataclass
class LDA(Decoder):
    """Class for applying Linear Discriminant Analysis using scikit-learn."""

    def __post_init__(self):
        # if self.balancing == "balance_weights":
        #     raise ValueError(
        #         "Sample weights cannot be balanced for Linear "
        #         "Discriminant Analysis. Please set `balance_weights` to"
        #         "either `oversample`, `undersample` or `None`."
        #     )
        if self.optimize:
            raise ValueError(
                "Hyperparameter optimization cannot be performed for this"
                " implementation of Linear Discriminant Analysis. Please"
                " set `optimize` to False."
            )

    def fit(
        self,
        data_train: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ):
        """Fit model to given training data and training labels."""
        (
            self.data_train,
            self.labels_train,
            _,
        ) = self.balance_samples(data_train, labels)
        if self.balancing == "balance_weights":
            priors = [0.5, 0.5]
        else:
            priors = None
        self.model = LinearDiscriminantAnalysis(
            solver="lsqr",
            priors=priors,
            # covariance_estimator=sklearn.covariance.OAS(
            #     store_precision=False, assume_centered=False
            # )
            shrinkage="auto",
        )
        self.model.fit(self.data_train, self.labels_train)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(data)

    def save_model(self, filename: Path | str) -> None:
        filename = Path(filename).with_suffix(".pickle")
        with open(filename, "wb") as file:
            pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass
class LR(Decoder):
    """Basic representation of class for finding and filtering files."""

    def fit_and_predict(
        self,
        data_train: np.ndarray,
        data_test: pd.DataFrame,
        labels: np.ndarray,
        groups,
    ) -> None:
        """Fit model to given training data and training labels."""
        self.data_train = data_train
        self.labels_train = labels
        self.groups_train = groups

        if self.optimize:
            self.model = self._bayesian_optimization()
        else:
            self.model = LogisticRegression(solver="newton-cg")

        self.data_train, self.labels_train, _ = self.balance_samples(
            data_train, labels
        )

        self.model.fit(self.data_train, self.labels_train)
        return self.model.predict(data_test)

    def _bayesian_optimization(self):
        """Estimate optimal model parameters using bayesian optimization."""
        optimizer = BayesianOptimization(
            self._bo_tune,
            {"C": (0.01, 1.0)},  # pylint: disable=invalid-name
        )
        optimizer.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = optimizer.max["params"]
        # params['max_iter'] = int(params['max_iter'])
        return LogisticRegression(
            solver="newton-cg", max_iter=500, C=params["C"]
        )

    def _bo_tune(self, C: float):  # pylint: disable=invalid-name
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []

        for train_index, test_index in cv_inner.split(
            self.data_train, self.labels_train, self.groups_train
        ):
            data_train_, data_test_ = (
                self.data_train[train_index],
                self.data_train[test_index],
            )
            y_tr, y_te = (
                self.labels_train[train_index],
                self.labels_train[test_index],
            )
            data_train_, y_tr, sample_weight = self.balance_samples(
                data_train_, y_tr
            )
            inner_model = LogisticRegression(
                solver="newton-cg", C=C, max_iter=500
            )
            inner_model.fit(data_train_, y_tr, sample_weight=sample_weight)
            y_probs = inner_model.predict_proba(data_test_)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)


@dataclass
class Dummy(Decoder):
    """Dummy classifier implementation from scikit learn"""

    def fit_and_predict(
        self,
        data_train: np.ndarray,
        data_test: pd.DataFrame,
        labels: np.ndarray,
        groups: pd.Series,
    ) -> np.ndarray:
        """Fit model to given training data and training labels."""
        self.data_train, self.labels_train, _ = self.balance_samples(
            data_train, labels
        )
        self.model = DummyClassifier(strategy="uniform")
        self.model.fit(self.data_train, self.labels_train)
        return self.model.predict(data_test)

    def get_score(self, data_test: np.ndarray, label_test: np.ndarray):
        """Calculate score."""
        scores = [
            self.scoring(self.model, data_test, label_test)
            for _ in range(0, 100)
        ]
        return np.mean(scores)


@dataclass
class QDA(Decoder):
    """Class for applying Linear Discriminant Analysis using scikit-learn."""

    def __post_init__(self):
        if self.balancing == "balance_weights":
            raise ValueError(
                "Sample weights cannot be balanced for Quadratic "
                "Discriminant Analysis. Please set `balance_weights` to"
                "either `oversample`, `undersample` or `None`."
            )
        if self.optimize:
            raise ValueError(
                "Hyperparameter optimization cannot be performed for this"
                " implementation of Quadratic Discriminant Analysis. Please"
                " set `optimize` to False."
            )

    def fit_and_predict(
        self,
        data_train: np.ndarray,
        labels: np.ndarray,
        data_test: pd.DataFrame,
        groups: pd.Series,
    ) -> None:
        """Fit model to given training data and training labels."""
        self.data_train, self.labels_train, _ = self.balance_samples(
            data_train, labels
        )
        self.model = QuadraticDiscriminantAnalysis()
        self.model.fit(self.data_train, self.labels_train)
        return self.model.predict(data_test)


@dataclass
class XGB(Decoder):
    """Basic representation of class for finding and filtering files."""

    def fit_and_predict(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """Fit model to given training data and training labels."""
        self.data_train = data_train
        self.labels_train = labels
        self.groups_train = groups

        if self.optimize:
            self.model = self._bayesian_optimization()
        else:
            self.model = xgb.XGBClassifier(
                objective="binary:logistic",
                booster="gbtree",
                use_label_encoder=False,
                n_estimators=200,
                eval_metric="logloss",
            )

        # Train outer model
        (
            self.data_train,
            self.labels_train,
            eval_set,
        ) = self._get_validation_split(
            self.data_train,
            self.labels_train,
            self.groups_train,
            train_size=0.8,
        )
        (
            self.data_train,
            self.labels_train,
            sample_weight,
        ) = self.balance_samples(data=data_train, labels=labels)

        self.model.fit(
            self.data_train,
            self.labels_train,
            eval_set=eval_set,
            early_stopping_rounds=20,
            sample_weight=sample_weight,
            verbose=False,
        )
        return self.model.predict(data_test)

    def _bayesian_optimization(self) -> xgb.XGBClassifier:
        """Estimate optimal model parameters using bayesian optimization."""
        optimizer = BayesianOptimization(
            self._bo_tune,
            {
                "learning_rate": (0.003, 0.3),
                "max_depth": (4, 10),
                "gamma": (0, 1),
                "colsample_bytree": (0.4, 1),
                "subsample": (0.4, 1),
            },
        )
        optimizer.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = optimizer.max["params"]
        return xgb.XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            n_estimators=200,
            eval_metric="logloss",
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            max_depth=int(params["max_depth"]),
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
        )

    def _bo_tune(
        self, learning_rate, gamma, max_depth, subsample, colsample_bytree
    ):
        cv_inner = GroupKFold(
            n_splits=3,
        )
        scores = []

        for train_index, test_index in cv_inner.split(
            self.data_train, self.labels_train, self.groups_train
        ):
            data_train_, data_test_ = (
                self.data_train.iloc[train_index],
                self.data_train.iloc[test_index],
            )
            y_tr, y_te = (
                self.labels_train[train_index],
                self.labels_train[test_index],
            )
            groups_tr = self.groups_train[train_index]

            (
                data_train_,
                y_tr,
                eval_set_inner,
            ) = self._get_validation_split(
                data=data_train_,
                labels=y_tr,
                groups=groups_tr,
                train_size=0.8,
            )
            (
                data_train_,
                y_tr,
                sample_weight,
            ) = self.balance_samples(data=data_train_, labels=y_tr)
            inner_model = xgb.XGBClassifier(
                objective="binary:logistic",
                booster="gbtree",
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=100,
                learning_rate=learning_rate,
                gamma=gamma,
                max_depth=int(max_depth),
                colsample_bytree=colsample_bytree,
                subsample=subsample,
            )
            inner_model.fit(
                X=data_train_,
                y=y_tr,
                eval_set=eval_set_inner,
                early_stopping_rounds=20,
                sample_weight=sample_weight,
                verbose=False,
            )
            y_probs = inner_model.predict_proba(X=data_test_)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    # @dataclass
    # class SVC_Lin(Decoder):
    #     """"""

    # @dataclass
    # class SVC_Poly(Decoder):
    #     """"""

    # @dataclass
    # class SVC_RBF(Decoder):
    #     """"""

    # @dataclass
    # class SVC_Sig(Decoder):
    #     """"""

    # def classify_svm_lin(data_train, y_train, group_train, optimize,
    #       balance):
    #     """"""

    #     def bo_tune(C, tol):
    #         # Cross validating with the specified parameters in 5 folds
    #         cv_inner = GroupShuffleSplit(
    #             n_splits=3, train_size=0.66, random_state=42
    #         )
    #         scores = []
    #         for train_index, test_index in cv_inner.split(
    #             data_train, y_train, group_train
    #         ):
    #             data_train_, data_test_ = data_train[train_index],
    #  data_train[test_index]
    #             y_tr, y_te = y_train[train_index], y_train[test_index]
    #             inner_model = SVC(
    #                 kernel="linear",
    #                 C=C,
    #                 max_iter=500,
    #                 tol=tol,
    #                 gamma="scale",
    #                 shrinking=True,
    #                 class_weight=None,
    #                 probability=True,
    #                 verbose=False,
    #             )
    #             inner_model.fit(data_train_, y_tr,
    # sample_weight=sample_weight)
    #             y_probs = inner_model.predict_proba(data_test_)
    #             score = log_loss(y_te, y_probs, labels=[0, 1])
    #             scores.append(score)
    #         # Return the negative MLOGLOSS
    #         return -1.0 * np.mean(scores)

    #     if optimize:
    #         # Perform Bayesian Optimization
    #         bo = BayesianOptimization(
    #             bo_tune, {"C": (pow(10, -1), pow(10, 1)),
    # "tol": (1e-4, 1e-2)}
    #         )
    #         bo.maximize(init_points=10, n_iter=20, acq="ei")
    #         # Train outer model with optimized parameters
    #         params = bo.max["params"]
    #         # params['max_iter'] = 500
    #         model = SVC(
    #             kernel="linear",
    #             C=params["C"],
    #             max_iter=500,
    #             tol=params["tol"],
    #             gamma="scale",
    #             shrinking=True,
    #             class_weight=None,
    #             verbose=False,
    #         )
    #     else:
    #         # Use default values
    #         model = SVC(
    #             kernel="linear",
    #             gamma="scale",
    #             shrinking=True,
    #             class_weight=None,
    #             verbose=False,
    #         )
    #     model.fit(data_train, y_train, sample_weight=sample_weight)
    #     return model

    # def classify_svm_rbf(data_train, y_train, group_train, optimize,
    # balance):
    #     """"""

    #     def bo_tune(C, tol):
    #         # Cross validating with the specified parameters in 5 folds
    #         cv_inner = GroupShuffleSplit(
    #             n_splits=3, train_size=0.66, random_state=42
    #         )
    #         scores = []
    #         for train_index, test_index in cv_inner.split(
    #             data_train, y_train, group_train
    #         ):
    #             data_train_, data_test_ = data_train[train_index],
    # data_train[test_index]
    #             y_tr, y_te = y_train[train_index], y_train[test_index]
    #             inner_model = SVC(
    #                 kernel="rbf",
    #                 C=C,
    #                 max_iter=500,
    #                 tol=tol,
    #                 gamma="scale",
    #                 shrinking=True,
    #                 class_weight=None,
    #                 probability=True,
    #                 verbose=False,
    #             )
    #             inner_model.fit(data_train_, y_tr,
    # sample_weight=sample_weight)
    #             y_probs = inner_model.predict_proba(data_test_)
    #             score = log_loss(y_te, y_probs, labels=[0, 1])
    #             scores.append(score)
    #         # Return the negative MLOGLOSS
    #         return -1.0 * np.mean(scores)

    #     if optimize:
    #         # Perform Bayesian Optimization
    #         bo = BayesianOptimization(
    #             bo_tune, {"C": (pow(10, -1), pow(10, 1)),
    # "tol": (1e-4, 1e-2)}
    #         )
    #         bo.maximize(init_points=10, n_iter=20, acq="ei")
    #         # Train outer model with optimized parameters
    #         params = bo.max["params"]
    #         model = SVC(
    #             kernel="rbf",
    #             C=params["C"],
    #             max_iter=500,
    #             tol=params["tol"],
    #             gamma="scale",
    #             shrinking=True,
    #             class_weight=None,
    #             verbose=False,
    #         )
    #     else:
    #         # Use default values
    #         model = SVC(
    #             kernel="rbf",
    #             gamma="scale",
    #             shrinking=True,
    #             class_weight=None,
    #             verbose=False,
    #         )
    #     model.fit(data_train, y_train, sample_weight=sample_weight)
    #     return model

    # def classify_svm_poly(data_train, y_train, group_train):
    #     """"""

    #     def bo_tune(C, tol):
    #         # Cross validating with the specified parameters in 5 folds
    #         cv_inner = GroupShuffleSplit(
    #             n_splits=3, train_size=0.66, random_state=42
    #         )
    #         scores = []
    #         for train_index, test_index in cv_inner.split(
    #             data_train, y_train, group_train
    #         ):
    #             data_train_, data_test_ = data_train[train_index],
    #               data_train[test_index]
    #             y_tr, y_te = y_train[train_index], y_train[test_index]
    #             inner_model = SVC(
    #                 kernel="poly",
    #                 C=C,
    #                 max_iter=500,
    #                 tol=tol,
    #                 gamma="scale",
    #                 shrinking=True,
    #                 class_weight=None,
    #                 probability=True,
    #                 verbose=False,
    #             )
    #             inner_model.fit(data_train_, y_tr,
    #               sample_weight=sample_weight)
    #             y_probs = inner_model.predict_proba(data_test_)
    #             score = log_loss(y_te, y_probs, labels=[0, 1])
    #             scores.append(score)
    #         # Return the negative MLOGLOSS
    #         return -1.0 * np.mean(scores)

    #     if optimize:
    #         # Perform Bayesian Optimization
    #         bo = BayesianOptimization(
    #             bo_tune, {"C": (pow(10, -1), pow(10, 1)),
    #                       "tol": (1e-4, 1e-2)}
    #         )
    #         bo.maximize(init_points=10, n_iter=20, acq="ei")
    #         # Train outer model with optimized parameters
    #         params = bo.max["params"]
    #         model = SVC(
    #             kernel="poly",
    #             C=params["C"],
    #             max_iter=500,
    #             tol=params["tol"],
    #             gamma="scale",
    #             shrinking=True,
    #             class_weight=None,
    #             verbose=False,
    #         )
    #     else:
    #         # Use default values
    #         model = SVC(
    #             kernel="poly",
    #             gamma="scale",
    #             shrinking=True,
    #             class_weight=None,
    #             verbose=False,
    #         )
    #     model.fit(data_train, y_train, sample_weight=sample_weight)
    #     return model

    # def classify_svm_sig(data_train, y_train, group_train, optimize,
    # balance):
    # """"""

    # def bo_tune(C, tol):
    #     # Cross validating with the specified parameters in 5 folds
    #     cv_inner = GroupShuffleSplit(
    #         n_splits=3, train_size=0.66, random_state=42
    #     )
    #     scores = []
    #     for train_index, test_index in cv_inner.split(
    #         data_train, y_train, group_train
    #     ):
    #         data_train_, data_test_ = data_train[train_index],
    # data_train[test_index]
    #         y_tr, y_te = y_train[train_index], y_train[test_index]
    #         inner_model = SVC(
    #             kernel="sigmoid",
    #             C=C,
    #             max_iter=500,
    #             tol=tol,
    #             gamma="auto",
    #             shrinking=True,
    #             class_weight=None,
    #             probability=True,
    #             verbose=False,
    #         )
    #         inner_model.fit(data_train_, y_tr, sample_weight=sample_weight)
    #         y_probs = inner_model.predict_proba(data_test_)
    #         score = log_loss(y_te, y_probs, labels=[0, 1])
    #         scores.append(score)
    #     # Return the negative MLOGLOSS
    #     return -1.0 * np.mean(scores)

    # if optimize:
    #     # Perform Bayesian Optimization
    #     bo = BayesianOptimization(
    #         bo_tune, {"C": (pow(10, -1), pow(10, 1)), "tol": (1e-4, 1e-2)}
    #     )
    #     bo.maximize(init_points=10, n_iter=20, acq="ei")
    #     # Train outer model with optimized parameters
    #     params = bo.max["params"]
    #     model = SVC(
    #         kernel="sigmoid",
    #         C=params["C"],
    #         max_iter=500,
    #         tol=params["tol"],
    #         gamma="auto",
    #         shrinking=True,
    #         class_weight=None,
    #         verbose=False,
    #     )
    # else:
    #     # Use default values
    #     model = SVC(
    #         kernel="sigmoid",
    #         gamma="scale",
    #         shrinking=True,
    #         class_weight=None,
    #         verbose=False,
    #     )

    # model.fit(data_train, y_train, sample_weight=sample_weight)
    # return model
