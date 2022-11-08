"""Module for extracting event-based features for decoding."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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
            out_path = str(Path(out_path).resolve())
            features.to_csv(
                out_path + r"/FeaturesEngineered.csv.gz", index=False
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
