"""Module for extracting event-based features for decoding."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineer:
    """Engineer features to use from given features."""

    use_times: int = 1
    verbose: bool = False

    def run(
        self,
        features: pd.DataFrame,
        out_path: Path | str | None = None,
    ) -> pd.DataFrame:
        features = self._process_use_times(features)

        if out_path:
            out_path = str(Path(out_path).resolve())
            features.to_csv(
                out_path + "_FeaturesEngineered.csv.gz", index=False
            )

        return features

    def _process_use_times(self, features: pd.DataFrame) -> pd.DataFrame:
        """Process time points used."""
        feat_processed = [
            features.rename(
                columns={col: col + "_0ms" for col in features.columns}
            )
        ]
        # Use additional features from previous time points
        # ``use_times = 1`` means no features from previous time points are used
        for use_time in np.arange(1, self.use_times):
            feat_processed.append(
                features.shift(use_time, axis=0).rename(
                    columns={
                        col: col + f"_{(use_time) * 100}ms"
                        for col in features.columns
                    }
                )
            )
        return pd.concat(feat_processed, axis=1).fillna(0.0)
