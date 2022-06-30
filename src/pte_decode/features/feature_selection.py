"""Module for feature selection."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass
class FeatureSelector:
    """Select features."""

    feature_keywords: Sequence[str] | None
    verbose: bool = False

    def run(
        self,
        features: pd.DataFrame,
        out_path: Path | str | None = None,
    ) -> pd.DataFrame:
        feature_picks = self._pick_by_keywords(features)

        if out_path:
            out_path = str(Path(out_path).resolve())

            feature_picks.to_csv(
                out_path + "_FeaturesSelected.csv.gz", index=False
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
