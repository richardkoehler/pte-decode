"""Module for extracting event-based features for decoding."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


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
            out_path = str(Path(out_path).resolve())

            feature_picks.to_csv(
                out_path + "_FeaturesCleaned.csv.gz", index=False
            )
            label.to_csv(out_path + "_Label.csv.gz", index=False)
            label.to_csv(out_path + "_PlottingTarget.csv.gz", index=False)

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


def _transform_side(side: str) -> str:
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
