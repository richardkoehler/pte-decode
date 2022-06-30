"""Modules for machine learning."""

__version__ = "0.1.0"

from .decoding.decoder_base import Decoder
from .decoding.decoder_factory import get_decoder
from .experiment.experiment_base import DecodingExperiment
from .experiment.experiment_factory import run_pipeline_decoding
from .features.feature_cleaning import FeatureCleaner
from .features.feature_engineering import FeatureEngineer
from .features.feature_epochs import FeatureEpochs
from .features.feature_selection import FeatureSelector
from .plotting.plot import (
    boxplot_results,
    lineplot_prediction,
    lineplot_prediction_single,
    violinplot_results,
)
from .results.load import (
    load_predictions,
    load_scores,
    load_results_singlechannel,
)
from .results.timepoint import get_earliest_timepoint
