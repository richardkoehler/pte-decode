"""Modules for machine learning."""

__version__ = "0.1.0"

from .decoding.decoder_factory import get_decoder
from .decoding.decoder_base import Decoder
from .experiment.experiment_base import Experiment
from .experiment.experiment_factory import run_experiment
from .plotting.plot import (
    boxplot_results,
    lineplot_prediction,
    lineplot_prediction_single,
    violinplot_results,
)
from .results.load import (
    load_predictions,
    load_results,
    load_results_singlechannel,
)
from .results.timepoint import get_earliest_timepoint
