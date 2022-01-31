"""Modules for machine learning."""

from .decoding.decoder import get_decoder
from .decoding.decoder_abc import Decoder
from .experiment.experiment_base import Experiment
from .experiment.experiment_factory import run_experiment
from .plotting.plot import (
    boxplot_results,
    lineplot_prediction,
    violinplot_results,
)
from .results.load import (
    load_predictions,
    load_results,
    load_results_singlechannel,
)
from .results.timepoint import get_earliest_timepoint
