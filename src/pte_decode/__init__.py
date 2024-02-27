"""Package for decoding of neurophysiology data."""

__version__ = "0.1.0"

from .decoders import Decoder, get_decoder
from .experiment import (
    DecodingExperiment,
    run_pipeline_multiproc,
)
from .features import (
    FeatureCleaner,
    FeatureEngineer,
    FeatureEpochs,
    FeatureSelector,
)
from .plotting import (
    boxplot_all_conds,
    boxplot_results,
    boxplot_updrs,
    lineplot_compare,
    lineplot_prediction,
    lineplot_single,
)
from .results import (
    get_earliest_timepoint,
    load_predictions,
    load_predictions_singlefile,
    load_results_singlechannel,
    load_scores,
)
