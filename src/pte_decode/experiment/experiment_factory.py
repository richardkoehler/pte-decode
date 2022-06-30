"""Module for running decoding experiments."""
from pathlib import Path
from typing import Sequence

import mne_bids
import pte_decode
from joblib import Parallel, delayed
from sklearn.model_selection import BaseCrossValidator


def run_pipeline_decoding(
    filepaths_features: list[str] | list[Path],
    n_jobs: int = 1,
    **kwargs,
) -> None:
    """Run decoding pipeline with given files."""
    if not filepaths_features:
        raise ValueError("No feature files specified.")
    if len(filepaths_features) == 1 or n_jobs in (0, 1):
        (
            pynm_experiment(
                feature_file=feature_file,
                **kwargs,
            )
            for feature_file in filepaths_features
        )
    Parallel(n_jobs=n_jobs)(
        delayed(pynm_experiment)(feature_file=feature_file, **kwargs)
        for feature_file in filepaths_features
    )


def pynm_experiment(
    feature_root: Path | str,
    feature_file: Path | str,
    classifier: str,
    label_channels: Sequence[str],
    target_begin: str | int | float,
    target_end: str | int | float,
    optimize: bool,
    balancing: str | None,
    out_root: Path | str,
    feature_keywords: Sequence[str],
    cross_validation: BaseCrossValidator,
    plotting_target_channels: list[str],
    channels_used: str = "single",
    types_used: str | Sequence[str] = "all",
    hemispheres_used: str = "both",
    scoring: str = "balanced_accuracy",
    bad_epochs_path: Path | str | None = None,
    pred_mode: str = "classify",
    pred_begin: int | float = -3.0,
    pred_end: int | float = 2.0,
    use_times: int = 1,
    dist_end: int | float = 2.0,
    excep_dist_end: int | float = 2.0,
    exception_keywords: Sequence[str] | None = None,
    feature_importance: bool | int = False,
    verbose: bool = True,
    rest_begin: int | float = -5.0,
    rest_end: int | float = -2.0,
) -> None:
    """Run experiment with single file."""
    import pte  # pylint: disable=import-outside-toplevel
    from py_neuromodulation import (  # pylint: disable=import-outside-toplevel
        nm_analysis,
    )

    print("Using file: ", feature_file)
    # Read features
    nm_reader = nm_analysis.Feature_Reader(
        feature_dir=str(feature_root), feature_file=str(feature_file)
    )
    features_raw = nm_reader.feature_arr

    ch_types_all = nm_reader.nm_channels[["new_name", "type"]].set_index(
        "new_name"
    )
    ch_names = nm_reader.sidecar["ch_names"]
    ch_types = ch_types_all.loc[ch_names, "type"].tolist()  # type: ignore
    side = _get_trial_side(fname=feature_file)

    outpath_feat_clean, file_suffix = _outpath_feat_clean(
        root=out_root,
        fname=feature_file,
        types_used=types_used,
        hemispheres_used=hemispheres_used,
    )
    # Clean features
    features_clean, label, plotting_target = pte_decode.FeatureCleaner(
        data_channel_names=ch_names,
        data_channel_types=ch_types,
        label_channels=label_channels,
        plotting_target_channels=plotting_target_channels,
        side=side,
        types_used=types_used,
        hemispheres_used=hemispheres_used,
    ).run(features=features_raw, out_path=outpath_feat_clean)

    # Engineer features
    outpath_feat_eng, file_suffix = _outpath_feat_eng(
        root=out_root,
        fname=feature_file,
        suffix=file_suffix,
        use_times=use_times,
    )
    features_eng = pte_decode.FeatureEngineer(
        use_times=use_times, verbose=verbose
    ).run(features_clean, out_path=outpath_feat_eng)

    # Select features
    outpath_feat_sel, file_suffix = _outpath_feat_sel(
        root=out_root,
        fname=feature_file,
        suffix=file_suffix,
    )
    features_sel = pte_decode.FeatureSelector(
        feature_keywords=feature_keywords, verbose=verbose
    ).run(features_eng, out_path=outpath_feat_sel)

    # Handle bad events file
    bad_epochs_df = pte.filetools.get_bad_epochs(
        bad_epochs_dir=bad_epochs_path, filename=feature_file
    )
    bad_epochs = bad_epochs_df.event_id.to_numpy()

    # Handle exception files
    dist_end = _handle_exception_files(
        fname=feature_file,
        dist_end=dist_end,
        excep_dist_end=excep_dist_end,
        exception_keywords=exception_keywords,
    )

    # Get feature epochs
    path_feat_epochs, file_suffix = _outpath_feat_epochs(
        root=out_root,
        fname=feature_file,
        suffix=file_suffix,
        target_begin=target_begin,
        target_end=target_end,
    )

    (features_concat, features_timelocked, _, _,) = pte_decode.FeatureEpochs(
        target_begin=target_begin,
        target_end=target_end,
        rest_begin=rest_begin,
        rest_end=rest_end,
        offset_rest_begin=dist_end,
        epoch_begin=pred_begin,
        epoch_end=pred_end,
        verbose=verbose,
    ).run(
        features=features_sel,
        label=label,
        plotting_target=plotting_target,
        sfreq=nm_reader.settings["sampling_rate_features_hz"],
        bad_epochs=bad_epochs,
        out_path=path_feat_epochs,
    )

    label_concat = features_concat["labels"]
    trial_ids = features_concat["trial_ids"]
    features_concat = features_concat.drop(
        columns=["time", "time relative", "labels", "trial_ids"]
    )

    # Generate output file name
    outpath_predict, file_suffix = _outpath_predict(
        root=out_root,
        fname=feature_file,
        suffix=file_suffix,
        classifier=classifier,
        optimize=optimize,
    )
    outpath_predict.parent.mkdir(exist_ok=True, parents=True)

    decoder = pte_decode.get_decoder(
        classifier=classifier,
        scoring=scoring,
        balancing=balancing,
        optimize=optimize,
    )
    # Initialize Experiment instance
    experiment = pte_decode.DecodingExperiment(
        features=features_concat,
        features_timelocked=features_timelocked,
        times_timelocked=features_timelocked["time"],
        labels=label_concat,
        trial_ids=trial_ids,
        plotting_data=plotting_target,
        ch_names=ch_names,
        decoder=decoder,
        bad_epochs=bad_epochs,
        scoring=scoring,
        feature_importance=feature_importance,
        channels_used=channels_used,
        prediction_mode=pred_mode,
        cv_outer=cross_validation,
        verbose=verbose,
    )
    experiment.run()
    experiment.save(
        path=outpath_predict,
        scores=True,
        predictions_concatenated=True,
        predictions_timelocked=True,
        feature_importances=True,
        final_models=True,
    )


def _handle_exception_files(
    fname: Path | str,
    dist_end: int | float,
    excep_dist_end: int | float,
    exception_keywords: Sequence[str] | None = None,
):
    """Check if current file is listed in exception files."""
    if exception_keywords:
        if any(keyw in str(fname) for keyw in exception_keywords):
            print("Exception file recognized: ", Path(fname).name)
            return excep_dist_end
    return dist_end


def _outpath_feat_clean(
    root: Path | str,
    fname: Path | str,
    types_used: str | Sequence[str],
    hemispheres_used: str,
) -> tuple[Path, str]:
    """Generate file name for output files."""
    if isinstance(types_used, str):
        types_used = [types_used]
    type_str = "_".join(("chs", *(_type for _type in types_used)))
    side_str = f"hem_{hemispheres_used}"
    dir_name = "_".join((type_str, side_str))

    out_path = Path(root, "12_features_cleaned", dir_name, fname, fname)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    return out_path, dir_name


def _outpath_feat_eng(
    root: Path | str,
    fname: Path | str,
    suffix: str,
    use_times: int,
) -> tuple[Path, str]:
    """Generate file name for output files."""
    feat_str = f"{use_times * 100}ms"
    dir_name = "_".join((feat_str, suffix))

    out_path = Path(root, "14_features_engineered", dir_name, fname, fname)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    return out_path, dir_name


def _outpath_feat_sel(
    root: Path | str,
    fname: Path | str,
    suffix: str,
) -> tuple[Path, str]:
    """Generate file name for output files."""
    dir_name = "_".join((suffix,))
    out_path = Path(root, "16_features_selected", dir_name, fname, fname)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    return out_path, dir_name


def _outpath_feat_epochs(
    root: Path | str,
    fname: Path | str,
    suffix: str,
    target_begin: str | int | float,
    target_end: str | int | float,
) -> tuple[Path, str]:
    """Generate file name for output files."""
    if target_begin == 0.0:
        target_begin = "trial_begin"
    if target_end == 0.0:
        target_end = "trial_begin"

    target_str = f"{target_begin}_{target_end}"
    dir_name = "_".join((target_str, suffix))

    out_path = Path(root, "18_features_epochs", dir_name, fname, fname)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    return out_path, dir_name


def _outpath_predict(
    root: Path | str,
    fname: Path | str,
    suffix: str,
    classifier: str,
    optimize: bool,
) -> tuple[Path, str]:
    """Generate file name for output files."""
    clf_str = str(classifier)
    opt_str = "opt_yes" if optimize else "opt_no"
    dir_name = "_".join((clf_str, opt_str, suffix))

    out_path = Path(root, "20_predict", dir_name, fname, fname)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    return out_path, dir_name


def _get_trial_side(fname: Path | str) -> str | None:
    """Get body side of given trial"""
    task: str = mne_bids.get_entities_from_fname(str(fname))["task"]
    if task.endswith("R"):
        return "right"
    if task.endswith("L"):
        return "left"
    return None
