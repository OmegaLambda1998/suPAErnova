import os
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import pytest
from astropy import cosmology as cosmo
import sncosmo

from suPAErnova.configs.steps.data import DataStepResult

if TYPE_CHECKING:
    from _pytest.tmpdir import TempPathFactory

    from . import (
        DataParams,
        DataResults,
        DataStepFactory,
        DataStepResults,
        DataResultFactory,
    )


def legacy_data_step(
    data_params: "DataParams",
) -> "DataResults":
    # Used cached result if it exists.
    savepath = (
        Path(data_params["cache_path"])
        / data_params["fname"]
        / "data"
        / "legacy"
        / "data_step.npz"
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    if savepath.exists():
        with np.load(savepath, allow_pickle=True) as io:
            return dict(io)

    # Import here to avoid dependency conflicts
    from supaernova_legacy.scripts.prep_data import prep_data

    # Except where indicated, this is running the `train_ae` script verbatim

    # Variation: train_ae script modified to allow passing args as a list of strings
    # Variation: train_ae script modified to return a dictionary of results per train stage
    # Variation: train_ae script modified to skip training stages which have already been run
    # Variation: Legacy code relied on the now deprecated tensorflow_addons package.
    #            Most of the functionality now resides in tensorflow.keras
    #            The Legacy code has been updated to reflect this
    #            The biggest change is to the AdamW optimiser, which can no longer take a function for its weight_decay argument
    # Variation: Legacy code used an old version of Tensorflow, and the syntax has since changed
    #            In particular:
    #             - `tf.tf_fn(x)` is no longer valid, so has been updated to `ks.layers.Lambda(tf.tf_fn)(x)`
    #             - Stricter dtype checks (int32 ~= int64 ~= float32 ~= float64). Using tf.cast where needed.

    yaml_config = Path(data_params["tmp_path"]) / "train.yaml"

    config = {"data": {**data_params, "savepath": str(savepath)}}

    with yaml_config.open("w") as io:
        yaml.safe_dump(config, io)

    args = [f"--yaml_config={yaml_config}", "--config=data"]
    results = prep_data(args)

    np.savez_compressed(savepath, **results)
    with np.load(savepath, allow_pickle=True) as io:
        return dict(io)


@pytest.fixture(scope="session")
def legacy_data_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
    tmp_path_factory: "TempPathFactory",
) -> "DataStepFactory":
    def _legacy_data_step(data_params: "DataParams") -> "DataResults":
        data_params["data_path"] = str(data_path)
        data_params["root_path"] = str(root_path)
        data_params["cache_path"] = str(cache_path)
        data_params["tmp_path"] = str(tmp_path_factory.mktemp("config"))
        return legacy_data_step(data_params)

    return _legacy_data_step


@pytest.fixture(scope="session")
def legacy_data_result_factory(
    legacy_data_step_factory: "DataStepFactory",
) -> "DataResultFactory":
    def _legacy_data_result(data_params: "DataParams") -> "DataStepResults":
        return DataStepResult.model_validate(legacy_data_step_factory(data_params))

    return _legacy_data_result
