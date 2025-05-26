from typing import TYPE_CHECKING

import yaml
import numpy as np
import pytest

from suPAErnova.configs.steps.pae import PAEStepResult
from suPAErnova.configs.steps.data import DataStepResult
from suPAErnova.configs.steps.nflow import NFlowStepResult

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from _pytest.tmpdir import TempPathFactory


def legacy_nflow_step(
    nflow_params: dict[str, "Any"],
    data: "DataStepResult",
    pae: "PAEStepResult",
) -> dict[str, dict[str, "Any"]]:
    # Used cached result if it exists.
    savepath = (
        nflow_params["cache_path"]
        / nflow_params["fname"]
        / "nflow"
        / "legacy"
        / nflow_params["seed"]
        / "nflow_step.npz"
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    if savepath.exists():
        with np.load(savepath, allow_pickle=True) as io:
            return {k: v.item() for k, v in io.items()}

    # Import here to avoid dependency conflicts
    from supaernova_legacy.scripts.train_flow import train_flow

    # Except where indicated, this is running the `train_ae` script verbatim

    # Variation: train_flow script modified to allow passing args as a list of strings
    # Variation: train_flow script modified to return a dictionary of results
    # Variation: train_flow script modified to skip training which has already been run
    # Variation: Legacy code used an old version of Tensorflow, and the syntax has since changed
    #            In particular:
    #             - `tf.tf_fn(x)` is no longer valid, so has been updated to `ks.layers.Lambda(tf.tf_fn)(x)`
    #             - Stricter dtype checks (int32 ~= int64 ~= float32 ~= float64). Using tf.cast where needed.

    data_out_path: Path = (
        nflow_params["cache_path"] / nflow_params["fname"] / "data" / "legacy"
    )
    pae_out_path: Path = (
        nflow_params["cache_path"]
        / nflow_params["fname"]
        / "pae"
        / "legacy"
        / nflow_params["seed"]
    )

    model_dir = pae_out_path / "tensorflow_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    param_dir = pae_out_path / "params"
    param_dir.mkdir(parents=True, exist_ok=True)

    nflow_out_path: Path = (
        nflow_params["cache_path"]
        / nflow_params["fname"]
        / "nflow"
        / "legacy"
        / nflow_params["seed"]
    )

    yaml_config = nflow_params["tmp_path"] / "train.yaml"

    config = {
        "nflow": {
            "PROJECT_DIR": str(nflow_params["root_path"]),
            "MODEL_DIR": str(model_dir),
            "PARAM_DIR": str(param_dir),
            "model_summary": False,
            "out_file_tail": "",
            "train_data_file": str(data_out_path / "train" / "kfold0.npz"),
            "test_data_file": str(data_out_path / "test" / "kfold0.npz"),
            "verbose": True,
            "kfold": 0,
            "prev_train_stage": "5",
            "set_data_min_val": 0,
            "checkpoint_flow_every": 10,
            "patience": 30,
            "encode_dims": (*nflow_params["encode_dims"], 32),
            "latent_dims": (nflow_params["n_z_latents"],),
            "overfit": not nflow_params["save_best"],
            "lr_flow": nflow_params["learning_rate"],
            "use_extrinsic_params": nflow_params["physical_latents"],
            "nlayers": nflow_params["n_layers"],
            "nunit": nflow_params["n_hidden_units"],
            "epochs_flow": nflow_params["epochs"],
            "batchnorm": nflow_params["batch_normalisation"],
            "val_frac_flow": nflow_params["validation_frac"],
            "batch_size": nflow_params["batch_size"],
        }
    }
    with yaml_config.open("w") as io:
        yaml.safe_dump(config, io)

    args = [f"--yaml_config={yaml_config}", "--config=nflow"]
    results = train_flow(args)

    nflow_step_results = {}

    np.savez_compressed(savepath, **nflow_step_results)
    with np.load(savepath, allow_pickle=True) as io:
        return {k: v.item() for k, v in io.items()}


@pytest.fixture(scope="session")
def legacy_nflow_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
    tmp_path_factory: "TempPathFactory",
    legacy_data_step_factory: "Callable[[dict[str, Any]], dict[str, Any]]",
    legacy_pae_step_factory: "Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]",
) -> "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, dict[str, Any]]]":
    def _legacy_nflow_step(
        data_params: "dict[str, Any]",
        pae_params: "dict[str, Any]",
        nflow_params: "dict[str, Any]",
    ) -> "dict[str, dict[str, Any]]":
        nflow_params["data_path"] = data_path
        nflow_params["root_path"] = root_path
        nflow_params["cache_path"] = cache_path
        nflow_params["tmp_path"] = tmp_path_factory.mktemp("config")

        data = DataStepResult.model_validate(legacy_data_step_factory(data_params))
        pae = PAEStepResult.model_validate(
            legacy_pae_step_factory(data_params, pae_params)["6"]
        )
        return legacy_nflow_step(nflow_params, data, pae)

    return _legacy_nflow_step


@pytest.fixture(scope="session")
def legacy_nflow_result_factory(
    legacy_nflow_step_factory: "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, dict[str, Any]]]",
) -> "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], NFlowStepResult]":
    def _legacy_nflow_result(
        data_params: dict[str, "Any"],
        pae_params: dict[str, "Any"],
        nflow_params: dict[str, "Any"],
    ) -> "NFlowStepResult":
        nflow_step_results = legacy_nflow_step_factory(
            data_params, pae_params, nflow_params
        )
        return NFlowStepResult.model_validate(nflow_step_results)

    return _legacy_nflow_result
