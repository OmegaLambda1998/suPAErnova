from typing import TYPE_CHECKING

import yaml
import numpy as np
import pytest

from suPAErnova.configs.steps.pae import PAEStepResult
from suPAErnova.configs.steps.data import DataStepResult
from suPAErnova.configs.steps.nflow import NFlowStepResult
from suPAErnova.configs.steps.posterior import PosteriorStepResult

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.tmpdir import TempPathFactory

    from tests.fixtures.pae import (
        PAEParams,
        PAEStepFactory,
    )
    from tests.fixtures.data import (
        DataParams,
        DataStepFactory,
    )
    from tests.fixtures.nflow import (
        NFlowParams,
        NFlowStepFactory,
    )

    from . import (
        PosteriorParams,
        PosteriorResults,
        PosteriorStepFactory,
        PosteriorStepResults,
        PosteriorResultFactory,
    )


def legacy_posterior_step(
    posterior_params: "PosteriorParams",
    data: "DataStepResult",
    pae: "PAEStepResult",
    nflow: "NFlowStepResult",
) -> "PosteriorResults":
    # Used cached result if it exists.
    savepath = (
        posterior_params["cache_path"]
        / posterior_params["fname"]
        / "posterior"
        / "legacy"
        / posterior_params["seed"]
        / "posterior_step.npz"
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    if savepath.exists():
        with np.load(savepath, allow_pickle=True) as io:
            return dict(io.items())

    # Import here to avoid dependency conflicts
    from supaernova_legacy.scripts.run_posterior_analysis import run_posterior_analysis

    # Except where indicated, this is running the `run_posterior_analysis` script verbatim

    # Variation: run_posterior_analysis script modified to allow passing args as a list of strings
    # Variation: run_posterior_analysis script modified to return a dictionary of results
    # Variation: run_posterior_analysis script modified to skip training which has already been run
    # Variation: Legacy code mixed numpy and tensorflow arrays, so use tf.convert_to_tensor to resolve
    # Variation: Legacy code used an old version of Tensorflow, and the syntax has since changed
    #            In particular:
    #             - `tf.tf_fn(x)` is no longer valid, so has been updated to `ks.layers.Lambda(tf.tf_fn)(x)`
    #             - Stricter dtype checks (int32 ~= int64 ~= float32 ~= float64). Using tf.cast where needed.
    # Variation: Legacy code used an old version of TensorflowProbability which is no longer compatible with tf.keras.
    #            Instead we need to use tf_keras
    # Variation: Legacy code was incorrectly loading models in, this has been fixed

    data_out_path: Path = (
        posterior_params["cache_path"] / posterior_params["fname"] / "data" / "legacy"
    )

    pae_out_path: Path = (
        posterior_params["cache_path"]
        / posterior_params["fname"]
        / "pae"
        / "legacy"
        / posterior_params["seed"]
    )
    pae_model_dir = pae_out_path / "tensorflow_models"
    pae_model_dir.mkdir(parents=True, exist_ok=True)
    param_dir = pae_out_path / "params"
    param_dir.mkdir(parents=True, exist_ok=True)

    nflow_out_path: Path = (
        posterior_params["cache_path"]
        / posterior_params["fname"]
        / "nflow"
        / "legacy"
        / posterior_params["seed"]
    )
    nflow_model_dir = nflow_out_path / "tensorflow_models"
    nflow_model_dir.mkdir(parents=True, exist_ok=True)

    posterior_out_path: Path = (
        posterior_params["cache_path"]
        / posterior_params["fname"]
        / "posterior"
        / "legacy"
        / posterior_params["seed"]
    )
    posterior_model_dir = posterior_out_path / "tensorflow_models/"
    posterior_model_dir.mkdir(parents=True, exist_ok=True)

    yaml_config = posterior_params["tmp_path"] / "train.yaml"

    config = {
        "posterior": {
            "PROJECT_DIR": str(posterior_params["root_path"]),
            "MODEL_DIR": str(pae_model_dir),
            "NFLOW_MODEL_DIR": str(nflow_model_dir) + "/",
            "OUTPUT_DIR": str(posterior_model_dir) + "/",
            "PARAM_DIR": str(param_dir),
            "train_data_file": str(
                data_out_path / "train" / f"kfold{posterior_params['kfold']}.npz"
            ),
            "test_data_file": str(
                data_out_path / "test" / f"kfold{posterior_params['kfold']}.npz"
            ),
            "prev_train_stage": "5",
            "latent_dims": (posterior_params["n_z_latents"],),
            "kfold": posterior_params["kfold"],
            "seed": int(posterior_params["seed"]),
            "encode_dims": (*posterior_params["encode_dims"], 32),
            "out_file_tail": "",
            "posterior_file_tail": "",
            "overfit": not posterior_params["save_best"],
            "verbose": True,
            "model_summary": False,
            "print_params": True,
            "nunit": posterior_params["n_hidden_units"],
            "nlayers": posterior_params["n_layers"],
            "use_extrinsic_params": posterior_params["nflow_physical_latents"],
            "physical_latent": posterior_params["pae_physical_latents"],
            "batchnorm": posterior_params["batch_normalisation"],
            "set_data_min_val": 0,
            "min_train_redshift": posterior_params["min_train_redshift"],
            "max_train_redshift": posterior_params["max_train_redshift"],
            "max_light_cut": (
                posterior_params["min_train_phase"],
                posterior_params["max_train_phase"],
            ),
            "max_light_cut_spectra": (
                posterior_params["min_train_phase"],
                posterior_params["max_train_phase"],
            ),
            "inverse_spectra_cut": False,
            "twins_cut": False,
            "batch_size": posterior_params["batch_size"],
            "rMAPini": posterior_params["random_initial_positions"],
            "find_MAP": True,
            "run_HMC": True,
            "ihmc": posterior_params["monte_carlo_method"] == "HMC",
            "train_amplitude": posterior_params["train_delta_m"],
            "use_amplitude": posterior_params["train_delta_m"],
            "train_dtime": posterior_params["train_delta_p"],
            "train_bias": posterior_params["train_bias"],
            "nchains": posterior_params["n_chains_early"]
            + posterior_params["n_chains_mid"]
            + posterior_params["n_chains_final"],
            "dtime_norm": 1.0,
            "tolerance": posterior_params["tolerance"],
            "max_iterations": posterior_params["max_iterations"],
            "num_burnin_steps": posterior_params["n_burnin"],
            "num_samples": posterior_params["n_samples"],
            "num_leapfrog_steps": posterior_params["n_leapfrog"],
            "step_size": posterior_params["step_size"],
            "target_accept_rate": posterior_params["target_acceptance_rate"],
        }
    }
    with yaml_config.open("w") as io:
        yaml.safe_dump(config, io)

    args = [f"--yaml_config={yaml_config}", "--config=posterior"]
    run_posterior_analysis(args)

    posterior_step_results = {}
    posterior_step_results["ind"] = data.ind
    posterior_step_results["sn_name"] = data.sn_name
    posterior_step_results["spectra_id"] = data.spectra_id

    np.savez_compressed(savepath, **posterior_step_results)
    with np.load(savepath, allow_pickle=True) as io:
        return dict(io.items())


@pytest.fixture(scope="session")
def legacy_posterior_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
    tmp_path_factory: "TempPathFactory",
    legacy_data_step_factory: "DataStepFactory",
    legacy_pae_step_factory: "PAEStepFactory",
    legacy_nflow_step_factory: "NFlowStepFactory",
) -> "PosteriorStepFactory":
    def _legacy_posterior_step(
        data_params: "DataParams",
        pae_params: "PAEParams",
        nflow_params: "NFlowParams",
        posterior_params: "PosteriorParams",
    ) -> "PosteriorResults":
        posterior_params["data_path"] = data_path
        posterior_params["root_path"] = root_path
        posterior_params["cache_path"] = cache_path
        posterior_params["tmp_path"] = tmp_path_factory.mktemp("config")

        data = DataStepResult.model_validate(legacy_data_step_factory(data_params))
        pae = PAEStepResult.model_validate(
            legacy_pae_step_factory(data_params, pae_params)["6"]
        )
        nflow = NFlowStepResult.model_validate(
            legacy_nflow_step_factory(data_params, pae_params, nflow_params)
        )
        return legacy_posterior_step(posterior_params, data, pae, nflow)

    return _legacy_posterior_step


@pytest.fixture(scope="session")
def legacy_posterior_result_factory(
    legacy_posterior_step_factory: "PosteriorStepFactory",
) -> "PosteriorResultFactory":
    def _legacy_posterior_result(
        data_params: "DataParams",
        pae_params: "PAEParams",
        nflow_params: "NFlowParams",
        posterior_params: "PosteriorParams",
    ) -> "PosteriorStepResults":
        posterior_step_results = legacy_posterior_step_factory(
            data_params, pae_params, nflow_params, posterior_params
        )
        return PosteriorStepResult.model_validate(posterior_step_results)

    return _legacy_posterior_result
