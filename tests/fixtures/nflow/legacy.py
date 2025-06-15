from typing import TYPE_CHECKING

import yaml
import numpy as np
import pytest

from suPAErnova.configs.steps.pae import PAEStepResult
from suPAErnova.configs.steps.data import DataStepResult
from suPAErnova.configs.steps.nflow import NFlowStepResult
from suPAErnova.analysis.distribution import DistributionPlot, DistributionPlotter

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

    from . import (
        NFlowParams,
        NFlowResults,
        NFlowStepFactory,
        NFlowStepResults,
        NFlowResultFactory,
    )


def legacy_nflow_step(
    pae_params: "PAEParams",
    nflow_params: "NFlowParams",
    data: "DataStepResult",
    pae: "PAEStepResult",
) -> "NFlowResults":
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
            return dict(io.items())

    # Import here to avoid dependency conflicts
    from supaernova_legacy.utils.data_loader import (
        get_train_mask,
        get_train_mask_spectra,
    )
    from supaernova_legacy.scripts.train_flow import train_flow

    # Except where indicated, this is running the `train_flow` script verbatim

    # Variation: train_flow script modified to allow passing args as a list of strings
    # Variation: train_flow script modified to return a dictionary of results
    # Variation: train_flow script modified to skip training which has already been run
    # Variation: Legacy code used an old version of Tensorflow, and the syntax has since changed
    #            In particular:
    #             - `tf.tf_fn(x)` is no longer valid, so has been updated to `ks.layers.Lambda(tf.tf_fn)(x)`
    #             - Stricter dtype checks (int32 ~= int64 ~= float32 ~= float64). Using tf.cast where needed.
    # Variation: Legacy code used an old version of TensorflowProbability which is no longer compatible with tf.keras.
    #            Instead we need to use tf_keras
    # Variation: Legacy code was incorrectly loading models in, this has been fixed

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

    pae_model_dir = pae_out_path / "tensorflow_models"
    pae_model_dir.mkdir(parents=True, exist_ok=True)
    param_dir = pae_out_path / "params"
    param_dir.mkdir(parents=True, exist_ok=True)

    nflow_out_path: Path = (
        nflow_params["cache_path"]
        / nflow_params["fname"]
        / "nflow"
        / "legacy"
        / nflow_params["seed"]
    )
    nflow_model_dir = nflow_out_path / "tensorflow_models"
    nflow_model_dir.mkdir(parents=True, exist_ok=True)

    yaml_config = nflow_params["tmp_path"] / "train.yaml"

    config = {
        "nflow": {
            "PROJECT_DIR": str(nflow_params["root_path"]),
            "MODEL_DIR": str(pae_model_dir),
            "NFLOW_MODEL_DIR": str(nflow_model_dir) + "/",
            "PARAM_DIR": str(param_dir),
            "model_summary": False,
            "out_file_tail": "",
            "train_data_file": str(
                data_out_path / "train" / f"kfold{nflow_params['kfold']}.npz"
            ),
            "test_data_file": str(
                data_out_path / "test" / f"kfold{nflow_params['kfold']}.npz"
            ),
            "verbose": False,
            "kfold": nflow_params["kfold"],
            "seed": int(nflow_params["seed"]),
            "prev_train_stage": "5",
            "set_data_min_val": 0,
            "checkpoint_flow_every": 10,
            "patience": nflow_params["patience"],
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
    model, flow, params = train_flow(args)

    params["min_train_redshift"] = pae_params["min_train_redshift"]
    params["max_train_redshift"] = pae_params["max_train_redshift"]
    params["max_light_cut"] = (
        pae_params["min_train_phase"],
        pae_params["max_train_phase"],
    )
    params["max_light_cut_spectra"] = (
        pae_params["min_train_phase"],
        pae_params["max_train_phase"],
    )
    params["twins_cut"] = False
    params["inverse_spectra_cut"] = False

    data_mask_dict = {"redshift": data.redshift, "times_orig": data.phase}
    sn_mask = get_train_mask(data_mask_dict, params)[:, 0, 0:1]

    nflow_step_results = {}
    nflow_step_results["ind"] = data.ind
    nflow_step_results["sn_name"] = data.sn_name
    nflow_step_results["spectra_id"] = data.spectra_id

    inds = np.squeeze(sn_mask.astype(np.bool), axis=-1)

    # SNPAE latent ordering:
    # ΔAᵥ -> zs -> Δℳ  -> Δ𝓅 ([0, 1, 2, 3, 4, 5])
    # Legacy latent ordering:
    # Δ𝓅 -> Δℳ  -> ΔAᵥ -> zs ([5, 4, 0, 1, 2, 3])
    z = pae.latents[:, [5, 4, 0, 1, 2, 3]][inds]

    z = z[:, -4:]
    if not params["use_extrinsic_params"]:
        z = z[:, 1:]
    nflow_step_results["latents"] = z

    nflow_step_results["log_prob"] = -model(z)

    u = flow.bijector.inverse(z)
    nflow_step_results["z_to_u"] = u
    nflow_step_results["u_to_z"] = flow.bijector.forward(u)

    np.savez_compressed(savepath, **nflow_step_results)
    with np.load(savepath, allow_pickle=True) as io:
        return dict(io.items())


@pytest.fixture(scope="session")
def legacy_nflow_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
    tmp_path_factory: "TempPathFactory",
    legacy_data_step_factory: "DataStepFactory",
    legacy_pae_step_factory: "PAEStepFactory",
) -> "NFlowStepFactory":
    def _legacy_nflow_step(
        data_params: "DataParams",
        pae_params: "PAEParams",
        nflow_params: "NFlowParams",
    ) -> "NFlowResults":
        nflow_params["data_path"] = data_path
        nflow_params["root_path"] = root_path
        nflow_params["cache_path"] = cache_path
        nflow_params["tmp_path"] = tmp_path_factory.mktemp("config")

        data = DataStepResult.model_validate(legacy_data_step_factory(data_params)[0])
        pae = PAEStepResult.model_validate(
            legacy_pae_step_factory(data_params, pae_params)[0]["6"]
        )
        return legacy_nflow_step(pae_params, nflow_params, data, pae), nflow_params

    return _legacy_nflow_step


@pytest.fixture(scope="session")
def legacy_nflow_result_factory(
    legacy_nflow_step_factory: "NFlowStepFactory",
) -> "NFlowResultFactory":
    def _legacy_nflow_result(
        data_params: "DataParams",
        pae_params: "PAEParams",
        nflow_params: "NFlowParams",
    ) -> "NFlowStepResults":
        nflow_step_results, params = legacy_nflow_step_factory(
            data_params, pae_params, nflow_params
        )

        results = NFlowStepResult.model_validate(nflow_step_results)

        savepath = (
            params["cache_path"]
            / params["fname"]
            / "nflow"
            / "legacy"
            / "plots"
            / params["seed"]
        )
        savepath.mkdir(parents=True, exist_ok=True)

        z_labels = {}
        u_labels = {}
        ind = 0
        if params["physical_latents"]:
            z_labels[0] = "ΔAᵥ"
            u_labels[0] = "μΔAᵥ"
            ind = 1
        for i in range(params["n_z_latents"]):
            z_labels[ind] = f"z{i}"
            u_labels[ind] = f"μ{i}"
            ind += 1

        plot_u_latents = DistributionPlot.model_validate({
            "labels": u_labels,
            "name": "u_latents",
            "savepath": savepath,
        })
        DistributionPlotter.plot_corner(results.z_to_u, plot_u_latents)

        plot_z_latents = DistributionPlot.model_validate({
            "labels": z_labels,
            "name": "z_latents",
            "savepath": savepath,
        })
        DistributionPlotter.plot_corner(results.u_to_z, plot_z_latents)

        return results

    return _legacy_nflow_result
