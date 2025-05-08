from typing import TYPE_CHECKING

import yaml
import numpy as np
import pytest
import tensorflow as tf

from suPAErnova.configs.steps.pae import PAEStepResult
from suPAErnova.configs.steps.data import DataStepResult

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from _pytest.tmpdir import TempPathFactory


def legacy_pae_step(
    pae_params: dict[str, "Any"],
    data: "DataStepResult",
) -> dict[str, dict[str, "Any"]]:
    # Used cached result if it exists.
    savepath = (
        pae_params["cache_path"]
        / pae_params["fname"]
        / "pae"
        / "legacy"
        / "pae_step.npz"
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    if savepath.exists():
        with np.load(savepath, allow_pickle=True) as io:
            return {k: v.item() for k, v in io.items()}

    # Import here to avoid dependency conflicts
    from supaernova_legacy.models.losses import compute_loss_ae
    from supaernova_legacy.scripts.train_ae import train_ae

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

    data_out_path: Path = (
        pae_params["cache_path"] / pae_params["fname"] / "data" / "legacy"
    )
    pae_out_path: Path = (
        pae_params["cache_path"] / pae_params["fname"] / "pae" / "legacy"
    )
    model_dir = pae_out_path / "tensorflow_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    param_dir = pae_out_path / "params"
    param_dir.mkdir(parents=True, exist_ok=True)

    yaml_config = pae_params["tmp_path"] / "train.yaml"

    config = {
        "pae": {
            "PROJECT_DIR": str(pae_params["root_path"]),
            "MODEL_DIR": str(model_dir),
            "PARAM_DIR": str(param_dir),
            "train_data_file": str(data_out_path / "train" / "kfold0.npz"),
            "test_data_file": str(data_out_path / "test" / "kfold0.npz"),
            "verbose": True,
            "overfit": not pae_params["save_best"],
            "epochs": pae_params["delta_av_epochs"],
            "epochs_latent": pae_params["zs_epochs"],
            "epochs_final": pae_params["final_epochs"],
            "lr": pae_params["delta_av_lr"],
            "lr_decay_steps": pae_params["delta_av_lr_decay_steps"],
            "lr_decay_rate": pae_params["delta_av_lr_decay_rate"],
            "lr_deltat": pae_params["delta_p_lr"],
            "weight_decay_rate": pae_params["delta_av_lr_weight_decay_rate"],
            "min_train_redshift": pae_params["min_train_redshift"],
            "max_train_redshift": pae_params["max_train_redshift"],
            "max_light_cut": (
                pae_params["min_train_phase"],
                pae_params["max_train_phase"],
            ),
            "max_light_cut_spectra": (
                pae_params["min_train_phase"],
                pae_params["max_train_phase"],
            ),
            "latent_dims": (pae_params["n_z_latents"],),
            "seed": pae_params["seed"],
            "layer_type": pae_params["architecture"],
            "physical_latent": pae_params["physical_latents"],
            "val_every": pae_params["val_every"],
            "activation": pae_params["activation"].upper(),
            "loss_fn": pae_params["loss"].upper(),
            "optimizer": pae_params["optimiser"].upper(),
            "scheduler": pae_params["scheduler"].upper().replace("DECAY", ""),
            "kernel_regularizer": pae_params["kernel_regulariser"] is not None,
            "colorlaw_file": str(pae_params["colourlaw"]),
            "colorlaw_preset": pae_params["colourlaw"] is not None,
            "kfold": 0,
            "out_file_tail": "",
            "savemodel": True,
            "model_summary": False,
            "inverse_spectra_cut": False,
            "twins_cut": False,
            "use_val": pae_params["validation_frac"] > 0,
            "cond_dim": 1,
            "data_dim": 288,  # Only correct for the SNFactory data
            "n_timestep": 32,  # Only correct for the SNFactory data
            "encode_dims": (*pae_params["encode_dims"], 32),
            "decode_dims": (32, *reversed(pae_params["encode_dims"])),
            "batch_size": pae_params["batch_size"],
            "dropout": pae_params["dropout"] > 0,
            "batchnorm": pae_params["batch_normalisation"],
            "set_data_min_val": 0,
            "train_noise": pae_params["amplitude_offset_scale"] > 0,
            "noise_scale": pae_params["amplitude_offset_scale"],
            "train_time_uncertainty": pae_params["phase_offset_scale"] != 0,
            "vary_mask": pae_params["mask_fraction"] > 0,
            "mask_vary_frac": pae_params["mask_fraction"],
            "clip_delta": pae_params["loss_clip_delta"],
            "iloss_amplitude_offset": pae_params["loss_residual_penalty"] > 0,
            "lambda_amplitude_offset": pae_params["loss_residual_penalty"],
            "use_amplitude": pae_params["use_amplitude"],
            "iloss_amplitude_parameter": pae_params["loss_delta_m_penalty"] > 0,
            "lambda_amplitude_parameter": pae_params["loss_delta_m_penalty"],
            "iloss_covariance": pae_params["loss_covariance_penalty"] > 0,
            "lambda_covariance": pae_params["loss_covariance_penalty"],
            "decorrelate_dust": pae_params["loss_decorrelate_dust"],
            "decorrelate_all": pae_params["loss_decorrelate_all"],
            "train_latent_individual": pae_params["seperate_z_latent_training"],
        }
    }
    with yaml_config.open("w") as io:
        yaml.safe_dump(config, io)

    args = [f"--yaml_config={yaml_config}", "--config=pae"]
    results = train_ae(args)
    pae_step_results = {}
    for stage, (model, _params) in results.items():
        pae_step = {}
        pae_step["stage"] = stage
        pae_step["ind"] = data.ind
        pae_step["sn_name"] = data.sn_name
        pae_step["spectra_id"] = data.spectra_id

        x = data.amplitude
        cond = data.phase
        sigma = data.sigma
        mask = data.mask

        x = tf.cast(x, tf.float32)
        cond = tf.cast(cond, tf.float32)
        mask = tf.cast(mask, tf.float32)

        z = model.encode(x, cond, mask)
        x_pred = model.decode(z, cond, mask)
        pae_step["latents"] = z.numpy()
        pae_step["output_amp"] = x_pred.numpy()

        loss, (pred_loss, resid_loss, delta_loss, cov_loss, model_loss) = (
            compute_loss_ae(model, x, cond, sigma, mask)
        )
        pae_step["loss"] = loss
        pae_step["pred_loss"] = pred_loss
        pae_step["resid_loss"] = resid_loss
        pae_step["delta_loss"] = delta_loss
        pae_step["cov_loss"] = cov_loss
        pae_step["model_loss"] = model_loss

        pae_step_results[str(pae_step["stage"])] = pae_step

    np.savez_compressed(savepath, **pae_step_results)
    with np.load(savepath, allow_pickle=True) as io:
        return {k: v.item() for k, v in io.items()}


@pytest.fixture(scope="session")
def legacy_pae_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
    tmp_path_factory: "TempPathFactory",
    legacy_data_step_factory: "Callable[[dict[str, Any]], dict[str, Any]]",
) -> "Callable[[dict[str, Any], dict[str, Any]], dict[str, dict[str, Any]]]":
    def _legacy_pae_step(
        data_params: "dict[str, Any]", pae_params: "dict[str, Any]"
    ) -> "dict[str, dict[str, Any]]":
        pae_params["data_path"] = data_path
        pae_params["root_path"] = root_path
        pae_params["cache_path"] = cache_path
        pae_params["tmp_path"] = tmp_path_factory.mktemp("config")

        data = DataStepResult.model_validate(legacy_data_step_factory(data_params))
        return legacy_pae_step(pae_params, data)

    return _legacy_pae_step


@pytest.fixture(scope="session")
def legacy_pae_result_factory(
    legacy_pae_step_factory: "Callable[[dict[str, Any], dict[str, Any]], dict[str, dict[str, Any]]]",
) -> "Callable[[dict[str, Any], dict[str, Any]], dict[str, PAEStepResult]]":
    def _legacy_pae_result(
        data_params: dict[str, "Any"], pae_params: dict[str, "Any"]
    ) -> "dict[str, PAEStepResult]":
        pae_step_results = legacy_pae_step_factory(data_params, pae_params)
        return {
            stage: PAEStepResult.model_validate(pae_step)
            for stage, pae_step in pae_step_results.items()
        }

    return _legacy_pae_result
