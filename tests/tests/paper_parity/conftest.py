from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt

    from suPAErnova.configs.steps.pae import PAEStepResult
    from suPAErnova.configs.steps.data import DataStepResult

pytestmark = pytest.mark.paper_parity

# --- Constants ---
SEEDS = ["12345"]  # , "23456", "34567", "45678", "56789"]


@pytest.fixture(scope="module", params=SEEDS)
def seed(request) -> str:
    return request.param


# --- Utilities ---
class PaperParityUtils:
    @staticmethod
    def compare_arrays(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        *,
        snpae_sigma: "float | npt.NDArray[Any] | None" = None,
        legacy_sigma: "float | npt.NDArray[Any] | None" = None,
    ) -> "npt.NDArray[np.bool]":
        if np.issubdtype(snpae.dtype, np.number) and np.issubdtype(
            legacy.dtype, np.number
        ):
            if (snpae_sigma is not None) and (legacy_sigma is not None):
                atol = np.sqrt(snpae_sigma * snpae_sigma + legacy_sigma * legacy_sigma)
            else:
                atol = 1e-8
            return np.isclose(snpae, legacy, atol=atol)
        return snpae == legacy

    @staticmethod
    def diff_arrays(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        diff_mask: "npt.NDArray[np.bool]",
        *,
        max_diffs: int = 5,
        sort: bool = True,
        spectra: "npt.NDArray[np.str_] | None" = None,
        snpae_sigma: "npt.NDArray[Any] | None" = None,
        legacy_sigma: "npt.NDArray[Any] | None" = None,
    ) -> str:
        eps = 1e-10

        # Element-wise comparison
        diff_indices = np.argwhere(diff_mask)

        # Summary Statistics
        diff = snpae - legacy
        abs_diff = np.abs(diff)
        min_abs_diff = abs_diff.min()
        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()
        std_abs_diff = abs_diff.std()
        abs_sort_mask = np.argsort(abs_diff[diff_mask])[::-1]

        rel_diff = 2 * diff / (snpae + legacy + eps)
        min_rel_diff = rel_diff.min()
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()
        std_rel_diff = rel_diff.std()
        rel_sort_mask = np.argsort(rel_diff[diff_mask])[::-1]

        s = f"{len(diff_indices)} differences ({int(100 * len(diff_indices) / diff_mask.size)}%):\n"
        s += f"Absolute Difference - min: {min_abs_diff:.2f}, max: {max_abs_diff:.2f}, mean±std: {mean_abs_diff:.2f}±{std_abs_diff:.2f}\n"
        s += f"Relative Difference - min: {min_rel_diff:.2%}, max: {max_rel_diff:.2%}, mean±std: {mean_rel_diff:.2%}±{std_rel_diff:.2%}\n"

        if sort:
            indices = (
                list(diff_indices[abs_sort_mask][:max_diffs])
                + list(diff_indices[rel_sort_mask][:max_diffs])
                + list(diff_indices[rel_sort_mask[::-1]][:max_diffs])
            )
        else:
            indices = diff_indices[:max_diffs]

        for idx in indices:
            snpae_val = snpae[tuple(idx)]
            legacy_val = legacy[tuple(idx)]
            snpae_val_sigma = None if snpae_sigma is None else snpae_sigma[tuple(idx)]
            legacy_val_sigma = (
                None if legacy_sigma is None else legacy_sigma[tuple(idx)]
            )

            # Summary Statistics
            val_diff = snpae_val - legacy_val
            abs_val_diff = abs(val_diff)
            rel_val_diff = 2 * val_diff / (snpae_val + legacy_val + eps)

            s += f"  At index {[int(i) for i in tuple(idx)]}{f' ({spectra[idx[0]][idx[1]][0]})' if spectra is not None else ''}:\n"
            s += f"    snpae = {snpae_val:.2f}{f'±{snpae_val_sigma:.4f}' if snpae_val_sigma is not None else ''}, legacy = {legacy_val:.2f}{f'±{legacy_val_sigma:.4f}' if legacy_val_sigma is not None else ''}\n"
            s += f"    abs diff = {abs_val_diff:.2f}, rel diff = {rel_val_diff:.2%}\n"

        if len(diff_indices) > max_diffs:
            s += f"... and {len(diff_indices) - max_diffs} more differences.\n"

        return s

    @staticmethod
    def assert_arrays(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        *,
        max_diffs: int = 5,
        sort: bool = True,
        spectra: "npt.NDArray[np.str_] | None" = None,
        snpae_sigma: "npt.NDArray[Any] | None" = None,
        legacy_sigma: "npt.NDArray[Any] | None" = None,
    ) -> None:
        diff_mask = PaperParityUtils.compare_arrays(
            snpae, legacy, snpae_sigma=snpae_sigma, legacy_sigma=legacy_sigma
        )
        assert diff_mask.all(), PaperParityUtils.diff_arrays(
            snpae,
            legacy,
            np.logical_not(diff_mask),
            max_diffs=max_diffs,
            sort=sort,
            spectra=spectra,
            snpae_sigma=snpae_sigma,
            legacy_sigma=legacy_sigma,
        )


@pytest.fixture(scope="module")
def utils() -> PaperParityUtils:
    return PaperParityUtils()


# --- Data Step ---


@pytest.fixture(scope="module")
def data_params() -> dict[str, "Any"]:
    return {
        "min_phase": -10,
        "max_phase": 40,
        "train_frac": 0.75,
        "seed": 12345,
        "fname": "paper_parity",
        "cosmological_model": "WMAP7",
        "salt_model": "salt2",
    }


@pytest.fixture(scope="module")
def snpae_data(
    data_params: dict[str, "Any"],
    snpae_data_result_factory: "Callable[[dict[str, Any]], DataStepResult]",
) -> "DataStepResult":
    return snpae_data_result_factory(data_params)


@pytest.fixture(scope="module")
def legacy_data(
    data_params: dict[str, "Any"],
    legacy_data_result_factory: "Callable[[dict[str, Any]], DataStepResult]",
) -> "DataStepResult":
    return legacy_data_result_factory(data_params)


# --- PAE Step ---


@pytest.fixture(scope="module")
def pae_params(
    data_path: "Path",
) -> dict[str, "Any"]:
    return {
        "fname": "paper_parity",
        "validation_frac": 0,
        "save_best": False,
        "batch_size": 57,  # Only correct for the SNFactory data
        "val_every": 100,
        # Epochs
        "delta_av_epochs": 1000,
        "zs_epochs": 1000,
        "delta_m_epochs": 5000,
        "delta_p_epochs": 5000,
        "final_epochs": 5000,
        # Learning Rate
        "delta_av_lr": 0.005,
        "zs_lr": 0.005,
        "delta_m_lr": 0.005,
        "delta_p_lr": 0.001,
        "final_lr": 0.001,
        # Decay Steps
        "delta_av_lr_decay_steps": 300,
        "zs_lr_decay_steps": 300,
        "delta_m_lr_decay_steps": 300,
        "delta_p_lr_decay_steps": 300,
        "final_lr_decay_steps": 300,
        # Decay Rate
        "delta_av_lr_decay_rate": 0.95,
        "zs_lr_decay_rate": 0.95,
        "delta_m_lr_decay_rate": 0.95,
        "delta_p_lr_decay_rate": 0.95,
        "final_lr_decay_rate": 0.95,
        # Weight Decay Rate
        "delta_av_lr_weight_decay_rate": 0.0001,
        "zs_lr_weight_decay_rate": 0.0001,
        "delta_m_lr_weight_decay_rate": 0.0001,
        "delta_p_lr_weight_decay_rate": 0.0001,
        "final_lr_weight_decay_rate": 0.0001,
        # Redshift
        "min_train_redshift": 0.02,
        "min_test_redshift": 0.02,
        "min_val_redshift": 0.02,
        "max_train_redshift": 1.0,
        "max_test_redshift": 1.0,
        "max_val_redshift": 1.0,
        # Phase
        "min_train_phase": -10,
        "min_test_phase": -10,
        "min_val_phase": -10,
        "max_train_phase": 40,
        "max_test_phase": 40,
        "max_val_phase": 40,
        # Latents
        "n_z_latents": 3,
        "physical_latents": True,
        "seperate_latent_training": True,
        "seperate_z_latent_training": False,
        # Architecture
        "architecture": "dense",
        "encode_dims": (256, 128),
        "dropout": 0,
        "batch_normalisation": False,
        # NN Functions
        "activation": "relu",
        "loss": "WHuber",
        "optimiser": "AdamW",
        "scheduler": "ExponentialDecay",
        "colourlaw": data_path / "colourlaws" / "F99_colourlaw.txt",
        "kernel_regulariser": "L2",
        "kernel_regulariser_penalty": 100,  # Not used if kernel_regulariser is None
        # Noise
        "amplitude_offset_scale": 1.0,
        "phase_offset_scale": -0.02,
        "mask_fraction": 0.1,
        # Loss
        "use_amplitude": True,
        "loss_clip_delta": 25,
        "loss_residual_penalty": 0,
        "loss_delta_m_penalty": 0,
        "loss_delta_av_penalty": 0,
        "loss_delta_p_penalty": 0,
        "loss_covariance_penalty": 50000,  # Seems excessive
        "loss_decorrelate_all": True,
        "loss_decorrelate_dust": True,
    }


@pytest.fixture(scope="module")
def snpae_pae(
    seed: str,
    data_params: dict[str, "Any"],
    pae_params: dict[str, "Any"],
    snpae_pae_result_factory: "Callable[[dict[str, Any], dict[str, Any]], PAEStepResult]",
) -> "PAEStepResult":
    pae_params["seed"] = seed
    return snpae_pae_result_factory(data_params, pae_params)


@pytest.fixture(scope="module")
def legacy_pae(
    seed: str,
    data_params: dict[str, "Any"],
    pae_params: dict[str, "Any"],
    legacy_pae_result_factory: "Callable[[dict[str, Any], dict[str, Any]], list[PAEStepResult]]",
) -> "list[PAEStepResult]":
    pae_params["seed"] = seed
    return legacy_pae_result_factory(data_params, pae_params)


# --- NFlow Step ---


@pytest.fixture(scope="module")
def nflow_params(pae_params: dict[str, "Any"]) -> dict[str, "Any"]:
    return {
        "fname": "paper_parity",
        "n_z_latents": pae_params["n_z_latents"],
        "encode_dims": pae_params["encode_dims"],
        "save_best": pae_params["save_best"],
        "batch_size": pae_params["batch_size"],
        "learning_rate": 0.001,
        "physical_latents": True,
        "n_hidden_units": 8,
        "n_layers": 12,
        "epochs": 500,
        "batch_normalisation": False,
        "validation_frac": 0.22,
    }


# @pytest.fixture(scope="module")
# def snpae_nflow(
#     seed: str,
#     data_params: dict[str, "Any"],
#     pae_params: dict[str, "Any"],
#     nflow_params: dict[str, "Any"],
#     snpae_nflow_result_factory: "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], PAEStepResult]",
# ) -> "PAEStepResult":
#     pae_params["seed"] = seed
#     nflow_params["seed"] = seed
#     return snpae_nflow_result_factory(data_params, pae_params, nflow_params)
#
#
@pytest.fixture(scope="module")
def legacy_nflow(
    seed: str,
    data_params: dict[str, "Any"],
    pae_params: dict[str, "Any"],
    nflow_params: dict[str, "Any"],
    legacy_nflow_result_factory: "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], list[PAEStepResult]]",
) -> "list[PAEStepResult]":
    pae_params["seed"] = seed
    nflow_params["seed"] = seed
    return legacy_nflow_result_factory(data_params, pae_params, nflow_params)
