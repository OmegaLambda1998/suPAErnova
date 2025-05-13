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
SEEDS = ["12345", "23456", "34567", "45678", "56789"]


# --- Utilities ---
class PaperParityUtils:
    @staticmethod
    def compare_arrays(snpae: "npt.NDArray[Any]", legacy: "npt.NDArray[Any]") -> bool:
        compare = (
            np.allclose
            if np.issubdtype(snpae.dtype, np.number)
            and np.issubdtype(legacy.dtype, np.number)
            else np.array_equal
        )
        return compare(snpae, legacy)

    @staticmethod
    def diff_arrays(
        snpae: "npt.NDArray[Any]", legacy: "npt.NDArray[Any]", max_diffs: int = 10
    ) -> str:
        # Element-wise comparison
        diff_mask = snpae != legacy
        sort_mask = np.argsort(np.abs(snpae - legacy)[diff_mask])[::-1]
        diff_indices = np.argwhere(diff_mask)[sort_mask]

        # Summary Statistics
        diff = snpae - legacy
        abs_diff = abs(diff)

        min_abs_diff = abs_diff.min()
        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()
        std_abs_diff = abs_diff.std()

        s = f"Arrays differ at {len(diff_indices)} positions ({int(100 * len(diff_indices) / diff_mask.size)}%):\n"
        s += f"Absolute Difference - min: {min_abs_diff:2f}, max: {max_abs_diff:2f}, mean±std: {mean_abs_diff:2f}±{std_abs_diff:2f}\n"
        for idx in diff_indices[:max_diffs]:
            snpae_val = snpae[tuple(idx)]
            legacy_val = legacy[tuple(idx)]

            # Summary Statistics
            val_diff = snpae_val - legacy_val
            abs_val_diff = abs(val_diff)

            s += f"  At index {tuple(idx)}: snpae = {snpae_val:2f}, legacy = {legacy_val:2f}, abs diff = {abs_val_diff:2f}\n"

        if len(diff_indices) > max_diffs:
            s += f"... and {len(diff_indices) - max_diffs} more differences.\n"

        return s

    @staticmethod
    def assert_arrays(snpae: "npt.NDArray[Any]", legacy: "npt.NDArray[Any]") -> None:
        assert PaperParityUtils.compare_arrays(snpae, legacy), (
            PaperParityUtils.diff_arrays(snpae, legacy)
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
        "seed": 12345,
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
        "kernel_regulariser": None,
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


@pytest.fixture(scope="module", params=SEEDS)
def pae_seed(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def snpae_pae(
    pae_seed: str,
    data_params: dict[str, "Any"],
    pae_params: dict[str, "Any"],
    snpae_pae_result_factory: "Callable[[dict[str, Any], dict[str, Any]], PAEStepResult]",
) -> "PAEStepResult":
    pae_params["seed"] = pae_seed
    return snpae_pae_result_factory(data_params, pae_params)


@pytest.fixture(scope="module")
def legacy_pae(
    pae_seed: str,
    data_params: dict[str, "Any"],
    pae_params: dict[str, "Any"],
    legacy_pae_result_factory: "Callable[[dict[str, Any], dict[str, Any]], list[PAEStepResult]]",
) -> "list[PAEStepResult]":
    pae_params["seed"] = pae_seed
    return legacy_pae_result_factory(data_params, pae_params)
