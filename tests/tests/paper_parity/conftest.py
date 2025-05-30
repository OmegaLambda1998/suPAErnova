from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path

    from numpy import typing as npt

    from tests.fixtures.pae import PAEParams, PAEStepResults, PAEResultFactory
    from tests.fixtures.data import DataParams, DataStepResults, DataResultFactory
    from tests.fixtures.nflow import NFlowParams, NFlowStepResults, NFlowResultFactory
    from tests.fixtures.posterior import (
        PosteriorParams,
        PosteriorStepResults,
        PosteriorResultFactory,
    )

pytestmark = pytest.mark.paper_parity

# --- Constants ---
SEEDS = list(enumerate(("12345", "23456", "34567", "45678")))


@pytest.fixture(scope="module", params=SEEDS)
def seed(request) -> tuple[int, str]:
    return request.param


# --- Utilities ---
class PaperParityUtils:
    @staticmethod
    def equal(
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
                atol = 0.5 * (snpae_sigma + legacy_sigma)
            else:
                atol = 1e-8
            return np.isclose(snpae, legacy, atol=atol)
        return snpae == legacy

    @staticmethod
    def le(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        *,
        _snpae_sigma: "float | npt.NDArray[Any] | None" = None,
        _legacy_sigma: "float | npt.NDArray[Any] | None" = None,
    ) -> "npt.NDArray[np.bool]":
        return snpae <= legacy

    @staticmethod
    def lt(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        *,
        _snpae_sigma: "float | npt.NDArray[Any] | None" = None,
        _legacy_sigma: "float | npt.NDArray[Any] | None" = None,
    ) -> "npt.NDArray[np.bool]":
        return snpae < legacy

    @staticmethod
    def ge(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        *,
        _snpae_sigma: "float | npt.NDArray[Any] | None" = None,
        _legacy_sigma: "float | npt.NDArray[Any] | None" = None,
    ) -> "npt.NDArray[np.bool]":
        return snpae >= legacy

    @staticmethod
    def gt(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        *,
        _snpae_sigma: "float | npt.NDArray[Any] | None" = None,
        _legacy_sigma: "float | npt.NDArray[Any] | None" = None,
    ) -> "npt.NDArray[np.bool]":
        return snpae < legacy

    @staticmethod
    def context(
        snpae: "npt.NDArray[Any]",
        legacy: "npt.NDArray[Any]",
        diff_mask: "npt.NDArray[np.bool]",
        metadata: dict[str, "Any"],
        *,
        max_diffs: int = 5,
        sort: bool = True,
        spectra: "npt.NDArray[np.str_] | None" = None,
        snpae_sigma: "npt.NDArray[Any] | None" = None,
        legacy_sigma: "npt.NDArray[Any] | None" = None,
    ) -> str:
        s = ""
        for k, v in metadata.items():
            s += f"{k}: {v}\n"

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

        s += f"{len(diff_indices)} differences ({int(100 * len(diff_indices) / diff_mask.size)}%):\n"
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
            s += f"    snpae = {snpae_val:.4f}{f'±{snpae_val_sigma:.4f}' if snpae_val_sigma is not None else ''}, legacy = {legacy_val:.4f}{f'±{legacy_val_sigma:.4f}' if legacy_val_sigma is not None else ''}\n"
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
        metadata: dict[str, "Any"] | None = None,
        compare: "Any" = None,
    ) -> None:
        if metadata is None:
            metadata = {}

        if compare is None:
            compare = PaperParityUtils.equal

        diff_mask = compare(
            snpae, legacy, snpae_sigma=snpae_sigma, legacy_sigma=legacy_sigma
        )

        assert diff_mask.all(), PaperParityUtils.context(
            snpae,
            legacy,
            np.logical_not(diff_mask),
            metadata,
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
def data_params() -> "DataParams":
    return {
        "min_phase": -10,
        "max_phase": 40,
        "train_frac": 0.75,
        "fname": "paper_parity",
        "cosmological_model": "WMAP7",
        "salt_model": "salt2",
        "seed": SEEDS[0][-1],
    }


@pytest.fixture(scope="module")
def snpae_data(
    data_params: "DataParams",
    snpae_data_result_factory: "DataResultFactory",
) -> "DataStepResults":
    result = snpae_data_result_factory(data_params)
    if result.metadata is None:
        result.metadata = {}
    return result


@pytest.fixture(scope="module")
def legacy_data(
    data_params: "DataParams",
    legacy_data_result_factory: "DataResultFactory",
) -> "DataStepResults":
    result = legacy_data_result_factory(data_params)
    if result.metadata is None:
        result.metadata = {}
    return result


# --- PAE Step ---


@pytest.fixture(scope="module")
def pae_params(
    data_path: "Path",
) -> "PAEParams":
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
    seed: tuple[int, str],
    data_params: "DataParams",
    pae_params: "PAEParams",
    snpae_pae_result_factory: "PAEResultFactory",
) -> "PAEStepResults":
    pae_params["kfold"], pae_params["seed"] = seed
    results = snpae_pae_result_factory(data_params, pae_params)
    for result in results.values():
        if result.metadata is None:
            result.metadata = {}
        result.metadata["seed"] = pae_params["seed"]
    return results


@pytest.fixture(scope="module")
def legacy_pae(
    seed: tuple[int, str],
    data_params: "DataParams",
    pae_params: "PAEParams",
    legacy_pae_result_factory: "PAEResultFactory",
) -> "PAEStepResults":
    pae_params["kfold"], pae_params["seed"] = seed
    results = legacy_pae_result_factory(data_params, pae_params)
    for result in results.values():
        if result.metadata is None:
            result.metadata = {}
        result.metadata["seed"] = pae_params["seed"]
    return results


# --- NFlow Step ---


@pytest.fixture(scope="module")
def nflow_params(pae_params: "PAEParams") -> "NFlowParams":
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
        "activation": "relu",
        "optimiser": "Adam",
    }


@pytest.fixture(scope="module")
def snpae_nflow(
    seed: tuple[int, str],
    data_params: "DataParams",
    pae_params: "PAEParams",
    nflow_params: "NFlowParams",
    snpae_nflow_result_factory: "NFlowResultFactory",
) -> "NFlowStepResults":
    pae_params["kfold"], pae_params["seed"] = seed
    nflow_params["kfold"], nflow_params["seed"] = seed
    result = snpae_nflow_result_factory(data_params, pae_params, nflow_params)
    if result.metadata is None:
        result.metadata = {}
    result.metadata["seed"] = nflow_params["seed"]
    return result


@pytest.fixture(scope="module")
def legacy_nflow(
    seed: tuple[int, str],
    data_params: "DataParams",
    pae_params: "PAEParams",
    nflow_params: "NFlowParams",
    legacy_nflow_result_factory: "NFlowResultFactory",
) -> "NFlowStepResults":
    pae_params["kfold"], pae_params["seed"] = seed
    nflow_params["kfold"], nflow_params["seed"] = seed
    result = legacy_nflow_result_factory(data_params, pae_params, nflow_params)
    if result.metadata is None:
        result.metadata = {}
    result.metadata["seed"] = nflow_params["seed"]
    return result


# --- Posterior Step ---


@pytest.fixture(scope="module")
def posterior_params(
    pae_params: "PAEParams", nflow_params: "NFlowParams"
) -> "PosteriorParams":
    return {"fname": "paper_parity"}


@pytest.fixture(scope="module")
def snpae_posterior(
    seed: tuple[int, str],
    data_params: "DataParams",
    pae_params: "PAEParams",
    nflow_params: "NFlowParams",
    posterior_params: "PosteriorParams",
    snpae_posterior_result_factory: "PosteriorResultFactory",
) -> "PosteriorStepResults":
    pae_params["kfold"], pae_params["seed"] = seed
    nflow_params["kfold"], nflow_params["seed"] = seed
    posterior_params["kfold"], posterior_params["seed"] = seed
    result = snpae_posterior_result_factory(
        data_params, pae_params, nflow_params, posterior_params
    )
    if result.metadata is None:
        result.metadata = {}
    result.metadata["seed"] = nflow_params["seed"]
    return result


@pytest.fixture(scope="module")
def legacy_posterior(
    seed: tuple[int, str],
    data_params: "DataParams",
    pae_params: "PAEParams",
    nflow_params: "NFlowParams",
    posterior_params: "PosteriorParams",
    legacy_posterior_result_factory: "PosteriorResultFactory",
) -> "PosteriorStepResults":
    pae_params["kfold"], pae_params["seed"] = seed
    nflow_params["kfold"], nflow_params["seed"] = seed
    posterior_params["kfold"], posterior_params["seed"] = seed
    result = legacy_posterior_result_factory(
        data_params, pae_params, nflow_params, posterior_params
    )
    if result.metadata is None:
        result.metadata = {}
    result.metadata["seed"] = nflow_params["seed"]
    return result
