from typing import TYPE_CHECKING

import numpy as np
import pytest

from suPAErnova.configs.steps.pae import PAEStepResult

if TYPE_CHECKING:
    from tests.fixtures.pae import PAEStepResults
    from tests.tests.paper_parity.conftest import PaperParityUtils

pytestmark = pytest.mark.pae

STAGES = ["1", "4", "5", "6"]
KEYS = list(PAEStepResult.model_fields.keys())
LOSSES = [k for k in KEYS if "loss" in k]
AMPS = [k for k in KEYS if "_amp" in k]
WEIGHTS = [k for k in KEYS if "_weights" in k]
REMAINING = [
    k for k in KEYS if k not in LOSSES + AMPS + WEIGHTS + ["latents", "metadata"]
]


@pytest.mark.setup("snpae")
def test_snpae_pae_setup(snpae_pae: "PAEStepResults") -> None:
    pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_pae_setup(legacy_pae: "PAEStepResults") -> None:
    pass


@pytest.mark.parametrize("stage", STAGES)
def test_snpae_stages(
    stage: str,
    snpae_pae: "PAEStepResults",
) -> None:
    assert stage in snpae_pae


@pytest.mark.parametrize("stage", STAGES)
def test_legacy_stages(
    stage: str,
    legacy_pae: "PAEStepResults",
) -> None:
    assert stage in legacy_pae


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("key", REMAINING)
def test_shapes(
    stage: str,
    key: str,
    snpae_pae: "PAEStepResults",
    legacy_pae: "PAEStepResults",
) -> None:
    snpae_vals = np.array(getattr(snpae_pae[stage], key))
    legacy_vals = np.array(getattr(legacy_pae[stage], key))
    assert snpae_vals.shape == legacy_vals.shape


@pytest.mark.parametrize("key", REMAINING)
@pytest.mark.parametrize("stage", STAGES)
def test_matching_remaining(
    stage: str,
    key: str,
    snpae_pae: "PAEStepResults",
    legacy_pae: "PAEStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_pae[stage], key))
    legacy_vals = np.array(getattr(legacy_pae[stage], key))
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        spectra=snpae_pae[stage].spectra_id,
        metadata={
            **snpae_pae[stage].metadata,
            **legacy_pae[stage].metadata,
            "stage": stage,
            "key": key,
        },
    )


@pytest.mark.parametrize("key", LOSSES)
@pytest.mark.parametrize("stage", STAGES)
def test_matching_losses(
    stage: str,
    key: str,
    snpae_pae: "PAEStepResults",
    legacy_pae: "PAEStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_pae[stage], key))
    legacy_vals = np.array(getattr(legacy_pae[stage], key))
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={
            **snpae_pae[stage].metadata,
            **legacy_pae[stage].metadata,
            "stage": stage,
            "key": key,
        },
    )


@pytest.mark.parametrize("key", AMPS)
@pytest.mark.parametrize("stage", STAGES)
def test_matching_amps(
    stage: str,
    key: str,
    snpae_pae: "PAEStepResults",
    legacy_pae: "PAEStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_pae[stage], key))
    snpae_sigma = np.array(snpae_pae[stage].input_d_amp)
    legacy_vals = np.array(getattr(legacy_pae[stage], key))
    legacy_sigma = np.array(legacy_pae[stage].input_d_amp)
    data_vals = np.array(snpae_pae[stage].input_amp)
    data_sigma = np.array(snpae_pae[stage].input_d_amp)
    legacy_sigma = np.array(legacy_pae[stage].input_d_amp)
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        spectra=snpae_pae[stage].spectra_id,
        snpae_sigma=snpae_sigma,
        legacy_sigma=legacy_sigma,
        data=data_vals,
        data_sigma=data_sigma,
        metadata={
            **snpae_pae[stage].metadata,
            **legacy_pae[stage].metadata,
            "stage": stage,
            "key": key,
        },
    )


@pytest.mark.parametrize("key", AMPS)
@pytest.mark.parametrize("stage", STAGES)
def test_matching_amp_means(
    stage: str,
    key: str,
    snpae_pae: "PAEStepResults",
    legacy_pae: "PAEStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_pae[stage], key))
    legacy_vals = np.array(getattr(legacy_pae[stage], key))
    data_vals = np.array(snpae_pae[stage].input_amp)
    utils.assert_arrays(
        snpae_vals.mean(axis=(0, 1)),
        legacy_vals.mean(axis=(0, 1)),
        snpae_sigma=snpae_vals.std(axis=(0, 1)),
        legacy_sigma=legacy_vals.std(axis=(0, 1)),
        data=data_vals.mean(axis=(0, 1)),
        data_sigma=data_vals.std(axis=(0, 1)),
        metadata={
            **snpae_pae[stage].metadata,
            **legacy_pae[stage].metadata,
            "stage": stage,
            "key": key,
        },
    )


@pytest.mark.parametrize("stage", STAGES)
def test_matching_latents(
    stage: str,
    snpae_pae: "PAEStepResults",
    legacy_pae: "PAEStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_pae[stage].latents)[:, : int(stage)]
    legacy_vals = np.array(legacy_pae[stage].latents)[:, : int(stage)]
    utils.assert_arrays(
        abs(snpae_vals),
        abs(legacy_vals),
        metadata={
            **snpae_pae[stage].metadata,
            **legacy_pae[stage].metadata,
            "stage": stage,
        },
    )


@pytest.mark.parametrize("stage", STAGES)
def test_matching_latent_means(
    stage: str,
    snpae_pae: "PAEStepResults",
    legacy_pae: "PAEStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_pae[stage].latents)[:, : int(stage)]
    legacy_vals = np.array(legacy_pae[stage].latents)[:, : int(stage)]
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        max_diffs=6,
        sort=False,
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
        metadata={
            **snpae_pae[stage].metadata,
            **legacy_pae[stage].metadata,
            "stage": stage,
        },
    )
