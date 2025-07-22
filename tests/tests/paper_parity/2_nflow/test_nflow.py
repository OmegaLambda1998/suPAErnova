from typing import TYPE_CHECKING

import numpy as np
import pytest

from supaernova.configs.steps.nflow import NFlowStepResult

if TYPE_CHECKING:
    from tests.fixtures.nflow import NFlowStepResults
    from tests.tests.paper_parity.conftest import PaperParityUtils

pytestmark = pytest.mark.nflow

KEYS = [
    k
    for k in NFlowStepResult.model_fields
    if k not in {"log_prob", "z_to_u", "u_to_z", "latents"}
]


@pytest.mark.setup("snpae")
def test_snpae_nflow_setup(snpae_nflow: "NFlowStepResults") -> None:
    pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_nflow_setup(legacy_nflow: "NFlowStepResults") -> None:
    pass


@pytest.mark.parametrize("key", KEYS)
def test_shapes(
    key: str,
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
) -> None:
    snpae_vals = np.array(getattr(snpae_nflow, key))
    legacy_vals = np.array(getattr(legacy_nflow, key))
    assert snpae_vals.shape == legacy_vals.shape


@pytest.mark.parametrize("key", KEYS)
def test_matching_remaining(
    key: str,
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_nflow, key))
    legacy_vals = np.array(getattr(legacy_nflow, key))
    utils.assert_arrays(snpae_vals, legacy_vals, metadata={"key": key})


def test_matching_latents(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.latents)
    legacy_vals = np.array(legacy_nflow.latents)
    utils.assert_arrays(
        abs(snpae_vals),
        abs(legacy_vals),
        metadata={
            **snpae_nflow.metadata,
            **legacy_nflow.metadata,
            "key": "latents",
        },
    )


def test_matching_latents_means(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.latents)
    legacy_vals = np.array(legacy_nflow.latents)
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
        metadata={
            **snpae_nflow.metadata,
            **legacy_nflow.metadata,
            "key": "mean latents",
        },
        sort=False,
    )


def test_matching_neg_log_prob(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.log_prob)
    legacy_vals = np.array(legacy_nflow.log_prob)
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={
            **snpae_nflow.metadata,
            **legacy_nflow.metadata,
            "key": "neg_log_prob",
        },
    )


def test_matching_neg_log_prob_means(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.log_prob)
    legacy_vals = np.array(legacy_nflow.log_prob)
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
        metadata={
            **snpae_nflow.metadata,
            **legacy_nflow.metadata,
            "key": "mean neg_log_prob",
        },
    )


def test_matching_z_to_u(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.z_to_u)
    legacy_vals = np.array(legacy_nflow.z_to_u)
    utils.assert_arrays(
        abs(snpae_vals),
        abs(legacy_vals),
        metadata={**snpae_nflow.metadata, **legacy_nflow.metadata, "key": "z_to_u"},
    )


def test_matching_z_to_u_means(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.z_to_u)
    legacy_vals = np.array(legacy_nflow.z_to_u)
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
        metadata={
            **snpae_nflow.metadata,
            **legacy_nflow.metadata,
            "key": "mean z_to_u",
        },
        sort=False,
    )


def test_matching_u_to_z(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.u_to_z)
    legacy_vals = np.array(legacy_nflow.u_to_z)
    utils.assert_arrays(
        abs(snpae_vals),
        abs(legacy_vals),
        metadata={**snpae_nflow.metadata, **legacy_nflow.metadata, "key": "u_to_z"},
    )


def test_matching_u_to_z_means(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.u_to_z)
    legacy_vals = np.array(legacy_nflow.u_to_z)
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
        metadata={
            **snpae_nflow.metadata,
            **legacy_nflow.metadata,
            "key": "mean u_to_z",
        },
        sort=False,
    )


def test_matching_z_to_uz(
    snpae_nflow: "NFlowStepResults",
    legacy_nflow: "NFlowStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_nflow.u_to_z) - np.array(snpae_nflow.latents)
    legacy_vals = np.array(legacy_nflow.u_to_z) - np.array(legacy_nflow.latents)
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={**snpae_nflow.metadata, **legacy_nflow.metadata, "key": "z_to_uz"},
    )
