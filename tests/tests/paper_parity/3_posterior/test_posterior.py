from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.tests.paper_parity.conftest import PaperParityUtils
from supaernova.configs.steps.posterior import (
    PosteriorStepResult,
    PosteriorStepHMCResult,
    PosteriorStepMAPResult,
)

if TYPE_CHECKING:
    from tests.fixtures.posterior import PosteriorStepResults

pytestmark = pytest.mark.posterior

KEYS = {
    k for k in PosteriorStepResult.model_fields if k not in {"map", "hmc", "metadata"}
}

INIT_KEYS = {k for k in PosteriorStepMAPResult.model_fields if "init_" in k}

BEST_KEYS = {k for k in PosteriorStepMAPResult.model_fields if "best_" in k}

MAP_KEYS = {
    k
    for k in PosteriorStepMAPResult.model_fields
    if k
    not in {
        "converged",
        "num_evaluations",
        "negative_log_prob",
        "metadata",
        *INIT_KEYS,
        *BEST_KEYS,
    }
}

HMC_KEYS = {
    k
    for k in PosteriorStepHMCResult.model_fields
    if k not in {"is_accepted", "samples", "metadata"}
}


@pytest.mark.setup("snpae")
def test_snpae_posterior_setup(snpae_posterior: "PosteriorStepResults") -> None:
    pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_posterior_setup(legacy_posterior: "PosteriorStepResults") -> None:
    pass


# === General Tests ===


@pytest.mark.parametrize("key", KEYS)
def test_shapes(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior, key))
    legacy_vals = np.array(getattr(legacy_posterior, key))
    assert snpae_vals.shape == legacy_vals.shape


@pytest.mark.parametrize("key", KEYS)
def test_matching_remaining(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior, key))
    legacy_vals = np.array(getattr(legacy_posterior, key))
    utils.assert_arrays(snpae_vals, legacy_vals, metadata={"key": key})


# === MAP Tests ===


@pytest.mark.parametrize("key", MAP_KEYS)
def test_matching_map(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior.map, key))
    legacy_vals = np.array(getattr(legacy_posterior.map, key))
    utils.assert_arrays(snpae_vals, legacy_vals, metadata={"key": key})


@pytest.mark.parametrize("key", INIT_KEYS)
def test_matching_init(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior.map, key))
    legacy_vals = np.array(getattr(legacy_posterior.map, key))
    utils.assert_arrays(snpae_vals, legacy_vals, metadata={"key": key})


@pytest.mark.parametrize("key", INIT_KEYS)
def test_matching_init_means(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior.map, key))
    legacy_vals = np.array(getattr(legacy_posterior.map, key))
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        metadata={"key": key + "_mean"},
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
    )


@pytest.mark.parametrize("key", BEST_KEYS)
def test_matching_best(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior.map, key))
    legacy_vals = np.array(getattr(legacy_posterior.map, key))
    utils.assert_arrays(snpae_vals, legacy_vals, metadata={"key": key})


@pytest.mark.parametrize("key", BEST_KEYS)
def test_matching_best_means(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior.map, key))
    legacy_vals = np.array(getattr(legacy_posterior.map, key))
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        metadata={"key": key + "_mean"},
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
    )


def test_matching_converged(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.map.converged)
    legacy_vals = np.array(legacy_posterior.map.converged)
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={"key": "converged"},
        compare=PaperParityUtils.logical_nor,
    )


def test_matching_num_evaluations(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.map.num_evaluations)[None, ...]
    legacy_vals = np.array(legacy_posterior.map.num_evaluations)[None, ...]
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={"key": "num_evaluations"},
    )


def test_matching_num_evaluations_means(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.map.num_evaluations)[None, ...]
    legacy_vals = np.array(legacy_posterior.map.num_evaluations)[None, ...]
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        metadata={"key": "num_evaluations_mean"},
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
    )


def test_matching_neg_log_prob(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.map.negative_log_prob)
    legacy_vals = np.array(legacy_posterior.map.negative_log_prob)
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={"key": "neg_log_prob"},
    )


def test_matching_neg_log_prob_means(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.map.negative_log_prob)
    legacy_vals = np.array(legacy_posterior.map.negative_log_prob)
    utils.assert_arrays(
        snpae_vals.mean(axis=0),
        legacy_vals.mean(axis=0),
        metadata={"key": "neg_log_prob_means"},
        snpae_sigma=snpae_vals.std(axis=0),
        legacy_sigma=legacy_vals.std(axis=0),
    )


# === HMC Tests ===


@pytest.mark.parametrize("key", HMC_KEYS)
def test_matching_hmc(
    key: str,
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_posterior.hmc, key))
    legacy_vals = np.array(getattr(legacy_posterior.hmc, key))
    utils.assert_arrays(snpae_vals, legacy_vals, metadata={"key": key})


def test_matching_is_accepted(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.hmc.is_accepted)
    legacy_vals = np.array(legacy_posterior.hmc.is_accepted)
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={"key": "is_accepted"},
        compare=PaperParityUtils.logical_nor,
    )


def test_matching_samples(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.hmc.samples)
    legacy_vals = np.array(legacy_posterior.hmc.samples)
    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        metadata={"key": "samples"},
    )


def test_matching_samples_means(
    snpae_posterior: "PosteriorStepResults",
    legacy_posterior: "PosteriorStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(snpae_posterior.hmc.samples)
    legacy_vals = np.array(legacy_posterior.hmc.samples)
    utils.assert_arrays(
        snpae_vals.mean(axis=(0, 1)),
        legacy_vals.mean(axis=(0, 1)),
        metadata={"key": "neg_log_prob_means"},
        snpae_sigma=snpae_vals.std(axis=(0, 1)),
        legacy_sigma=legacy_vals.std(axis=(0, 1)),
    )
