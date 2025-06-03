from typing import TYPE_CHECKING

import numpy as np
import pytest

from suPAErnova.configs.steps.posterior import PosteriorStepResult

if TYPE_CHECKING:
    from tests.fixtures.posterior import PosteriorStepResults
    from tests.tests.paper_parity.conftest import PaperParityUtils

pytestmark = pytest.mark.posterior

KEYS = list(PosteriorStepResult.model_fields)


@pytest.mark.setup("legacy_snpae")
def test_legacy_posterior_setup(legacy_posterior: "PosteriorStepResults") -> None:
    pass


# @pytest.mark.setup("snpae")
# def test_snpae_posterior_setup(snpae_posterior: "PosteriorStepResults") -> None:
#     pass


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
