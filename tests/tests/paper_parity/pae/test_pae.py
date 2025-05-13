from typing import TYPE_CHECKING

import numpy as np
import pytest

from suPAErnova.configs.steps.pae import PAEStepResult

if TYPE_CHECKING:
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Layer

    from tests.tests.paper_parity.conftest import PaperParityUtils

pytestmark = pytest.mark.pae

KEYS = list(PAEStepResult.model_fields.keys())
STAGES = ["1", "4", "5", "6"]
SEEDS = ["12345", "23456", "34567", "45678", "56789"]


@pytest.mark.setup("snpae")
def test_snpae_pae_setup(snpae_pae: "dict[str, PAEStepResult]") -> None:
    pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_pae_setup(legacy_pae: "dict[str, PAEStepResult]") -> None:
    pass


@pytest.mark.parametrize("stage", STAGES)
def test_snpae_stages(
    stage: str,
    snpae_pae: "dict[str, PAEStepResult]",
) -> None:
    assert stage in snpae_pae


@pytest.mark.parametrize("stage", STAGES)
def test_legacy_stages(
    stage: str,
    legacy_pae: "dict[str, PAEStepResult]",
) -> None:
    assert stage in legacy_pae


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("key", KEYS)
def test_shapes(
    stage: str,
    key: str,
    snpae_pae: "dict[str, PAEStepResult]",
    legacy_pae: "dict[str, PAEStepResult]",
) -> None:
    snpae_vals = np.array(getattr(snpae_pae[stage], key))
    legacy_vals = np.array(getattr(legacy_pae[stage], key))
    assert snpae_vals.shape == legacy_vals.shape


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("key", KEYS)
def test_matching_values(
    stage: str,
    key: str,
    snpae_pae: "dict[str, PAEStepResult]",
    legacy_pae: "dict[str, PAEStepResult]",
    utils: "PaperParityUtils",
) -> None:
    snpae_vals = np.array(getattr(snpae_pae[stage], key))
    legacy_vals = np.array(getattr(legacy_pae[stage], key))
    utils.assert_arrays(snpae_vals, legacy_vals)
