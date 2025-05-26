from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from suPAErnova.configs.steps.nflow import NFlowStepResult
    from tests.tests.paper_parity.conftest import PaperParityUtils

pytestmark = pytest.mark.pae

# STAGES = ["1", "4", "5", "6"]
# KEYS = list(PAEStepResult.model_fields.keys())
# LOSSES = [k for k in KEYS if "loss" in k]
# AMPS = [k for k in KEYS if "_amp" in k]
# WEIGHTS = [k for k in KEYS if "_weights" in k]
# REMAINING = [k for k in KEYS if k not in LOSSES + AMPS + WEIGHTS + ["latents"]]


# @pytest.mark.setup("snpae")
# def test_snpae_pae_setup(snpae_pae: "dict[str, PAEStepResult]") -> None:
#     pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_nflow_setup(legacy_nflow: "dict[str, NFlowStepResult]") -> None:
    pass


# @pytest.mark.parametrize("stage", STAGES)
# def test_snpae_stages(
#     stage: str,
#     snpae_pae: "dict[str, PAEStepResult]",
# ) -> None:
#     assert stage in snpae_pae
#
#
# @pytest.mark.parametrize("stage", STAGES)
# def test_legacy_stages(
#     stage: str,
#     legacy_pae: "dict[str, PAEStepResult]",
# ) -> None:
#     assert stage in legacy_pae
#
#
# @pytest.mark.parametrize("stage", STAGES)
# @pytest.mark.parametrize("key", REMAINING)
# def test_shapes(
#     stage: str,
#     key: str,
#     snpae_pae: "dict[str, PAEStepResult]",
#     legacy_pae: "dict[str, PAEStepResult]",
# ) -> None:
#     snpae_vals = np.array(getattr(snpae_pae[stage], key))
#     legacy_vals = np.array(getattr(legacy_pae[stage], key))
#     assert snpae_vals.shape == legacy_vals.shape
#
#
# @pytest.mark.parametrize("key", REMAINING)
# @pytest.mark.parametrize("stage", STAGES)
# def test_matching_remaining(
#     stage: str,
#     key: str,
#     snpae_pae: "dict[str, PAEStepResult]",
#     legacy_pae: "dict[str, PAEStepResult]",
#     utils: "PaperParityUtils",
# ) -> None:
#     snpae_vals = np.array(getattr(snpae_pae[stage], key))
#     legacy_vals = np.array(getattr(legacy_pae[stage], key))
#     utils.assert_arrays(snpae_vals, legacy_vals)
#
#
# # @pytest.mark.parametrize("key", LOSSES)
# # @pytest.mark.parametrize("stage", STAGES)
# # def test_matching_losses(
# #     stage: str,
# #     key: str,
# #     snpae_pae: "dict[str, PAEStepResult]",
# #     legacy_pae: "dict[str, PAEStepResult]",
# #     utils: "PaperParityUtils",
# # ) -> None:
# #     snpae_vals = np.array(getattr(snpae_pae[stage], key))
# #     legacy_vals = np.array(getattr(legacy_pae[stage], key))
# #     utils.assert_arrays(snpae_vals, legacy_vals, sort=False)
#
#
# @pytest.mark.parametrize("key", AMPS)
# @pytest.mark.parametrize("stage", STAGES)
# def test_matching_amps(
#     stage: str,
#     key: str,
#     snpae_pae: "dict[str, PAEStepResult]",
#     legacy_pae: "dict[str, PAEStepResult]",
#     utils: "PaperParityUtils",
# ) -> None:
#     snpae_vals = np.array(getattr(snpae_pae[stage], key))
#     legacy_vals = np.array(getattr(legacy_pae[stage], key))
#     utils.assert_arrays(snpae_vals, legacy_vals, spectra=snpae_pae[stage].spectra_id)
#
#
# @pytest.mark.parametrize("key", AMPS)
# @pytest.mark.parametrize("stage", STAGES)
# def test_matching_amp_means(
#     stage: str,
#     key: str,
#     snpae_pae: "dict[str, PAEStepResult]",
#     legacy_pae: "dict[str, PAEStepResult]",
#     utils: "PaperParityUtils",
# ) -> None:
#     snpae_vals = np.array(getattr(snpae_pae[stage], key))
#     legacy_vals = np.array(getattr(legacy_pae[stage], key))
#     utils.assert_arrays(snpae_vals.mean(axis=(0, 1)), legacy_vals.mean(axis=(0, 1)))
#
#
# # @pytest.mark.parametrize("stage", STAGES)
# # def test_matching_latents(
# #     stage: str,
# #     snpae_pae: "dict[str, PAEStepResult]",
# #     legacy_pae: "dict[str, PAEStepResult]",
# #     utils: "PaperParityUtils",
# # ) -> None:
# #     snpae_vals = np.array(snpae_pae[stage].latents)
# #     legacy_vals = np.array(legacy_pae[stage].latents)
# #     utils.assert_arrays(abs(snpae_vals), abs(legacy_vals))
# #
# #
# # @pytest.mark.parametrize("stage", STAGES)
# # def test_matching_latent_means(
# #     stage: str,
# #     snpae_pae: "dict[str, PAEStepResult]",
# #     legacy_pae: "dict[str, PAEStepResult]",
# #     utils: "PaperParityUtils",
# # ) -> None:
# #     snpae_vals = np.array(snpae_pae[stage].latents)
# #     legacy_vals = np.array(legacy_pae[stage].latents)
# #     utils.assert_arrays(
# #         snpae_vals.mean(axis=0), legacy_vals.mean(axis=0), max_diffs=6, sort=False
# #     )
