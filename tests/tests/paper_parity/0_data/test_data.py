from typing import TYPE_CHECKING

import numpy as np
import pytest

from supaernova.configs.steps.data import DataStepResult

if TYPE_CHECKING:
    from tests.fixtures.data import DataStepResults
    from tests.tests.paper_parity.conftest import PaperParityUtils

pytestmark = pytest.mark.data

DTS = ["train"]  # , "test"]
KFOLDS = [0]  # , 1, 2, 3]
KEYS = [k for k in DataStepResult.model_fields if k != "metadata"]


@pytest.mark.setup("snpae")
def test_snpae_data_setup(snpae_data: "DataStepResults") -> None:
    pass


@pytest.mark.setup("legacy_snpae")
def test_legacy_data_setup(legacy_data: "DataStepResults") -> None:
    pass


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("kfold", KFOLDS)
@pytest.mark.parametrize("dt", DTS)
def test_shapes(
    key: str,
    kfold: int,
    dt: str,
    snpae_data: "DataStepResults",
    legacy_data: "DataStepResults",
) -> None:
    snpae_dt = getattr(snpae_data, f"{dt}_data")[kfold]
    legacy_dt = legacy_data[f"{dt}_data"][kfold]

    snpae_shape = getattr(snpae_dt, key).shape
    legacy_shape = getattr(legacy_dt, key).shape

    assert snpae_shape == legacy_shape, f"dt: {dt}, kfold: {kfold}, key: {key}"


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("kfold", KFOLDS)
@pytest.mark.parametrize("dt", DTS)
def test_matching_values(
    key: str,
    kfold: int,
    dt: str,
    snpae_data: "DataStepResults",
    legacy_data: "DataStepResults",
    utils: "PaperParityUtils",
) -> None:
    snpae_dt = getattr(snpae_data, f"{dt}_data")[kfold]
    legacy_dt = legacy_data[f"{dt}_data"][kfold]

    snpae_vals = getattr(snpae_dt, key)
    legacy_vals = getattr(legacy_dt, key)

    snpae_mask = snpae_dt.sn_name[:, 0, 0].argsort()
    legacy_mask = legacy_dt.sn_name[:, 0, 0].argsort()

    snpae_vals = snpae_vals[snpae_mask]
    legacy_vals = legacy_vals[legacy_mask]

    if snpae_dt.metadata is None:
        snpae_dt.metadata = {}
    if legacy_dt.metadata is None:
        legacy_dt.metadata = {}

    utils.assert_arrays(
        snpae_vals,
        legacy_vals,
        spectra=snpae_dt.spectra_id + "---" + legacy_dt.spectra_id,
        metadata={
            **snpae_dt.metadata,
            **legacy_dt.metadata,
            "dt": dt,
            "kfold": kfold,
            "key": key,
        },
    )
