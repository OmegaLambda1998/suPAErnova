from typing import TYPE_CHECKING

import pytest

import suPAErnova
from suPAErnova.configs.steps.data import DataStepConfig

if TYPE_CHECKING:
    from pathlib import Path

    from . import (
        DataParams,
        DataResults,
        DataStepFactory,
        DataStepResults,
        DataResultFactory,
    )


@pytest.fixture(scope="session")
def snpae_data_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "DataStepFactory":
    def _snpae_data_step(data_params: "DataParams") -> "DataResults":
        config = {
            "data": {
                **{
                    key: val
                    for key, val in data_params.items()
                    if key in DataStepConfig.model_fields
                },
                "data_dir": data_path,
                "meta": "meta.csv",
                "idr": "IDR_eTmax.txt",
                "mask": "mask_info_wmin_wmax.txt",
            }
        }
        snpae = suPAErnova.prepare_config(
            config,
            base_path=root_path,
            out_path=cache_path / data_params["fname"] / "data" / "snpae",
        )
        snpae.run()
        datastep = snpae.data_step
        assert datastep is not None, "Error running DataStep"
        return datastep

    return _snpae_data_step


@pytest.fixture(scope="session")
def snpae_data_result_factory(
    snpae_data_step_factory: "DataStepFactory",
) -> "DataResultFactory":
    def _snpae_data_result(data_params: "DataParams") -> "DataStepResults":
        return snpae_data_step_factory(data_params).data

    return _snpae_data_result
