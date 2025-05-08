from typing import TYPE_CHECKING, Literal

import pytest

import suPAErnova
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.pae.model import PAEModelConfig

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.pae import PAEStep
    from suPAErnova.configs.steps.pae import PAEStepResult

    PAE = PAEStep[Literal["tf"]]


@pytest.fixture(scope="session")
def snpae_pae_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "Callable[[dict[str, Any], dict[str, Any]], PAE]":
    def _snpae_pae_step(
        data_params: "dict[str, Any]", pae_params: "dict[str, Any]"
    ) -> "PAE":
        from suPAErnova.configs.steps.pae.tf import (
            TFPAEModelConfig,  # Import here to avoid dependency conflicts
        )

        config: "dict[str, Any]" = {
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
            },
            "pae": {
                "validation_frac": pae_params["validation_frac"],
                "seed": pae_params["seed"],
                "model": {
                    **{
                        key: val
                        for key, val in pae_params.items()
                        if key
                        in {
                            *PAEStepConfig.model_fields.keys(),
                            *PAEModelConfig.model_fields.keys(),
                            *TFPAEModelConfig.model_fields.keys(),
                        }
                    },
                    "backend": "tf",
                },
            },
        }
        snpae = suPAErnova.prepare_config(
            config,
            base_path=root_path,
            out_path=cache_path / pae_params["fname"] / "pae" / "snpae",
        )
        snpae.run()
        paestep = snpae.pae_step
        assert paestep is not None, "Error running PAEStep"
        return paestep

    return _snpae_pae_step


@pytest.fixture(scope="session")
def snpae_pae_result_factory(
    snpae_pae_step_factory: "Callable[[dict[str, Any], dict[str, Any]], PAE]",
) -> "Callable[[dict[str, Any], dict[str, Any]], dict[str, PAEStepResult]]":
    def _snpae_pae_result(
        data_params: dict[str, "Any"], pae_params: dict[str, "Any"]
    ) -> "dict[str, PAEStepResult]":
        return snpae_pae_step_factory(data_params, pae_params).models[0].results

    return _snpae_pae_result
