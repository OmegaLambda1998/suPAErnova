from typing import TYPE_CHECKING

import pytest

import suPAErnova
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.pae.model import PAEModelConfig

if TYPE_CHECKING:
    from pathlib import Path

    from tests.fixtures.data import (
        DataParams,
    )

    from . import (
        PAEParams,
        PAEResults,
        PAEStepFactory,
        PAEStepResults,
        PAEResultFactory,
    )


@pytest.fixture(scope="session")
def snpae_pae_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "PAEStepFactory":
    def _snpae_pae_step(
        data_params: "DataParams", pae_params: "PAEParams"
    ) -> "PAEResults":
        from suPAErnova.configs.steps.pae.tf import (
            TFPAEModelConfig,  # Import here to avoid dependency conflicts
        )

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
            },
            "pae": {
                "validation_frac": pae_params["validation_frac"],
                "seed": pae_params["seed"],
                "kfolds": [pae_params["kfold"]],
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
            out_path=cache_path
            / pae_params["fname"]
            / "pae"
            / "snpae"
            / pae_params["seed"],
        )
        assert snpae.data is not None, "Error setting up DataStep"
        snpae.data.paths.out = (
            cache_path / pae_params["fname"] / "data" / "snpae" / snpae.data.name
        )
        snpae.run()
        paestep = snpae.pae_step
        assert paestep is not None, "Error running PAEStep"
        return paestep

    return _snpae_pae_step


@pytest.fixture(scope="session")
def snpae_pae_result_factory(
    snpae_pae_step_factory: "PAEStepFactory",
) -> "PAEResultFactory":
    def _snpae_pae_result(
        data_params: "DataParams", pae_params: "PAEParams"
    ) -> "PAEStepResults":
        pae_step = snpae_pae_step_factory(data_params, pae_params).models[0]
        pae_step._result()
        return pae_step.results

    return _snpae_pae_result
