from typing import TYPE_CHECKING

import pytest

import supaernova
from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.data import DataStepConfig
from supaernova.configs.steps.nflow import NFlowStepConfig
from supaernova.configs.steps.pae.model import PAEModelConfig
from supaernova.configs.steps.nflow.model import NFlowModelConfig

if TYPE_CHECKING:
    from pathlib import Path

    from tests.fixtures.pae import (
        PAEParams,
    )
    from tests.fixtures.data import (
        DataParams,
    )

    from . import (
        NFlowParams,
        NFlowResults,
        NFlowStepFactory,
        NFlowStepResults,
        NFlowResultFactory,
    )


@pytest.fixture(scope="session")
def snpae_nflow_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "NFlowStepFactory":
    def _snpae_nflow_step(
        data_params: "DataParams",
        pae_params: "PAEParams",
        nflow_params: "NFlowParams",
    ) -> "NFlowResults":
        # Import here to avoid dependency conflicts
        from supaernova.configs.steps.pae.tf import (
            TFPAEModelConfig,
        )
        from supaernova.configs.steps.nflow.tf import (
            TFNFlowModelConfig,
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
                "seed": nflow_params["seed"],
                "kfolds": [nflow_params["kfold"]],
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
            "nflow": {
                "validation_frac": nflow_params["validation_frac"],
                "seed": nflow_params["seed"],
                "model": {
                    **{
                        key: val
                        for key, val in nflow_params.items()
                        if key
                        in {
                            *NFlowStepConfig.model_fields.keys(),
                            *NFlowModelConfig.model_fields.keys(),
                            *TFNFlowModelConfig.model_fields.keys(),
                        }
                    },
                    "backend": "tf",
                },
            },
        }
        snpae = supaernova.prepare_config(
            config,
            verbose=False,
            base_path=root_path,
            out_path=cache_path
            / nflow_params["fname"]
            / "nflow"
            / "snpae"
            / nflow_params["seed"],
        )
        assert snpae.data is not None, "Error setting up DataStep"
        snpae.data.paths.out = (
            cache_path / nflow_params["fname"] / "data" / "snpae" / snpae.data.name
        )
        assert snpae.pae is not None, "Error setting up PAEStep"
        snpae.pae.paths.out = (
            cache_path / nflow_params["fname"] / "pae" / "snpae" / nflow_params["seed"]
        )
        for model in snpae.pae.models:
            model.paths.out = snpae.pae.paths.out / "PAEStepConfig" / "PaperParityPAE"
        snpae.run()
        nflowstep = snpae.nflow_step
        assert nflowstep is not None, "Error running NFlowStep"
        return nflowstep

    return _snpae_nflow_step


@pytest.fixture(scope="session")
def snpae_nflow_result_factory(
    snpae_nflow_step_factory: "NFlowStepFactory",
) -> "NFlowResultFactory":
    def _snpae_nflow_result(
        data_params: "DataParams", pae_params: "PAEParams", nflow_params: "NFlowParams"
    ) -> "NFlowStepResults":
        nflow_step = snpae_nflow_step_factory(
            data_params, pae_params, nflow_params
        ).models[0]
        nflow_step._result()
        return nflow_step.results

    return _snpae_nflow_result
