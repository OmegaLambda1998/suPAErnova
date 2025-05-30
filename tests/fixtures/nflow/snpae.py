from typing import TYPE_CHECKING, Literal

import pytest
import tensorflow as tf

import suPAErnova
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.pae.model import PAEModelConfig
from suPAErnova.configs.steps.nflow.model import NFlowModelConfig

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.nflow import NFlowStep
    from suPAErnova.configs.steps.nflow import NFlowStepResult

    NFLOW = NFlowStep[Literal["tf"]]


@pytest.fixture(scope="session")
def snpae_nflow_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], NFLOW]":
    def _snpae_nflow_step(
        data_params: "dict[str, Any]",
        pae_params: "dict[str, Any]",
        nflow_params: "dict[str, Any]",
    ) -> "NFLOW":
        # Import here to avoid dependency conflicts
        from suPAErnova.configs.steps.pae.tf import (
            TFPAEModelConfig,
        )
        from suPAErnova.configs.steps.nflow.tf import (
            TFNFlowModelConfig,
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
            "nflow": {
                "validation_frac": pae_params["validation_frac"],
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
        snpae = suPAErnova.prepare_config(
            config,
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
            cache_path / nflow_params["fname"] / "pae" / "snpae" / pae_params["seed"]
        )
        for model in snpae.pae.models:
            model.paths.out = snpae.pae.paths.out / "PAEStepConfig" / "TFPAEModelConfig"
        snpae.run()
        nflowstep = snpae.nflow_step
        assert nflowstep is not None, "Error running NFlowStep"
        return nflowstep

    return _snpae_nflow_step


@pytest.fixture(scope="session")
def snpae_nflow_result_factory(
    snpae_nflow_step_factory: "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], NFLOW]",
) -> "Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, NFlowStepResult]]":
    def _snpae_nflow_result(
        data_params: dict[str, "Any"],
        pae_params: dict[str, "Any"],
        nflow_params: dict[str, "Any"],
    ) -> "dict[str, NFlowStepResult]":
        nflow_step = snpae_nflow_step_factory(
            data_params, pae_params, nflow_params
        ).models[0]
        nflow_step._result()
        return nflow_step.results

    return _snpae_nflow_result
