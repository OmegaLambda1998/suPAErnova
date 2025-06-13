from typing import TYPE_CHECKING

import pytest

import suPAErnova
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.data import DataStepConfig
from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.pae.model import PAEModelConfig
from suPAErnova.configs.steps.posterior import PosteriorStepConfig
from suPAErnova.configs.steps.nflow.model import NFlowModelConfig
from suPAErnova.configs.steps.posterior.model import PosteriorModelConfig

if TYPE_CHECKING:
    from pathlib import Path

    from tests.fixtures.pae import (
        PAEParams,
    )
    from tests.fixtures.data import (
        DataParams,
    )
    from tests.fixtures.nflow import (
        NFlowParams,
    )

    from . import (
        PosteriorParams,
        PosteriorResults,
        PosteriorStepFactory,
        PosteriorStepResults,
        PosteriorResultFactory,
    )


@pytest.fixture(scope="session")
def snpae_posterior_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "PosteriorStepFactory":
    def _snpae_posterior_step(
        data_params: "DataParams",
        pae_params: "PAEParams",
        nflow_params: "NFlowParams",
        posterior_params: "PosteriorParams",
    ) -> "PosteriorResults":
        # Import here to avoid dependency conflicts
        from suPAErnova.configs.steps.pae.tf import (
            TFPAEModelConfig,
        )
        from suPAErnova.configs.steps.nflow.tf import (
            TFNFlowModelConfig,
        )
        from suPAErnova.configs.steps.posterior.tf import (
            TFPosteriorModelConfig,
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
                "seed": posterior_params["seed"],
                "kfolds": [posterior_params["kfold"]],
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
                "seed": posterior_params["seed"],
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
            "posterior": {
                "seed": posterior_params["seed"],
                "model": {
                    **{
                        key: val
                        for key, val in posterior_params.items()
                        if key
                        in {
                            *PosteriorStepConfig.model_fields.keys(),
                            *PosteriorModelConfig.model_fields.keys(),
                            *TFPosteriorModelConfig.model_fields.keys(),
                        }
                    },
                    "backend": "tf",
                },
            },
        }
        snpae = suPAErnova.prepare_config(
            config,
            verbose=False,
            base_path=root_path,
            out_path=cache_path
            / posterior_params["fname"]
            / "posterior"
            / "snpae"
            / posterior_params["seed"],
        )
        assert snpae.data is not None, "Error setting up DataStep"
        snpae.data.paths.out = (
            cache_path / posterior_params["fname"] / "data" / "snpae" / snpae.data.name
        )

        assert snpae.pae is not None, "Error setting up PAEStep"
        snpae.pae.paths.out = (
            cache_path
            / posterior_params["fname"]
            / "pae"
            / "snpae"
            / posterior_params["seed"]
        )
        for model in snpae.pae.models:
            model.paths.out = snpae.pae.paths.out / "PAEStepConfig" / "TFPAEModelConfig"

        assert snpae.nflow is not None, "Error setting up NFlowStep"
        snpae.nflow.paths.out = (
            cache_path
            / posterior_params["fname"]
            / "nflow"
            / "snpae"
            / posterior_params["seed"]
        )
        for model in snpae.nflow.models:
            model.paths.out = (
                snpae.nflow.paths.out / "NFlowStepConfig" / "TFNFlowModelConfig"
            )

        snpae.run()
        posteriorstep = snpae.posterior_step
        assert posteriorstep is not None, "Error running PosteriorStep"
        return posteriorstep

    return _snpae_posterior_step


@pytest.fixture(scope="session")
def snpae_posterior_result_factory(
    snpae_posterior_step_factory: "PosteriorStepFactory",
) -> "PosteriorResultFactory":
    def _snpae_posterior_result(
        data_params: "DataParams",
        pae_params: "PAEParams",
        nflow_params: "NFlowParams",
        posterior_params: "PosteriorParams",
    ) -> "PosteriorStepResults":
        posterior_step = snpae_posterior_step_factory(
            data_params, pae_params, nflow_params, posterior_params
        ).models[0]
        posterior_step._result()
        return posterior_step.results

    return _snpae_posterior_result
