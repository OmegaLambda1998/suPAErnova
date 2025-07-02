from typing import TYPE_CHECKING

import numpy as np
import pytest

import suPAErnova
from suPAErnova.configs.steps.data import DataStepConfig, DataStepResult

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
            verbose=False,
            base_path=root_path,
            out_path=cache_path / data_params["fname"] / "data" / "snpae",
        )
        snpae.run()
        datastep = snpae.data_step
        assert datastep is not None, "Error running DataStep"

        orig_data_path = data_path / "legacy"
        for dt in ("train", "test"):
            dt_data = []
            for kfold in range(datastep.n_kfolds):
                data = datastep.data.model_copy().model_dump()
                orig_data_file = orig_data_path / f"{dt}_data_kfold{kfold}.npy"
                orig_data = np.load(orig_data_file, allow_pickle=True).item()
                kfold_mask = np.logical_or.reduce([
                    datastep.data.sn_name == name for name in orig_data["names"]
                ])[:, 0, 0]
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = v[kfold_mask]
                dt_data.append(DataStepResult.model_validate(data))
            setattr(datastep, f"{dt}_data", dt_data)
        datastep.result()

        return datastep

    return _snpae_data_step


@pytest.fixture(scope="session")
def snpae_data_result_factory(
    snpae_data_step_factory: "DataStepFactory",
) -> "DataResultFactory":
    def _snpae_data_result(data_params: "DataParams") -> "DataStepResults":
        data_step_factory = snpae_data_step_factory(data_params)
        return data_step_factory.data

    return _snpae_data_result
