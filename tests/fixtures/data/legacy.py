from typing import TYPE_CHECKING

import numpy as np
import pytest

import supaernova
from supaernova.configs.steps.data import DataStepConfig, DataStepResult

if TYPE_CHECKING:
    from pathlib import Path

    from . import (
        DataParams,
        DataResults,
        DataStepFactory,
        DataStepResults,
        DataResultFactory,
    )

KEY_MAP = {
    "ind": "ID",
    "nspectra": "Nspectra_ID",
    "sn_name": "names",
    "dphase": "dphase",
    "redshift": "redshift",
    "x0": "x0",
    "x1": "x1",
    "c": "c",
    "MB": "MB",
    "hubble_residual": "hubble_resid",
    "luminosity_distance": "luminosity_distance",
    "spectra_id": "spectra_ids",
    "phase": "times_orig",
    "wl_mask_min": ("wavelength_mask", 0),
    "wl_mask_max": ("wavelength_mask", 1),
    "amplitude": "spectra",
    "sigma": "sigma",
    "salt_flux": "spectra_salt",
    "wavelength": "wavelengths",
    "mask": "mask",
    "time": "times",
}

SHAPE_MAP = {
    "ind": 1,
    "nspectra": 1,
    "sn_name": 1,
    "dphase": 1,
    "redshift": 1,
    "x0": 1,
    "x1": 1,
    "c": 1,
    "MB": 1,
    "hubble_residual": 1,
    "luminosity_distance": 1,
    "spectra_id": 1,
    "phase": 0,
    "wl_mask_min": 1,
    "wl_mask_max": 1,
    "amplitude": 0,
    "sigma": 0,
    "salt_flux": 0,
    "wavelength": 2,
    "mask": 0,
    "time": 0,
}


@pytest.fixture(scope="session")
def legacy_data_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "DataStepFactory":
    def _legacy_data_step(data_params: "DataParams") -> "DataResults":
        n_kfolds = int(1 / (1 - data_params["train_frac"]))
        legacy_data_path = data_path / "legacy"
        datastep = {}

        for dt in ("train", "test"):
            dt_data = []
            for kfold in range(n_kfolds):
                kfold_data = {}
                legacy_data_file = legacy_data_path / f"{dt}_data_kfold{kfold}.npy"
                legacy_data = np.load(legacy_data_file, allow_pickle=True).item()

                legacy_data["spectra_ids"] = (
                    legacy_data["names"] + "_" + legacy_data["spectra_ids"]
                )

                n_sn, n_spec, _n_wl = legacy_data["spectra"].shape
                for k, legacy_k in KEY_MAP.items():
                    if isinstance(legacy_k, tuple):
                        key, i = legacy_k
                        val = legacy_data[key][..., i]
                    else:
                        val = legacy_data[legacy_k]

                    shape = SHAPE_MAP[k]
                    # (n_sn, 1, 1)
                    if shape == 1:
                        val = np.tile(val[:, None, None], (1, n_spec, 1))
                    elif shape == 2:
                        val = np.tile(val, (n_sn, n_spec, 1))
                    kfold_data[k] = val
                dt_data.append(DataStepResult.model_validate(kfold_data))

                kfold_data["wl_mask_min"][np.isnan(kfold_data["wl_mask_min"])] = np.inf

            datastep[f"{dt}_data"] = dt_data
        return datastep

    return _legacy_data_step


@pytest.fixture(scope="session")
def legacy_data_result_factory(
    legacy_data_step_factory: "DataStepFactory",
) -> "DataResultFactory":
    def _legacy_data_result(data_params: "DataParams") -> "DataStepResults":
        return legacy_data_step_factory(data_params)

    return _legacy_data_result
