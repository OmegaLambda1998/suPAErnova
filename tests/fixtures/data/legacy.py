import os
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import cosmology as cosmo
import sncosmo

from suPAErnova.configs.steps.data import DataStepResult

if TYPE_CHECKING:
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


def legacy_data_step(data_params: "DataParams") -> "DataResults":
    # Except where indicated, this is taken verbatim from the `salt_model_make_dataset` notebook
    # Comments have been removed
    # Print statements have been removed
    # Plotting has been removed
    # Unused variables have been removed

    # Used cached result if it exists.
    savepath = (
        data_params["cache_path"]
        / data_params["fname"]
        / "data"
        / "legacy"
        / "data_step.npz"
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    if savepath.exists():
        with np.load(savepath, allow_pickle=True) as io:
            return dict(io)

    # Variation: `data_dir` points to tests data path
    data_dir = str(data_params["data_path"])
    meta = pd.read_csv(os.path.join(data_dir, "meta.csv"), index_col=0)
    df_dt = np.genfromtxt(
        os.path.join(data_dir, "IDR_eTmax.txt"), dtype=None, names=("name", "_", "days")
    )
    deltat_name = df_dt["name"].astype(str)
    deltat_days = df_dt["days"]
    ind = np.where(deltat_name == meta["sn"].iloc[0])[0][0]
    meta_deltat = [
        deltat_days[np.where(deltat_name == meta["sn"].iloc[i])[0][0]]
        for i in range(meta["sn"].shape[0])
    ]
    meta["dphase"] = meta_deltat

    # Variation: Point to tests data path
    spectra_mask = np.genfromtxt(
        os.path.join(data_dir, "mask_info_wmin_wmax.txt"),
        delimiter=" ",
        dtype=("<U32", "<U32", int, float, float),
        names=(
            "sn_name",
            "spectra_id",
            "flag",
            "wavelength_min",
            "wavelength_max",
        ),
    )

    # Variation: Legacy code did not handle nan values, instead expecting them to be -1
    spectra_mask["wavelength_min"][np.isnan(spectra_mask["wavelength_min"])] = -1
    spectra_mask["wavelength_max"][np.isnan(spectra_mask["wavelength_max"])] = -1

    def get_all_spectra(sn_name):
        sn_meta = meta[meta.sn == sn_name]
        waves, fluxes, phases, sigma, spec_id = [], [], [], [], []
        for _i, spec_info in sn_meta.iterrows():
            data = pd.read_csv(os.path.join(data_dir, spec_info.path))
            spec_id.append(
                str(os.path.splitext(os.path.basename(spec_info.path))[0][-14:])
            )
            waves.append(data.wave)
            fluxes.append(data.flux)
            sigma.append(data.sigma)
            phases.append(spec_info.phase)
        return (
            np.array(waves),
            np.array(fluxes),
            np.array(phases),
            np.array(sigma),
            spec_id,
        )

    nspectra = 0
    nwaves = 0
    for sn_name in meta.sn.unique()[:1]:
        waves, fluxes, phases, sigmas, spec_ids = get_all_spectra(sn_name)
        nwaves = waves.shape[-1]

    nspectra = meta.shape[0]
    params = ["dphase", "z", "x0", "x1", "c", "MB", "hubble_resid"]
    nparams = len(params)
    param_labels = ["ID", "Nspectra_ID", "phase", *params]
    data = {}
    data["spectra"] = np.zeros((nspectra, nwaves))
    data["wavelength_mask"] = np.zeros((nspectra, 2))
    data["sigma"] = np.zeros((nspectra, nwaves))
    data["params"] = np.zeros((nspectra, nparams + 3))
    sn_names = []
    spectra_ids = []
    ispectra_prev = 0
    ispectra = 0
    sn_meta = meta.drop_duplicates(subset="sn")
    for i, sn_name in enumerate(meta.sn.unique()):
        waves, fluxes, phases, sigmas, spec_ids = get_all_spectra(sn_name)
        nspectra = len(phases)
        ispectra += nspectra
        for j in range(nparams):
            data["params"][ispectra_prev:ispectra, j + 3] = sn_meta[params[j]][i]
        sn_names += [sn_name] * nspectra
        spectra_ids += spec_ids
        data["params"][ispectra_prev:ispectra, 0] = i
        data["params"][ispectra_prev:ispectra, 1] = nspectra
        data["params"][ispectra_prev:ispectra, 2] = phases
        data["spectra"][ispectra_prev:ispectra, :] = fluxes
        data["sigma"][ispectra_prev:ispectra, :] = sigmas
        for j, spec_id in enumerate(spec_ids):
            try:
                ind = np.where(spectra_mask["spectra_id"] == spec_id)[0][0]
                data["wavelength_mask"][ispectra_prev + j, 0] = spectra_mask[
                    "wavelength_min"
                ][ind]
                data["wavelength_mask"][ispectra_prev + j, 1] = spectra_mask[
                    "wavelength_max"
                ][ind]
            except:
                (
                    data["wavelength_mask"][ispectra_prev + j, 0],
                    data["wavelength_mask"][ispectra_prev + j, 1],
                ) = 3298.68, 9701.23  # use whole spectra
        ispectra_prev += len(phases)
    data["names"] = np.array(sn_names)
    data["spectra_ids"] = np.array(spectra_ids)
    data["wavelengths"] = waves[0, :]
    for i, param in enumerate(param_labels):
        data[param] = data["params"][:, i]
    data["ID"] = data["ID"].astype(int)
    data["Nspectra_ID"] = data["Nspectra_ID"].astype(int)
    data["redshift"] = data.pop("z")  # rename redshift field
    data.pop("params")

    # Variation: Removed unused optional params, made params which are always passed required
    # Variation: Precompute some values (including snmodel) which were unecessarily computed every run
    cosmology = getattr(cosmo, data_params["cosmological_model"])
    zref = 0.05
    zref_dist = cosmology.luminosity_distance(zref)

    # Variation: salt2-4 -> salt2-k21-frag
    # Variation: Make path relative to user home (~)
    snmodel = sncosmo.SALT2Source(
        modeldir=str(
            (
                Path.home() / ".astropy/cache/sncosmo/models/salt2/salt2-k21-frag/"
            ).resolve()
        )
    )

    def get_flux(
        wavelength,
        tobs,
        z,
        x0,
        x1,
        c,
    ):
        snmodel.set(x0=x0, x1=x1, c=c)
        return (
            snmodel.flux(phase=tobs, wave=wavelength)
            * (cosmology.luminosity_distance(z) / zref_dist) ** 2
            * (1 + z)
            / (1 + zref)
            * 1e15
        )

    tmin = data_params["min_phase"]
    tmax = data_params["max_phase"]
    dm = (tmin < data["phase"]) & (tmax > data["phase"])
    for k, v in data.items():
        if v.shape[0] == dm.shape[0]:
            data[k] = v[dm]

    data["spectra_salt"] = np.zeros(data["spectra"].shape)

    # Note that without the `get_flux` variations above, this takes ~10 minutes.
    #   With the variations it takes a quarter of a second.
    for i in range(data["spectra_salt"].shape[0]):
        data["spectra_salt"][i] = get_flux(
            data["wavelengths"],
            tobs=data["phase"][i],
            z=data["redshift"][i],
            x0=data["x0"][i],
            x1=data["x1"][i],
            c=data["c"][i],
        )

    # The following is taken from `make_datasets/make_train_test_data.py`
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    class time_normalization:
        def __init__(self, times, normed=False) -> None:
            self.tmin = tmin
            self.tmax = tmax
            self.minmax = self.tmax - self.tmin
            # only normalize existing times, not null value (-100)
            self.dm = times > -100
            self.normed = normed

        def scale(self, times):
            if self.normed:
                times[self.dm] = times[self.dm] * self.minmax + self.tmin
                self.normed = False
            else:
                times[self.dm] = (times[self.dm] - self.tmin) / self.minmax
                times[~self.dm] = -1.0
                self.normed = True
            return times

    data_out = {}
    data_out["wavelengths"] = data["wavelengths"]
    data_out["names"] = data["names"]
    data_out["spectra_ids"] = data["spectra_ids"]

    n_timestep = np.max(np.bincount(data["ID"]))
    n_sn = len(np.unique(data["ID"]))
    n_wavelength = len(data["wavelengths"])

    data_out["spectra"] = (
        np.zeros((n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1
    )
    data_out["spectra_salt"] = (
        np.zeros((n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1
    )
    data_out["sigma"] = np.ones((n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1
    data_out["mask"] = np.full((n_sn, n_timestep, n_wavelength), False)
    data_out["times"] = np.zeros((n_sn, n_timestep, 1), dtype=np.float32) - 100
    n_spec_each = np.zeros(n_sn)

    spectra_ids = np.full((n_sn, n_timestep, 1), "-" * 32)
    wavelength_mask = np.full((n_sn, n_timestep, 2), -1.0)
    for i, idi in enumerate(np.unique(data["ID"])):
        ids = np.where(data["ID"] == idi)[0]
        n_speci = min(n_timestep, len(ids))
        n_spec_each[i] = n_speci
        for k in ["spectra", "spectra_salt"]:
            data_out[k][i, :n_speci] = data[k][ids[:n_timestep]]
        data_out["sigma"][i, :n_speci] = data["sigma"][ids[:n_timestep]]
        spectra_ids[i, :n_speci] = (
            data["names"][ids[:n_timestep]][:, np.newaxis]
            + "_"
            + data["spectra_ids"][ids[:n_timestep]][:, np.newaxis]
        )
        wavelength_mask[i, :n_speci] = data["wavelength_mask"][ids[:n_timestep]]
        for ispec in range(n_speci):
            keep_min, keep_max = data["wavelength_mask"][ids[ispec]]
            ind_keep_min = 0
            ind_keep_max = data["wavelengths"].shape[0]
            if keep_min != -1.0:
                if keep_min > data["wavelengths"][0]:
                    ind_keep_min = find_nearest_idx(data["wavelengths"], keep_min)
                # Variation: In the legacy code, keep_max was incorrectly compared to the minimum wavelength not the maximum wavelength
                if keep_max < data["wavelengths"][-1]:
                    ind_keep_max = find_nearest_idx(data["wavelengths"], keep_max)
                # Variation: In the legacy code, valid maximum wavelength where masked as ind_keep_min:ind_keep_max results in an array that does not include ind_keep_max
                data_out["mask"][i, ispec, ind_keep_min : ind_keep_max + 1] = True
            wavelength_bin_start = find_nearest_idx(data["wavelengths"], 5000.0)

            # Variation: In the legacy code, some wavelengths where ignored as wavelength_bin_start:wavelength_bin_end results in an array that does not include wavelength_bin_end
            wavelength_bin_end = find_nearest_idx(data["wavelengths"], 8000.0) + 1
            laser_width = 2
            laser_height = 0.4
            speci = data_out["spectra"][
                i,
                ispec,
                wavelength_bin_start:wavelength_bin_end,
            ]
            speci_smooth = (
                data_out["spectra"][
                    i,
                    ispec,
                    wavelength_bin_start - laser_width : wavelength_bin_end
                    - laser_width,
                ]
                + data_out["spectra"][
                    i,
                    ispec,
                    wavelength_bin_start + laser_width : wavelength_bin_end
                    + laser_width,
                ]
            ) / 2
            laser_mask = (speci - speci_smooth) > laser_height

            # Variation: i->ii. Doesn't change the code since python can keep track of list-comprehension vs non list-comprehension variables, but avoids confusion
            # Variation: In the legacy code, invalid wavelengths where unmasked as ii-laser_width:ii+laser_width results in an array that does not include ii+laser_width
            laser_mask = np.array([
                np.any(laser_mask[ii - laser_width : ii + laser_width + 1])
                for ii in range(laser_mask.shape[0])
            ])

            # Variation: Legacy code originally overwrote the mask rather than &-ing the laser_mask with the mask, meaning non-laser points were unmasked even if they were originally masked to begin with.
            data_out["mask"][
                i,
                ispec,
                wavelength_bin_start:wavelength_bin_end,
            ] &= ~laser_mask

        data_out["times"][i, :n_speci] = data["phase"][ids[:n_timestep], None]
    data_out["mask"] = data_out["mask"].astype(np.float32)
    data.pop("wavelengths")
    data.pop("spectra")
    data.pop("spectra_salt")
    data.pop("sigma")
    data.pop("phase")

    _unique_IDs, inds = np.unique(data["ID"], return_index=True)
    for k, v in data.items():
        data_out[k] = v[inds]

    # Variation: Update spectra_ids formatting to match new code
    data_out["spectra_ids"] = spectra_ids

    # Variation: Original code did not correctly update Nspectra_ID after the phase cut
    data_out["Nspectra_ID"] = n_spec_each

    # Variation: Original code did not correctly update wavelength mask after mask created
    data_out["wavelength_mask"] = wavelength_mask

    data_out["times_orig"] = data_out["times"].copy()
    time_normalizer = time_normalization(data_out["times_orig"])
    data_out["times"] = time_normalizer.scale(data_out["times"])
    data_out["luminosity_distance"] = cosmology.luminosity_distance(
        data_out["redshift"],
    ).value

    # The following is taken from `utils/data_loader.py`
    # Variation: rename `data` to `data_out`
    dm = data_out["mask"] == 1
    data_out["spectra"][dm] = np.clip(data_out["spectra"][dm], 0.0, np.inf)
    data_out["sigma"] = 1.4 * data_out["sigma"] + 4e-10

    # Variation: Correct shape of some legacy data arrays to enable easy comparison
    n_sn, n_timestep, _n_wavelength = data_out["spectra"].shape
    config = {}
    for key, legacy_key in KEY_MAP.items():
        if isinstance(legacy_key, tuple):
            k, i = legacy_key
            val = data_out[k][:, :, i : i + 1]
        else:
            val = data_out[legacy_key]
        if val.ndim == 1:
            if val.shape[0] == n_sn:
                val = np.tile(val[:, np.newaxis, np.newaxis], (1, n_timestep, 1))
            else:
                val = np.tile(val[np.newaxis, np.newaxis, :], (n_sn, n_timestep, 1))
        config[key] = val

    # Match default values between legacy and new code
    config["wl_mask_min"][config["wl_mask_min"] == -1] = np.inf
    config["wl_mask_max"][config["wl_mask_max"] == -1] = -np.inf

    # The following is taken from `make_datasets/make_train_test_data.py`
    test_frac = 1 - data_params["train_frac"]
    nkfold = int(1.0 / test_frac)
    ind_split = int(n_sn * data_params["train_frac"])
    np.random.seed(int(data_params["seed"]))
    inds = np.arange(n_sn)
    np.random.shuffle(inds)
    for kfold in range(nkfold):
        inds_k = np.roll(inds, kfold * inds.shape[0] // nkfold)
        inds_train = inds_k[:ind_split]
        inds_test = inds_k[ind_split:]
        train_data = {}
        test_data = {}
        for k, v in data_out.items():
            if v.shape[0] != n_sn:
                train_data[k] = v
                test_data[k] = v
            else:
                train_data[k] = v[inds_train]
                test_data[k] = v[inds_test]

        # Variation: Savepath
        # Variation: save -> savez_compressed
        kfold_train_savepath = (
            savepath.parent / "train" / f"kfold{kfold:d}{savepath.suffix}"
        )
        kfold_test_savepath = (
            savepath.parent / "test" / f"kfold{kfold:d}{savepath.suffix}"
        )

        kfold_train_savepath.parent.mkdir(parents=True, exist_ok=True)
        kfold_test_savepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(kfold_train_savepath, **train_data)
        np.savez_compressed(kfold_test_savepath, **test_data)

    # Variation: Savepath
    # Variation: save -> savez_compressed
    np.savez_compressed(savepath, **config)
    with np.load(savepath, allow_pickle=True) as io:
        return dict(io)


@pytest.fixture(scope="session")
def legacy_data_step_factory(
    data_path: "Path",
    root_path: "Path",
    cache_path: "Path",
) -> "DataStepFactory":
    def _legacy_data_step(data_params: "DataParams") -> "DataResults":
        data_params["data_path"] = data_path
        data_params["root_path"] = root_path
        data_params["cache_path"] = cache_path
        return legacy_data_step(data_params)

    return _legacy_data_step


@pytest.fixture(scope="session")
def legacy_data_result_factory(
    legacy_data_step_factory: "DataStepFactory",
) -> "DataResultFactory":
    def _legacy_data_result(data_params: "DataParams") -> "DataStepResults":
        return DataStepResult.model_validate(legacy_data_step_factory(data_params))

    return _legacy_data_result
