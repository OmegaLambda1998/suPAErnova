from typing import TYPE_CHECKING, Literal, ClassVar, cast, override
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
from astropy import cosmology as cosmo
import sncosmo

from supaernova.steps import Step
from supaernova.utils import pp, resolve_path, SNR
from supaernova.steps.variants import Variant
from supaernova.analysis.spectra import SpectraPlotter
from supaernova.analysis.analysis import Plotter
from supaernova.configs.callbacks import callback
from supaernova.configs.steps.data import (
    DataConfig,
    LazySNPAEData,
    DataStepConfig,
    DataStepResult,
    DataStepAnalysis,
    LazySNPAEDataTuple,
)

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable, Sequence

    from numpy import typing as npt

    from supaernova.typing import T

    SNeDataFrame = pd.DataFrame


class Data(Step[DataConfig]):
    def __init__(self, config: DataConfig) -> None:
        super().__init__(config)

        # === Config Variables ===
        # --- Required ---
        self.data_dir: Path = self.options.data_dir
        self.meta: Path = self.options.meta
        self.idr: Path = self.options.idr
        self.mask: Path = self.options.mask

        # --- Optional ---
        self.min_redshift: float = self.options.min_redshift
        self.max_redshift: float = self.options.max_redshift
        self.min_phase: float = self.options.min_phase
        self.max_phase: float = self.options.max_phase
        self.min_wavelength: float = self.options.min_wavelength
        self.max_wavelength: float = self.options.max_wavelength
        self.seed: int = self.options.seed

        self.data_splits: Path | None = self.options.data_splits
        self.train_frac: float = self.options.train_frac

        # Output Paths
        self.out_data: Path = self.paths.results / "data.npz"
        self.out_sne: Path = self.paths.results / "sne.pkl"
        self.out_train: Path = self.paths.results / "train"
        self.out_train.mkdir(parents=True, exist_ok=True)
        self.out_test: Path = self.paths.results / "test"
        self.out_test.mkdir(parents=True, exist_ok=True)

        # === Setup Variables ===
        self.setup_attributes: set[str] = {
            "colourlaw",
            "cosmological_model",
            "salt_model",
            "test_frac",
            "n_kfolds",
            "data",
            "train_data",
            "test_data",
        }

        self.colourlaw: npt.NDArray[float] | None
        self.cosmological_model: cosmo.FLRW
        self.salt_model: sncosmo.SALT2Source | sncosmo.SALT3Source

        # Train / Test Split
        self.splits: dict[str, dict[Literal["train | test | validate"], list[str]]]
        self.test_frac: float
        self.n_kfolds: int
        if self.options.n_kfolds is not None:
            self.n_kfolds = self.options.n_kfolds

        self.data: LazySNPAEData
        self.train_data: LazySNPAEDataTuple
        self.test_data: LazySNPAEDataTuple

        # === Run / Save / Load Variables ===
        self.run_attributes: set[str] = {
            "sne",
            "wavelength",
            "nspecta_per_sn",
            "sn_dim",
            "spec_dim",
            "wl_dim",
        }
        self.save_attributes: set[str] = self.run_attributes
        self.load_attributes: set[str] = self.save_attributes

        self.sne: SNeDataFrame
        self.wavelength: npt.NDArray[float]

        # Data Dimensions
        self.nspectra_per_sn: npt.NDArray[int]  # Created in self.get_dims
        self.sn_dim: int  # Created in self.get_dims
        self.spec_dim: int  # Created in self.get_dims
        self.wl_dim: int  # Created in self.get_dims

        # === Result Variables ===
        self.results: DataStepResult

        # === Analysis Variables ===
        self.analysis: DataStepAnalysis = self.options.analysis or DataStepAnalysis()

    @override
    def _is_setup(self, *args: "Any", **kwargs: "Any") -> bool:
        for attr in self.setup_attributes:
            if not self.has_attributes([attr]):
                self.log.debug(f"{self.name} is not setup because {attr} is missing")
                return False
        return True

    @override
    def _setup(self, *args: "Any", **kwargs: "Any") -> None:
        colourlaw = self.options.colourlaw
        if colourlaw is not None:
            _, colourlaw = np.loadtxt(colourlaw, unpack=True)
        self.colourlaw = colourlaw

        # Get astropy.cosmology model associated with provided cosmological_model string
        self.cosmological_model = getattr(cosmo, self.options.cosmological_model)

        # Get sncosmo SALTSource associated with provided salt_model string
        #   If salt_model is a valid Path, pass it to the SALTSource as the modeldir
        salt_model = self.options.salt_model
        if isinstance(salt_model, Path):
            if "salt2" in str(salt_model):
                self.salt_model = sncosmo.SALT2Source(salt_model)
            elif "salt3" in str(salt_model):
                self.salt_model = sncosmo.SALT3Source(salt_model)
        else:
            self.salt_model = sncosmo.get_source(salt_model)

        # === Config Variables ===

        # Train / Test Split
        if self.data_splits is not None:
            with self.data_splits.open("rb") as io:
                self.splits = tomllib.load(io)
            nfold_0 = self.splits.get("0", {})
            n_train = len(nfold_0.get("train", []))
            n_test = len(nfold_0.get("test", []))
            self.train_frac = n_train / (n_train + n_test)

        self.test_frac = 1 - self.train_frac
        self.n_kfolds = (
            int(1 / self.test_frac)
            if self.options.n_kfolds is None
            else self.options.n_kfolds
        )

        self.data = LazySNPAEData(self.out_data)
        self.train_data = LazySNPAEDataTuple(
            self.out_train / f"kfold_{i:d}.npz" for i in range(self.n_kfolds)
        )
        self.test_data = LazySNPAEDataTuple(
            self.out_test / f"kfold_{i:d}.npz" for i in range(self.n_kfolds)
        )

    @override
    def _has_run(self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(self.run_attributes)

    @override
    def _run(self, *args: "Any", **kwargs: "Any") -> None:
        self.load_sne()
        self.calculate_salt_flux()
        self.get_dims()
        self.prepare_data_arrays()
        self.split_train_test()

    @override
    def _is_saved(self, *args: "Any", **kwargs: "Any") -> bool:
        if not self.out_data.exists():
            self.log.debug(
                f"{self.name} is not saved as {self.out_data} does not exist"
            )
            return False
        if not self.out_sne.exists():
            self.log.debug(f"{self.name} is not saved as {self.out_sne} does not exist")
            return False
        if not self.out_train.exists():
            self.log.debug(
                f"{self.name} is not saved as {self.out_train} does not exist"
            )
            return False
        if not self.out_test.exists():
            self.log.debug(
                f"{self.name} is not saved as {self.out_test} does not exist"
            )
            return False
        if len(list(self.out_train.iterdir())) == 0:
            self.log.debug(
                f"{self.name} is not saved as {self.out_train} does not contain any files"
            )
            return False
        if len(list(self.out_test.iterdir())) == 0:
            self.log.debug(
                f"{self.name} is not saved as {self.out_test} does not contain any files"
            )
            return False
        return True

    @override
    def _save(self, *args: "Any", **kwargs: "Any") -> None:
        if self.force or not self.out_sne.exists():
            self.log.debug(f"Saving SNe DataFrame to {self.out_sne}")
            self.sne.to_pickle(self.out_sne)

        if self.force or not self.out_data.exists():
            self.log.debug(f"Saving data arrays to {self.out_data}")
            np.savez_compressed(
                self.out_data,
                **self.data.model_dump(exclude={"name"}),
            )

        self.data.clear()

        for i, train_data in enumerate(self.train_data):
            out_train = self.out_train / f"kfold_{i:d}.npz"
            if self.force or not out_train.exists():
                self.log.debug(f"Saving #{i} training data array to {out_train}")
                np.savez_compressed(
                    out_train,
                    **train_data.model_dump(exclude={"name"}),
                )
                train_data.clear()

        for i, test_data in enumerate(self.test_data):
            out_test = self.out_test / f"kfold_{i:d}.npz"
            if self.force or not out_test.exists():
                self.log.debug(f"Saving #{i} testing data array to {out_test}")
                np.savez_compressed(
                    out_test,
                    **test_data.model_dump(exclude={"name"}),
                )
                test_data.clear()

    @override
    def _load(self, *args: "Any", **kwargs: "Any") -> None:
        # Load SNe DataFrames
        self.log.debug(f"Loading SNe dataframe from {self.out_sne}")
        self.sne = pd.read_pickle(self.out_sne)

        # Calculate data dimensions
        self.get_dims()

    @override
    def _has_results(self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(["results"])

    @override
    def _result(self, *args: "Any", **kwargs: "Any") -> None:
        results = {}
        results["data"] = self.data
        results["dir"] = self.data_dir
        results["train_data"] = self.train_data
        results["test_data"] = self.test_data
        results["colourlaw"] = self.colourlaw
        results["min_redshift"] = self.min_redshift
        results["max_redshift"] = self.max_redshift
        results["min_phase"] = self.min_phase
        results["max_phase"] = self.max_phase
        results["min_wavelength"] = self.min_wavelength
        results["max_wavelength"] = self.max_wavelength
        results["train_frac"] = self.train_frac
        results["sn_dim"] = self.sn_dim
        results["spec_dim"] = self.spec_dim
        results["wl_dim"] = self.wl_dim
        self.results = DataStepResult.model_validate(results)
        self.data.clear()
        self.train_data.clear()
        self.test_data.clear()

    @override
    def _was_analysed(self, *args: "Any", **kwargs: "Any") -> bool:
        if self.analysis.plot_spectra is not None:
            if not isinstance(self.analysis.plot_spectra, list):
                self.analysis.plot_spectra = [self.analysis.plot_spectra]
            for opts in self.analysis.plot_spectra:
                name = "spectra" if opts.name is None else opts.name
                savepath = (
                    self.paths.plots / str(self.seed) / f"{name}.{opts.ext}"
                    if opts.savepath is None
                    else opts.savepath
                )
                if not savepath.exists():
                    self.log.debug(
                        f"{self.name} is missing analyses as {savepath} does not exist"
                    )
                    return False

        if self.analysis.plot_summary is not None:
            if not isinstance(self.analysis.plot_summary, list):
                self.analysis.plot_summary = [self.analysis.plot_summary]
            for opts in self.analysis.plot_summary:
                name = "summary" if opts.name is None else opts.name
                savepath = (
                    self.paths.plots / str(self.seed) / f"{name}.{opts.ext}"
                    if opts.savepath is None
                    else opts.savepath
                )
                if not savepath.exists():
                    self.log.debug(
                        f"{self.name} is missing analyses as {savepath} does not exist"
                    )
                    return False

        if self.analysis.plot_comparison is not None:
            if not isinstance(self.analysis.plot_comparison, list):
                self.analysis.plot_comparison = [self.analysis.plot_comparison]
            for opts in self.analysis.plot_comparison:
                name = "comparison" if opts.name is None else opts.name
                savepath = (
                    self.paths.plots / str(self.seed) / f"{name}.{opts.ext}"
                    if opts.savepath is None
                    else opts.savepath
                )
                if not savepath.exists():
                    self.log.debug(
                        f"{self.name} is missing analyses as {savepath} does not exist"
                    )
                    return False

        return not self.analysis.force

    def _plot_spectra(self) -> None:
        if self.analysis.plot_spectra is not None:
            if not isinstance(self.analysis.plot_spectra, list):
                self.analysis.plot_spectra = [self.analysis.plot_spectra]
            for opts in self.analysis.plot_spectra:
                o = opts.model_copy(deep=True)
                if o.name is None:
                    o.name = "spectra"
                if o.savepath is None:
                    o.savepath = self.paths.plots / str(self.seed)
                o.savepath.mkdir(parents=True, exist_ok=True)
                if (o.savepath / f"{o.name}.{o.ext}").exists():
                    continue
                self.log.debug(f"Plotting {o.name}")
                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": self.name}
                SpectraPlotter.plot_spectra(
                    self.results.data,
                    o,
                    mask=self.results.data.mask,
                    sn_mask=self.results.data.sn_mask,
                    spec_mask=self.results.data.spec_mask,
                    wl_mask=self.results.data.wl_mask,
                )

    def _plot_summary(self) -> None:
        if self.analysis.plot_summary is not None:
            if not isinstance(self.analysis.plot_summary, list):
                self.analysis.plot_summary = [self.analysis.plot_summary]
            for opts in self.analysis.plot_summary:
                for dataset in ["", "train_", "test_"]:
                    o = opts.model_copy(deep=True)
                    if o.name is None:
                        o.name = f"{dataset}summary"
                    if o.savepath is None:
                        o.savepath = self.paths.plots / str(self.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if (o.savepath / f"{o.name}.{o.ext}").exists():
                        continue
                    self.log.debug(f"Plotting {o.name}")
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"label": f"{dataset}{self.name}"}
                    data = getattr(self.results, f"{dataset}data")
                    if dataset:
                        data = data[0]
                    data.load()
                    SpectraPlotter.plot_summary(
                        data,
                        o,
                        mask=data.mask,
                        sn_mask=data.sn_mask,
                        spec_mask=data.spec_mask,
                        wl_mask=data.wl_mask,
                    )

    def _plot_comparison(self) -> None:
        if self.analysis.plot_comparison is not None:
            if not isinstance(self.analysis.plot_comparison, list):
                self.analysis.plot_comparison = [self.analysis.plot_comparison]
            for opts in self.analysis.plot_comparison:
                for dataset in ["", "train_", "test_"]:
                    o = opts.model_copy(deep=True)
                    if o.name is None:
                        o.name = f"{dataset}comparison"
                    if o.savepath is None:
                        o.savepath = self.paths.plots / str(self.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if (o.savepath / f"{o.name}.{o.ext}").exists():
                        continue
                    self.log.debug(f"Plotting {o.name}")
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"label": f"{dataset}{self.name}"}
                    data = getattr(self.results, f"{dataset}data")
                    if dataset:
                        data = data[0]
                    SpectraPlotter.plot_comparison(
                        data,
                        o,
                        mask=data.mask,
                        sn_mask=data.sn_mask,
                        spec_mask=data.spec_mask,
                        wl_mask=data.wl_mask,
                    )

    @override
    def _analyse(self, *args: "Any", **kwargs: "Any") -> None:
        if self.analysis.skip:
            return
        self._plot_spectra()
        self._plot_summary()
        self._plot_comparison()

    @override
    def _is_cleaned(self, *args: "Any", **kwargs: "Any") -> bool:
        return True

    @override
    def _clear(
        self,
        *args: "Any",
        setup: bool = False,
        run: bool = False,
        save: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        **kwargs: "Any",
    ) -> None:
        if setup:
            self.clear_attributes(self.setup_attributes)

        if run:
            self.clear_attributes(self.run_attributes)

        if save:
            self.clear_attributes(self.save_attributes)

        if load:
            self.clear_attributes(self.load_attributes)

        if result:
            self.clear_attributes("results")

        if analyse:
            self.analysis = self.options.analysis or DataStepAnalysis()

    #
    # === Instance Methods ===
    #

    def load_sne(self) -> None:
        self.log.debug(f"Loading data from `meta` file: {self.meta}")
        sne_dtypes = {
            "id": str,
            "sn": str,
            "phase": float,
            "z": float,
            "mB": float,
            "x0": float,
            "x1": float,
            "c": float,
            "path": str,
            "hubble_resid": float,
        }
        sne_data = pd.read_csv(self.meta, header=0, dtype=sne_dtypes)
        # Update paths relative to self.meta
        sne_data.path = sne_data.path.apply(
            lambda path: str(resolve_path(Path(path), relative_path=self.meta.parent))
        )

        self.log.debug(f"Loading data from `idr` file: {self.idr}")
        dphase_dtypes = {
            "sn": str,
            "mjd": float,
            "dphase": float,
        }
        dphase = pd.read_csv(
            self.idr, sep="\\s+", names=["sn", "mjd", "dphase"], dtype=dphase_dtypes
        )
        sne_data = sne_data.merge(dphase, on="sn", how="left")

        self.log.debug(f"Loading data from `mask` file: {self.mask}")
        mask_dtypes = {
            "sn": str,
            "id": str,
            "flag": int,
            "wl_mask_min": float,
            "wl_mask_max": float,
        }
        mask = pd.read_csv(
            self.mask,
            sep="\\s+",
            names=["sn", "id", "flag", "wl_mask_min", "wl_mask_max"],
            dtype=mask_dtypes,
        )

        # Fill NaN values
        mask.wl_mask_min = mask.wl_mask_min.fillna(np.inf)
        mask.wl_mask_max = mask.wl_mask_max.fillna(-np.inf)

        mask.id = mask.sn + "_" + mask.id
        sne_data = sne_data.merge(mask, on=["sn", "id"], how="left")

        # Fill missing values with default values
        sne_data.wl_mask_min = sne_data.wl_mask_min.fillna(self.min_wavelength)
        sne_data.loc[sne_data.wl_mask_min < self.min_wavelength, "wl_mask_min"] = (
            self.min_wavelength
        )
        sne_data.wl_mask_max = sne_data.wl_mask_max.fillna(self.max_wavelength)
        sne_data.loc[sne_data.wl_mask_max > self.max_wavelength, "wl_mask_max"] = (
            self.max_wavelength
        )
        sne_data.flag = sne_data.flag.fillna(1)

        self.log.debug("Merging SNe data")
        # Split data into two dataframes

        # A SN dataframe which contains one row per SN, and the following columns
        sne_cols = ["sn", "MB", "x0", "x1", "c", "z", "hubble_resid", "mjd", "dphase"]
        sne = sne_data[sne_cols].drop_duplicates().reset_index(drop=True)

        # A dataframe which contains one row per spectra, and the following columns
        # Note that we keep the sn column so that we can match each spectra with their SN
        spec_cols = [
            "sn",
            "id",
            "phase",
            "path",
            "flag",
            "wl_mask_min",
            "wl_mask_max",
        ]
        spectra = sne_data[spec_cols].reset_index(drop=True)

        self.log.debug("Loading spectra data")
        spectra_dtype = {"wave": float, "flux": float, "sigma": float}
        spectra["data"] = [
            pd.read_csv(spec.path, dtype=spectra_dtype)
            for _, spec in spectra.iterrows()
        ]

        self.log.debug("Linking spectra to their associated SNe")
        sne["spectra"] = sne.sn.apply(
            lambda sn_name: spectra[spectra.sn == sn_name].reset_index(drop=True),
        )

        # Final structure is 1 row per SN with columns:
        #   sn:           str       = SN Name
        #   mB:           float     = Redshift-dependant absolute magnitude of a ``standard'' SN Ia
        #   x0:           float     = SALT $x_{0}$ parameter, with the SN apparent magnitude $m_{b}=\log_{10}(x0)$
        #   x1:           float     = SALT $x_{1}$ stretch parameter
        #   c:            float     = SALT $\mathcal{C}$ colour parameter
        #   z:            float     = Redshift of SN
        #   hubble_resid: float     = Hubble Residual
        #   dphase:       float     = Phase offset
        #   spectra:      DataFrame = SN Spectra with columns:
        #
        #       sn:             str       = SN Name
        #       id:             str       = Spectra Id
        #       phase:          float     = Spectral phase relative to peak mag in days
        #       path:           str       = Path to spectra, relative to metapath
        #       flag:           int       = Quality of spectra
        #       wl_mask_min:    float     = Min wavelength of spectra
        #       wl_mask_max:    float     = Max wavelength of spectra
        #       data:           DataFrame = Spectral data with columns:
        #
        #           wave:  Series[float]  = wavelength in AA
        #           flux:  Series[float]  = flux
        #           sigma: Series[float]  = flux error

        self.sne = sne

    def calculate_salt_flux(self) -> None:
        self.log.debug("Calculating SALT fluxes")

        def get_salt_flux(
            wavelength: "npt.NDArray[float]",
            tobs: float = 0.0,
            z: float = 0.0,
            x0: float = 1.0,
            x1: float = 0.0,
            c: float = 0.0,
            zref: float = 0.05,
        ) -> "npt.NDArray[float]":
            self.salt_model.set(x0=x0, x1=x1, c=c)
            return (
                self.salt_model.flux(phase=tobs, wave=wavelength)
                * (
                    (
                        self.cosmological_model.luminosity_distance(z)
                        / self.cosmological_model.luminosity_distance(zref)
                    )
                    ** 2
                )
                * ((1 + z) / (1 + zref))
                * 1e15
            )

        for _, sn in self.sne.iterrows():
            for _, spectra in sn["spectra"].iterrows():
                spectra["data"]["salt_flux"] = get_salt_flux(
                    spectra["data"]["wave"].to_numpy(),
                    tobs=spectra["phase"],
                    z=sn["z"],
                    x0=sn["x0"],
                    x1=sn["x1"],
                    c=sn["c"],
                )

    def get_dims(self) -> None:
        self.log.debug("Calculating data dimensions")
        self.sn_dim = len(self.sne)
        self.log.debug(f"Number of SNe: {self.sn_dim}")

        # Maximum number of observations for any given SN
        self.nspectra_per_sn = np.array(
            [len(spectra) for spectra in self.sne["spectra"]],
        )
        self.spec_dim = int(self.nspectra_per_sn.max())
        self.log.debug(
            f"Maximum number of observations for any given SN: {self.spec_dim}",
        )

        # Wavelength grid
        # Since all spectra share the same wavelength grid
        # Just get the wavelength grid of the first spectrum
        self.wavelength = self.sne["spectra"][0]["data"][0]["wave"].to_numpy()
        self.wl_dim = len(self.wavelength)
        self.log.debug(f"Length of wavelength grid: {self.wl_dim}")

    def prepare_data_arrays(self) -> None:
        self.log.debug("Preparing data arrays")
        # Each element of data is a 3D Array of shape (SNDim x SpecDim x DataDim) where:
        #   SNDim = Number of SNe
        #   SpecDim = Maximum number of observations for any given SN (padded if needed)
        #   DataDim = Length of datatype

        # Allows for filling an array with padding
        phase_axis = self.nspectra_per_sn.copy()
        phase_axis.fill(self.spec_dim)

        # --- Get Parameters ---
        data = {}

        # Given an array of shape (sn_dim x N <= spec_dim)
        # Create an array of shape sn_dim by spec_dim padding if needed
        def pad[T: np.generic](
            arr: "Iterable[Sequence[T | npt.NDArray[T]]]",
            padding: "T | npt.NDArray[T]",
        ) -> "npt.NDArray[T]":
            if isinstance(padding, np.ndarray):
                padded_arr: npt.NDArray[T] = np.full(
                    (self.sn_dim, self.spec_dim, *padding.shape),
                    padding,
                )
            else:
                padded_arr = np.full((self.sn_dim, self.spec_dim), padding)
            for i, row in enumerate(arr):
                row_length = len(row)
                padded_arr[i, :row_length] = row
            return padded_arr

        # Given a list of value-per-row of length sn_dim
        # Fill each row with spec_dim repeats of that row's value
        def fill_rows[T: np.generic](values: "npt.NDArray[T]") -> "npt.NDArray[T]":
            return np.repeat(values, phase_axis).reshape((self.sn_dim, self.spec_dim))

        # Index of each SNe
        data["ind"] = fill_rows(np.array(range(self.sn_dim)))

        # Number of spectra per SNe
        data["nspectra"] = fill_rows(self.nspectra_per_sn)

        # Get SNe parameters
        sne_params = {
            "sn_name": "sn",
            "dphase": "dphase",
            "redshift": "z",
            "x0": "x0",
            "x1": "x1",
            "c": "c",
            "mB": "MB",
            "hubble_residual": "hubble_resid",
        }

        for data_key, sne_key in sne_params.items():
            data[data_key] = fill_rows(
                self.sne[sne_key].to_numpy(),
            )

        data["luminosity_distance"] = self.cosmological_model.luminosity_distance(
            data["redshift"],
        ).value

        # Get Parameters from spectra
        max_id_len = max(len(spectra["id"]) for spectra in self.sne["spectra"])
        spectra_params = {
            "spectra_id": ("id", np.str_("-" * max_id_len)),
            "phase": ("phase", np.float32(-np.inf)),
            "wl_mask_min": ("wl_mask_min", np.float32(np.inf)),
            "wl_mask_max": ("wl_mask_max", np.float32(-np.inf)),
        }

        for data_key, (spectra_key, padding) in spectra_params.items():
            data[data_key] = pad(
                [spectra[spectra_key].to_numpy() for spectra in self.sne["spectra"]],
                padding,
            )
        # Get spectral data parameters
        spectral_data_params = {
            "amplitude": ("flux", -np.inf * np.ones(self.wl_dim)),
            "sigma": ("sigma", -np.inf * np.ones(self.wl_dim)),
            "salt_flux": ("salt_flux", -np.inf * np.ones(self.wl_dim)),
        }

        for data_key, (spectral_data_key, padding) in spectral_data_params.items():
            data[data_key] = pad(
                [
                    [
                        spectral_data[spectral_data_key].to_numpy()
                        for spectral_data in spectra["data"]
                    ]
                    for spectra in self.sne["spectra"]
                ],
                padding,
            )

        data["wavelength"] = np.tile(self.wavelength, (self.sn_dim, self.spec_dim, 1))

        # Ensure everything has the right number of axes
        n_axes = 2
        for k, v in data.items():
            if len(v.shape) == n_axes:
                data[k] = v[..., np.newaxis]

        # def nearest_mask[T: np.number[Any]](
        #     arr: "npt.NDArray[T]",
        #     min_val: "T | npt.NDArray[T]",
        #     max_val: "T | npt.NDArray[T]",
        # ) -> "npt.NDArray[bool]":
        #     if not isinstance(min_val, np.ndarray):
        #         min_val = cast("npt.NDArray[T]", np.array(min_val))
        #     if not isinstance(max_val, np.ndarray):
        #         max_val = cast("npt.NDArray[T]", np.array(max_val))
        #     base_mask = (min_val <= arr) & (arr <= max_val)
        #
        #     # Pad left and right
        #     pad_left = np.pad(
        #         base_mask[:, :, :-1],
        #         ((0, 0), (0, 0), (1, 0)),
        #         mode="constant",
        #         constant_values=False,
        #     )
        #     pad_right = np.pad(
        #         base_mask[:, :, 1:],
        #         ((0, 0), (0, 0), (0, 1)),
        #         mode="constant",
        #         constant_values=False,
        #     )
        #
        #     # Compute distances to the boundaries
        #     dist_to_min = np.abs(arr - min_val)
        #     dist_to_max = np.abs(arr - max_val)
        #
        #     # Left edge logic
        #     expand_left = (
        #         (~base_mask) & pad_right & (dist_to_min < np.roll(dist_to_min, -1))
        #     )
        #     # Right edge logic
        #     expand_right = (
        #         (~base_mask) & pad_left & (dist_to_max < np.roll(dist_to_max, 1))
        #     )
        #     # Combine the masks
        #     return base_mask | expand_left | expand_right

        # Create a mask of wavelength outside of the wavelength limits
        data["mask"] = np.full(
            (self.sn_dim, self.spec_dim, self.wl_dim), fill_value=True
        )

        self.log.debug("Initial:")
        self.get_unmasked_dims(data["mask"])

        valid_redshift_mask = (self.min_redshift <= data["redshift"]) & (
            self.max_redshift >= data["redshift"]
        )
        data["sn_mask"] = sn_mask = valid_redshift_mask[:, :1, :]
        data["mask"] &= sn_mask

        self.log.debug(
            f"Valid Redshifts ({self.min_redshift} <= z <= {self.max_redshift}):"
        )
        self.get_unmasked_dims(data["mask"])

        valid_phase_mask = (self.min_phase <= data["phase"]) & (
            self.max_phase >= data["phase"]
        )
        data["spec_mask"] = spec_mask = (
            valid_phase_mask & (data["phase"] > -np.inf) & (data["phase"] < np.inf)
        )
        data["mask"] &= spec_mask

        self.log.debug(f"Valid Phases ({self.min_phase} <= p <= {self.max_phase}):")
        self.get_unmasked_dims(data["mask"])

        # valid_wavelength_mask = nearest_mask(
        #     data["wavelength"], data["wl_mask_min"], data["wl_mask_max"]
        # )
        valid_wavelength_mask = (data["wavelength"] >= data["wl_mask_min"]) & (
            data["wavelength"] <= data["wl_mask_max"]
        )
        data["wl_mask"] = wl_mask = valid_wavelength_mask
        data["mask"] &= wl_mask

        self.log.debug(
            f"Valid Wavelengths ({self.min_wavelength} <= wl <= {self.max_wavelength}):"
        )
        self.get_unmasked_dims(data["mask"])

        # Mask any huge laser lines, Na D (5674 - 5692A)
        # TODO: Make these options
        # these are large jumps in flux, localized over a few wavelength bins
        laser_wl_start = np.float32(5000.0)
        laser_wl_end = np.float32(8000.0)
        laser_width = 2  # in units of wavelength bins
        laser_height = 0.4  # fractional increase in amplitude over neighbours to be considered laser

        # laser_wl_mask = nearest_mask(data["wavelength"], laser_wl_start, laser_wl_end)
        laser_wl_mask = (data["wavelength"] >= laser_wl_start) & (
            data["wavelength"] <= laser_wl_end
        )

        laser_amp = np.full(data["amplitude"].shape, np.nan)
        laser_amp[laser_wl_mask] = data["amplitude"][laser_wl_mask]

        laser_amp_min = np.roll(laser_amp, (0, 0, -laser_width))
        laser_amp_max = np.roll(laser_amp, (0, 0, laser_width))

        laser_amp_smooth = (
            0.5 * (laser_amp_min + laser_amp_max) * laser_wl_mask.astype(np.float32)
        )

        laser_amp = np.where(
            np.isfinite(laser_amp) & np.isfinite(laser_amp_smooth),
            laser_amp,
            np.zeros_like(laser_amp),
        )
        laser_amp_smooth = np.where(
            np.isfinite(laser_amp) & np.isfinite(laser_amp_smooth),
            laser_amp_smooth,
            np.zeros_like(laser_amp_smooth),
        )

        laser_mask = (laser_amp - laser_amp_smooth) > laser_height

        while laser_width > 0:
            laser_mask_min = np.roll(laser_mask, (0, 0, -1))
            laser_mask_max = np.roll(laser_mask, (0, 0, 1))
            laser_mask = laser_mask | laser_mask_min | laser_mask_max
            laser_width -= 1

        data["laser_mask"] = laser_mask
        data["mask"] &= ~laser_mask

        self.log.debug("Laser Lines:")
        self.get_unmasked_dims(data["mask"])

        # --- Finalise Data ---
        # Rescale phase to time such that:
        #   time = 0 -> phase = min_phase
        #   time = 1 -> phase = max_phase
        time = data["phase"][spec_mask]
        min_phase = time.min()
        if not np.isinf(self.min_phase):
            min_phase = min(self.min_phase, min_phase)
        self.log.debug(f"{self.min_phase = }, {time.min() = }, {min_phase = }")
        max_phase = time.max()
        if not np.isinf(self.max_phase):
            max_phase = max(self.max_phase, max_phase)
        self.log.debug(f"{self.max_phase = }, {time.max() = }, {max_phase = }")
        data["time"] = (data["phase"] - min_phase) / (max_phase - min_phase)
        # data["time"][~spec_mask] = -np.inf

        # Scale observed uncertainty to account for fitting degrees of freedom, and an error floor
        data["sigma"] = 1.4 * data["sigma"] + 4e-10

        data["mask"] = data["mask"].astype(np.int32)

        self.data.model_validate(data)

    def get_unmasked_dims(
        self, mask: "npt.NDArray[np.int32] | None" = None
    ) -> tuple[int, int, int]:
        self.log.debug("Calculating unmasked data dimensions")
        if mask is None:
            mask = self.data.mask

        total_mask = np.ones_like(mask).astype(bool)

        unmasked_sn_dim = mask.max(axis=-1).max(axis=-1).sum()
        total_sn_dim = total_mask.max(axis=-1).max(axis=-1).sum()
        self.log.debug(
            f"Number of unmasked SNe: {unmasked_sn_dim} ({unmasked_sn_dim / total_sn_dim:.2%})"
        )

        unmasked_spec_dim = mask.max(axis=-1).sum()
        total_spec_dim = total_mask.max(axis=-1).sum()
        self.log.debug(
            f"Number of unmasked spectra: {unmasked_spec_dim} ({unmasked_spec_dim / total_spec_dim:.2%})"
        )

        unmasked_wl_dim = mask.sum()
        total_wl_dim = total_mask.sum()
        self.log.debug(
            f"Number of unmasked WL bins: {unmasked_wl_dim} ({unmasked_wl_dim / total_wl_dim:.2%})"
        )

        return unmasked_sn_dim, unmasked_spec_dim, unmasked_wl_dim

    def split_train_test(self) -> None:
        if not hasattr(self, "splits"):
            self.splits = {
                str(i): {"train": [], "test": [], "validate": []}
                for i in range(self.n_kfolds)
            }

            # Train test split
            ind_split = int(self.sn_dim * self.train_frac)

            # Select train_frac for training, the rest for testing
            inds = np.arange(0, self.sn_dim)
            self.rng.shuffle(inds)

            # Split into k cross validation sets
            for kfold in range(self.n_kfolds):
                inds_k = np.roll(inds, kfold * inds.shape[0] // self.n_kfolds)
                inds_train = inds_k[:ind_split]
                inds_test = inds_k[ind_split:]
                self.splits[str(kfold)]["train"] = self.data.model_dump()["sn_name"][
                    inds_train, 0, 0
                ]
                self.splits[str(kfold)]["test"] = self.data.model_dump()["sn_name"][
                    inds_test, 0, 0
                ]

        # Split into k cross validation sets
        for kfold in range(self.n_kfolds):
            inds_train = np.nonzero(
                np.isin(
                    self.data.model_dump()["sn_name"][:, 0, 0],
                    self.splits[str(kfold)]["train"],
                )
            )[0]
            inds_test = np.nonzero(
                np.isin(
                    self.data.model_dump()["sn_name"][:, 0, 0],
                    self.splits[str(kfold)]["test"],
                )
            )[0]

            n_axes = 3
            self.train_data[kfold].model_validate({
                key: val[inds_train, :, :] if val.ndim == n_axes else val[inds_train, :]
                for key, val in self.data.model_dump().items()
                if isinstance(val, np.ndarray)
            })

            self.test_data[kfold].model_validate({
                key: val[inds_test, :, :] if val.ndim == n_axes else val[inds_test, :]
                for key, val in self.data.model_dump().items()
                if isinstance(val, np.ndarray)
            })

        n_train_sn = self.train_data[0].amplitude.shape[0]
        n_test_sn = self.test_data[0].amplitude.shape[0]
        self.log.debug(
            f"n_train_sn: {n_train_sn} ({100 * n_train_sn / self.sn_dim}%) + n_test_sn: {n_test_sn} ({100 * n_test_sn / self.sn_dim}%) = {n_train_sn + n_test_sn} ({100 * (n_train_sn + n_test_sn) / self.sn_dim}%)",
        )


class DataStep(Variant[DataStepConfig, Data]):
    id: ClassVar[str] = "data"

    def __init__(self, config: "DataStepConfig") -> None:
        super().__init__(config)

        self.bases: dict[str, dict[str, Any]] = {}
        self.plots: dict[str, dict[str, Any]] = {}

    def _plot_comparison_pre(
        self, variant: Data, *args: "Any", **kwargs: "Any"
    ) -> None:
        if variant.analysis.plot_comparison is not None:
            if not isinstance(variant.analysis.plot_comparison, list):
                variant.analysis.plot_comparison = [variant.analysis.plot_comparison]
            for opts in variant.analysis.plot_comparison:
                for dataset in ["", "train_", "test_"]:
                    self._setup(*args, **{**kwargs, "variants": [opts.base]})
                    if opts.name is None:
                        opts.name = f"{dataset}comparison"
                    name = f"{opts.name}.{opts.ext}"
                    self.bases[name] = self.bases.get(
                        name, {"wl": None, "amp": None, "sigma": None, "mask": None}
                    )
                    base_wl = self.bases[name]["wl"]
                    base_amp = self.bases[name]["amp"]
                    base_sigma = self.bases[name]["sigma"]
                    base_mask = self.bases[name]["mask"]
                    if base_amp is None:
                        data = getattr(self.results[opts.base], f"{dataset}data")
                        if dataset:
                            data = data[0]
                        data.load()
                        (
                            wl,
                            amplitude,
                            sigma,
                            _sn_name,
                            _time,
                            mask,
                            _sn_mask,
                            _spec_mask,
                            _wl_mask,
                        ) = SpectraPlotter.prep(
                            data,
                            opts,
                            mask=data.mask,
                            sn_mask=data.sn_mask,
                            spec_mask=data.spec_mask,
                            wl_mask=data.wl_mask,
                        )
                        base_wl = wl
                        base_amp = amplitude
                        base_sigma = sigma
                        base_mask = np.logical_not(mask)
                    self.bases[name]["wl"] = base_wl
                    self.bases[name]["amp"] = base_amp
                    self.bases[name]["sigma"] = base_sigma
                    self.bases[name]["mask"] = base_mask
                    if not dataset:
                        opts.base_wl = base_wl
                        opts.base_amp = base_amp
                        opts.base_sigma = base_sigma
                        opts.base_mask = base_mask
                        opts.plot_base = True

    def _plot_summary(self, variant: Data) -> None:
        if variant.analysis.plot_summary is not None:
            for opts in variant.analysis.plot_summary:
                for dataset in ["", "train_", "test_"]:
                    o = opts.model_copy(deep=True)
                    if o.name is None:
                        o.name = f"{dataset}summary"
                    name = f"{o.name}.{o.ext}"
                    self.plots[name] = self.plots.get(name, {"fig": None, "ax": None})
                    fig = self.plots[name]["fig"]
                    ax = self.plots[name]["ax"]
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"label": f"{dataset}{variant.name}"}
                    data = getattr(variant.results, f"{dataset}data")
                    if dataset:
                        data = data[0]
                    data.load()
                    fig, ax = SpectraPlotter.plot_summary(
                        data,
                        o,
                        mask=data.mask,
                        sn_mask=data.sn_mask,
                        spec_mask=data.spec_mask,
                        wl_mask=data.wl_mask,
                        fig=fig,
                        ax=ax,
                        save=False,
                        force=True,
                    )
                    self.plots[name]["fig"] = fig
                    self.plots[name]["ax"] = ax

    def _plot_comparison_post(self, variant: Data) -> None:
        if variant.analysis.plot_comparison is not None:
            for opts in variant.analysis.plot_comparison:
                for dataset in ["", "train_", "test_"]:
                    o = opts.model_copy(deep=True)

                    if o.name is None:
                        o.name = f"{dataset}comparison"
                    name = f"{o.name}.{o.ext}"
                    self.plots[name] = self.plots.get(
                        name, {"fig": None, "ax": None, "base": True}
                    )
                    fig = self.plots[name]["fig"]
                    ax = self.plots[name]["ax"]
                    o.plot_base = self.plots[name]["base"]
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"label": f"{dataset}{variant.name}"}
                    data = getattr(variant.results, f"{dataset}data")
                    if dataset:
                        data = data[0]
                    data.load()

                    o.base_wl = self.bases[name]["wl"]
                    o.base_amp = self.bases[name]["amp"]
                    o.base_sigma = self.bases[name]["sigma"]
                    o.base_mask = self.bases[name]["mask"]

                    fig, ax = SpectraPlotter.plot_comparison(
                        data,
                        o,
                        mask=data.mask,
                        sn_mask=data.sn_mask,
                        spec_mask=data.spec_mask,
                        wl_mask=data.wl_mask,
                        fig=fig,
                        ax=ax,
                        save=False,
                        force=True,
                    )
                    self.plots[name]["fig"] = fig
                    self.plots[name]["ax"] = ax
                    self.plots[name]["base"] = False

    @override
    def _analyse(
        self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant_name in variants:
            variant = self.variants[variant_name]
            variant.set_seed()
            variant.log.info(f"Analysing {variant.name}")
            variant.result(*args, **{**kwargs, "variants": [variant_name]})
            if variant.analysis.skip:
                continue

            self._plot_comparison_pre(variant, *args, **kwargs)

            variant._analyse(*args, **{**kwargs, "variants": [variant_name]})

            self._plot_summary(variant)

            self._plot_comparison_post(variant)

            variant.log.info(f"Finished analysing {variant.name}")

    @override
    @callback
    def analyse(self, *args: "Any", **kwargs: "Any") -> None:
        super().analyse(*args, **kwargs)
        if len(self.variants) > 1 and (
            not all(variant.analysis.skip for variant in self.variants.values())
        ):
            for name, opts in self.plots.items():
                savepath = self.paths.plots / name
                if savepath.exists():
                    continue
                self.log.debug(f"Plotting {name}")
                fig = opts["fig"]
                ax = opts["ax"]
                fig = Plotter.save(fig, savepath)
                Plotter.close(fig, ax)


DataStep.register_step(Data)
