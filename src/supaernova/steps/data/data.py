from typing import TYPE_CHECKING, ClassVar, cast, override
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import cosmology as cosmo
import sncosmo

from supaernova.steps import Step
from supaernova.utils import resolve_path
from supaernova.steps.variants import Variant
from supaernova.analysis.spectra import SpectraPlotter
from supaernova.analysis.analysis import Plotter
from supaernova.configs.steps.data import DataStepResult, DataStepAnalysis

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable, Sequence

    from numpy import typing as npt

    from supaernova.configs.steps.data import DataStepConfig

    SNeDataFrame = pd.DataFrame


class Data(Step):
    def __init__(self, config: "DataStepConfig") -> None:
        super().__init__(config)

        # === Previous Step Variables ===

        # === Config Variables ===
        # --- Required ---
        self.data_dir: Path = self.options.data_dir
        self.meta: Path = self.options.meta
        self.idr: Path = self.options.idr
        self.mask: Path = self.options.mask
        self.colourlaw: npt.NDArray[np.float64] | None
        self.train_frac: float = self.options.train_frac

        # --- Optional ---
        self.cosmological_model: cosmo.FlatLambdaCDM
        self.salt_model: sncosmo.SALT2Source | sncosmo.SALT3Source
        self.min_phase: float = self.options.min_phase
        self.max_phase: float = self.options.max_phase
        self.min_redshift: float = self.options.min_redshift
        self.max_redshift: float = self.options.max_redshift
        self.min_wavelength: float = self.options.min_wavelength
        self.max_wavelength: float = self.options.max_wavelength
        self.seed: int = self.options.seed

        # === Setup Variables ===
        # Output Paths
        self.out_data: Path
        self.out_sne: Path
        self.out_train: Path
        self.out_test: Path

        # Train / Test Split
        self.test_frac: float
        self.n_kfolds: int

        # === Run Variables ===
        self.wavelength: npt.NDArray[np.float32]
        self.nspectra_per_sn: npt.NDArray[np.int32]

        # === Result Variables ===
        self.results: DataStepResult
        self.sne: SNeDataFrame  # Created in self.load_sne
        self.data: DataStepResult  # Created in self.prepare_data_arrays
        self.train_data: list[DataStepResult]  # Created in self.split_train_test
        self.test_data: list[DataStepResult]  # Created in self.split_train_test

        # Data Dimensions
        self.sn_dim: int  # Created in self.get_dims
        self.nspectra_per_sn: npt.NDArray[int]  # Created in self.get_dims
        self.spec_dim: int  # Created in self.get_dims
        self.wl_dim: int  # Created in self.get_dims

        # === Analysis Variables ===
        self.analysis: DataStepAnalysis = self.options.analysis or DataStepAnalysis()

    @override
    def _setup(self) -> None:
        # === Previous Step Variables ===

        # === Config Variables ===
        # --- Required ---
        colourlaw = self.options.colourlaw
        if colourlaw is not None:
            _, colourlaw = np.loadtxt(colourlaw, unpack=True)
        self.colourlaw = colourlaw

        # --- Optional ---
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
        # Output Paths
        self.out_data = self.paths.results / "data.npz"
        self.out_sne = self.paths.results / "sne.pkl"
        self.out_train = self.paths.results / "train"
        self.out_train.mkdir(parents=True, exist_ok=True)
        self.out_test = self.paths.results / "test"
        self.out_test.mkdir(parents=True, exist_ok=True)

        # Train / Test Split
        self.test_frac = 1 - self.train_frac
        self.n_kfolds = int(1 / self.test_frac)

    @override
    def _completed(self) -> bool:
        if not self.out_data.exists():
            self.log.debug(
                f"{self.name} has not completed as {self.out_data} does not exist"
            )
            return False
        if not self.out_sne.exists():
            self.log.debug(
                f"{self.name} has not completed as {self.out_sne} does not exist"
            )
            return False
        if not self.out_train.exists():
            self.log.debug(
                f"{self.name} has not completed as {self.out_train} does not exist"
            )
            return False
        if not self.out_test.exists():
            self.log.debug(
                f"{self.name} has not completed as {self.out_test} does not exist"
            )
            return False
        if len(list(self.out_train.iterdir())) == 0:
            self.log.debug(
                f"{self.name} has not completed as {self.out_train} does not contain any files"
            )
            return False
        if len(list(self.out_test.iterdir())) == 0:
            self.log.debug(
                f"{self.name} has not completed as {self.out_test} does not contain any files"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        # Load SNe DataFrames
        self.log.debug(f"Loading SNe dataframe from {self.out_sne}")
        self.sne = pd.read_pickle(self.out_sne)

        # Calculate data dimensions
        self.get_dims()
        # Load data from files
        # Open the file, read each key into a dictionary, then close the file
        self.log.debug(f"Loading data arrays from {self.out_data}")
        with np.load(self.out_data, allow_pickle=True) as io:
            data = dict(io.items())
        self.data = DataStepResult.model_validate(data)
        self.results = self.data

        # Load in training and testing data
        self.log.debug(f"Loading training data arrays from {self.out_train}")
        self.train_data = []
        for train_data in self.out_train.iterdir():
            if train_data.is_file():
                with np.load(train_data, allow_pickle=True) as io:
                    data = dict(io.items())
                self.train_data.append(DataStepResult.model_validate(data))

        self.log.debug(f"Loading testing data arrays from {self.out_test}")
        self.test_data = []
        for test_data in self.out_test.iterdir():
            if test_data.is_file():
                with np.load(test_data, allow_pickle=True) as io:
                    data = dict(io.items())
                self.test_data.append(DataStepResult.model_validate(data))

    @override
    def _run(self) -> None:
        self.load_sne()
        self.calculate_salt_flux()
        self.get_dims()
        self.prepare_data_arrays()
        self.split_train_test()

    @override
    def _result(self) -> None:
        if self.force or not self.out_sne.exists():
            self.log.debug(f"Saving SNe DataFrame to {self.out_sne}")
            self.sne.to_pickle(self.out_sne)

        if self.force or not self.out_data.exists():
            self.log.debug(f"Saving data arrays to {self.out_data}")
            np.savez_compressed(
                self.out_data, **self.data.model_dump(exclude={"metadata", "name"})
            )

        for i, train_data in enumerate(self.train_data):
            out_train = self.out_train / f"kfold_{i:d}.npz"
            if self.force or not out_train.exists():
                self.log.debug(f"Saving #{i} training data array to {out_train}")
                np.savez_compressed(
                    out_train,
                    **train_data.model_dump(exclude={"metadata", "name"}),
                )

        for i, test_data in enumerate(self.test_data):
            out_test = self.out_test / f"kfold_{i:d}.npz"
            if self.force or not out_test.exists():
                self.log.debug(f"Saving #{i} testing data array to {out_test}")
                np.savez_compressed(
                    out_test,
                    **test_data.model_dump(exclude={"metadata", "name"}),
                )

    @override
    def _analyse(self) -> None:
        if self.analysis.plot_spectra is not None:
            if not isinstance(self.analysis.plot_spectra, list):
                self.analysis.plot_spectra = [self.analysis.plot_spectra]
            for opts in self.analysis.plot_spectra:
                if opts.name is None:
                    opts.name = "spectra"
                self.log.debug(f"Plotting {opts.name}")
                if opts.savepath is None:
                    opts.savepath = self.paths.plots / str(self.seed)
                opts.savepath.mkdir(parents=True, exist_ok=True)
                SpectraPlotter.plot_spectra(self.data, opts)

        if self.analysis.plot_summary is not None:
            if not isinstance(self.analysis.plot_summary, list):
                self.analysis.plot_summary = [self.analysis.plot_summary]
            for opts in self.analysis.plot_summary:
                if opts.name is None:
                    opts.name = "summary"
                self.log.debug(f"Plotting {opts.name}")
                if opts.savepath is None:
                    opts.savepath = self.paths.plots / str(self.seed)
                opts.savepath.mkdir(parents=True, exist_ok=True)
                savepath = (opts.savepath or Path()) / f"{opts.name}.{opts.ext}"
                if savepath.exists():
                    continue
                fig, ax = SpectraPlotter.plot_summary(self.data, opts, save=False)
                for dt in ("train", "test"):
                    for kfold in range(self.n_kfolds):
                        fig, ax = SpectraPlotter.plot_summary(
                            getattr(self, f"{dt}_data")[kfold],
                            opts,
                            fig=fig,
                            ax=ax,
                            save=False,
                        )
                fig = Plotter.save(fig, savepath)
                Plotter.close(fig, ax)

    #
    # === DataStep Specific Functions ===
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

        # self.log.debug(
        #     f"Cutting spectra with phases outside the range {self.min_phase} <= phase <= {self.max_phase}",
        # )
        # self.log.debug(f"Number of spectra before phase-cut: {len(spectra)}")
        # spectra = spectra[spectra.phase.between(self.min_phase, self.max_phase)]
        # self.log.debug(f"Number of spectra after phase-cut: {len(spectra)}")

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
            wavelength: "npt.NDArray[np.float32]",
            tobs: float = 0.0,
            z: float = 0.0,
            x0: float = 1.0,
            x1: float = 0.0,
            c: float = 0.0,
            zref: float = 0.05,
        ) -> "npt.NDArray[np.float32]":
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
        self.spec_dim = self.nspectra_per_sn.max()
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
            "amplitude": ("flux", np.zeros(self.wl_dim) - 1),
            "sigma": ("sigma", np.zeros(self.wl_dim)),
            "salt_flux": ("salt_flux", np.zeros(self.wl_dim) - 1),
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

        def nearest_mask[T: np.number[Any]](
            arr: "npt.NDArray[T]",
            min_val: "T | npt.NDArray[T]",
            max_val: "T | npt.NDArray[T]",
        ) -> "npt.NDArray[np.bool_]":
            if not isinstance(min_val, np.ndarray):
                min_val = cast("npt.NDArray[T]", np.array(min_val))
            if not isinstance(max_val, np.ndarray):
                max_val = cast("npt.NDArray[T]", np.array(max_val))
            base_mask = (min_val <= arr) & (arr <= max_val)

            # Pad left and right
            pad_left = np.pad(
                base_mask[:, :, :-1],
                ((0, 0), (0, 0), (1, 0)),
                mode="constant",
                constant_values=False,
            )
            pad_right = np.pad(
                base_mask[:, :, 1:],
                ((0, 0), (0, 0), (0, 1)),
                mode="constant",
                constant_values=False,
            )

            # Compute distances to the boundaries
            dist_to_min = np.abs(arr - min_val)
            dist_to_max = np.abs(arr - max_val)

            # Left edge logic
            expand_left = (
                (~base_mask) & pad_right & (dist_to_min < np.roll(dist_to_min, -1))
            )
            # Right edge logic
            expand_right = (
                (~base_mask) & pad_left & (dist_to_max < np.roll(dist_to_max, 1))
            )
            # Combine the masks
            return base_mask | expand_left | expand_right

        # Create a mask of wavelength outside of the wavelength limits
        data["mask"] = np.full(
            (self.sn_dim, self.spec_dim, self.wl_dim), fill_value=True
        )

        self.log.debug("Initial:")
        self.get_unmasked_dims(data["mask"])

        valid_redshift_mask = (self.min_redshift <= data["redshift"]) & (
            self.max_redshift >= data["redshift"]
        )
        data["mask"][~valid_redshift_mask[:, 0, 0]] = False
        self.log.debug("Valid Redshifts:")
        self.get_unmasked_dims(data["mask"])

        valid_phase_mask = (self.min_phase <= data["phase"]) & (
            self.max_phase >= data["phase"]
        )
        data["mask"][~valid_phase_mask[:, :, 0]] = False
        self.log.debug("Valid phases:")
        self.get_unmasked_dims(data["mask"])

        valid_wavelength_mask = nearest_mask(
            data["wavelength"], data["wl_mask_min"], data["wl_mask_max"]
        )

        data["wl_mask"] = valid_wavelength_mask

        data["mask"][~valid_wavelength_mask] = False

        self.log.debug("Valid Wavelengths:")
        self.get_unmasked_dims(data["mask"])

        # Mask any huge laser lines, Na D (5674 - 5692A)
        # TODO: Make these options
        # these are large jumps in flux, localized over a few wavelength bins
        laser_wl_start = np.float32(5000.0)
        laser_wl_end = np.float32(8000.0)
        laser_width = 2  # in units of wavelength bins
        laser_height = 0.4  # fractional increase in amplitude over neighbours to be considered laser

        laser_wl_mask = nearest_mask(data["wavelength"], laser_wl_start, laser_wl_end)
        laser_amp = np.full(data["amplitude"].shape, np.nan)
        laser_amp[laser_wl_mask] = data["amplitude"][laser_wl_mask]

        laser_amp_min = np.roll(laser_amp, (0, 0, -laser_width))
        laser_amp_max = np.roll(laser_amp, (0, 0, laser_width))

        laser_amp_smooth = (
            0.5 * (laser_amp_min + laser_amp_max) * laser_wl_mask.astype(np.float32)
        )
        laser_mask = (laser_amp - laser_amp_smooth) > laser_height

        while laser_width > 0:
            laser_mask_min = np.roll(laser_mask, (0, 0, -1))
            laser_mask_max = np.roll(laser_mask, (0, 0, 1))
            laser_mask = laser_mask | laser_mask_min | laser_mask_max
            laser_width -= 1

        data["mask"] &= ~laser_mask

        self.log.debug("Laser Lines:")
        self.get_unmasked_dims(data["mask"])

        # --- Finalise Data ---
        # Rescale phase to time such that:
        #   time = 0 -> phase = min_phase
        #   time = 1 -> phase = max_phase
        time_mask = data["phase"] > -np.inf
        data["time"] = data["phase"].copy()
        min_phase = max(self.min_phase, data["time"][time_mask].min())
        max_phase = min(self.max_phase, data["time"][time_mask].max())
        data["time"][time_mask] = (data["time"][time_mask] - min_phase) / (
            max_phase - min_phase
        )
        data["time"][~time_mask] = -1

        # Remove negative amplitude from unmasked amplitudes
        data["amplitude"][data["mask"]] = np.clip(
            data["amplitude"][data["mask"]],
            0,
            np.inf,
        )

        # Scale observed uncertainty to account for fitting degrees of freedom, and an error floor
        data["sigma"] = 1.4 * data["sigma"] + 4e-10

        data["mask"] = data["mask"].astype(np.int32)

        # Mask spectra without any valid wavelength
        data["spec_mask"] = data["mask"].max(axis=-1, keepdims=True)

        # Mask SNe without any valid spectra
        data["sn_mask"] = data["spec_mask"].max(axis=-2, keepdims=True)

        self.data = DataStepResult.model_validate(data)
        self.results = self.data

    def get_unmasked_dims(self, mask: np.ndarray | None = None) -> tuple[int, int, int]:
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
        self.train_data = []
        self.test_data = []

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

            n_axes = 3
            self.train_data.append(
                DataStepResult.model_validate({
                    key: val[inds_train, :, :]
                    if val.ndim == n_axes
                    else val[inds_train, :]
                    for key, val in self.data.model_dump().items()
                    if isinstance(val, np.ndarray)
                })
            )
            self.test_data.append(
                DataStepResult.model_validate({
                    key: val[inds_test, :, :]
                    if val.ndim == n_axes
                    else val[inds_test, :]
                    for key, val in self.data.model_dump().items()
                    if isinstance(val, np.ndarray)
                })
            )
        n_train_sn = self.train_data[0].amplitude.shape[0]
        n_test_sn = self.test_data[0].amplitude.shape[0]
        self.log.debug(
            f"n_train_sn: {n_train_sn} ({100 * n_train_sn / self.sn_dim}%) + n_test_sn: {n_test_sn} ({100 * n_test_sn / self.sn_dim}%) = {n_train_sn + n_test_sn} ({100 * (n_train_sn + n_test_sn) / self.sn_dim}%)",
        )


class DataStep(Variant):
    id: ClassVar[str] = "data"


DataStep.register_step(Data)
