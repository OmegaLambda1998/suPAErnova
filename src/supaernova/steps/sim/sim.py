from supaernova.utils import pp
from supaernova.analysis.spectra import SpectraPlotter
from supaernova.analysis import Plotter
from supaernova.configs.callbacks import callback
from typing import TYPE_CHECKING, Any, ClassVar, override
import numpy as np

from supaernova.steps import Step
from supaernova.steps.data import DataStep, Data
from supaernova.steps.variants import Variant
from supaernova.configs.steps.sim import (
    SimConfig,
    SimStepConfig,
    SimStepResult,
    SimStepAnalysis,
)
from supaernova.configs.steps.data import (
    LazySNPAEData,
    LazySNPAEDataTuple,
    DataStepResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import typing as npt

    from supaernova.steps.pae import PAEModel
    from supaernova.steps.nflow import NFlowModel
    from supaernova.configs.steps.pae import PAEStepResult
    from supaernova.configs.steps.nflow import NFlowStepResult


class Sim(Step[SimConfig]):
    def __init__(self, config: SimConfig) -> None:
        super().__init__(config)

        self.seed: int = self.options.seed
        self.cadence: int = self.options.cadence
        self.n_sn: int

        # Output Paths
        self.out_data: Path = self.paths.results / "data.npz"
        self.out_train: Path = self.paths.results / "train"
        self.out_train.mkdir(parents=True, exist_ok=True)
        self.out_test: Path = self.paths.results / "test"
        self.out_test.mkdir(parents=True, exist_ok=True)

        # === Setup Variables ===
        self.setup_attributes: set[str] = {
            "nflow",
            "pae",
            "real_data",
            "n_sn",
            "min_redshift",
            "max_redshift",
            "min_phase",
            "max_phase",
            "min_wavelength",
            "max_wavelength",
            "test_frac",
            "n_kfolds",
            "colourlaw",
            "data",
            "train_data",
            "test_data",
        }

        self.data: LazySNPAEData
        self.train_data: LazySNPAEDataTuple
        self.test_data: LazySNPAEDataTuple

        self.nflow: NFlowModel
        self.pae: PAEModel
        self.real_data: DataStepResult

        self.data_dir: Path
        self.train_frac: float
        self.test_frac: float
        self.n_kfolds: int
        self.colourlaw: npt.NDArray[float] | None

        self.min_redshift: float
        self.max_redshift: float
        self.min_phase: float
        self.max_phase: float
        self.min_wavelength: float
        self.max_wavelength: float

        # === Run / Save / Load Variables ===
        self.run_attributes: set[str] = {
            "sn_dim",
            "spec_dim",
            "wl_dim",
        }
        self.save_attributes: set[str] = self.run_attributes
        self.load_attributes: set[str] = self.save_attributes

        # Data Dimensions
        self.sn_dim: int
        self.spec_dim: int
        self.wl_dim: int

        # === Result Variables ===
        self.results: SimStepResult

        # === Analysis Variables ===
        self.analysis: SimStepAnalysis = self.options.analysis or SimStepAnalysis()

    @override
    def _is_setup(self, *args: "Any", **kwargs: "Any") -> bool:
        for attr in self.setup_attributes:
            if not self.has_attributes([attr]):
                self.log.debug(f"{self.name} is not setup because {attr} is missing")
                return False
        return True

    @override
    def _setup(
        self,
        *args: "Any",
        data: "DataStepResult",
        pae: "PAEStepResult",
        nflow: "NFlowStepResult",
        **kwargs: "Any",
    ) -> None:
        self.nflow = nflow.model
        self.pae = pae.model
        self.real_data = data

        self.n_sn = (
            self.options.n_sn
            if self.options.n_sn is not None
            else self.real_data.sn_dim
        )

        self.data_dir = self.real_data.dir
        self.train_frac = self.real_data.train_frac
        self.test_frac = 1 - self.train_frac
        self.n_kfolds = int(1 / self.test_frac)
        self.colourlaw = self.real_data.colourlaw

        self.min_redshift = self.real_data.min_redshift
        self.max_redshift = self.real_data.max_redshift
        self.min_phase = self.real_data.min_phase
        self.max_phase = self.real_data.max_phase
        self.min_wavelength = self.real_data.min_wavelength
        self.max_wavelength = self.real_data.max_wavelength

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
        data = {}
        self.get_dims()

        synth_wl = self.real_data.data.wavelength[:1, ...].repeat(self.sn_dim, axis=0)
        us = self.rng.normal(
            np.zeros((self.sn_dim, self.nflow.n_flow_latents)),
            1 + np.zeros((self.sn_dim, self.nflow.n_flow_latents)),
        )
        us = np.astype(us, np.float32)
        synth_zs = self.nflow.u_to_z(us)

        arr = np.concatenate((us, synth_zs), axis=-1)

        def encode(x, sigfigs):
            return f"{x:.{sigfigs}g}".replace(".", "-")

        n, d = arr.shape
        sigfigs = np.full(arr.shape, 2)
        while True:
            encoded = np.empty(arr.shape, dtype=object)
            for i in range(n):
                encoded[i] = [encode(x, sf) for x, sf in zip(arr[i], sigfigs[i])]
            synth_names = np.array(["_".join(row) for row in encoded])
            _, inverse, counts = np.unique(
                synth_names, return_inverse=True, return_counts=True
            )
            collisions = counts[inverse] > 1
            if not np.any(collisions):
                break
            sigfigs[collisions] += 1

        synth_zs = np.repeat(synth_zs[:, None, :], self.spec_dim, axis=-2)
        synth_zs = np.concatenate(
            (synth_zs, np.zeros((self.sn_dim, self.spec_dim, 2))), axis=-1
        )
        synth_phase = (
            np.arange(0, self.spec_dim * self.cadence, self.cadence)[
                None, :, None
            ].repeat(self.sn_dim, axis=0)
            + self.min_phase
        )
        synth_time = (synth_phase - self.min_phase) / (self.max_phase - self.min_phase)
        cadence_time = 0.5 * self.cadence / (self.max_phase - self.min_phase)

        synth_mask = np.ones((self.sn_dim, self.spec_dim, self.wl_dim), dtype=np.bool)
        synth_sn_mask = synth_mask[..., :1, :1]
        synth_spec_mask = synth_mask[..., :1]
        synth_wl_mask = synth_mask

        decoder_inputs = np.concatenate((synth_time, synth_zs), axis=-1)
        synth_amp = self.pae.decoder(
            decoder_inputs,
            mask=synth_mask,
            sn_mask=synth_sn_mask,
            spec_mask=synth_spec_mask,
            wl_mask=synth_wl_mask,
            training=False,
        ).numpy()

        snr_per_time = np.zeros((self.sn_dim, self.spec_dim, self.wl_dim))
        data_mask = self.real_data.data.mask
        data_amp = self.real_data.data.amplitude
        data_sigma = self.real_data.data.sigma
        data_time = self.real_data.data.time
        data_snr = np.abs(data_amp / data_sigma)
        data_snr[~data_mask] = 0

        for t in synth_time[0, :, 0]:
            tlo = max(0, t - cadence_time)
            thi = min(1, t + cadence_time)
            t_mask = np.logical_and(data_time >= tlo, data_time < thi)
            t_snr = np.where(t_mask, data_snr, np.zeros_like(data_snr))
            t_snr = np.sum(t_snr, axis=(0, 1), keepdims=True) / np.count_nonzero(
                t_mask, axis=(0, 1), keepdims=True
            )
            t_snr[~np.isfinite(t_snr)] = 0
            t_snr.repeat(self.sn_dim, axis=0).repeat(self.spec_dim, axis=1)
            synth_t_mask = np.logical_and(synth_time >= tlo, synth_time < thi).repeat(
                self.wl_dim, axis=-1
            )
            snr_per_time += np.where(synth_t_mask, t_snr, np.zeros_like(t_snr))
        snr_per_time[snr_per_time == 0] = 1
        synth_sigma = np.abs(synth_amp / snr_per_time)
        synth_amp += self.rng.normal(np.zeros_like(synth_amp), synth_sigma)

        data["ind"] = np.arange(self.sn_dim)[:, None, None]
        data["nspectra"] = np.ones((self.sn_dim, self.spec_dim)) * self.spec_dim
        data["sn_name"] = synth_names[:, None, None]
        data["dphase"] = np.zeros((self.sn_dim, self.spec_dim))
        data["redshift"] = self.rng.normal(
            self.real_data.data.redshift.mean() + np.zeros((self.sn_dim, 1, 1)),
            self.real_data.data.redshift.std() + np.zeros((self.sn_dim, 1, 1)),
        )
        data["x0"] = np.zeros((self.sn_dim, 1, 1))
        data["x1"] = np.zeros((self.sn_dim, 1, 1))
        data["c"] = np.zeros((self.sn_dim, 1, 1))
        data["mb"] = np.zeros((self.sn_dim, 1, 1))
        data["hubble_residual"] = np.zeros((self.sn_dim, 1, 1))
        data["luminosity_distance"] = np.zeros((self.sn_dim, 1, 1))
        data["spectra_id"] = np.arange(self.spec_dim)[None, :].repeat(
            self.sn_dim, axis=0
        )
        data["phase"] = synth_phase
        data["wl_mask_min"] = self.min_wavelength * np.ones((
            self.sn_dim,
            self.spec_dim,
            1,
        ))
        data["wl_mask_max"] = self.max_wavelength * np.ones((
            self.sn_dim,
            self.spec_dim,
            1,
        ))
        data["amplitude"] = synth_amp
        data["sigma"] = synth_sigma
        data["salt_flux"] = synth_amp
        data["wavelength"] = synth_wl
        data["mask"] = synth_mask
        data["laser_mask"] = synth_mask
        data["sn_mask"] = synth_sn_mask
        data["spec_mask"] = synth_spec_mask
        data["wl_mask"] = synth_wl_mask
        data["time"] = synth_time

        self.data.model_validate(data)

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
            for key, val in self.data.model_dump().items():
                if isinstance(val, np.ndarray):
                    print(key, val.shape)
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

    @override
    def _is_saved(self, *args: "Any", **kwargs: "Any") -> bool:
        if not self.out_data.exists():
            self.log.debug(
                f"{self.name} is not saved as {self.out_data} does not exist"
            )
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
        self.get_dims()

    @override
    def _has_results(self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(["results"])

    @override
    def _result(self, *args: Any, **kwargs: Any) -> None:
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
        self.results = SimStepResult.model_validate(results)

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
            self.analysis = self.options.analysis or SimStepAnalysis()

    def get_dims(self):
        self.sn_dim = self.n_sn
        self.spec_dim = self.real_data.spec_dim
        self.wl_dim = self.real_data.wl_dim


class SimStep(Variant[SimStepConfig, Sim]):
    id: ClassVar[str] = "sim"

    def __init__(self, config: "SimStepConfig") -> None:
        super().__init__(config)

        self.bases: dict[str, dict[str, Any]] = {}
        self.plots: dict[str, dict[str, Any]] = {}

    def _plot_comparison_pre(self, variant: Sim, *args: "Any", **kwargs: "Any") -> None:
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

    def _plot_summary(self, variant: Sim) -> None:
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

    def _plot_comparison_post(
        self,
        variant: Sim,
    ) -> None:
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

            self._plot_comparison_pre(variant, *args, **kwargs)

            super()._analyse(*args, **{**kwargs, "variants": [variant_name]})

            self._plot_summary(variant)

            self._plot_comparison_post(variant)

            variant.log.info(f"Finished analysing {variant.name}")

    @override
    @callback
    def analyse(
        self,
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().analyse(*args, **kwargs)
        if len(self.variants) > 1:
            for name, opts in self.plots.items():
                savepath = self.paths.plots / name
                if savepath.exists():
                    continue
                self.log.debug(f"Plotting {name}")
                fig = opts["fig"]
                ax = opts["ax"]
                fig = Plotter.save(fig, savepath)
                Plotter.close(fig, ax)


SimStep.register_step(Sim)
DataStep.register_proxy(SimStep)
