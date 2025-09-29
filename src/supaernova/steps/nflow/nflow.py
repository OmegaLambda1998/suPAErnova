from typing import TYPE_CHECKING, Any, Self, ClassVar, override
import importlib

import numpy as np

from supaernova.analysis import Plotter
from supaernova.steps.models import Model, ModelStep
from supaernova.configs.steps.nflow import (
    NFlowConfig,
    NFlowStepConfig,
    NFlowStepResult,
    NFlowModelResult,
    NFlowStepAnalysis,
)
from supaernova.analysis.distribution import DistributionPlotter

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.steps.pae import PAEModel
    from supaernova.configs.steps.pae import PAEStepResult
    from supaernova.configs.steps.data import LazySNPAEData, DataStepResult

    from .tf import TFNFlowModel

    NFlowModel = TFNFlowModel


class NFlow(ModelStep[NFlowConfig]):
    # Class Variables
    id: ClassVar[str] = "nflow"

    def __init__(self: Self, config: "NFlowConfig") -> None:
        super().__init__(config)

        # === Config Variables ===
        # --- Required ---
        self.validation_frac: float = self.options.validation_frac
        # --- Optional ---
        self.debug: bool = self.options.debug
        self.profile: bool = self.options.profile
        self.kfold = self.options.kfold
        self.save_best: bool = self.options.save_best
        self.patience: float = self.options.patience
        self.lr: float = self.options.lr
        self.lr_decay_steps: float = self.options.lr_decay_steps
        self.lr_decay_rate: float = self.options.lr_decay_rate
        self.lr_weight_decay_rate: float | None = self.options.lr_weight_decay_rate
        self.repeats: int = self.options.repeats
        self.epochs: int = self.options.epochs
        self.batch_normalisation: bool = self.options.batch_normalisation
        self.n_hidden_units: int = self.options.n_hidden_units
        self.n_layers: int = self.options.n_layers

        # === Setup Variables ===
        self.setup_attributes: set[str] = {
            "pae",
            "data",
            "mask",
            "sn_mask",
            "spec_mask",
            "wl_mask",
            "train_data",
            "train_mask",
            "train_sn_mask",
            "train_spec_mask",
            "train_wl_mask",
            "test_data",
            "test_mask",
            "test_sn_mask",
            "test_spec_mask",
            "test_wl_mask",
            "val_data",
            "val_mask",
            "val_sn_mask",
            "val_spec_mask",
            "val_wl_mask",
            "min_redshift",
            "max_redshift",
            "min_train_redshift",
            "max_train_redshift",
            "min_test_redshift",
            "max_test_redshift",
            "min_val_redshift",
            "max_val_redshift",
            "min_phase",
            "max_phase",
            "min_train_phase",
            "max_train_phase",
            "min_test_phase",
            "max_test_phase",
            "min_val_phase",
            "max_val_phase",
            "min_wavelength",
            "max_wavelength",
            "min_train_wavelength",
            "max_train_wavelength",
            "min_test_wavelength",
            "max_test_wavelength",
            "min_val_wavelength",
            "max_val_wavelength",
            "batch_size",
            "sn_dim",
            "spec_dim",
            "wl_dim",
            "model",
        }

        # --- Previous Step Variables ---
        self.pae: PAEModel

        self.data: LazySNPAEData
        self.mask: npt.NDArray[bool]
        self.sn_mask: npt.NDArray[bool]
        self.spec_mask: npt.NDArray[bool]
        self.wl_mask: npt.NDArray[bool]

        self.train_data: LazySNPAEData
        self.train_mask: npt.NDArray[bool]
        self.train_sn_mask: npt.NDArray[bool]
        self.train_spec_mask: npt.NDArray[bool]
        self.train_wl_mask: npt.NDArray[bool]

        self.test_data: LazySNPAEData
        self.test_mask: npt.NDArray[bool]
        self.test_sn_mask: npt.NDArray[bool]
        self.test_spec_mask: npt.NDArray[bool]
        self.test_wl_mask: npt.NDArray[bool]

        self.val_data: LazySNPAEData
        self.val_mask: npt.NDArray[bool]
        self.val_sn_mask: npt.NDArray[bool]
        self.val_spec_mask: npt.NDArray[bool]
        self.val_wl_mask: npt.NDArray[bool]

        # --- Bounds ---
        self.min_redshift: float
        self.max_redshift: float
        self.min_train_redshift: float
        self.max_train_redshift: float
        self.min_test_redshift: float
        self.max_test_redshift: float
        self.min_val_redshift: float
        self.max_val_redshift: float

        self.min_phase: float
        self.max_phase: float
        self.min_train_phase: float
        self.max_train_phase: float
        self.min_test_phase: float
        self.max_test_phase: float
        self.min_val_phase: float
        self.max_val_phase: float

        self.min_wavelength: float
        self.max_wavelength: float
        self.min_train_wavelength: float
        self.max_train_wavelength: float
        self.min_test_wavelength: float
        self.max_test_wavelength: float
        self.min_val_wavelength: float
        self.max_val_wavelength: float

        self.batch_size: int

        # Data Dimensions
        self.sn_dim: int
        self.spec_dim: int
        self.wl_dim: int

        self.model: NFlowModel

        # === Run / Save / Load Variables ===
        self._run_flag: bool
        self.run_attributes: set[str] = {"_run_flag"}
        self.save_attributes: set[str] = self.run_attributes
        self.load_attributes: set[str] = self.save_attributes

        # === Result Variables ===
        self.results: NFlowStepResult

        # === Analysis Variables ===
        self.analysis: NFlowStepAnalysis = self.options.analysis or NFlowStepAnalysis()

    @override
    def _is_setup(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        for attr in self.setup_attributes:
            if not self.has_attributes([attr]):
                self.log.debug(f"{self.name} is not setup because {attr} is missing")
                return False
        return True

    @override
    def _setup(
        self: Self,
        *args: Any,
        data: "DataStepResult",
        pae: "PAEStepResult",
        **kwargs: Any,
    ) -> None:
        super()._setup()
        # --- Previous Step Variables ---
        self.data = data.data
        self.pae = pae.model
        self.train_data = data.train_data[self.kfold % len(data.train_data)]
        self.test_data = data.test_data[self.kfold % len(data.test_data)]
        self.val_data = self.test_data
        if self.validation_frac > 0:
            ind_split = int(data.sn_dim * self.validation_frac)
            self.val_data.model_validate({
                k: v[-ind_split:] for k, v in self.train_data.model_dump().items()
            })
            self.train_data.model_validate({
                k: v[:-ind_split] for k, v in self.train_data.model_dump().items()
            })

        n_batches = self.options.n_batches
        self.batch_size = max(int(data.train_frac * data.sn_dim / n_batches), 1)

        # --- Bounds ---
        self.min_redshift = self.options.min_redshift or max(
            pae.min_redshift, data.min_redshift
        )
        self.max_redshift = self.options.max_redshift or min(
            pae.max_redshift, data.max_redshift
        )
        self.min_train_redshift = self.options.min_train_redshift or self.min_redshift
        self.max_train_redshift = self.options.max_train_redshift or self.max_redshift
        self.min_test_redshift = self.options.min_test_redshift or self.min_redshift
        self.max_test_redshift = self.options.max_test_redshift or self.max_redshift
        self.min_val_redshift = self.options.min_val_redshift or self.min_redshift
        self.max_val_redshift = self.options.max_val_redshift or self.max_redshift

        self.min_phase = self.options.min_phase or max(pae.min_phase, data.min_phase)
        self.max_phase = self.options.max_phase or min(pae.max_phase, data.max_phase)
        self.min_train_phase = self.options.min_train_phase or self.min_phase
        self.max_train_phase = self.options.max_train_phase or self.max_phase
        self.min_test_phase = self.options.min_test_phase or self.min_phase
        self.max_test_phase = self.options.max_test_phase or self.max_phase
        self.min_val_phase = self.options.min_val_phase or self.min_phase
        self.max_val_phase = self.options.max_val_phase or self.max_phase

        self.min_wavelength = self.options.min_wavelength or max(
            pae.min_wavelength, data.min_wavelength
        )
        self.max_wavelength = self.options.max_wavelength or min(
            pae.max_wavelength, data.max_wavelength
        )
        self.min_train_wavelength = (
            self.options.min_train_wavelength or self.min_wavelength
        )
        self.max_train_wavelength = (
            self.options.max_train_wavelength or self.max_wavelength
        )
        self.min_test_wavelength = (
            self.options.min_test_wavelength or self.min_wavelength
        )
        self.max_test_wavelength = (
            self.options.max_test_wavelength or self.max_wavelength
        )
        self.min_val_wavelength = self.options.min_val_wavelength or self.min_wavelength
        self.max_val_wavelength = self.options.max_val_wavelength or self.max_wavelength

        self.setup_data_masks()

        # Data Dimensions
        self.sn_dim = data.sn_dim
        self.spec_dim = data.spec_dim
        self.wl_dim = data.wl_dim

    @override
    def _has_run(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(self.run_attributes)

    @override
    def _run(self: Self, *args: Any, **kwargs: Any) -> None:
        savepath = self.paths.results / self.model.name
        self.model = self.model.__class__(self)
        best_model = self.model
        best_loss = np.inf
        for i in range(self.repeats):
            self.log.debug(f"{self.name}: ({i}/{self.repeats})")
            self.model = self.model.__class__(self)
            history = self.model.train_model(savepath=savepath / str(i))
            val_loss = history.history["val_loss"][-1]
            if val_loss < best_loss:
                best_model = self.model
                best_loss = val_loss
        self.model = best_model
        self.model.save_checkpoint(savepath)
        self._run_flag = True

    @override
    def _is_saved(self: Self, *args: Any, **kwargs: Any) -> bool:
        savepath = self.paths.results / self.model.name / self.model.ckpt_path
        if not (savepath.exists() and any(savepath.iterdir())):
            self.log.debug(f"{self.name} is not saved as {savepath} does not exist")
            return False
        return True

    @override
    def _save(self: Self, *args: "Any", **kwargs: "Any") -> None:
        savepath = self.paths.results / self.model.name
        self.model = self.model.__class__(self)
        self.model.load_checkpoint(savepath)
        self.log.debug(f"Saving final NFlow model weights to {savepath}")
        self.model.save_checkpoint(savepath)

    @override
    def _load(self: Self, *args: Any, **kwargs: Any) -> None:
        savepath = self.paths.results / self.model.name
        self.model = self.model.__class__(self)
        self.log.debug(f"Loading final NFlow model weights from {savepath}")
        self.model.load_checkpoint(savepath)
        self._run_flag = True

    @override
    def _has_results(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(["results"])

    @override
    def _result(self: Self, *args: Any, **kwargs: Any) -> None:
        nflow_results = {}
        nflow_results["min_redshift"] = self.min_redshift
        nflow_results["max_redshift"] = self.max_redshift
        nflow_results["min_phase"] = self.min_phase
        nflow_results["max_phase"] = self.max_phase
        nflow_results["min_wavelength"] = self.min_wavelength
        nflow_results["max_wavelength"] = self.max_wavelength

        self._load(*args, **kwargs)

        nflow_results["model"] = self.model

        dt_results: dict[str, NFlowStepResult] = {}
        for dt in ["train_", "test_"]:
            data = getattr(self, f"{dt}data")
            input_ind = data.ind
            input_sn_name = data.sn_name
            input_spectra_id = data.spectra_id
            data.clear()

            mask = getattr(self.model, f"{dt}mask")
            z_latents = getattr(self.model, f"{dt}latents")

            nflow_inputs = np.concatenate((z_latents, mask), axis=-1)
            log_prob = self.model(nflow_inputs, training=False)

            u_latents = self.model.z_to_u(z_latents)
            u_to_z_latents = self.model.u_to_z(u_latents)

            model_results = {
                "ind": input_ind,
                "sn_name": input_sn_name,
                "spectra_id": input_spectra_id,
                "z_latents": z_latents.numpy(),
                "u_latents": u_latents.numpy(),
                "u_to_z_latents": u_to_z_latents.numpy(),
                "log_prob": -log_prob.numpy(),
            }

            dt_results[dt[:-1]] = NFlowModelResult.model_validate(model_results)

        nflow_results["models"] = dt_results
        self.results = NFlowStepResult.model_validate(nflow_results)

    @override
    def _was_analysed(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        for dt in ["train_", "test_"]:
            if self.analysis.plot_u_latents is not None:
                if not isinstance(self.analysis.plot_u_latents, list):
                    self.analysis.plot_u_latents = [self.analysis.plot_u_latents]
                for opts in self.analysis.plot_u_latents:
                    name = "u_latents" if opts.name is None else opts.name
                    savepath = (
                        self.paths.plots
                        / dt[:-1]
                        / str(self.options.seed)
                        / f"{name}.{opts.ext}"
                        if opts.savepath is None
                        else opts.savepath
                    )
                    if not savepath.exists():
                        self.log.debug(
                            f"{self.name} is missing analyses as {savepath} does not exist"
                        )
                        return False

            if self.analysis.plot_z_latents is not None:
                if not isinstance(self.analysis.plot_z_latents, list):
                    self.analysis.plot_z_latents = [self.analysis.plot_z_latents]
                for opts in self.analysis.plot_z_latents:
                    name = "z_latents" if opts.name is None else opts.name
                    savepath = (
                        self.paths.plots
                        / dt[:-1]
                        / str(self.seed)
                        / f"{name}.{opts.ext}"
                        if opts.savepath is None
                        else opts.savepath
                    )
                    if not savepath.exists():
                        self.log.debug(
                            f"{self.name} is missing analyses as {savepath} does not exist"
                        )
                        return False

            if self.analysis.plot_latents is not None:
                if not isinstance(self.analysis.plot_latents, list):
                    self.analysis.plot_latents = [self.analysis.plot_latents]
                for opts in self.analysis.plot_latents:
                    name = "latents" if opts.name is None else opts.name
                    savepath = (
                        self.paths.plots
                        / dt[:-1]
                        / str(self.seed)
                        / f"{name}.{opts.ext}"
                        if opts.savepath is None
                        else opts.savepath
                    )
                    if not savepath.exists():
                        self.log.debug(
                            f"{self.name} is missing analyses as {savepath} does not exist"
                        )
                        return False

            if self.analysis.plot_latent_steps is not None:
                if not isinstance(self.analysis.plot_latent_steps, list):
                    self.analysis.plot_latent_steps = [self.analysis.plot_latent_steps]
                for opts in self.analysis.plot_latent_steps:
                    num_steps = 2 * self.n_layers
                    for step in range(num_steps):
                        if step != 0 and step % 2 == 0:
                            continue
                        name = (
                            f"step_{step}_latent_steps"
                            if opts.name is None
                            else opts.name
                        )
                        savepath = (
                            self.paths.plots
                            / dt[:-1]
                            / str(self.seed)
                            / "steps"
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False
        return not self.analysis.force

    def _plot_u_latents(
        self: Self, gaussian, z_to_u, dt, u_labels, n_latents, n_bins
    ) -> None:
        if self.analysis.plot_u_latents is not None:
            if not isinstance(self.analysis.plot_u_latents, list):
                self.analysis.plot_u_latents = [self.analysis.plot_u_latents]
            for opts in self.analysis.plot_u_latents:
                o = opts.model_copy(deep=True)
                if o.labels is None:
                    o.labels = {
                        "gaussian": u_labels,
                        "u_latents": u_labels,
                        "u_latents_smoothed": u_labels,
                    }
                if o.name is None:
                    o.name = "u_latents"
                self.log.debug(f"Plotting {o.name}")
                if o.savepath is None:
                    o.savepath = self.paths.plots / dt[:-1] / str(self.seed)
                o.savepath.mkdir(parents=True, exist_ok=True)
                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": f"{dt}{self.name}"}
                DistributionPlotter.plot_corner(
                    {
                        "gaussian": gaussian,
                        "u_latents": z_to_u,
                        "u_latents_smoothed": z_to_u,
                    },
                    o,
                    statistics="max_central",
                    shade_alpha=0.0,
                    plot_cloud={"u_latents": True},
                    smooth={"u_latents": 0},
                    bins={
                        "u_latents": n_latents,
                        "u_latents_smoothed": n_bins,
                    },
                    color={
                        "gaussian": Plotter.colour_sequence.colors[0],
                        "u_latents": Plotter.colour_sequence.colors[1],
                        "u_latents_smoothed": Plotter.colour_sequence.colors[1],
                    },
                )

    def _plot_z_latents(self: Self, z, u_to_z, dt, z_labels, n_latents, n_bins) -> None:
        if self.analysis.plot_z_latents is not None:
            if not isinstance(self.analysis.plot_z_latents, list):
                self.analysis.plot_z_latents = [self.analysis.plot_z_latents]
            for opts in self.analysis.plot_z_latents:
                o = opts.model_copy(deep=True)
                if o.labels is None:
                    o.labels = {"z_latents": z_labels, "u_to_z_latents": z_labels}
                if o.name is None:
                    o.name = "z_latents"
                self.log.debug(f"Plotting {o.name}")
                if o.savepath is None:
                    o.savepath = self.paths.plots / dt[:-1] / str(self.seed)
                o.savepath.mkdir(parents=True, exist_ok=True)
                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": f"{dt}{self.name}"}
                DistributionPlotter.plot_corner(
                    {"z_latents": z, "u_to_z_latents": u_to_z},
                    o,
                    statistics="max_central",
                    shade_alpha=0.0,
                    plot_cloud=True,
                    smooth={"z_latents": 0, "u_to_z_latents": 0},
                    bins={
                        "z_latents": n_latents,
                        "u_to_z_latents": n_latents,
                    },
                )

    def _plot_latents(
        self: Self, z_to_u, u_to_z, dt, labels, n_latents, n_bins
    ) -> None:
        if self.analysis.plot_latents is not None:
            if not isinstance(self.analysis.plot_latents, list):
                self.analysis.plot_latents = [self.analysis.plot_latents]
            for opts in self.analysis.plot_latents:
                o = opts.model_copy(deep=True)
                if o.labels is None:
                    o.labels = {
                        "u_latents": labels,
                        "u_latents_smoothed": labels,
                        "z_latents": labels,
                        "z_latents_smoothed": labels,
                    }
                if o.name is None:
                    o.name = "latents"
                self.log.debug(f"Plotting {o.name}")
                if o.savepath is None:
                    o.savepath = self.paths.plots / dt[:-1] / str(self.seed)
                o.savepath.mkdir(parents=True, exist_ok=True)
                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": f"{dt}{self.name}"}
                DistributionPlotter.plot_corner(
                    {
                        "u_latents": z_to_u,
                        "u_latents_smoothed": z_to_u,
                        "z_latents": u_to_z,
                        "z_latents_smoothed": u_to_z,
                    },
                    o,
                    statistics="max_central",
                    shade_alpha=0.0,
                    plot_cloud={"u_latents": True, "z_latents": True},
                    smooth={"u_latents": 0, "z_latents": 0},
                    bins={
                        "u_latents": n_latents,
                        "u_latents_smoothed": n_bins,
                        "z_latents": n_latents,
                        "z_latents_smoothed": n_bins,
                    },
                    color={
                        "u_latents": Plotter.colour_sequence.colors[0],
                        "u_latents_smoothed": Plotter.colour_sequence.colors[0],
                        "z_latents": Plotter.colour_sequence.colors[1],
                        "z_latents_smoothed": Plotter.colour_sequence.colors[1],
                    },
                )

    def _plot_latent_steps(
        self: Self, gaussian, results, mask, labels, dt, n_latents, n_bins
    ) -> None:
        if self.analysis.plot_latent_steps is not None:
            if not isinstance(self.analysis.plot_latent_steps, list):
                self.analysis.plot_latent_steps = [self.analysis.plot_latent_steps]
            for opts in self.analysis.plot_latent_steps:
                num_steps = len(self.model.flow.bijector.bijectors) + 1

                for step in range(num_steps):
                    step_latents, is_shift = self.model.z_to_u_steps(
                        results.z_latents, step, permute=True
                    )

                    step_u_latents = step_latents.numpy()[mask]

                    if is_shift:
                        continue
                    o = opts.model_copy(deep=True)
                    if o.labels is None:
                        o.labels = {
                            "gaussian": labels,
                            f"step_{step}_latents": labels,
                            f"step_{step}_latents_smoothed": labels,
                        }
                    if o.name is None:
                        o.name = f"step_{step}_latent_steps"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.plots / dt[:-1] / str(self.seed) / "steps"
                        )
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"title": f"{dt}{self.name}"}

                    DistributionPlotter.plot_corner(
                        {
                            "gaussian": gaussian,
                            f"step_{step}_latents": step_u_latents,
                            f"step_{step}_latents_smoothed": step_u_latents,
                        },
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud={f"step_{step}_latents": True},
                        smooth={
                            f"step_{step}_latents": 0,
                        },
                        bins={
                            f"step_{step}_latents": n_latents,
                            f"step_{step}_latents_smoothed": n_bins,
                        },
                        color={
                            "gaussian": Plotter.colour_sequence.colors[0],
                            f"step_{step}_latents": Plotter.colour_sequence.colors[1],
                            f"step_{step}_latents_smoothed": Plotter.colour_sequence.colors[
                                1
                            ],
                        },
                    )

    @override
    def _analyse(self: Self, *args: Any, **kwargs: Any) -> None:
        z_labels = {}
        u_labels = {}
        labels = {}
        ind = 0
        if self.model.physical_latents:
            z_labels[0] = "ΔAᵥ"
            u_labels[0] = "μΔAᵥ"
            labels[0] = "z/μΔAᵥ"
            ind = 1
        for i in range(self.model.n_u_latents):
            z_labels[ind] = f"z{i + 1}"
            u_labels[ind] = f"μ{i + 1}"
            labels[ind] = f"z/μ{i + 1}"
            ind += 1

        for dt in ["train_", "test_"]:
            results = self.results.models[dt[:-1]]
            gaussian = self.rng.normal(0, 1, (results.u_latents.size**2, ind))

            mask = getattr(self.model, f"{dt}mask")[:, 0].numpy().astype(bool)

            u_latents = self.model.z_to_u(results.z_latents, permute=True).numpy()
            z_latents = self.model.u_to_z(u_latents, permute=True).numpy()

            z = results.z_latents[mask]
            z_to_u = u_latents[mask]
            u_to_z = z_latents[mask]

            n_latents = z.shape[0]
            latents_num = np.log10(n_latents)
            latents_scale_min = np.floor(latents_num)
            latents_scale_max = np.ceil(latents_num)
            latents_scale = latents_scale_min if n_latents > 100 else latents_scale_max
            n_bins = int(np.sqrt(10**latents_scale))

            self._plot_u_latents(gaussian, z_to_u, dt, u_labels, n_latents, n_bins)

            self._plot_z_latents(z, u_to_z, dt, z_labels, n_latents, n_bins)

            self._plot_latents(z_to_u, u_to_z, dt, labels, n_latents, n_bins)

            self._plot_latent_steps(
                gaussian, results, mask, labels, dt, n_latents, n_bins
            )

    @override
    def _clear(
        self: Self,
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
            self.analysis = self.options.analysis or NFlowStepAnalysis()

        super()._clear(
            *args,
            setup=setup,
            run=run,
            save=save,
            load=load,
            result=result,
            analyse=analyse,
            **kwargs,
        )

    # === Instance Methods ===

    def setup_data_masks(self: Self) -> None:
        for mask_type in ["train_", "test_", "val_", ""]:
            data: LazySNPAEData = getattr(self, f"{mask_type}data")
            input_redshift = data.redshift
            input_phase = data.phase
            input_wavelength = data.wavelength
            input_mask = data.mask.astype(bool)
            data.clear()

            min_redshift: float = getattr(self, f"min_{mask_type}redshift")
            max_redshift: float = getattr(self, f"max_{mask_type}redshift")
            redshift_mask = (
                (input_redshift >= min_redshift) & (input_redshift <= max_redshift)
            )[:, 0:1, 0:1]
            # Mask out SNe outside the redshift range
            sn_mask = redshift_mask

            min_phase: float = getattr(self, f"min_{mask_type}phase")
            max_phase: float = getattr(self, f"max_{mask_type}phase")
            phase_mask = ((input_phase >= min_phase) & (input_phase <= max_phase))[
                ..., 0:1
            ]
            # Mask out spectra outside the phase range
            spec_mask = phase_mask

            min_wavelength: float = getattr(self, f"min_{mask_type}wavelength")
            max_wavelength: float = getattr(self, f"max_{mask_type}wavelength")
            wavelength_mask = (input_wavelength >= min_wavelength) & (
                input_wavelength <= max_wavelength
            )
            # Mask out wavelengths outside the wavelength range
            wl_mask = wavelength_mask

            setattr(self, f"{mask_type}mask", input_mask)
            setattr(self, f"{mask_type}sn_mask", sn_mask)
            setattr(self, f"{mask_type}spec_mask", spec_mask)
            setattr(self, f"{mask_type}wl_mask", wl_mask)


class NFlowStep(Model[NFlowStepConfig, NFlow]):
    id: "ClassVar[str]" = "nflow"
    model_backend: "ClassVar[dict[str, Callable[[], type[NFlowModel]]]]" = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFNFlowModel,
    }


NFlowStep.register_step(NFlow)
