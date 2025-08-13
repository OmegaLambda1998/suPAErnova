from typing import TYPE_CHECKING, ClassVar, override
import importlib

import numpy as np

from supaernova.steps import Step
from supaernova.steps.models import Model
from supaernova.configs.steps.data import DataStepResult
from supaernova.configs.steps.nflow import NFlowStepResult, NFlowStepAnalysis
from supaernova.analysis.distribution import DistributionPlotter

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.steps.pae import PAE, PAEModel
    from supaernova.steps.data import Data
    from supaernova.configs.steps.nflow import NFlowStepConfig

    from .tf import TFNFlowModel

    NFlowModel = TFNFlowModel


class NFlow(Step):
    # Class Variables
    id: ClassVar[str] = "nflow"

    def __init__(self, config: "NFlowStepConfig") -> None:
        super().__init__(config)

        # === Previous Step Variables ===
        self.data_step: Data
        self.kfold: int
        self.pae_step: PAE
        self.pae: PAEModel

        self.data: DataStepResult
        self.mask: npt.NDArray[bool]
        self.sn_mask: npt.NDArray[bool]
        self.spec_mask: npt.NDArray[bool]
        self.wl_mask: npt.NDArray[bool]

        self.train_data: DataStepResult
        self.train_mask: npt.NDArray[bool]
        self.train_sn_mask: npt.NDArray[bool]
        self.train_spec_mask: npt.NDArray[bool]
        self.train_wl_mask: npt.NDArray[bool]

        self.test_data: DataStepResult
        self.test_mask: npt.NDArray[bool]
        self.test_sn_mask: npt.NDArray[bool]
        self.test_spec_mask: npt.NDArray[bool]
        self.test_wl_mask: npt.NDArray[bool]

        self.val_data: DataStepResult
        self.val_mask: npt.NDArray[bool]
        self.val_sn_mask: npt.NDArray[bool]
        self.val_spec_mask: npt.NDArray[bool]
        self.val_wl_mask: npt.NDArray[bool]

        # === Config Variables ===
        # --- Required ---
        self.physical_latents: bool = self.options.physical_latents
        self.validation_frac: float = self.options.validation_frac
        self.batch_size: int
        # --- Optional ---
        self.debug: bool = self.options.debug
        self.profile: bool = self.options.profile
        self.save_best: bool = self.options.save_best
        self.patience: float = self.options.patience
        self.lr: float = self.options.lr
        self.lr_decay_steps: float = self.options.lr_decay_steps
        self.lr_decay_rate: float = self.options.lr_decay_rate
        self.lr_weight_decay_rate: float = self.options.lr_weight_decay_rate
        self.epochs: int = self.options.epochs
        self.batch_normalisation: bool = self.options.batch_normalisation
        self.n_hidden_units: int = self.options.n_hidden_units
        self.n_layers: int = self.options.n_layers

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

        # === Setup Variables ===
        self.model: NFlowModel
        self.savepath: Path
        # === Run Variables ===
        # === Result Variables ===
        self.results: NFlowStepResult

        # === Analysis Variables ===
        self.analysis: NFlowStepAnalysis = self.options.analysis or NFlowStepAnalysis()

    @override
    def _setup(self, *, data: "Data", pae: "PAE") -> None:
        super()._setup()
        # === Previous Step Variables ===
        self.data_step = data
        self.data = data.data
        self.pae_step = pae
        self.pae_step.load()
        self.pae = self.pae_step.model
        self.kfold = self.options.kfold or self.pae_step.kfold
        self.train_data = data.train_data[self.kfold % len(data.train_data)]
        self.test_data = data.test_data[self.kfold % len(data.test_data)]
        self.val_data = self.test_data
        if self.validation_frac > 0:
            ind_split = int(self.data_step.sn_dim * self.validation_frac)
            self.val_data = DataStepResult.model_validate({
                k: v[-ind_split:] for k, v in self.train_data.model_dump().items()
            })
            self.train_data = DataStepResult.model_validate({
                k: v[:-ind_split] for k, v in self.train_data.model_dump().items()
            })

        self.batch_size = max(
            int(
                self.data_step.train_frac
                * self.data_step.sn_dim
                / self.options.n_batches
            ),
            1,
        )

        self.min_redshift = self.options.min_redshift or max(
            self.pae_step.min_redshift, self.data_step.min_redshift
        )
        self.max_redshift = self.options.max_redshift or min(
            self.pae_step.max_redshift, self.data_step.max_redshift
        )
        self.min_train_redshift = self.options.min_train_redshift or self.min_redshift
        self.max_train_redshift = self.options.max_train_redshift or self.max_redshift
        self.min_test_redshift = self.options.min_test_redshift or self.min_redshift
        self.max_test_redshift = self.options.max_test_redshift or self.max_redshift
        self.min_val_redshift = self.options.min_val_redshift or self.min_redshift
        self.max_val_redshift = self.options.max_val_redshift or self.max_redshift

        self.min_phase = self.options.min_phase or max(
            self.pae_step.min_phase, self.data_step.min_phase
        )
        self.max_phase = self.options.max_phase or min(
            self.pae_step.max_phase, self.data_step.max_phase
        )
        self.min_train_phase = self.options.min_train_phase or self.min_phase
        self.max_train_phase = self.options.max_train_phase or self.max_phase
        self.min_test_phase = self.options.min_test_phase or self.min_phase
        self.max_test_phase = self.options.max_test_phase or self.max_phase
        self.min_val_phase = self.options.min_val_phase or self.min_phase
        self.max_val_phase = self.options.max_val_phase or self.max_phase

        self.min_wavelength = self.options.min_wavelength or max(
            self.pae_step.min_wavelength, self.data_step.min_wavelength
        )
        self.max_wavelength = self.options.max_wavelength or min(
            self.pae_step.max_wavelength, self.data_step.max_wavelength
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

    @override
    def _completed(self) -> bool:
        self.savepath = self.paths.results / self.model.name
        savepath = self.savepath / self.model.ckpt_path

        if not (savepath.exists() and any(savepath.iterdir())):
            self.log.debug(
                f"{self.name} has not completed as {savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self.log.debug(f"Loading final NFlow model weights from {self.savepath}")
        self.model.load_checkpoint(self.savepath)

    @override
    def _run(self) -> None:
        self.model.train_model(savepath=self.savepath)

    @override
    def _result(self) -> None:
        self.log.debug(f"Saving final NFlow model weights to {self.savepath}")
        self.model.save_checkpoint(self.savepath)

        dt_results: dict[str, NFlowStepResult] = {}
        for dt in ["train_", "test_", ""]:
            data = getattr(self, f"{dt}data")
            mask = getattr(self.model, f"{dt}mask")
            z_latents = getattr(self.model, f"{dt}latents")

            nflow_inputs = np.concatenate((z_latents, mask), axis=-1)
            log_prob = self.model(nflow_inputs, training=False)

            u_latents = self.model.z_to_u(z_latents)
            u_to_z_latents = self.model.u_to_z(u_latents)

            model_results = {
                "ind": data.ind,
                "sn_name": data.sn_name,
                "spectra_id": data.spectra_id,
                "z_latents": z_latents.numpy(),
                "u_latents": u_latents.numpy(),
                "u_to_z_latents": u_to_z_latents.numpy(),
                "log_prob": -log_prob.numpy(),
            }

            dt_results[dt[:-1]] = NFlowStepResult.model_validate(model_results)

        self.results = dt_results

    @override
    def _analyse(self) -> None:
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

        for dt in ["train_", "test_", ""]:
            results = self.results[dt[:-1]]
            gaussian = self.rng.normal(0, 1, (results.u_latents.size**2, ind))

            mask = np.repeat(
                np.logical_not(getattr(self.model, f"{dt}mask").numpy()),
                self.model.n_flow_latents,
                axis=-1,
            )
            u_latents = self.model.z_to_u(results.z_latents, permute=True).numpy()
            z_latents = self.model.u_to_z(u_latents, permute=True).numpy()

            z = np.ma.masked_array(results.z_latents, mask)
            z = results.z_latents
            z_to_u = np.ma.masked_array(u_latents, mask)
            z_to_u = u_latents
            u_to_z = np.ma.masked_array(z_latents, mask)
            u_to_z = z_latents

            if self.analysis.plot_u_latents is not None:
                if not isinstance(self.analysis.plot_u_latents, list):
                    self.analysis.plot_u_latents = [self.analysis.plot_u_latents]
                for opts in self.analysis.plot_u_latents:
                    o = opts.model_copy(deep=True)
                    if o.labels is None:
                        o.labels = {"gaussian": u_labels, "u_latents": u_labels}
                    if o.name is None:
                        o.name = "u_latents"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt[:-1] / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"title": f"{dt}{self.name}"}
                    DistributionPlotter.plot_corner(
                        {"gaussian": gaussian, "u_latents": z_to_u},
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud={"u_latents": True},
                        smooth={"u_latents": 0},
                        bins={"u_latents": u_latents.shape[0]},
                    )

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
                        o.savepath = self.paths.plots / dt[:-1] / str(self.model.seed)
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
                            "z_latents": results.z_latents.shape[0],
                            "u_to_z_latents": z_latents.shape[0],
                        },
                    )

            if self.analysis.plot_latents is not None:
                if not isinstance(self.analysis.plot_latents, list):
                    self.analysis.plot_latents = [self.analysis.plot_latents]
                for opts in self.analysis.plot_latents:
                    o = opts.model_copy(deep=True)
                    if o.labels is None:
                        o.labels = {
                            "z_latents": labels,
                            "u_latents": labels,
                        }
                    if o.name is None:
                        o.name = "latents"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt[:-1] / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"title": f"{dt}{self.name}"}
                    DistributionPlotter.plot_corner(
                        {"u_latents": z_to_u, "z_latents": u_to_z},
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud=True,
                        smooth={"u_latents": 0, "z_latents": 0},
                        bins={
                            "u_latents": u_latents.shape[0],
                            "z_latents": z_latents.shape[0],
                        },
                    )

            if self.analysis.plot_latent_steps is not None:
                if not isinstance(self.analysis.plot_latent_steps, list):
                    self.analysis.plot_latent_steps = [self.analysis.plot_latent_steps]
                for opts in self.analysis.plot_latent_steps:
                    num_steps = len(self.model.flow.bijector.bijectors) + 1

                    for step in range(num_steps):
                        step_latents, is_shift = self.model.z_to_u_steps(
                            results.z_latents, step, permute=True
                        )

                        step_u_latents = np.ma.masked_array(step_latents.numpy(), mask)
                        step_u_latents = step_latents.numpy()

                        if is_shift:
                            continue
                        o = opts.model_copy(deep=True)
                        if o.labels is None:
                            o.labels = {
                                "gaussian": labels,
                                f"step_{step}_latents": labels,
                            }
                        if o.name is None:
                            o.name = f"step_{step}_latent_steps"
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
                                / dt[:-1]
                                / str(self.model.seed)
                                / "steps"
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {"title": f"{dt}{self.name}"}

                        DistributionPlotter.plot_corner(
                            {
                                "gaussian": gaussian,
                                f"step_{step}_latents": step_u_latents,
                            },
                            o,
                            statistics="max_central",
                            shade_alpha=0.0,
                            plot_cloud={f"step_{step}_latents": True},
                            smooth={
                                f"step_{step}_latents": 0,
                            },
                            bins={
                                f"step_{step}_latents": step_latents.shape[0],
                            },
                        )

    # === Instance Methods ===

    def setup_data_masks(self) -> None:
        for mask_type in ["train_", "test_", "val_", ""]:
            data: DataStepResult = getattr(self, f"{mask_type}data")
            mask = data.mask

            min_redshift: float = getattr(self, f"min_{mask_type}redshift")
            max_redshift: float = getattr(self, f"max_{mask_type}redshift")
            redshift_mask = (
                (data.redshift >= min_redshift) & (data.redshift <= max_redshift)
            )[:, 0:1, 0:1]
            # Mask out SNe outside the redshift range
            sn_mask = redshift_mask.astype(np.int32)

            min_phase: float = getattr(self, f"min_{mask_type}phase")
            max_phase: float = getattr(self, f"max_{mask_type}phase")
            phase_mask = ((data.phase >= min_phase) & (data.phase <= max_phase))[
                ..., 0:1
            ]
            # Mask out spectra outside the phase range
            spec_mask = phase_mask.astype(np.int32)

            min_wavelength: float = getattr(self, f"min_{mask_type}wavelength")
            max_wavelength: float = getattr(self, f"max_{mask_type}wavelength")
            wavelength_mask = (data.wavelength >= min_wavelength) & (
                data.wavelength <= max_wavelength
            )
            # Mask out wavelengths outside the wavelength range
            wl_mask = wavelength_mask.astype(np.int32)

            setattr(self, f"{mask_type}mask", mask)
            setattr(self, f"{mask_type}sn_mask", sn_mask)
            setattr(self, f"{mask_type}spec_mask", spec_mask)
            setattr(self, f"{mask_type}wl_mask", wl_mask)


class NFlowStep(Model):
    id: "ClassVar[str]" = "nflow"
    model_backend: "ClassVar[dict[str, Callable[[], type[NFlowModel]]]]" = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFNFlowModel,
    }


NFlowStep.register_step(NFlow)
