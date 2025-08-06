from typing import TYPE_CHECKING, ClassVar, override
import importlib

import numpy as np

from supaernova.steps import Step
from supaernova.steps.models import Model
from supaernova.configs.steps.nflow import NFlowStepResult, NFlowStepAnalysis
from supaernova.analysis.distribution import DistributionPlotter

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from supaernova.steps.pae import PAE
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
        self.pae_step: PAE
        self.data_step: Data
        self.kfold = self.options.kfold

        # === Config Variables ===
        # --- Required ---
        self.physical_latents: bool = self.options.physical_latents
        self.validation_frac: float = self.options.validation_frac
        self.batch_size: int
        # --- Optional ---
        self.pae: str | int = 0
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
        self.pae_step = pae
        self.data_step = data
        self.batch_size = max(
            int(
                self.data_step.train_frac
                * self.data_step.sn_dim
                / self.options.n_batches
            ),
            1,
        )

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
        for dt in ["train", "test"]:
            data = getattr(self.data_step, f"{dt}_data")[self.kfold]
            spec_mask = getattr(self.model.pae.stage, f"{dt}_spec_mask")
            sn_mask = getattr(self.model.pae.stage, f"{dt}_sn_mask")

            input_phase = data.time
            input_amplitude = data.amplitude
            input_mask = data.mask

            latents = self.model.pae(
                (input_phase, input_amplitude),
                training=False,
                mask=input_mask * spec_mask,
            )[0]

            inds = np.squeeze(np.array(sn_mask).astype(bool), axis=(1, 2))
            latents = latents[inds]

            z = latents[:, 0, :4]
            if not self.model.physical_latents:
                z = z[:, 1:]

            log_prob = self.model(z)
            u = self.model.z_to_u(z)
            uz = self.model.u_to_z(u)

            model_results = {
                "ind": data.ind,
                "sn_name": data.sn_name,
                "spectra_id": data.spectra_id,
                "latents": z.numpy(),
                "log_prob": -log_prob.numpy(),
                "z_to_u": u.numpy(),
                "u_to_z": uz.numpy(),
            }

            dt_results[dt] = NFlowStepResult.model_validate(model_results)

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

        for dt in ["train", "test"]:
            results = self.results[dt]
            gaussian = self.rng.normal(0, 1, (results.z_to_u.size**2, ind))
            u_latents = self.model.z_to_u(results.latents, permute=True).numpy()
            z_latents = self.model.u_to_z(u_latents, permute=True).numpy()

            if self.analysis.plot_u_latents is not None:
                if not isinstance(self.analysis.plot_u_latents, list):
                    self.analysis.plot_u_latents = [self.analysis.plot_u_latents]
                for opts in self.analysis.plot_u_latents:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = {"gaussian": u_labels, "u_latents": u_labels}
                    if o.name is None:
                        o.name = "u_latents"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"title": f"{self.name}_{dt}"}
                    DistributionPlotter.plot_corner(
                        {"gaussian": gaussian, "u_latents": u_latents},
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
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = {"z_latents": z_labels, "u_to_z_latents": z_labels}
                    if o.name is None:
                        o.name = "z_latents"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"title": f"{self.name}_{dt}"}
                    DistributionPlotter.plot_corner(
                        {"z_latents": results.latents, "u_to_z_latents": z_latents},
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud=True,
                        smooth={"z_latents": 0, "u_to_z_latents": 0},
                        bins={
                            "z_latents": results.latents.shape[0],
                            "u_to_z_latents": z_latents.shape[0],
                        },
                    )

            if self.analysis.plot_latents is not None:
                if not isinstance(self.analysis.plot_latents, list):
                    self.analysis.plot_latents = [self.analysis.plot_latents]
                for opts in self.analysis.plot_latents:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = {
                            "z_latents": labels,
                            "u_latents": labels,
                        }
                    if o.name is None:
                        o.name = "latents"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"title": f"{self.name}_{dt}"}
                    DistributionPlotter.plot_corner(
                        {"u_latents": u_latents, "z_latents": z_latents},
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
                            results.latents, step, permute=True
                        )
                        if is_shift:
                            continue
                        o = opts.model_copy()
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
                                self.paths.plots / dt / str(self.model.seed) / "steps"
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {"title": f"{self.name}_{dt}"}

                        DistributionPlotter.plot_corner(
                            {
                                "gaussian": gaussian,
                                f"step_{step}_latents": step_latents.numpy(),
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


class NFlowStep(Model):
    id: "ClassVar[str]" = "nflow"
    model_backend: "ClassVar[dict[str, Callable[[], type[NFlowModel]]]]" = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFNFlowModel,
    }


NFlowStep.register_step(NFlow)
