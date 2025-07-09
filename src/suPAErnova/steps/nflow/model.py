# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, ClassVar, override
import importlib

import numpy as np

from suPAErnova.steps.backends import AbstractModel
from suPAErnova.configs.steps.nflow import NFlowStepResult
from suPAErnova.analysis.distribution import DistributionPlotter

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.pae.model import PAEModelStep
    from suPAErnova.configs.steps.nflow.model import NFlowModelConfig, NFlowStepAnalysis

    from .tf import TFNFlowModel

    NFlowModel = TFNFlowModel


class NFlowModelStep[Backend: str](AbstractModel[Backend]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[NFlowModel]]"]] = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFNFlowModel,
    }
    id: ClassVar[str] = "nflow_model"

    def __init__(self, config: "NFlowModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: NFlowModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool
        self.profile: bool
        self.savepath: Path

        self.pae: PAEModelStep

        self.results: dict[str, NFlowStepResult]
        self.analysis: tuple[NFlowStepAnalysis] = self.options.analysis

    @override
    def _setup(self, *, pae: "PAEModelStep") -> None:
        self.debug = self.options.debug
        self.profile = self.options.profile
        self.pae = pae
        self._model(force=True)
        self.savepath = self.paths.out / self.model.name

    @override
    def _completed(self) -> bool:
        self._model(force=True)
        savepath = self.savepath / self.model.ckpt_path

        if not (savepath.exists() and any(savepath.iterdir())):
            self.log.debug(
                f"{self.name} has not completed as {savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self._model(force=True)
        self.log.debug(f"Loading final NFlow model weights from {self.savepath}")
        self.model.load_checkpoint(self.savepath)

    @override
    def _run(self) -> None:
        self._model(force=True)
        self.model.train_model(savepath=self.savepath)

    @override
    def _result(self) -> None:
        self.log.debug(f"Saving final NFlow model weights to {self.savepath}")
        self.model.save_checkpoint(self.savepath)

        dt_results: dict[str, NFlowStepResult] = {}
        for dt in ["train", "test"]:
            data = getattr(self.model.pae.stage, f"{dt}_data")
            sn_mask = getattr(self.model.pae.stage, f"{dt}_sn_mask")
            spec_mask = getattr(self.model.pae.stage, f"{dt}_spec_mask")

            input_phase = data.time
            input_amplitude = data.amplitude
            input_mask = data.mask

            latents = self.model.pae(
                (input_phase, input_amplitude),
                training=False,
                mask=input_mask * spec_mask,
            )[0]

            inds = np.squeeze(np.array(sn_mask).astype(np.bool_), axis=(1, 2))
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
        self._model()
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
            z_labels[ind] = f"z{i}"
            u_labels[ind] = f"μ{i}"
            labels[ind] = f"z/μ{i}"
            ind += 1

        for dt in ["train", "test"]:
            results = self.results[dt]
            gaussian = np.random.normal(0, 1, (results.z_to_u.size**2, ind))

            if self.analysis.plot_u_latents is not None:
                if not isinstance(self.analysis.plot_u_latents, list):
                    self.analysis.plot_u_latents = [self.analysis.plot_u_latents]
                for opts in self.analysis.plot_u_latents:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = {"gaussian": u_labels, "u_latents": u_labels}
                    if o.name is None:
                        o.name = "u_latents"
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / dt / str(self.model.seed)
                        )
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    DistributionPlotter.plot_corner(
                        {"gaussian": gaussian, "u_latents": results.z_to_u},
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud=True,
                    )

            if self.analysis.plot_z_latents is not None:
                if not isinstance(self.analysis.plot_z_latents, list):
                    self.analysis.plot_z_latents = [self.analysis.plot_z_latents]
                for opts in self.analysis.plot_z_latents:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = z_labels
                    if o.name is None:
                        o.name = "z_latents"
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / dt / str(self.model.seed)
                        )
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    DistributionPlotter.plot_corner(
                        results.u_to_z,
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud=True,
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
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / dt / str(self.model.seed)
                        )
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    u_latents = self.model.z_to_u(results.latents, permute=True).numpy()
                    z_latents = self.model.u_to_z(u_latents, permute=True).numpy()
                    DistributionPlotter.plot_corner(
                        {"u_latents": u_latents, "z_latents": z_latents},
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud=True,
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
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.out
                                / "plots"
                                / dt
                                / str(self.model.seed)
                                / "steps"
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)

                        DistributionPlotter.plot_corner(
                            {
                                "gaussian": gaussian,
                                f"step_{step}_latents": step_latents.numpy(),
                            },
                            o,
                            statistics="max_central",
                            shade_alpha=0.0,
                            plot_cloud=True,
                        )

    #
    # === NFlowModel Specific Functions ===
    #
