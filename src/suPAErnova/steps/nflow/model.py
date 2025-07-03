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
        self.savepath: Path

        self.pae: PAEModelStep

        self.results: NFlowStepResult
        self.analysis: tuple[NFlowStepAnalysis] = self.options.analysis

    @override
    def _setup(self, *, pae: "PAEModelStep") -> None:
        self.debug = self.options.debug
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

        data = self.model.pae.stage.all_data
        all_sn_mask = self.model.pae.stage.all_sn_mask
        all_spec_mask = self.model.pae.stage.all_spec_mask

        input_phase = data.time
        input_amplitude = data.amplitude
        input_mask = data.mask

        latents = self.model.pae.encoder(
            (input_phase, input_amplitude),
            training=False,
            mask=input_mask * all_spec_mask,
        )

        inds = np.squeeze(all_sn_mask.astype(np.bool_), axis=(1, 2))
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

        self.results = NFlowStepResult.model_validate(model_results)

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

        gaussian = np.random.normal(0, 1, (self.results.z_to_u.size**2, ind))

        if self.analysis.plot_u_latents is not None:
            if not isinstance(self.analysis.plot_u_latents, list):
                self.analysis.plot_u_latents = [self.analysis.plot_u_latents]
            for opts in self.analysis.plot_u_latents:
                if opts.labels is None:
                    opts.labels = {"gaussian": u_labels, "u_latents": u_labels}
                if opts.name is None:
                    opts.name = "u_latents"
                if opts.savepath is None:
                    opts.savepath = self.paths.out / "plots" / str(self.model.seed)
                opts.savepath.mkdir(parents=True, exist_ok=True)
                DistributionPlotter.plot_corner(
                    {"gaussian": gaussian, "u_latents": self.results.z_to_u},
                    opts,
                    statistics="max_central",
                    shade_alpha=0.0,
                    plot_cloud=True,
                )

        if self.analysis.plot_z_latents is not None:
            if not isinstance(self.analysis.plot_z_latents, list):
                self.analysis.plot_z_latents = [self.analysis.plot_z_latents]
            for opts in self.analysis.plot_z_latents:
                if opts.labels is None:
                    opts.labels = z_labels
                if opts.name is None:
                    opts.name = "z_latents"
                if opts.savepath is None:
                    opts.savepath = self.paths.out / "plots" / str(self.model.seed)
                opts.savepath.mkdir(parents=True, exist_ok=True)
                DistributionPlotter.plot_corner(
                    self.results.u_to_z,
                    opts,
                    statistics="max_central",
                    shade_alpha=0.0,
                    plot_cloud=True,
                )

        if self.analysis.plot_latents is not None:
            if not isinstance(self.analysis.plot_latents, list):
                self.analysis.plot_latents = [self.analysis.plot_latents]
            for opts in self.analysis.plot_latents:
                if opts.labels is None:
                    opts.labels = {
                        "z_latents": labels,
                        "u_latents": labels,
                    }
                if opts.name is None:
                    opts.name = "latents"
                if opts.savepath is None:
                    opts.savepath = self.paths.out / "plots" / str(self.model.seed)
                opts.savepath.mkdir(parents=True, exist_ok=True)
                u_latents = self.model.z_to_u(
                    self.results.latents, permute=True
                ).numpy()
                z_latents = self.model.u_to_z(u_latents, permute=True).numpy()
                DistributionPlotter.plot_corner(
                    {"u_latents": u_latents, "z_latents": z_latents},
                    opts,
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
                        self.results.latents, step, permute=True
                    )
                    if is_shift:
                        continue
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = {"gaussian": labels, f"step_{step}_latents": labels}
                    if o.name is None:
                        o.name = f"step_{step}_latent_steps"
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / str(self.model.seed) / "steps"
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
