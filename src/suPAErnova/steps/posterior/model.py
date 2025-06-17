# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, ClassVar, override
import importlib

import numpy as np

from suPAErnova.steps.backends import AbstractModel
from suPAErnova.analysis.dispersion import DispersionPlotter
from suPAErnova.analysis.distribution import DistributionPlotter
from suPAErnova.configs.steps.posterior import (
    PosteriorStepResult,
)
from suPAErnova.configs.steps.posterior.posterior import PosteriorMapStage

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.nflow.model import NFlowModel
    from suPAErnova.configs.steps.posterior import (
        PosteriorStepAnalysis,
    )
    from suPAErnova.configs.steps.posterior.model import PosteriorModelConfig

    from .tf import TFPosteriorModel

    PosteriorModel = TFPosteriorModel


class PosteriorModelStep[Backend: str](AbstractModel[Backend]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[PosteriorModel]]"]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPosteriorModel,
    }
    id: ClassVar[str] = "posterior_model"

    def __init__(self, config: "PosteriorModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: PosteriorModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool
        self.savepath: Path

        self.nflow: NFlowModel

        self.seeds: list[int] = self.options.seeds
        self.results: dict[int, PosteriorStepResult]
        self.analysis: tuple[PosteriorStepAnalysis] = self.options.analysis

        # --- Setup Variables ---
        self.n_chains_early: int = self.options.n_chains_early
        self.n_chains_mid: int = self.options.n_chains_mid
        self.n_chains_final: int = self.options.n_chains_final

        self.map_stage_init: PosteriorMapStage
        self.map_stage_early: PosteriorMapStage
        self.map_stage_mid: PosteriorMapStage
        self.map_stage_final: PosteriorMapStage
        self.map_stages: list[PosteriorMapStage]

    @override
    def _setup(self, *, nflow: "NFlowModel") -> None:
        self.debug = self.options.debug

        self.nflow = nflow
        self.nflow.load()

        for seed in self.seeds:
            self.options.seed = seed
            self._model(force=True)

        self.savepath = self.paths.out / self.model.name

        # --- Stages ---
        self.map_stage_init = PosteriorMapStage.model_validate({
            "stage": 0,
            "name": "init",
            "fname": "init",
            "n_chains": 1,
            "init": True,
        })
        self.map_stage_early = PosteriorMapStage.model_validate({
            "stage": 1,
            "name": "random",
            "fname": "random",
            "n_chains": self.n_chains_early,
            "init_u_delta_av": "random",
            "init_latents": "u_random",
            "init_delta_av": "data",
            "init_delta_m": "random",
            "init_delta_p": "random",
            "init_bias": "current",
        })
        self.map_stage_mid = PosteriorMapStage.model_validate({
            "stage": 2,
            "name": "delta_m",
            "fname": "delta_m",
            "n_chains": self.n_chains_mid,
            "init_u_delta_av": "constant",
            "init_latents": "u_constant",
            "init_delta_av": "data",
            "init_delta_m": "scale",
            "init_delta_p": "random",
            "init_bias": "current",
        })
        self.map_stage_final = PosteriorMapStage.model_validate({
            "stage": 3,
            "name": "delta_av",
            "fname": "delta_av",
            "n_chains": self.n_chains_final,
            "init_u_delta_av": "data",
            "init_latents": "z_constant",
            "init_delta_av": "scale",
            "init_delta_m": "constant",
            "init_delta_p": "random",
            "init_bias": "current",
        })

        self.map_stages = [
            self.map_stage_init,
            self.map_stage_early,
            self.map_stage_mid,
            self.map_stage_final,
        ]

    @override
    def _completed(self) -> bool:
        for seed in self.seeds:
            self.options.seed = seed
            self._model(force=True)
            savepath = self.savepath / str(seed) / self.model.model_path
            self.log.debug(
                f"{self.name} has not completed as {savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        for seed in self.seeds:
            self.options.seed = seed
            self._model(force=True)
            self.log.debug(
                f"Loading final Posterior model weights from {self.savepath / str(seed)}"
            )
            self.model.load_checkpoint(self.savepath / str(seed))

        self._result()

    @override
    def _run(self) -> None:
        for seed in self.seeds:
            self.options.seed = seed
            self._model(force=True)
            model_path = self.savepath / str(seed) / self.model.model_path
            weights_path = self.savepath / str(seed) / self.model.weights_path
            if model_path.exists() and not self.force:
                # Don't retrain stages if you don't need to
                self.log.debug(f"Loading weights from {weights_path}")
                self.model.load_checkpoint(self.savepath / str(seed))
            else:
                self.model.train_model(
                    self.map_stages, savepath=self.savepath / str(seed)
                )
            self.model.save_checkpoint(self.savepath / str(seed))

    @override
    def _result(self) -> None:
        data = self.nflow.pae.data.data
        results = {}

        for seed in self.seeds:
            self.options.seed = seed
            self._model(force=True)
            self.model.load_checkpoint(self.savepath / str(seed))
            self.log.debug(
                f"Saving final Posterior model weights to {self.savepath / str(seed)}"
            )
            self.model.save_checkpoint(self.savepath / str(seed))

            map_results = {
                "chain_min": self.model.map.chain_min.numpy(),
                "converged": self.model.map.converged.numpy(),
                "num_evaluations": self.model.map.num_evaluations.numpy(),
                "negative_log_prob": self.model.map.negative_log_prob.numpy(),
                "init_u_delta_av": self.model.map.u_delta_av.initial.numpy(),
                "init_u_latents": self.model.map.u_latents.initial.numpy(),
                "init_delta_av": self.model.map.delta_av.initial.numpy(),
                "init_delta_m": self.model.map.delta_m.initial.numpy(),
                "init_delta_p": self.model.map.delta_p.initial.numpy(),
                "init_z_latents": self.model.map.z_latents.initial.numpy(),
                "best_u_delta_av": self.model.map.u_delta_av.best.numpy(),
                "best_u_latents": self.model.map.u_latents.best.numpy(),
                "best_delta_av": self.model.map.delta_av.best.numpy(),
                "best_delta_m": self.model.map.delta_m.best.numpy(),
                "best_delta_p": self.model.map.delta_p.best.numpy(),
                "best_z_latents": self.model.map.z_latents.best.numpy(),
            }

            hmc_results = {
                "samples": self.model.hmc.samples.numpy(),
                "step_sizes_final": self.model.hmc.step_sizes_final.numpy(),
                "is_accepted": self.model.hmc.is_accepted.numpy(),
                "u_delta_av": self.model.hmc.u_delta_av.numpy(),
                "u_latents": self.model.hmc.u_latents.numpy(),
                "delta_av": self.model.hmc.delta_av.numpy(),
                "z_latents": self.model.hmc.z_latents.numpy(),
                "delta_m": self.model.hmc.delta_m.numpy(),
                "delta_p": self.model.hmc.delta_p.numpy(),
            }

            model_results = {
                "ind": data.ind,
                "sn_name": data.sn_name,
                "spectra_id": data.spectra_id,
                "map": map_results,
                "hmc": hmc_results,
            }

            results[seed] = model_results

        self.results = {
            seed: PosteriorStepResult.model_validate(model_results)
            for seed, model_results in results.items()
        }

    @override
    def _analyse(self) -> None:
        for seed in self.seeds:
            self.options.seed = seed
            self._model(force=True)
            self.model.load_checkpoint(self.savepath / str(seed))

            map_init_results = []
            map_best_results = []
            map_labels = {}
            ind = 0
            if self.model.map.nflow.physical_latents:
                map_init_results.append(self.results[seed].map.init_u_delta_av)
                map_best_results.append(self.results[seed].map.best_u_delta_av)
                map_labels[0] = "μΔAᵥ"
                ind = 1
            for i in range(self.model.map.n_u_latents):
                map_labels[ind] = f"μ{i}"
                ind += 1
            map_init_results.append(self.results[seed].map.init_u_latents)
            map_best_results.append(self.results[seed].map.best_u_latents)
            if self.model.map.pae.physical_latents:
                map_init_results.append(self.results[seed].map.init_delta_av)
                map_best_results.append(self.results[seed].map.best_delta_av)
                map_labels[ind] = "ΔAᵥ"
                ind += 1
            for i in range(self.model.map.n_z_latents):
                map_labels[ind] = f"z{i}"
                ind += 1
            map_init_results.append(self.results[seed].map.init_z_latents)
            map_best_results.append(self.results[seed].map.best_z_latents)
            if self.model.map.pae.physical_latents:
                map_init_results.extend((
                    self.results[seed].map.init_delta_m,
                    self.results[seed].map.init_delta_p,
                ))
                map_best_results.extend((
                    self.results[seed].map.best_delta_m,
                    self.results[seed].map.best_delta_p,
                ))
                map_labels[ind] = "Δℳ"
                ind += 1
                map_labels[ind] = "Δp"
            map_init_results = np.concat(map_init_results, axis=-1)
            map_best_results = np.concat(map_best_results, axis=-1)

            hmc_labels = {}
            hmc_ind = 0
            if self.model.map.train_delta_m:
                hmc_labels[hmc_ind] = "Δℳ"
                hmc_ind += 1
            if self.model.map.train_delta_p:
                hmc_labels[hmc_ind] = "Δp"
                hmc_ind += 1
            if self.model.map.nflow.physical_latents:
                hmc_labels[hmc_ind] = "μΔAᵥ"
                hmc_ind += 1
            for i in range(self.model.map.n_u_latents):
                hmc_labels[hmc_ind + i] = f"μ{i}"

            if self.analysis.plot_map_init is not None:
                if not isinstance(self.analysis.plot_map_init, list):
                    self.analysis.plot_map_init = [self.analysis.plot_map_init]
                for opts in self.analysis.plot_map_init:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = map_labels
                    if o.name is None:
                        o.name = "map_init"
                    if o.savepath is None:
                        o.savepath = self.paths.out / "plots" / str(seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    DistributionPlotter.plot_corner(map_init_results, o)

            if self.analysis.plot_map_best is not None:
                if not isinstance(self.analysis.plot_map_best, list):
                    self.analysis.plot_map_best = [self.analysis.plot_map_best]
                for opts in self.analysis.plot_map_best:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = map_labels
                    if o.name is None:
                        o.name = "map_best"
                    if o.savepath is None:
                        o.savepath = self.paths.out / "plots" / str(seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    DistributionPlotter.plot_corner(map_best_results, o)

            if self.analysis.plot_hmc is not None:
                if not isinstance(self.analysis.plot_hmc, list):
                    self.analysis.plot_hmc = [self.analysis.plot_hmc]
                for opts in self.analysis.plot_hmc:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = hmc_labels
                    if o.name is None:
                        o.name = "hmc"
                    if o.savepath is None:
                        o.savepath = self.paths.out / "plots" / str(seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    samples = self.results[seed].hmc.samples
                    chains = [samples[:, i, :] for i in range(samples.shape[1])]
                    DistributionPlotter.plot_corner(chains, o)

        if self.analysis.plot_dispersion is not None:
            if not isinstance(self.analysis.plot_dispersion, list):
                self.analysis.plot_dispersion = [self.analysis.plot_dispersion]
            for opts in self.analysis.plot_dispersion:
                o = opts.model_copy()
                if o.name is None:
                    o.name = f"{o.subset}_dispersion"
                if o.savepath is None:
                    o.savepath = self.paths.out / "plots" / str(self.seeds[0])
                o.savepath.mkdir(parents=True, exist_ok=True)
                data = (
                    self.nflow.pae.model.stage.train_data
                    if o.subset == "train"
                    else self.nflow.pae.model.stage.test_data
                )
                data.mask *= (
                    (
                        self.nflow.pae.model.stage.train_sn_mask
                        * self.nflow.pae.model.stage.train_spec_mask
                    )
                    if o.subset == "train"
                    else (
                        self.nflow.pae.model.stage.test_sn_mask
                        * self.nflow.pae.model.stage.test_spec_mask
                    )
                )
                hmc = list(self.results.values())
                DispersionPlotter.plot_dispersion(data, hmc, o)
