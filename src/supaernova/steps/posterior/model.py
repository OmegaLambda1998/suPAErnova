# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, ClassVar, override
import importlib

import numpy as np
import pandas as pd

from supaernova.steps.backends import AbstractModel
from supaernova.analysis.dispersion import DispersionPlotter
from supaernova.analysis.distribution import DistributionPlotter
from supaernova.configs.steps.posterior import (
    PosteriorStepResult,
)
from supaernova.configs.steps.posterior.posterior import PosteriorMapStage

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from supaernova.configs.paths import PathConfig
    from supaernova.configs.globals import GlobalConfig
    from supaernova.steps.nflow.model import NFlowModel
    from supaernova.configs.steps.posterior import (
        PosteriorStepAnalysis,
    )
    from supaernova.configs.steps.posterior.model import PosteriorModelConfig

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
        self.profile: bool
        self.savepath: Path

        self.nflow: NFlowModel

        self.subsets = (["train"] if self.options.train_subset else []) + (
            ["test"] if self.options.test_subset else []
        )
        self.seeds: list[int] = self.options.seeds
        self.results: dict[str, dict[int, PosteriorStepResult]]
        self.models: dict[str, dict[int, PosteriorModel]]
        self.analysis: PosteriorStepAnalysis = self.options.analysis

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
        self.profile = self.options.profile

        self.nflow = nflow
        self.nflow.load()

        for subset in self.subsets:
            for seed in self.seeds:
                self.options.subset = subset
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
        for subset in self.subsets:
            for seed in self.seeds:
                self.options.subset = subset
                self.options.seed = seed
                self._model(force=True)
                savepath = self.savepath / subset / str(seed) / self.model.ckpt_path
                if not (savepath.exists() and any(savepath.iterdir())):
                    self.log.debug(
                        f"{self.name} has not completed as {savepath} does not exist"
                    )
                    return False
        return True

    @override
    def _load(self) -> None:
        models = {}
        for subset in self.subsets:
            models[subset] = {}
            for seed in self.seeds:
                self.options.subset = subset
                self.options.seed = seed
                self._model(force=True)
                self.log.debug(
                    f"Loading final Posterior model weights from {self.savepath / subset / str(seed)}"
                )
                self.model.load_checkpoint(
                    self.savepath / subset / str(seed), load_map=True, load_hmc=True
                )
                models[subset][seed] = self.model
        self.models = models

    @override
    def _run(self) -> None:
        models = {}
        for subset in self.subsets:
            models[subset] = {}
            for seed in self.seeds:
                self.options.subset = subset
                self.options.seed = seed
                self._model(force=True)
                ckpt_path = self.savepath / subset / str(seed) / self.model.ckpt_path
                # Don't retrain stages if you don't need to
                if self.force or not (ckpt_path.exists() and any(ckpt_path.iterdir())):
                    self.model.train_model(
                        self.map_stages, savepath=self.savepath / subset / str(seed)
                    )
                else:
                    self.log.debug(f"Loading weights from {ckpt_path}")
                    self.model.load_checkpoint(
                        self.savepath / subset / str(seed), load_map=True, load_hmc=True
                    )
                self.model.save_checkpoint(
                    self.savepath / subset / str(seed), save_map=True, save_hmc=True
                )
                models[subset][seed] = self.model
        self.models = models

    @override
    def _result(self) -> None:
        data = self.nflow.pae.data.data
        results = {}

        for subset in self.subsets:
            results[subset] = {}
            for seed in self.seeds:
                self.options.subset = subset
                self.options.seed = seed
                model = self.models[subset][seed]

                map_results = {
                    "chain_min": model.map.chain_min.numpy(),
                    "converged": model.map.converged.numpy(),
                    "num_evaluations": model.map.num_evaluations.numpy(),
                    "negative_log_prob": model.map.negative_log_prob.numpy(),
                    "init_u_delta_av": model.map.u_delta_av.initial.numpy(),
                    "init_u_latents": model.map.u_latents.initial.numpy(),
                    "init_delta_av": model.map.delta_av.initial.numpy(),
                    "init_delta_m": model.map.delta_m.initial.numpy(),
                    "init_delta_p": model.map.delta_p.initial.numpy(),
                    "init_z_latents": model.map.z_latents.initial.numpy(),
                    "best_u_delta_av": model.map.u_delta_av.best.numpy(),
                    "best_u_latents": model.map.u_latents.best.numpy(),
                    "best_delta_av": model.map.delta_av.best.numpy(),
                    "best_delta_m": model.map.delta_m.best.numpy(),
                    "best_delta_p": model.map.delta_p.best.numpy(),
                    "best_z_latents": model.map.z_latents.best.numpy(),
                }

                hmc_results = {
                    "samples": model.hmc.samples.numpy(),
                    "step_sizes_final": model.hmc.step_sizes_final.numpy(),
                    "is_accepted": model.hmc.is_accepted.numpy(),
                    "u_delta_av": model.hmc.u_delta_av.numpy(),
                    "u_latents": model.hmc.u_latents.numpy(),
                    "delta_av": model.hmc.delta_av.numpy(),
                    "z_latents": model.hmc.z_latents.numpy(),
                    "delta_m": model.hmc.delta_m.numpy(),
                    "delta_p": model.hmc.delta_p.numpy(),
                }

                model_results = {
                    "ind": data.ind,
                    "sn_name": data.sn_name,
                    "spectra_id": data.spectra_id,
                    "map": map_results,
                    "hmc": hmc_results,
                }
                results[subset][seed] = PosteriorStepResult.model_validate(
                    model_results
                )

        self.results = results

    @override
    def _analyse(self) -> None:
        for subset in self.subsets:
            subset_map_init_results = {}
            subset_map_best_results = {}
            subset_map_labels = {}
            subset_hmc_samples = {}
            subset_hmc_labels = {}

            for seed in self.seeds:
                self.options.subset = subset
                self.options.seed = seed

                model = self.models[subset][seed]
                results = self.results[subset][seed]

                map_init_results = []
                map_best_results = []
                map_labels = {}
                ind = 0
                if model.map.nflow.physical_latents:
                    map_init_results.append(results.map.init_u_delta_av)
                    map_best_results.append(results.map.best_u_delta_av)
                    map_labels[0] = "μΔAᵥ"
                    ind = 1
                for i in range(model.map.n_u_latents):
                    map_labels[ind] = f"μ{i}"
                    ind += 1
                map_init_results.append(results.map.init_u_latents)
                map_best_results.append(results.map.best_u_latents)
                if model.map.pae.physical_latents:
                    map_init_results.append(results.map.init_delta_av)
                    map_best_results.append(results.map.best_delta_av)
                    map_labels[ind] = "ΔAᵥ"
                    ind += 1
                for i in range(model.map.n_z_latents):
                    map_labels[ind] = f"z{i}"
                    ind += 1
                map_init_results.append(results.map.init_z_latents)
                map_best_results.append(results.map.best_z_latents)
                if model.map.pae.physical_latents:
                    map_init_results.extend((
                        results.map.init_delta_m,
                        results.map.init_delta_p,
                    ))
                    map_best_results.extend((
                        results.map.best_delta_m,
                        results.map.best_delta_p,
                    ))
                    map_labels[ind] = "Δℳ"
                    ind += 1
                    map_labels[ind] = "Δp"
                map_init_results = np.concatenate(map_init_results, axis=-1)
                map_best_results = np.concatenate(map_best_results, axis=-1)
                subset_map_init_results[seed] = map_init_results
                subset_map_best_results[seed] = map_best_results
                subset_map_labels[seed] = map_labels

                hmc_labels = {}
                hmc_ind = 0
                if model.map.train_delta_m:
                    hmc_labels[hmc_ind] = "Δℳ"
                    hmc_ind += 1
                if model.map.train_delta_p:
                    hmc_labels[hmc_ind] = "Δp"
                    hmc_ind += 1
                if model.map.nflow.physical_latents:
                    hmc_labels[hmc_ind] = "μΔAᵥ"
                    hmc_ind += 1
                for i in range(model.map.n_u_latents):
                    hmc_labels[hmc_ind + i] = f"μ{i}"
                subset_hmc_labels[seed] = hmc_labels

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
                            o.savepath = (
                                self.paths.out
                                / "plots"
                                / str(self.seeds[0])
                                / subset
                                / str(seed)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        DistributionPlotter.plot_corner(
                            map_init_results,
                            o,
                            statistics="max_central",
                        )

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
                            o.savepath = (
                                self.paths.out
                                / "plots"
                                / str(self.seeds[0])
                                / subset
                                / str(seed)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        DistributionPlotter.plot_corner(
                            map_best_results,
                            o,
                            statistics="max_central",
                        )

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
                            o.savepath = (
                                self.paths.out
                                / "plots"
                                / str(self.seeds[0])
                                / subset
                                / str(seed)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        samples = results.hmc.samples
                        chains = [samples[:, i, :] for i in range(samples.shape[1])]
                        subset_hmc_samples[seed] = np.mean(chains, axis=0)
                        DistributionPlotter.plot_corner(
                            chains,
                            o,
                            statistics="max_central",
                        )

            # === Subset Plots ===

            if self.analysis.plot_map_init is not None:
                if not isinstance(self.analysis.plot_map_init, list):
                    self.analysis.plot_map_init = [self.analysis.plot_map_init]
                for opts in self.analysis.plot_map_init:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = subset_map_labels
                    if o.name is None:
                        o.name = "map_init"
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / str(self.seeds[0]) / subset
                        )
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    DistributionPlotter.plot_corner(
                        subset_map_init_results,
                        o,
                        statistics="max_central",
                    )

            if self.analysis.plot_map_best is not None:
                if not isinstance(self.analysis.plot_map_best, list):
                    self.analysis.plot_map_best = [self.analysis.plot_map_best]
                for opts in self.analysis.plot_map_best:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = subset_map_labels
                    if o.name is None:
                        o.name = "map_best"
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / str(self.seeds[0]) / subset
                        )
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    DistributionPlotter.plot_corner(
                        subset_map_best_results,
                        o,
                        statistics="max_central",
                    )

            if self.analysis.plot_hmc is not None:
                if not isinstance(self.analysis.plot_hmc, list):
                    self.analysis.plot_hmc = [self.analysis.plot_hmc]
                for opts in self.analysis.plot_hmc:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = subset_hmc_labels
                    if o.name is None:
                        o.name = "hmc"
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / str(self.seeds[0]) / subset
                        )
                    o.mean = False
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    DistributionPlotter.plot_corner(
                        subset_hmc_samples,
                        o,
                        statistics="max_central",
                    )

            if self.analysis.plot_dispersion is not None:
                if not isinstance(self.analysis.plot_dispersion, list):
                    self.analysis.plot_dispersion = [self.analysis.plot_dispersion]
                for opts in self.analysis.plot_dispersion:
                    o = opts.model_copy()
                    if o.subset != subset:
                        continue
                    if o.name is None:
                        o.name = "dispersion"
                    if o.savepath is None:
                        o.savepath = (
                            self.paths.out / "plots" / str(self.seeds[0]) / subset
                        )
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    data = (
                        self.nflow.pae.model.stage.train_data
                        if subset == "train"
                        else self.nflow.pae.model.stage.test_data
                    )
                    hmc = list(self.results[subset].values())

                    twins = None
                    if o.twins is not None:
                        twins_path = self.nflow.pae.data.data_dir / o.twins
                        if twins_path.exists():
                            twins = pd.read_csv(twins_path, delimiter=",")
                        else:
                            self.log.error(
                                f"{twins_path} does not exist, can not load twins data."
                            )

                    legacy_data = None
                    if o.legacy is not None:
                        legacy_data = {}
                        for p in o.legacy:
                            legacy_path = self.nflow.pae.data.data_dir / p
                            if legacy_path.exists():
                                l_d = np.load(legacy_path, allow_pickle=True).item()
                                for k, v in l_d.items():
                                    if k not in legacy_data:
                                        legacy_data[k] = v
                                    else:
                                        found = False
                                        for dim in range(len(v.shape)):
                                            if (
                                                not found
                                                and legacy_data[k].shape[dim]
                                                != v.shape[dim]
                                            ):
                                                legacy_data[k] = np.concatenate(
                                                    (legacy_data[k], v), axis=dim
                                                )
                                                found = True
                            else:
                                self.log.error(
                                    f"{legacy_path} does not exist, can not load legacy data."
                                )

                    DispersionPlotter.plot_dispersion(
                        data, hmc, o, twins=twins, legacy=legacy_data
                    )
