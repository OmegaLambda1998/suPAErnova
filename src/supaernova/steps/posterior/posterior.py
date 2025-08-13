from typing import TYPE_CHECKING, ClassVar, override
import importlib

import numpy as np
import pandas as pd

from supaernova.steps import Step
from supaernova.steps.models import Model
from supaernova.analysis.spectra import SpectraPlotter
from supaernova.configs.steps.data import DataStepResult
from supaernova.analysis.dispersion import DispersionPlotter
from supaernova.analysis.distribution import DistributionPlotter
from supaernova.configs.steps.posterior import (
    PosteriorMAPStage,
    PosteriorStepResult,
    PosteriorStepAnalysis,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.steps.pae import PAE, PAEModel
    from supaernova.steps.data import Data
    from supaernova.steps.nflow import NFlow, NFlowModel
    from supaernova.configs.steps.posterior import PosteriorStepConfig

    from .tf import TFPosteriorModel

    PosteriorModel = TFPosteriorModel


class Posterior(Step):
    def __init__(self, config: "PosteriorStepConfig") -> None:
        super().__init__(config)

        # === Previous Step Variables ===
        self.data_step: Data
        self.kfold: int
        self.pae_step: PAE
        self.pae: PAEModel
        self.nflow_step: NFlow
        self.nflow: NFlowModel

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
        self.iterations: int
        self.validation_frac: float = self.options.validation_frac
        self.seeds: list[int] = [self.seed + i for i in range(self.options.iterations)]
        self.n_chains_early: int = self.options.n_chains_early
        self.n_chains_mid: int = self.options.n_chains_mid
        self.n_chains_final: int = self.options.n_chains_final
        self.n_burnin: int
        self.n_samples: int
        self.n_leapfrog: int
        self.train_delta_m: bool
        self.train_delta_p: bool
        self.train_bias: bool
        # --- Optional ---
        self.debug: bool = self.options.debug
        self.profile: bool = self.options.profile
        self.save_best: bool = self.options.save_best
        self.subsets = (["train"] if self.options.train_subset else []) + (
            ["test"] if self.options.test_subset else []
        )
        self.tolerance: float = self.options.tolerance
        self.target_acceptance_rate: float = self.options.target_acceptance_rate
        self.random_initial_positions: bool = self.options.random_initial_positions

        self.u_delta_av_min: float = self.options.u_delta_av_min
        self.u_delta_av_max: float = self.options.u_delta_av_max
        self.u_delta_av_start: float = self.options.u_delta_av_start
        self.u_delta_av_end: float = self.options.u_delta_av_end
        self.u_delta_av_mean: float = self.options.u_delta_av_mean
        self.u_delta_av_std: float = self.options.u_delta_av_std

        self.u_latents_min: float = self.options.u_latents_min
        self.u_latents_max: float = self.options.u_latents_max
        self.u_latents_mean: float = self.options.u_latents_mean
        self.u_latents_std: float = self.options.u_latents_std

        self.delta_av_min: float = self.options.delta_av_min
        self.delta_av_max: float = self.options.delta_av_max
        self.delta_av_start: float = self.options.delta_av_start
        self.delta_av_end: float = self.options.delta_av_end
        self.delta_av_mean: float = self.options.delta_av_mean
        self.delta_av_std: float = self.options.delta_av_std

        self.delta_m_min: float = self.options.delta_m_min
        self.delta_m_max: float = self.options.delta_m_max
        self.delta_m_start: float = self.options.delta_m_start
        self.delta_m_end: float = self.options.delta_m_end
        self.delta_m_mean: float = self.options.delta_m_mean
        self.delta_m_std: float = self.options.delta_m_std

        self.delta_p_min: float = self.options.delta_p_min
        self.delta_p_max: float = self.options.delta_p_max
        self.delta_p_start: float = self.options.delta_p_start
        self.delta_p_end: float = self.options.delta_p_end
        self.delta_p_mean: float = self.options.delta_p_mean
        self.delta_p_std: float = self.options.delta_p_std

        self.bias_min: float = self.options.bias_min
        self.bias_max: float = self.options.bias_max
        self.bias_start: float = self.options.bias_start
        self.bias_end: float = self.options.bias_end
        self.bias_mean: float = self.options.bias_mean
        self.bias_std: float = self.options.bias_std

        # === Setup Variables ===
        self.savepath: Path
        self.model: PosteriorModel

        # MAPStages
        self.map_stage_init: PosteriorMAPStage
        self.map_stage_early: PosteriorMAPStage
        self.map_stage_mid: PosteriorMAPStage
        self.map_stage_final: PosteriorMAPStage
        self.map_stages: list[PosteriorMAPStage]

        # === Run Variables ===
        self.models: dict[str, dict[str, PosteriorModel]]

        # === Result Variables ===
        self.results: PosteriorStepResult

        # === Analysis Variables ===
        self.analysis: PosteriorStepAnalysis = (
            self.options.analysis or PosteriorStepAnalysis()
        )

    @override
    def _setup(self, *, data: "Data", pae: "PAE", nflow: "NFlow") -> None:
        # === Previous Step Variables ===
        self.data_step = data
        self.data = data.data
        self.pae_step = pae
        self.pae_step.load()
        self.pae = self.pae_step.model
        self.nflow_step = nflow
        self.nflow_step.load()
        self.nflow = self.nflow_step.model
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

        self.min_redshift = self.options.min_redshift or max(
            self.nflow_step.min_redshift,
            self.pae_step.min_redshift,
            self.data_step.min_redshift,
        )
        self.max_redshift = self.options.max_redshift or min(
            self.nflow_step.max_redshift,
            self.pae_step.max_redshift,
            self.data_step.max_redshift,
        )
        self.min_train_redshift = self.options.min_train_redshift or self.min_redshift
        self.max_train_redshift = self.options.max_train_redshift or self.max_redshift
        self.min_test_redshift = self.options.min_test_redshift or self.min_redshift
        self.max_test_redshift = self.options.max_test_redshift or self.max_redshift
        self.min_val_redshift = self.options.min_val_redshift or self.min_redshift
        self.max_val_redshift = self.options.max_val_redshift or self.max_redshift

        self.min_phase = self.options.min_phase or max(
            self.nflow_step.min_phase, self.pae_step.min_phase, self.data_step.min_phase
        )
        self.max_phase = self.options.max_phase or min(
            self.nflow_step.max_phase, self.pae_step.max_phase, self.data_step.max_phase
        )
        self.min_train_phase = self.options.min_train_phase or self.min_phase
        self.max_train_phase = self.options.max_train_phase or self.max_phase
        self.min_test_phase = self.options.min_test_phase or self.min_phase
        self.max_test_phase = self.options.max_test_phase or self.max_phase
        self.min_val_phase = self.options.min_val_phase or self.min_phase
        self.max_val_phase = self.options.max_val_phase or self.max_phase

        self.min_wavelength = self.options.min_wavelength or max(
            self.nflow_step.min_wavelength,
            self.pae_step.min_wavelength,
            self.data_step.min_wavelength,
        )
        self.max_wavelength = self.options.max_wavelength or min(
            self.nflow_step.max_wavelength,
            self.pae_step.max_wavelength,
            self.data_step.max_wavelength,
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

        # --- Stages ---
        self.map_stage_init = PosteriorMAPStage.model_validate({
            "stage": 0,
            "name": "init",
            "fname": "init",
            "n_chains": 1,
            "init": True,
        })
        self.map_stage_early = PosteriorMAPStage.model_validate({
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
        self.map_stage_mid = PosteriorMAPStage.model_validate({
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
        self.map_stage_final = PosteriorMAPStage.model_validate({
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
                self.model = self.model.__class__(self)
                self.savepath = self.paths.results / self.model.name
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
                self.model = self.model.__class__(self)
                self.savepath = self.paths.results / self.model.name
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
                self.model = self.model.__class__(self)
                self.savepath = self.paths.results / self.model.name
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
        results = {}
        for subset in self.subsets:
            results[subset] = {}
            for seed in self.seeds:
                self.options.subset = subset
                self.options.seed = seed
                model = self.models[subset][seed]
                data = getattr(self.data_step, f"{subset}_data")[self.kfold]

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
                if self.nflow_step.physical_latents:
                    map_init_results.append(results.map.init_u_delta_av)
                    map_best_results.append(results.map.best_u_delta_av)
                    map_labels[0] = "μΔAᵥ"
                    ind = 1
                for i in range(model.map.n_u_latents):
                    map_labels[ind] = f"μ{i}"
                    ind += 1
                map_init_results.append(results.map.init_u_latents)
                map_best_results.append(results.map.best_u_latents)
                if self.pae_step.physical_latents:
                    map_init_results.append(results.map.init_delta_av)
                    map_best_results.append(results.map.best_delta_av)
                    map_labels[ind] = "ΔAᵥ"
                    ind += 1
                for i in range(model.map.n_z_latents):
                    map_labels[ind] = f"z{i}"
                    ind += 1
                map_init_results.append(results.map.init_z_latents)
                map_best_results.append(results.map.best_z_latents)
                if self.pae_step.physical_latents:
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
                if self.nflow_step.physical_latents:
                    hmc_labels[hmc_ind] = "μΔAᵥ"
                    hmc_ind += 1
                for i in range(model.map.n_u_latents):
                    hmc_labels[hmc_ind + i] = f"μ{i}"
                subset_hmc_labels[seed] = hmc_labels

                if self.analysis.plot_spectra is not None:
                    if not isinstance(self.analysis.plot_spectra, list):
                        self.analysis.plot_spectra = [self.analysis.plot_spectra]
                    for opts in self.analysis.plot_spectra:
                        o = opts.model_copy(deep=True)
                        if o.name is None:
                            o.name = "summary"
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
                                / str(self.seeds[0])
                                / subset
                                / str(seed)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)

                        data = model.data

                        fig, ax = SpectraPlotter.plot_summary(data, o, save=False)

                        pae_data = data.model_copy(deep=True)
                        pae_results = self.pae_step.results[subset][
                            str(self.pae_step.run_stages[-1].stage)
                        ]
                        pae_data.amplitude = pae_results.output_amp

                        fig, ax = SpectraPlotter.plot_summary(
                            pae_data, o, fig=fig, ax=ax, save=False
                        )

                        map_data = data.model_copy(deep=True)

                        decoder_inputs = np.concatenate(
                            (
                                data.time[..., :1],
                                np.repeat(
                                    results.map.best_delta_av[:, np.newaxis, :],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                                np.repeat(
                                    results.map.best_z_latents[:, np.newaxis, :],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                                np.repeat(
                                    results.map.best_delta_m[:, np.newaxis, :],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                                np.repeat(
                                    results.map.best_delta_p[:, np.newaxis, :],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                            ),
                            axis=-1,
                        )
                        map_amp = model.map.pae.decoder(decoder_inputs, mask=data.mask)

                        if model.map.train_delta_m and not model.pae.physical_latents:
                            map_amp *= results.map.best_delta_m

                        if model.map.train_bias:
                            map_amp += results.map.best_bias

                        map_data.amplitude = map_amp.numpy()

                        fig, ax = SpectraPlotter.plot_summary(
                            map_data, o, fig=fig, ax=ax, save=False
                        )

                        posterior_data = data.model_copy(deep=True)

                        decoder_inputs = np.concatenate(
                            (
                                data.time,
                                np.repeat(
                                    np.mean(results.hmc.delta_av, axis=0)[
                                        :, np.newaxis, :
                                    ],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                                np.repeat(
                                    np.mean(results.hmc.z_latents, axis=0)[
                                        :, np.newaxis, :
                                    ],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                                np.repeat(
                                    np.mean(results.hmc.delta_m, axis=0)[
                                        :, np.newaxis, :
                                    ],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                                np.repeat(
                                    np.mean(results.hmc.delta_p, axis=0)[
                                        :, np.newaxis, :
                                    ],
                                    data.time.shape[1],
                                    axis=1,
                                ),
                            ),
                            axis=-1,
                        )
                        posterior_amp = model.map.pae.decoder(
                            decoder_inputs, mask=data.mask
                        )

                        if model.map.train_delta_m and not model.pae.physical_latents:
                            map_amp *= results.hmc.delta_m

                        if model.map.train_bias:
                            map_amp += results.hmc.bias

                        posterior_data.amplitude = posterior_amp.numpy()

                        SpectraPlotter.plot_summary(posterior_data, o, fig=fig, ax=ax)

                if self.analysis.plot_map_init is not None:
                    if not isinstance(self.analysis.plot_map_init, list):
                        self.analysis.plot_map_init = [self.analysis.plot_map_init]
                    for opts in self.analysis.plot_map_init:
                        o = opts.model_copy(deep=True)
                        if o.labels is None:
                            o.labels = map_labels
                        if o.name is None:
                            o.name = "map_init"
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
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
                        o = opts.model_copy(deep=True)
                        if o.labels is None:
                            o.labels = map_labels
                        if o.name is None:
                            o.name = "map_best"
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
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
                        o = opts.model_copy(deep=True)
                        if o.labels is None:
                            o.labels = hmc_labels
                        if o.name is None:
                            o.name = "hmc"
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
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
                    o = opts.model_copy(deep=True)
                    if o.labels is None:
                        o.labels = subset_map_labels
                    if o.name is None:
                        o.name = "map_init"
                    if o.savepath is None:
                        o.savepath = self.paths.plots / str(self.seeds[0]) / subset
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
                    o = opts.model_copy(deep=True)
                    if o.labels is None:
                        o.labels = subset_map_labels
                    if o.name is None:
                        o.name = "map_best"
                    if o.savepath is None:
                        o.savepath = self.paths.plots / str(self.seeds[0]) / subset
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
                    o = opts.model_copy(deep=True)
                    if o.labels is None:
                        o.labels = subset_hmc_labels
                    if o.name is None:
                        o.name = "hmc"
                    if o.savepath is None:
                        o.savepath = self.paths.plots / str(self.seeds[0]) / subset
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
                    o = opts.model_copy(deep=True)
                    if o.subset != subset:
                        continue
                    if o.name is None:
                        o.name = "dispersion"
                    if o.savepath is None:
                        o.savepath = self.paths.plots / str(self.seeds[0]) / subset
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    data = (
                        self.data_step.train_data[
                            self.kfold % len(self.data_step.train_data)
                        ]
                        if subset == "train"
                        else self.data_step.test_data[
                            self.kfold % len(self.data_step.train_data)
                        ]
                    )
                    hmc = list(self.results[subset].values())

                    twins = None
                    if o.twins is not None:
                        twins_path = self.data_step.data_dir / o.twins
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
                            legacy_path = self.data_step.data_dir / p
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


class PosteriorStep(Model):
    id: "ClassVar[str]" = "posterior"
    model_backend: "ClassVar[dict[str, Callable[[], type[PosteriorModel]]]]" = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPosteriorModel,
    }


PosteriorStep.register_step(Posterior)
