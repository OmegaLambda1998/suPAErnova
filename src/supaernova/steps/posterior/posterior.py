from typing import TYPE_CHECKING, Any, Self, ClassVar, override
import importlib

import numpy as np
import pandas as pd

from supaernova.steps.models import Model, ModelStep
from supaernova.configs.steps.data import SNPAEData, DataStepResult
from supaernova.analysis.dispersion import DispersionPlotter
from supaernova.analysis.distribution import DistributionPlotter
from supaernova.configs.steps.posterior import (
    PosteriorConfig,
    PosteriorMAPStage,
    PosteriorStepConfig,
    PosteriorStepResult,
    PosteriorStepAnalysis,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.steps.pae import PAEModel, PAEStepResult
    from supaernova.steps.nflow import NFlowModel, NFlowStepResult
    from supaernova.configs.steps.data import DataStepResult

    from .tf import TFPosteriorModel

    PosteriorModel = TFPosteriorModel


class Posterior(ModelStep[PosteriorConfig]):
    def __init__(self: Self, config: "PosteriorConfig") -> None:
        super().__init__(config)

        # === Previous Step Variables ===
        self.nflow: NFlowModel
        self.pae: PAEModel

        self.data: SNPAEData
        self.mask: npt.NDArray[bool]
        self.sn_mask: npt.NDArray[bool]
        self.spec_mask: npt.NDArray[bool]
        self.wl_mask: npt.NDArray[bool]

        self.train_data: SNPAEData
        self.train_mask: npt.NDArray[bool]
        self.train_sn_mask: npt.NDArray[bool]
        self.train_spec_mask: npt.NDArray[bool]
        self.train_wl_mask: npt.NDArray[bool]

        self.test_data: SNPAEData
        self.test_mask: npt.NDArray[bool]
        self.test_sn_mask: npt.NDArray[bool]
        self.test_spec_mask: npt.NDArray[bool]
        self.test_wl_mask: npt.NDArray[bool]

        self.val_data: SNPAEData
        self.val_mask: npt.NDArray[bool]
        self.val_sn_mask: npt.NDArray[bool]
        self.val_spec_mask: npt.NDArray[bool]
        self.val_wl_mask: npt.NDArray[bool]

        self.step_sizes: dict[str, npt.NDArray[float]]

        self.data_dir: Path
        self.kfold: int = self.options.kfold

        self.sn_dim: int
        self.spec_dim: int
        self.wl_dim: int

        # === Config Variables ===
        # --- Required ---
        self.iterations: int
        self.validation_frac: float = self.options.validation_frac
        self.seeds: list[int] = [self.seed + i for i in range(self.options.iterations)]
        self.n_random_chains: int = self.options.n_random_chains
        self.n_delta_m_chains: int = self.options.n_delta_m_chains
        self.n_delta_av_chains: int = self.options.n_delta_av_chains
        self.n_burnin: int
        self.n_samples: int
        self.n_leapfrog: int
        self.train_delta_m: bool = self.options.train_delta_m
        self.train_delta_p: bool = self.options.train_delta_p
        self.train_bias: bool = self.options.train_bias
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
        self.map_stage_random: PosteriorMAPStage
        self.map_stage_delta_m: PosteriorMAPStage
        self.map_stage_delta_av: PosteriorMAPStage
        self.map_stages: list[PosteriorMAPStage]

        # === Run Variables ===
        self.models: dict[str, dict[str, PosteriorModel]]

        # === Result Variables ===
        self.results: dict[str, dict[str, PosteriorStepResult]]

        # === Analysis Variables ===
        self.analysis: PosteriorStepAnalysis = (
            self.options.analysis or PosteriorStepAnalysis()
        )

    @override
    def _setup(
        self: Self,
        *args: Any,
        data: "DataStepResult",
        pae: "PAEStepResult",
        nflow: "NFlowStepResult",
        **kwargs: Any,
    ) -> None:
        # === Previous Step Variables ===
        self.data_dir = data.dir
        self.data = data.data
        self.pae = pae.model
        self.nflow = nflow.model
        self.train_data = data.train_data[self.kfold % len(data.train_data)]
        self.test_data = data.test_data[self.kfold % len(data.test_data)]
        self.val_data = self.test_data

        self.sn_dim = data.sn_dim
        self.spec_dim = data.spec_dim
        self.wl_dim = data.wl_dim

        if self.validation_frac > 0:
            ind_split = int(data.sn_dim * self.validation_frac)
            self.val_data = SNPAEData.model_validate({
                k: v[-ind_split:] for k, v in self.train_data.model_dump().items()
            })
            self.train_data = SNPAEData.model_validate({
                k: v[:-ind_split] for k, v in self.train_data.model_dump().items()
            })

        self.min_redshift = self.options.min_redshift or max(
            nflow.min_redshift,
            pae.min_redshift,
            data.min_redshift,
        )
        self.max_redshift = self.options.max_redshift or min(
            nflow.max_redshift,
            pae.max_redshift,
            data.max_redshift,
        )
        self.min_train_redshift = self.options.min_train_redshift or self.min_redshift
        self.max_train_redshift = self.options.max_train_redshift or self.max_redshift
        self.min_test_redshift = self.options.min_test_redshift or self.min_redshift
        self.max_test_redshift = self.options.max_test_redshift or self.max_redshift
        self.min_val_redshift = self.options.min_val_redshift or self.min_redshift
        self.max_val_redshift = self.options.max_val_redshift or self.max_redshift

        self.min_phase = self.options.min_phase or max(
            nflow.min_phase, pae.min_phase, data.min_phase
        )
        self.max_phase = self.options.max_phase or min(
            nflow.max_phase, pae.max_phase, data.max_phase
        )
        self.min_train_phase = self.options.min_train_phase or self.min_phase
        self.max_train_phase = self.options.max_train_phase or self.max_phase
        self.min_test_phase = self.options.min_test_phase or self.min_phase
        self.max_test_phase = self.options.max_test_phase or self.max_phase
        self.min_val_phase = self.options.min_val_phase or self.min_phase
        self.max_val_phase = self.options.max_val_phase or self.max_phase

        self.min_wavelength = self.options.min_wavelength or max(
            nflow.min_wavelength,
            pae.min_wavelength,
            data.min_wavelength,
        )
        self.max_wavelength = self.options.max_wavelength or min(
            nflow.max_wavelength,
            pae.max_wavelength,
            data.max_wavelength,
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

        self.step_sizes = {}
        for subset in self.subsets:
            z_latents = pae.stages[subset][
                str(list(pae.stages[subset].keys())[-1])
            ].latents
            zs = z_latents[..., :-2] if pae.model.physical_latents else z_latents
            u_latents = self.nflow.z_to_u(zs, permute=True)

            step_sizes = []
            if self.train_delta_m:
                if pae.model.physical_latents:
                    delta_m = z_latents[..., -2:-1]
                    delta_m_step_size = np.std(delta_m, axis=0)
                else:
                    delta_m_step_size = np.array(self.delta_m_std)
                step_sizes.append(delta_m_step_size)
            if self.train_delta_p:
                if pae.model.physical_latents:
                    delta_p = z_latents[..., -1:]
                    delta_p_step_size = np.std(delta_p, axis=0)
                else:
                    delta_p_step_size = np.array(self.delta_p_std)
                step_sizes.append(delta_p_step_size)
            if self.train_bias:
                bias_step_size = np.array(self.bias_std)
                step_sizes.append(bias_step_size)
            u_latent_step_size = np.std(u_latents, axis=0)
            step_sizes.append(u_latent_step_size)

            self.step_sizes[subset] = np.concatenate(step_sizes, axis=-1)

        # --- Stages ---
        self.map_stage_init = PosteriorMAPStage.model_validate({
            "stage": 0,
            "name": "init",
            "fname": "init",
            "n_chains": 1,
            "init": True,
        })
        self.map_stage_random = PosteriorMAPStage.model_validate({
            "stage": 1,
            "name": "random",
            "fname": "random",
            "n_chains": self.n_random_chains,
            "init_u_delta_av": "random",
            "init_latents": "u_random",
            "init_delta_av": "data",
            "init_delta_m": "random",
            "init_delta_p": "random",
            "init_bias": "current",
        })
        self.map_stage_delta_m = PosteriorMAPStage.model_validate({
            "stage": 2,
            "name": "delta_m",
            "fname": "delta_m",
            "n_chains": self.n_delta_m_chains,
            "init_u_delta_av": "constant",
            "init_latents": "u_constant",
            "init_delta_av": "data",
            "init_delta_m": "scale",
            "init_delta_p": "random",
            "init_bias": "current",
        })
        self.map_stage_delta_av = PosteriorMAPStage.model_validate({
            "stage": 3,
            "name": "delta_av",
            "fname": "delta_av",
            "n_chains": self.n_delta_av_chains,
            "init_u_delta_av": "data",
            "init_latents": "z_constant",
            "init_delta_av": "scale",
            "init_delta_m": "constant",
            "init_delta_p": "random",
            "init_bias": "current",
        })

        self.map_stages = [
            self.map_stage_init,
            self.map_stage_random,
            self.map_stage_delta_m,
            self.map_stage_delta_av,
        ]

        self.is_setup = True

    @override
    def _completed(self: Self, *args: Any, **kwargs: Any) -> bool:
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
    def _load(self: Self, *args: Any, **kwargs: Any) -> None:
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
                models[subset][str(seed)] = self.model
        self.models = models

        self.is_loaded = True

    @override
    def _run(self: Self, *args: Any, **kwargs: Any) -> None:
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
                models[subset][str(seed)] = self.model
        self.models = models

        self.is_loaded = True

    @override
    def _result(self: Self, *args: Any, **kwargs: Any) -> None:
        results = {}
        for subset in self.subsets:
            results[subset] = {}
            for seed in self.seeds:
                self.options.subset = subset
                self.options.seed = seed
                model = self.models[subset][str(seed)]
                data = getattr(self, f"{subset}_data")

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

                samples = model.hmc.samples.numpy()
                mean_samples = samples.mean(axis=0)
                input_position = model.map.get_position(mean_samples)
                log_prob = model(
                    (
                        input_position,
                        model.data.time,
                        model.data.amplitude,
                        model.data.sigma,
                    ),
                    training=False,
                    mask=model.data_mask,
                    sn_mask=model.sn_mask,
                    spec_mask=model.spec_mask,
                    wl_mask=model.wl_mask,
                )

                hmc_results = {
                    "samples": samples,
                    "log_prob": log_prob.numpy(),
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
                results[subset][str(seed)] = PosteriorStepResult.model_validate(
                    model_results
                )

        self.results = results

        self.has_results = True

    @override
    def _analyse(self: Self, *args: Any, **kwargs: Any) -> None:
        for subset in self.subsets:
            subset_map_init_results = {}
            subset_map_best_results = {}
            subset_map_labels = {}
            subset_hmc_samples = {}
            subset_hmc_labels = {}
            self.options.subset = subset

            for seed in self.seeds:
                self.options.seed = seed

                model = self.models[subset][str(seed)]
                results = self.results[subset][str(seed)]

                map_init_results = []
                map_best_results = []
                map_labels = {}
                ind = 0
                if self.nflow.physical_latents:
                    map_init_results.append(results.map.init_u_delta_av)
                    map_best_results.append(results.map.best_u_delta_av)
                    map_labels[0] = "μΔAᵥ"
                    ind = 1
                for i in range(model.map.n_u_latents):
                    map_labels[ind] = f"μ{i + 1}"
                    ind += 1
                map_init_results.append(results.map.init_u_latents)
                map_best_results.append(results.map.best_u_latents)
                if self.pae.physical_latents:
                    map_init_results.append(results.map.init_delta_av)
                    map_best_results.append(results.map.best_delta_av)
                    map_labels[ind] = "ΔAᵥ"
                    ind += 1
                for i in range(model.map.n_z_latents):
                    map_labels[ind] = f"z{i + 1}"
                    ind += 1
                map_init_results.append(results.map.init_z_latents)
                map_best_results.append(results.map.best_z_latents)
                if self.pae.physical_latents:
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
                subset_map_init_results[str(seed)] = map_init_results
                subset_map_best_results[str(seed)] = map_best_results
                subset_map_labels[str(seed)] = map_labels

                hmc_labels = {}
                hmc_ind = 0
                if model.map.train_delta_m:
                    hmc_labels[hmc_ind] = "Δℳ"
                    hmc_ind += 1
                if model.map.train_delta_p:
                    hmc_labels[hmc_ind] = "Δp"
                    hmc_ind += 1
                if self.nflow.physical_latents:
                    hmc_labels[hmc_ind] = "μΔAᵥ"
                    hmc_ind += 1
                for i in range(model.map.n_u_latents):
                    hmc_labels[hmc_ind + i] = f"μ{i + 1}"
                subset_hmc_labels[str(seed)] = hmc_labels

                if self.analysis.plot_map_init is not None:
                    if not isinstance(self.analysis.plot_map_init, list):
                        self.analysis.plot_map_init = [self.analysis.plot_map_init]
                    for opts in self.analysis.plot_map_init:
                        o = opts.model_copy(deep=True)
                        if o.labels is None:
                            o.labels = map_labels
                        if o.name is None:
                            o.name = "map_init"
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {"title": f"{subset}_{self.name}_{o.name}"}
                        self.log.debug(f"Plotting {o.name}")
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
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {"title": f"{subset}_{self.name}_{o.name}"}
                        self.log.debug(f"Plotting {o.name}")
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
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {"title": f"{subset}_{self.name}_{o.name}"}
                        self.log.debug(f"Plotting {o.name}")
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
                        subset_hmc_samples[str(seed)] = np.mean(chains, axis=0)
                        DistributionPlotter.plot_corner(
                            chains,
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
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {"title": f"{subset}_{self.name}"}
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
                                / str(self.seeds[0])
                                / subset
                                / str(seed)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)

                        data = getattr(self, f"{o.subset}_data").model_copy(deep=True)
                        mask = getattr(self, f"{o.subset}_mask")
                        sn_mask = getattr(self, f"{o.subset}_sn_mask")
                        spec_mask = getattr(self, f"{o.subset}_spec_mask")
                        wl_mask = getattr(self, f"{o.subset}_wl_mask")

                        twins = None
                        if o.twins is not None:
                            twins_path = self.data_dir / o.twins
                            twins = pd.read_csv(twins_path, header=0)

                        legacy_keys = {
                            ("names", 0),
                            ("redshift", 0),
                            ("amplitude_mcmc", 0),
                            ("amplitude_mcmc_err", 0),
                            ("mask", 0),
                        }
                        legacy = {}
                        for path in o.legacy or []:
                            legacy_path = self.data_dir / path
                            legacy_data = np.load(legacy_path, allow_pickle=True).item()
                            legacy = {
                                k: (
                                    legacy_data[k]
                                    if k not in legacy
                                    else np.concatenate(
                                        (legacy[k], legacy_data[k]), axis=axis
                                    )
                                )
                                for (k, axis) in legacy_keys
                            }

                        if len(legacy) == 0:
                            legacy = None

                        DispersionPlotter.plot_dispersion(
                            data,
                            list(self.results[subset].values()),
                            o,
                            twins=twins,
                            legacy=legacy,
                            mask=mask,
                            sn_mask=sn_mask,
                            spec_mask=spec_mask,
                            wl_mask=wl_mask,
                        )

            # === Subset Plots ===

            if len(self.seeds) > 1:
                if self.analysis.plot_map_init is not None:
                    if not isinstance(self.analysis.plot_map_init, list):
                        self.analysis.plot_map_init = [self.analysis.plot_map_init]
                    for opts in self.analysis.plot_map_init:
                        o = opts.model_copy(deep=True)
                        if o.labels is None:
                            o.labels = subset_map_labels
                        if o.name is None:
                            o.name = "map_init"
                        self.log.debug(f"Plotting {o.name}")
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
                        self.log.debug(f"Plotting {o.name}")
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
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = self.paths.plots / str(self.seeds[0]) / subset
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        o.mean = False
                        DistributionPlotter.plot_corner(
                            subset_hmc_samples,
                            o,
                            statistics="max_central",
                        )

        self.was_analysed = True

    @override
    def _clear(
        self: Self,
        *args: "Any",
        setup: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        **kwargs: "Any",
    ) -> None:
        if setup:
            self.clear_attributes([
                "data",
                "pae",
                "nflow",
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
                "map_stage_init",
                "map_stage_random",
                "map_stage_delta_m",
                "map_stage_delta_av",
                "map_stages",
            ])

        if load:
            self.clear_attributes("model")
            self.clear_attributes("models")

        if result:
            self.clear_attributes("results")

        if analyse:
            self.analysis = self.options.analysis or PosteriorStepAnalysis()

        super()._clear(
            *args,
            setup=setup,
            load=load,
            result=result,
            analyse=analyse,
            **kwargs,
        )

    # === Instance Methods ===

    def setup_data_masks(self: Self) -> None:
        for mask_type in ["train_", "test_", "val_", ""]:
            data: SNPAEData = getattr(self, f"{mask_type}data")
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


class PosteriorStep(Model[PosteriorStepConfig, Posterior]):
    id: "ClassVar[str]" = "posterior"
    model_backend: "ClassVar[dict[str, Callable[[], type[PosteriorModel]]]]" = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPosteriorModel,
    }


PosteriorStep.register_step(Posterior)
