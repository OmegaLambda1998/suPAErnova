import json
import shutil
from typing import TYPE_CHECKING, Any, Literal, ClassVar, override
import importlib

import numpy as np
import pandas as pd

from supaernova.utils import pp, max_central
from supaernova.analysis import Plotter
from supaernova.steps.models import Model, ModelStep
from supaernova.analysis.spectra import SpectraPlotter
from supaernova.configs.callbacks import callback
from supaernova.configs.steps.data import DataStepResult
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

    from supaernova.analysis import Axis, Figure
    from supaernova.steps.pae import PAEModel, PAEStepResult
    from supaernova.steps.nflow import NFlowModel, NFlowStepResult
    from supaernova.configs.steps.data import LazySNPAEData, DataStepResult

    from .tf import TFPosteriorModel

    PosteriorModel = TFPosteriorModel


class Posterior(ModelStep[PosteriorConfig]):
    def __init__(self, config: "PosteriorConfig") -> None:
        super().__init__(config)

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
        self.debug: bool = self.config.debug or self.options.debug
        self.profile: bool = self.options.profile
        self.kfold: int = self.options.kfold
        self.save_best: bool = self.options.save_best
        self.subsets: list[Literal["test", "train"]] = (
            ["test"] if self.options.test_subset else []
        ) + (["train"] if self.options.train_subset else [])
        self.tolerance: float = self.options.tolerance
        self.target_acceptance_rate: float = self.options.target_acceptance_rate
        self.random_initial_positions: bool = self.options.random_initial_positions
        self.legacy_path: tuple[Path] | None = self.options.legacy

        self.fractional_error: bool = self.options.fractional_error
        self.weighted_error: bool = self.options.weighted_error
        self.measurement_error: bool = self.options.measurement_error
        self.reconstruction_error: Literal["train", "test", "match", "combined"] = (
            self.options.reconstruction_error
        )

        self.bounded_u_latents: bool = self.options.bounded_u_latents
        self.generalised_u_latents: float = self.options.generalised_u_latents
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
        self.setup_attributes: set[str] = {
            "nflow",
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
            "data_dir",
            "sn_dim",
            "spec_dim",
            "wl_dim",
            "step_sizes",
            "model",
            "map_stage_init",
            "map_stage_constant",
            "map_stage_legacy",
            "map_stage_random",
            "map_stage_delta_m",
            "map_stage_delta_av",
            "map_stages",
        }

        self.nflow: NFlowModel
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

        self.data_dir: Path

        self.sn_dim: int
        self.spec_dim: int
        self.wl_dim: int

        self.step_sizes: dict[str, npt.NDArray[float]]

        self.model: PosteriorModel

        # MAP Stages
        self.map_stage_setup: PosteriorMAPStage
        self.map_stage_init: PosteriorMAPStage
        self.map_stage_constant: PosteriorMAPStage
        self.map_stage_legacy: PosteriorMAPStage | None
        self.map_stage_random: PosteriorMAPStage
        self.map_stage_delta_m: PosteriorMAPStage
        self.map_stage_delta_av: PosteriorMAPStage
        self.map_stages: list[PosteriorMAPStage]

        # === Run / Save / Load Variables ===
        self.run_attributes: set[str] = {"models"}
        self.save_attributes: set[str] = self.run_attributes
        self.load_attributes: set[str] = self.save_attributes

        self.models: dict[str, dict[str, PosteriorModel]]
        self.model_name: str = self.name.rsplit()[-1]
        self.ckpt_path: str = (
            f"{'best' if self.options.save_best else 'latest'}.model.checkpoint/"
        )

        # === Result Variables ===
        self.results: dict[str, dict[str, PosteriorStepResult]]

        # === Analysis Variables ===
        self.analysis: PosteriorStepAnalysis = (
            self.options.analysis or PosteriorStepAnalysis()
        )

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
        *args: Any,
        data: "DataStepResult",
        pae: "PAEStepResult",
        nflow: "NFlowStepResult",
        **kwargs: Any,
    ) -> None:
        super()._setup()
        # === Previous Step Variables ===
        self.nflow = nflow.model
        self.pae = pae.model

        self.data = data.data
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

        self.data_dir = data.dir

        self.sn_dim = data.sn_dim
        self.spec_dim = data.spec_dim
        self.wl_dim = data.wl_dim

        self.u_latent_bounds = {}
        self.step_sizes = {}
        self.recon_error = {}
        self.recon_error_centers = {}
        for subset in self.subsets:
            # --- ULatent Bounds ---
            if self.bounded_u_latents:
                nearest = 5
                nflow_subset = nflow.models[subset]
                z_latents = nflow_subset.z_latents
                u_latents = nflow_subset.u_latents
                mask = getattr(self.nflow, f"{subset}_mask")[..., None]
                sn_mask = getattr(self.nflow, f"{subset}_sn_mask")
                spec_mask = getattr(self.nflow, f"{subset}_spec_mask")
                wl_mask = getattr(self.nflow, f"{subset}_wl_mask")
                mask_sn = np.any(
                    np.any(mask & wl_mask & spec_mask & sn_mask, axis=-1), axis=-1
                )
                u_latents_min = np.min(u_latents[mask_sn], axis=0) / nearest
                u_latents_max = np.max(u_latents[mask_sn], axis=0) / nearest
                u_latents_min = (
                    np.where(
                        u_latents_min > 0,
                        np.ceil(u_latents_min),
                        np.floor(u_latents_min),
                    )
                    * nearest
                )
                u_latents_max = (
                    np.where(
                        u_latents_max > 0,
                        np.ceil(u_latents_max),
                        np.floor(u_latents_max),
                    )
                    * nearest
                )
                u_latents_min = np.min(u_latents_min)
                u_latents_max = np.max(u_latents_max)
                u_latents_bounds = (u_latents_min, u_latents_max)
            else:
                u_latents_bounds = (-np.inf, np.inf)
            self.u_latent_bounds[subset] = u_latents_bounds

            # --- Step Sizes ---
            pae_subset = pae.stages[subset][str(pae.model.stage.stage)]
            z_latents = pae_subset.latents
            zs = z_latents[..., :-2] if self.pae.physical_latents else z_latents
            u_latents = self.nflow.z_to_u(zs, permute=True)
            mask = getattr(self.pae.stage, f"{subset}_mask")
            sn_mask = getattr(self.pae.stage, f"{subset}_sn_mask")
            spec_mask = getattr(self.pae.stage, f"{subset}_spec_mask")
            wl_mask = getattr(self.pae.stage, f"{subset}_wl_mask")
            mask_sn = np.any(
                np.any(mask & wl_mask & spec_mask & sn_mask, axis=-1), axis=-1
            )
            step_sizes = []
            if self.train_delta_m:
                if pae.model.physical_latents:
                    delta_m = z_latents[..., -2:-1]
                    delta_m_step_size = np.std(delta_m[mask_sn], axis=0)
                else:
                    delta_m_step_size = np.array(self.delta_m_std)
                step_sizes.append(delta_m_step_size)
            if self.train_delta_p:
                if pae.model.physical_latents:
                    delta_p = z_latents[..., -1:]
                    delta_p_step_size = np.std(delta_p[mask_sn], axis=0)
                else:
                    delta_p_step_size = np.array(self.delta_p_std)
                step_sizes.append(delta_p_step_size)
            if self.train_bias:
                bias_step_size = np.array(self.bias_std)
                step_sizes.append(bias_step_size)
            u_latent_step_size = np.std(u_latents[mask_sn], axis=0)
            step_sizes.append(u_latent_step_size)
            step_sizes = np.concatenate(step_sizes, axis=-1)
            self.step_sizes[subset] = step_sizes

            # --- Reconstruction Error ---
            stage = self.pae.stage
            stage_subset = self.reconstruction_error
            if stage_subset == "match":
                stage_subset = subset
            if stage_subset == "combined":
                stage_subset = ""
            else:
                stage_subset += "_"

            subset_data = getattr(stage, f"{stage_subset}data")
            time = subset_data.time
            amplitude = subset_data.amplitude
            sigma = subset_data.sigma
            mask = getattr(stage, f"{stage_subset}mask")
            sn_mask = getattr(stage, f"{stage_subset}sn_mask")
            spec_mask = getattr(stage, f"{stage_subset}spec_mask")
            wl_mask = getattr(stage, f"{stage_subset}wl_mask")

            recon_error, _, recon_error_centers = pae.model.recon_error(
                (
                    time,
                    amplitude,
                    sigma,
                    mask,
                    sn_mask,
                    spec_mask,
                    wl_mask,
                ),
                fractional_error=self.fractional_error,
                weighted_error=self.weighted_error,
                measurement_error=self.measurement_error,
            )

            self.recon_error[subset] = recon_error
            self.recon_error_centers[subset] = recon_error_centers

        print(self.u_latent_bounds)

        # --- Stages ---
        i = 0
        self.map_stage_setup = PosteriorMAPStage.model_validate({
            "stage": i,
            "name": "setup",
            "fname": "setup",
            "n_chains": 1,
            "init": True,
            "setup": True,
        })
        i += 1

        self.map_stage_init = PosteriorMAPStage.model_validate({
            "stage": i,
            "name": "init",
            "fname": "init",
            "n_chains": 1,
            "init": True,
        })
        i += 1

        self.map_stage_constant = PosteriorMAPStage.model_validate({
            "stage": i,
            "name": "constant",
            "fname": "constant",
            "n_chains": 1,
            "init_u_delta_av": "constant",
            "init_latents": "u_constant",
            "init_delta_av": "constant",
            "init_delta_m": "constant",
            "init_delta_p": "constant",
            "init_bias": "constant",
        })
        i += 1

        if self.legacy_path is not None:
            self.map_stage_legacy = PosteriorMAPStage.model_validate({
                "stage": i,
                "name": "legacy",
                "fname": "legacy",
                "n_chains": 1,
                "init_u_delta_av": "constant",
                "init_latents": "u_constant",
                "init_delta_av": "legacy",
                "init_delta_m": "legacy",
                "init_delta_p": "legacy",
                "init_bias": "constant",
            })
            i += 1
        else:
            self.map_stage_legacy = None
        self.map_stage_random = PosteriorMAPStage.model_validate({
            "stage": i,
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
        i += 1

        self.map_stage_delta_av = PosteriorMAPStage.model_validate({
            "stage": i,
            "name": "delta_av",
            "fname": "delta_av",
            "n_chains": self.n_delta_av_chains,
            "init_u_delta_av": "data",
            "init_latents": "z_constant",
            "init_delta_av": "scale",
            "init_delta_m": "constant",
            "init_delta_p": "constant",
            "init_bias": "current",
        })
        i += 1

        self.map_stage_delta_m = PosteriorMAPStage.model_validate({
            "stage": i,
            "name": "delta_m",
            "fname": "delta_m",
            "n_chains": self.n_delta_m_chains,
            "init_u_delta_av": "constant",
            "init_latents": "u_constant",
            "init_delta_av": "data",
            "init_delta_m": "scale",
            "init_delta_p": "constant",
            "init_bias": "current",
        })
        i += 1

        self.map_stages = (
            [self.map_stage_setup, self.map_stage_init, self.map_stage_constant]
            + ([self.map_stage_legacy] if self.map_stage_legacy is not None else [])
            + [
                self.map_stage_random,
                self.map_stage_delta_av,
                self.map_stage_delta_m,
            ]
        )

    @override
    def _has_run(self, *args: "Any", **kwargs: "Any") -> bool:
        if self.has_attributes(self.run_attributes):
            for subset in self.subsets:
                for seed in self.seeds:
                    model = self.models[subset][str(seed)]
                    if not hasattr(model, "map"):
                        return False
                    if not hasattr(model, "hmc"):
                        return False
        else:
            return False
        return True

    @override
    def _run(self, *args: Any, **kwargs: Any) -> None:
        models = {}
        for subset in self.subsets:
            models[subset] = {}
            for seed in self.seeds:
                self.model = self.model.__class__(self, subset, seed)
                savepath = self.paths.results / self.model.name / subset / str(seed)
                ckpt_path = savepath / self.model.ckpt_path
                # Don't retrain stages if you don't need to
                if self.force or not (ckpt_path.exists() and any(ckpt_path.iterdir())):
                    self.model.train_model(self.map_stages, savepath=savepath)
                self.log.debug(
                    f"Loading Posterior {subset}_{seed} weights from {ckpt_path}"
                )
                self.model.load_checkpoint(savepath, load_map=True, load_hmc=True)
                models[subset][str(seed)] = self.model
        self.models = models

    @override
    def _is_saved(self, *args: Any, **kwargs: Any) -> bool:
        for subset in self.subsets:
            for seed in self.seeds:
                self.model = self.model.__class__(self, subset, seed)
                savepath = self.paths.results / self.model.name
                ckpt_path = savepath / subset / str(seed) / self.model.ckpt_path
                if not (ckpt_path.exists() and any(ckpt_path.iterdir())):
                    self.log.debug(
                        f"{self.name} is not saved as {savepath} does not exist"
                    )
                    return False
        return True

    @override
    def _save(self, *args: "Any", **kwargs: "Any") -> None:
        for subset in self.subsets:
            for seed in self.seeds:
                self.model = self.models[subset][str(seed)]
                savepath = self.paths.results / self.model.name / subset / str(seed)
                self.model.load_checkpoint(savepath, load_map=True, load_hmc=True)
                self.log.debug(
                    f"Saving Posterior {subset}_{seed} model weights to {savepath}"
                )
                self.model.save_checkpoint(savepath, save_map=True, save_hmc=True)

    @override
    def _load(self, *args: Any, **kwargs: Any) -> None:
        models = {}
        for subset in self.subsets:
            models[subset] = {}
            for seed in self.seeds:
                self.model = self.model.__class__(self, subset, seed)
                savepath = self.paths.results / self.model.name / subset / str(seed)
                self.log.debug(
                    f"Loading Posterior {subset}_{seed} model weights from {savepath}"
                )
                self.model.load_checkpoint(savepath, load_map=True, load_hmc=True)
                models[subset][str(seed)] = self.model
        self.models = models

    @override
    def _has_results(self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(["results"])

    @override
    def _result(self, *args: Any, **kwargs: Any) -> None:
        results = {}
        for subset in self.subsets:
            results[subset] = {}
            for seed in self.seeds:
                model = self.models[subset][str(seed)]
                data = getattr(self, f"{subset}_data")
                input_ind = data.ind
                input_sn_name = data.sn_name
                input_spectra_id = data.spectra_id
                data.clear()

                map_results = {
                    "chain_min": model.map.chain_min.numpy(),
                    "converged": model.map.converged.numpy(),
                    "num_evaluations": model.map.num_evaluations.numpy(),
                    "negative_log_prior": model.map.negative_log_prior.numpy(),
                    "negative_log_like": model.map.negative_log_like.numpy(),
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

                samples = model.hmc.samples
                samples = samples.numpy().reshape((
                    samples.shape[0] * samples.shape[1],
                    *samples.shape[2:],
                ))

                delta_m = samples[..., 0]
                delta_p = samples[..., 1]
                u_delta_av = samples[..., 2]
                u_latents = samples[..., 3:]

                zs = model.hmc.zs.numpy()
                delta_av = zs[..., 0]
                z_latents = zs[..., 1:]

                hmc_results = {
                    "samples": samples,
                    "r_hat": model.r_hat.numpy(),
                    # "step_sizes_final": model.hmc.step_sizes_final.numpy(),
                    # "is_accepted": model.hmc.is_accepted.numpy(),
                    "delta_m": delta_m,
                    "delta_p": delta_p,
                    "u_delta_av": u_delta_av,
                    "u_latents": u_latents,
                    "delta_av": delta_av,
                    "z_latents": z_latents,
                    "log_prior": model.hmc.log_prior.numpy(),
                    "log_like": model.hmc.log_like.numpy(),
                    "log_prob": model.hmc.log_prob.numpy(),
                }

                model_results = {
                    "ind": input_ind,
                    "sn_name": input_sn_name,
                    "spectra_id": input_spectra_id,
                    "map": map_results,
                    "hmc": hmc_results,
                }
                results[subset][str(seed)] = PosteriorStepResult.model_validate(
                    model_results
                )

        self.results = results

    @override
    def _was_analysed(self, *args: "Any", **kwargs: "Any") -> bool:
        for subset in self.subsets:
            for seed in self.seeds:
                if self.analysis.plot_comparison is not None:
                    if not isinstance(self.analysis.plot_comparison, list):
                        self.analysis.plot_comparison = [self.analysis.plot_comparison]
                    for opts in self.analysis.plot_comparison:
                        name = (
                            f"comparison_{opts.reduce}"
                            if opts.name is None
                            else opts.name
                        )
                        savepath = (
                            self.paths.plots
                            / str(self.seeds[0])
                            / subset
                            / str(seed)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False

                if self.analysis.plot_comparison_spectra is not None:
                    if not isinstance(self.analysis.plot_comparison_spectra, list):
                        self.analysis.plot_comparison_spectra = [
                            self.analysis.plot_comparison_spectra
                        ]
                    for opts in self.analysis.plot_comparison_spectra:
                        name = "comparison_spectra" if opts.name is None else opts.name
                        savepath = (
                            self.paths.plots
                            / str(self.seeds[0])
                            / subset
                            / str(seed)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False

                if self.analysis.plot_comparison_array is not None:
                    if not isinstance(self.analysis.plot_comparison_array, list):
                        self.analysis.plot_comparison_array = [
                            self.analysis.plot_comparison_array
                        ]
                    for opts in self.analysis.plot_comparison_array:
                        name = (
                            f"comparison_array_{opts.reduce}"
                            if opts.name is None
                            else opts.name
                        )
                        savepath = (
                            self.paths.plots
                            / str(self.seeds[0])
                            / subset
                            / str(seed)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False

                if self.analysis.plot_map_init is not None:
                    if not isinstance(self.analysis.plot_map_init, list):
                        self.analysis.plot_map_init = [self.analysis.plot_map_init]
                    for opts in self.analysis.plot_map_init:
                        name = "map_init" if opts.name is None else opts.name
                        if opts.masked:
                            name += "_masked"
                        if opts.mean:
                            name += "_mean"
                        savepath = (
                            self.paths.plots
                            / str(self.seeds[0])
                            / subset
                            / str(seed)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False

                if self.analysis.plot_map_best is not None:
                    if not isinstance(self.analysis.plot_map_best, list):
                        self.analysis.plot_map_best = [self.analysis.plot_map_best]
                    for opts in self.analysis.plot_map_best:
                        name = "map_best" if opts.name is None else opts.name
                        if opts.masked:
                            name += "_masked"
                        if opts.mean:
                            name += "_mean"
                        savepath = (
                            self.paths.plots
                            / str(self.seeds[0])
                            / subset
                            / str(seed)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False

                if self.analysis.plot_hmc is not None:
                    if not isinstance(self.analysis.plot_hmc, list):
                        self.analysis.plot_hmc = [self.analysis.plot_hmc]
                    for opts in self.analysis.plot_hmc:
                        name = "hmc" if opts.name is None else opts.name
                        if opts.masked:
                            name += "_masked"
                        if opts.mean:
                            name += "_mean"
                        savepath = (
                            self.paths.plots
                            / str(self.seeds[0])
                            / subset
                            / str(seed)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False
                if self.analysis.plot_dispersion is not None:
                    if not isinstance(self.analysis.plot_dispersion, list):
                        self.analysis.plot_dispersion = [self.analysis.plot_dispersion]
                    for opts in self.analysis.plot_dispersion:
                        name = (
                            f"dispersion_{opts.reduce}"
                            if opts.name is None
                            else opts.name
                        )
                        savepath = (
                            self.paths.plots
                            / str(self.seeds[0])
                            / subset
                            / str(seed)
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

    def _plot_comparison(
        self,
        subset: str,
        seed: int,
        results: "PosteriorStepResult",
        model: "TFPosteriorModel",
        data: "LazySNPAEData",
        input_mask: "npt.NDArray[bool]",
        input_sn_mask: "npt.NDArray[bool]",
        input_spec_mask: "npt.NDArray[bool]",
        input_wl_mask: "npt.NDArray[bool]",
        *,
        figs: "list[Figure] | None" = None,
        axes: "list[Axis] | None" = None,
        save: bool = True,
        force: bool = False,
        decorate: bool = True,
    ) -> list[tuple["Figure | None", "list[Axis] | None"]]:
        rtn = []
        if self.analysis.plot_comparison is not None:
            if not isinstance(self.analysis.plot_comparison, list):
                self.analysis.plot_comparison = [self.analysis.plot_comparison]
            for i, opts in enumerate(self.analysis.plot_comparison):
                fig = figs[i] if figs is not None else None
                ax = axes[i] if axes is not None else None

                o = opts.model_copy(deep=True)
                if o.name is None:
                    o.name = f"comparison_{o.reduce}"
                if o.plot_kwargs is None:
                    o.plot_kwargs = {
                        "title": f"{subset}_{self.name}_{o.name}",
                    }
                if o.savepath is None:
                    o.savepath = (
                        self.paths.plots / str(self.seeds[0]) / subset / str(seed)
                    )
                o.savepath.mkdir(parents=True, exist_ok=True)
                if (o.savepath / f"{o.name}.{o.ext}").exists() and not force:
                    rtn.append((None, None))
                    continue
                if not force:
                    self.log.debug(f"Plotting {o.name}")

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
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                )
                if not np.any(mask) and not force:
                    continue

                o.base_wl = wl
                o.base_amp = amplitude
                o.base_sigma = sigma
                o.base_mask = np.logical_not(mask)

                # === PAE ===
                # --- PAE ---
                pae_pae_input = np.concatenate((data.time, data.amplitude), axis=-1)
                pae_pae_latents, _ = model.pae(
                    pae_pae_input,
                    training=False,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                )
                pae_pae_latents = pae_pae_latents[:, 0, :][None, ...]

                # --- NFlow ---
                pae_delta_m = pae_pae_latents[..., -2:-1]
                pae_delta_p = pae_pae_latents[..., -1:]
                pae_pae_latents = pae_pae_latents[..., :-2]
                pae_u_latents = model.nflow.z_to_u(pae_pae_latents, permute=True)

                # --- Posterior ---
                pae_position = model.map.unconstrain(
                    model.map.get_position(
                        np.concatenate(
                            (pae_delta_m, pae_delta_p, pae_u_latents),
                            axis=-1,
                        )
                    ),
                    full=True,
                )
                (
                    pae_log_prob,
                    pae_log_like,
                    pae_log_prior,
                    pae_amplitude,
                    pae_sigma,
                ) = model(
                    pae_position,
                    training=False,
                    input_phase=data.time,
                    input_amp=data.amplitude,
                    input_sigma=data.sigma,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    additional_outputs=True,
                )

                pae_amplitude = pae_amplitude[0, ...]
                pae_sigma = pae_sigma[0, ...]

                if not force:
                    pae_log_prob = pae_log_prob.numpy()
                    pae_log_like = pae_log_like.numpy()
                    pae_log_prior = pae_log_prior.numpy()

                    mean_pae_log_prob = float(
                        np.sum(
                            np.where(
                                np.isfinite(pae_log_prob),
                                pae_log_prob,
                                np.zeros_like(pae_log_prob),
                            )
                        )
                        / max(
                            np.sum(
                                np.where(
                                    np.isfinite(pae_log_prob),
                                    np.ones_like(pae_log_prob),
                                    np.zeros_like(pae_log_prob),
                                )
                            ),
                            1,
                        )
                    )
                    mean_pae_log_like = float(
                        np.sum(
                            np.where(
                                np.isfinite(pae_log_like),
                                pae_log_like,
                                np.zeros_like(pae_log_like),
                            )
                        )
                        / max(
                            np.sum(
                                np.where(
                                    np.isfinite(pae_log_like),
                                    np.ones_like(pae_log_like),
                                    np.zeros_like(pae_log_like),
                                )
                            ),
                            1,
                        )
                    )
                    mean_pae_log_prior = float(
                        np.sum(
                            np.where(
                                np.isfinite(pae_log_prior),
                                pae_log_prior,
                                np.zeros_like(pae_log_prior),
                            )
                        )
                        / max(
                            np.sum(
                                np.where(
                                    np.isfinite(pae_log_prior),
                                    np.ones_like(pae_log_prior),
                                    np.zeros_like(pae_log_prior),
                                )
                            ),
                            1,
                        )
                    )
                    o.plot_kwargs["label"] = (
                        f"PAE\n(log prob: {mean_pae_log_prior:.2E} + {mean_pae_log_like:.2E} = {mean_pae_log_prob:.2E})"
                    )

                data.amplitude = pae_amplitude.numpy()
                data.sigma = pae_sigma.numpy()

                fig, ax = SpectraPlotter.plot_comparison(
                    data,
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    fig=fig,
                    ax=ax,
                    save=False,
                    force=force,
                    decorate=decorate,
                )
                o.plot_base = False
                # print("pae")
                # pp(
                #     pae_position.numpy()[:, np.any(np.any(mask, axis=-1), axis=-1), :][
                #         0, 0, :
                #     ]
                # )

                # === MAP ===
                # --- PAE ---
                # --- NFlow ---
                # --- Posterior ---
                map_position = model.map.unconstrain(
                    model.map.get_position(model.map.position.best), full=True
                )
                (
                    map_log_prob,
                    map_log_like,
                    map_log_prior,
                    map_amplitude,
                    map_sigma,
                ) = model(
                    map_position,
                    training=False,
                    input_phase=data.time,
                    input_amp=data.amplitude,
                    input_sigma=data.sigma,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    additional_outputs=True,
                )

                map_amplitude = map_amplitude[0, ...]
                map_sigma = map_sigma[0, ...]

                if not force:
                    map_log_prob = map_log_prob.numpy()
                    map_log_like = map_log_like.numpy()
                    map_log_prior = map_log_prior.numpy()
                    mean_map_log_prob = float(
                        np.sum(
                            np.where(
                                np.isfinite(map_log_prob),
                                map_log_prob,
                                np.zeros_like(map_log_prob),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(map_log_prob),
                                np.ones_like(map_log_prob),
                                np.zeros_like(map_log_prob),
                            )
                        )
                    )
                    mean_map_log_like = float(
                        np.sum(
                            np.where(
                                np.isfinite(map_log_like),
                                map_log_like,
                                np.zeros_like(map_log_like),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(map_log_like),
                                np.ones_like(map_log_like),
                                np.zeros_like(map_log_like),
                            )
                        )
                    )
                    mean_map_log_prior = float(
                        np.sum(
                            np.where(
                                np.isfinite(map_log_prior),
                                map_log_prior,
                                np.zeros_like(map_log_prior),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(map_log_prior),
                                np.ones_like(map_log_prior),
                                np.zeros_like(map_log_prior),
                            )
                        )
                    )
                    o.plot_kwargs["label"] = (
                        f"MAP\n(log prob: {mean_map_log_prior:.2E} + {mean_map_log_like:.2E} = {mean_map_log_prob:.2E})"
                    )

                data.amplitude = map_amplitude.numpy()
                data.sigma = map_sigma.numpy()

                fig, ax = SpectraPlotter.plot_comparison(
                    data,
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    fig=fig,
                    ax=ax,
                    save=False,
                    force=force,
                    decorate=decorate,
                )
                # print("map")
                # pp(
                #     map_position.numpy()[:, np.any(np.any(mask, axis=-1), axis=-1), :][
                #         0, 0, :
                #     ]
                # )

                # === Posterior ===
                # --- PAE ---
                # --- NFlow ---
                # --- Posterior ---
                samples = results.hmc.samples
                log_prob = results.hmc.log_prob
                if o.reduce == "mean":
                    reduce_samples = samples.mean(axis=0, keepdims=True)
                elif o.reduce == "median":
                    reduce_samples = np.median(samples, axis=0, keepdims=True)
                else:
                    reduce_samples = np.array([
                        np.array([
                            max_central(
                                samples[..., sn, pos], weight=log_prob[..., sn]
                            )[1]
                            for pos in range(samples.shape[-1])
                        ])
                        for sn in range(samples.shape[-2])
                    ])[None, ...]
                pos_position = model.map.unconstrain(
                    model.map.get_position(reduce_samples), full=True
                )
                (
                    pos_log_prob,
                    pos_log_like,
                    pos_log_prior,
                    pos_amplitude,
                    pos_sigma,
                ) = model(
                    pos_position,
                    training=False,
                    input_phase=data.time,
                    input_amp=data.amplitude,
                    input_sigma=data.sigma,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    additional_outputs=True,
                )

                pos_amplitude = pos_amplitude[0, ...]
                pos_sigma = pos_sigma[0, ...]

                if not force:
                    pos_log_prob = pos_log_prob.numpy()
                    pos_log_like = pos_log_like.numpy()
                    pos_log_prior = pos_log_prior.numpy()

                    mean_pos_log_prob = float(
                        np.sum(
                            np.where(
                                np.isfinite(pos_log_prob),
                                pos_log_prob,
                                np.zeros_like(pos_log_prob),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(pos_log_prob),
                                np.ones_like(pos_log_prob),
                                np.zeros_like(pos_log_prob),
                            )
                        )
                    )
                    mean_pos_log_like = float(
                        np.sum(
                            np.where(
                                np.isfinite(pos_log_like),
                                pos_log_like,
                                np.zeros_like(pos_log_like),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(pos_log_like),
                                np.ones_like(pos_log_like),
                                np.zeros_like(pos_log_like),
                            )
                        )
                    )
                    mean_pos_log_prior = float(
                        np.sum(
                            np.where(
                                np.isfinite(pos_log_prior),
                                pos_log_prior,
                                np.zeros_like(pos_log_prior),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(pos_log_prior),
                                np.ones_like(pos_log_prior),
                                np.zeros_like(pos_log_prior),
                            )
                        )
                    )

                    o.plot_kwargs["label"] = (
                        f"Posterior\n(log prob: {mean_pos_log_prior:.2E} + {mean_pos_log_like:.2E} = {mean_pos_log_prob:.2E})"
                    )

                data.amplitude = pos_amplitude.numpy()
                data.sigma = pos_sigma.numpy()

                rtn.append(
                    SpectraPlotter.plot_comparison(
                        data,
                        o,
                        mask=input_mask,
                        sn_mask=input_sn_mask,
                        spec_mask=input_spec_mask,
                        wl_mask=input_wl_mask,
                        fig=fig,
                        ax=ax,
                        save=save,
                        force=force,
                        decorate=decorate,
                    )
                )
                # print("pos")
                # pp(
                #     pos_position.numpy()[:, np.any(np.any(mask, axis=-1), axis=-1), :][
                #         0, 0, :
                #     ]
                # )
        return rtn

    def _plot_comparison_spectra(
        self,
        subset: str,
        seed: int,
        results: "PosteriorStepResult",
        model: "TFPosteriorModel",
        data: "LazySNPAEData",
        input_mask: "npt.NDArray[bool]",
        input_sn_mask: "npt.NDArray[bool]",
        input_spec_mask: "npt.NDArray[bool]",
        input_wl_mask: "npt.NDArray[bool]",
        *,
        figs: "list[Figure] | None" = None,
        axes: "list[Axis] | None" = None,
        save: bool = True,
        force: bool = False,
    ) -> list[tuple["Figure | None", "list[Axis] | None"]]:
        rtn = []
        if self.analysis.plot_comparison_spectra is not None:
            if not isinstance(self.analysis.plot_comparison_spectra, list):
                self.analysis.plot_comparison_spectra = [
                    self.analysis.plot_comparison_spectra
                ]
            for i, opts in enumerate(self.analysis.plot_comparison_spectra):
                fig = figs[i] if figs is not None else None
                ax = axes[i] if axes is not None else None

                o = opts.model_copy(deep=True)
                if o.name is None:
                    o.name = "comparison_spectra"
                if o.plot_kwargs is None:
                    o.plot_kwargs = {
                        "title": f"{subset}_{self.name}_{o.name}",
                    }
                if o.savepath is None:
                    o.savepath = (
                        self.paths.plots / str(self.seeds[0]) / subset / str(seed)
                    )
                o.savepath.mkdir(parents=True, exist_ok=True)
                if (o.savepath / f"{o.name}.{o.ext}").exists() and not force:
                    rtn.append((None, None))
                    continue
                if not force:
                    self.log.debug(f"Plotting {o.name}")

                (
                    _wl,
                    _amplitude,
                    _sigma,
                    _sn_name,
                    _time,
                    mask,
                    _sn_mask,
                    _spec_mask,
                    _wl_mask,
                ) = SpectraPlotter.prep(
                    data,
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                )
                if not np.any(mask):
                    continue

                o.plot_kwargs["label"] = "Base"
                fig, ax = SpectraPlotter.plot_spectra(
                    data,
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    fig=fig,
                    ax=ax,
                    save=False,
                    force=force,
                    decorate=True,
                )

                # === PAE ===
                # --- PAE ---
                pae_pae_input = np.concatenate((data.time, data.amplitude), axis=-1)
                pae_pae_latents, _ = model.pae(
                    pae_pae_input,
                    training=False,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                )
                pae_pae_latents = pae_pae_latents[:, 0, :]

                # --- NFlow ---
                pae_delta_m = pae_pae_latents[..., -2:-1]
                pae_delta_p = pae_pae_latents[..., -1:]
                pae_pae_latents = pae_pae_latents[..., :-2]
                pae_u_latents = model.nflow.z_to_u(pae_pae_latents, permute=True)

                # --- Posterior ---
                pae_position = model.map.unconstrain(
                    model.map.get_position(
                        np.concatenate(
                            (pae_delta_m, pae_delta_p, pae_u_latents),
                            axis=-1,
                        )
                    ),
                    full=True,
                )
                (
                    pae_log_prob,
                    pae_log_like,
                    pae_log_prior,
                    pae_amplitude,
                    pae_sigma,
                ) = model(
                    pae_position,
                    training=False,
                    input_phase=data.time,
                    input_amp=data.amplitude,
                    input_sigma=data.sigma,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    additional_outputs=True,
                )

                if not force:
                    pae_log_prob = pae_log_prob.numpy()
                    pae_log_like = pae_log_like.numpy()
                    pae_log_prior = pae_log_prior.numpy()

                    mean_pae_log_prob = float(
                        np.sum(
                            np.where(
                                np.isfinite(pae_log_prob),
                                pae_log_prob,
                                np.zeros_like(pae_log_prob),
                            )
                        )
                        / max(
                            np.sum(
                                np.where(
                                    np.isfinite(pae_log_prob),
                                    np.ones_like(pae_log_prob),
                                    np.zeros_like(pae_log_prob),
                                )
                            ),
                            1,
                        )
                    )
                    mean_pae_log_like = float(
                        np.sum(
                            np.where(
                                np.isfinite(pae_log_like),
                                pae_log_like,
                                np.zeros_like(pae_log_like),
                            )
                        )
                        / max(
                            np.sum(
                                np.where(
                                    np.isfinite(pae_log_like),
                                    np.ones_like(pae_log_like),
                                    np.zeros_like(pae_log_like),
                                )
                            ),
                            1,
                        )
                    )
                    mean_pae_log_prior = float(
                        np.sum(
                            np.where(
                                np.isfinite(pae_log_prior),
                                pae_log_prior,
                                np.zeros_like(pae_log_prior),
                            )
                        )
                        / max(
                            np.sum(
                                np.where(
                                    np.isfinite(pae_log_prior),
                                    np.ones_like(pae_log_prior),
                                    np.zeros_like(pae_log_prior),
                                )
                            ),
                            1,
                        )
                    )
                    o.plot_kwargs["label"] = (
                        f"PAE\n(log prob: {mean_pae_log_prior:.2E} + {mean_pae_log_like:.2E} = {mean_pae_log_prob:.2E})"
                    )

                data.amplitude = pae_amplitude.numpy()
                data.sigma = pae_sigma.numpy()

                fig, ax = SpectraPlotter.plot_spectra(
                    data,
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    fig=fig,
                    ax=ax,
                    save=False,
                    force=force,
                    decorate=False,
                    offset=1,
                )

                # === MAP ===
                # --- PAE ---
                # --- NFlow ---
                # --- Posterior ---
                map_position = model.map.get_position(model.map.position.best)
                (
                    map_log_prob,
                    map_log_like,
                    map_log_prior,
                    map_amplitude,
                    map_sigma,
                ) = model(
                    model.map.unconstrain(map_position),
                    training=False,
                    input_phase=data.time,
                    input_amp=data.amplitude,
                    input_sigma=data.sigma,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    additional_outputs=True,
                )

                if not force:
                    map_log_prob = map_log_prob.numpy()
                    map_log_like = map_log_like.numpy()
                    map_log_prior = map_log_prior.numpy()
                    mean_map_log_prob = float(
                        np.sum(
                            np.where(
                                np.isfinite(map_log_prob),
                                map_log_prob,
                                np.zeros_like(map_log_prob),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(map_log_prob),
                                np.ones_like(map_log_prob),
                                np.zeros_like(map_log_prob),
                            )
                        )
                    )
                    mean_map_log_like = float(
                        np.sum(
                            np.where(
                                np.isfinite(map_log_like),
                                map_log_like,
                                np.zeros_like(map_log_like),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(map_log_like),
                                np.ones_like(map_log_like),
                                np.zeros_like(map_log_like),
                            )
                        )
                    )
                    mean_map_log_prior = float(
                        np.sum(
                            np.where(
                                np.isfinite(map_log_prior),
                                map_log_prior,
                                np.zeros_like(map_log_prior),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(map_log_prior),
                                np.ones_like(map_log_prior),
                                np.zeros_like(map_log_prior),
                            )
                        )
                    )
                    o.plot_kwargs["label"] = (
                        f"MAP\n(log prob: {mean_map_log_prior:.2E} + {mean_map_log_like:.2E} = {mean_map_log_prob:.2E})"
                    )

                data.amplitude = map_amplitude.numpy()
                data.sigma = map_sigma.numpy()

                fig, ax = SpectraPlotter.plot_spectra(
                    data,
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    fig=fig,
                    ax=ax,
                    save=False,
                    force=force,
                    decorate=False,
                    offset=2,
                )

                # === Posterior ===
                # --- PAE ---
                # --- NFlow ---
                # --- Posterior ---
                samples = results.hmc.samples
                log_prob = results.hmc.log_prob
                if o.reduce == "mean":
                    reduce_samples = samples.mean(axis=0)
                elif o.reduce == "median":
                    reduce_samples = np.median(samples, axis=0)
                else:
                    reduce_samples = np.array([
                        np.array([
                            max_central(samples[:, sn, pos], weight=log_prob[:, sn])[1]
                            for pos in range(samples.shape[-1])
                        ])
                        for sn in range(samples.shape[-2])
                    ])
                pos_position = model.map.unconstrain(
                    model.map.get_position(reduce_samples), full=True
                )
                (
                    pos_log_prob,
                    pos_log_like,
                    pos_log_prior,
                    pos_amplitude,
                    pos_sigma,
                ) = model(
                    model.map.unconstrain(pos_position),
                    training=False,
                    input_phase=data.time,
                    input_amp=data.amplitude,
                    input_sigma=data.sigma,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    additional_outputs=True,
                )

                if not force:
                    pos_log_prob = pos_log_prob.numpy()
                    pos_log_like = pos_log_like.numpy()
                    pos_log_prior = pos_log_prior.numpy()

                    mean_pos_log_prob = float(
                        np.sum(
                            np.where(
                                np.isfinite(pos_log_prob),
                                pos_log_prob,
                                np.zeros_like(pos_log_prob),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(pos_log_prob),
                                np.ones_like(pos_log_prob),
                                np.zeros_like(pos_log_prob),
                            )
                        )
                    )
                    mean_pos_log_like = float(
                        np.sum(
                            np.where(
                                np.isfinite(pos_log_like),
                                pos_log_like,
                                np.zeros_like(pos_log_like),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(pos_log_like),
                                np.ones_like(pos_log_like),
                                np.zeros_like(pos_log_like),
                            )
                        )
                    )
                    mean_pos_log_prior = float(
                        np.sum(
                            np.where(
                                np.isfinite(pos_log_prior),
                                pos_log_prior,
                                np.zeros_like(pos_log_prior),
                            )
                        )
                        / np.sum(
                            np.where(
                                np.isfinite(pos_log_prior),
                                np.ones_like(pos_log_prior),
                                np.zeros_like(pos_log_prior),
                            )
                        )
                    )

                    o.plot_kwargs["label"] = (
                        f"Posterior\n(log prob: {mean_pos_log_prior:.2E} + {mean_pos_log_like:.2E} = {mean_pos_log_prob:.2E})"
                    )

                data.amplitude = pos_amplitude.numpy()
                data.sigma = pos_sigma.numpy()

                rtn.append(
                    SpectraPlotter.plot_spectra(
                        data,
                        o,
                        mask=input_mask,
                        sn_mask=input_sn_mask,
                        spec_mask=input_spec_mask,
                        wl_mask=input_wl_mask,
                        fig=fig,
                        ax=ax,
                        save=save,
                        force=force,
                        decorate=False,
                        offset=3,
                    )
                )
        return rtn

    def _plot_comparison_array(
        self,
        subset: str,
        seed: int,
        model: "TFPosteriorModel",
        results: "PosteriorStepResult",
        data: "LazySNPAEData",
        input_mask: "npt.NDArray[bool]",
        input_sn_mask: "npt.NDArray[bool]",
        input_spec_mask: "npt.NDArray[bool]",
        input_wl_mask: "npt.NDArray[bool]",
    ) -> None:
        if self.analysis.plot_comparison_array is not None:
            if not isinstance(self.analysis.plot_comparison_array, list):
                self.analysis.plot_comparison_array = [
                    self.analysis.plot_comparison_array
                ]
            for opts in self.analysis.plot_comparison_array:
                o = opts.model_copy(deep=True)
                plot_comparison = self.analysis.plot_comparison
                self.analysis.plot_comparison = [o]
                if o.name is None:
                    o.name = f"comparison_array_{o.reduce}"
                if o.plot_kwargs is None:
                    o.plot_kwargs = {
                        "title": f"{subset}_{self.name}_{o.name}",
                    }
                if o.savepath is None:
                    o.savepath = (
                        self.paths.plots / str(self.seeds[0]) / subset / str(seed)
                    )
                o.savepath.mkdir(parents=True, exist_ok=True)
                if (o.savepath / f"{o.name}.{o.ext}").exists():
                    continue
                self.log.debug(f"Plotting {o.name}")

                (
                    _wl,
                    _amplitude,
                    _sigma,
                    _sn_name,
                    _time,
                    mask,
                    _sn_mask,
                    _spec_mask,
                    _wl_mask,
                ) = SpectraPlotter.prep(
                    data,
                    o,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                )
                if not np.any(mask):
                    continue

                plot_types = []
                if o.plot_best:
                    plot_types.append("Best")
                if o.plot_worst:
                    plot_types.append("Worst")
                if o.plot_mean:
                    plot_types.append("Mean")
                if o.plot_max_delta_m:
                    plot_types.append("Max DeltaM")
                if o.plot_min_delta_m:
                    plot_types.append("Min DeltaM")
                if o.plot_median:
                    plot_types.append("Median")
                if o.plot_max_delta_p:
                    plot_types.append("Max DeltaP")
                if o.plot_min_delta_p:
                    plot_types.append("Min DeltaP")
                if o.plot_names:
                    plot_types.extend(f"SN: {name}" for name in o.plot_names)
                n_plot_types = len(plot_types)
                n_random_plots = o.plot_random
                if o.plot_random == "auto":
                    n_random_plots = 9 if n_plot_types == 0 else 0
                elif o.plot_random == "none":
                    n_random_plots = 0

                n_plots = n_plot_types + n_random_plots

                n_plots = np.square(np.ceil(np.sqrt(n_plots)))

                if n_random_plots == 0 and o.plot_random != "none":
                    n_random_plots = n_plots - np.square(
                        np.floor(np.sqrt(n_plot_types))
                    )
                    if n_random_plots == 0:
                        n_plots = np.square(np.sqrt(n_plots) + 1)
                        n_random_plots = n_plots - np.square(
                            np.floor(np.sqrt(n_plot_types))
                        )
                n_plot_types = int(n_plot_types)
                n_random_plots = int(n_random_plots)
                n_plots = int(n_plots)

                plot_types += ["Random"] * n_random_plots
                plot_types += ["Blank"] * (n_plots - len(plot_types))

                nrows = ncols = int(np.sqrt(n_plots))

                supfig = Plotter.figure(scale=nrows)
                subfigs = np.array(Plotter.subfig(supfig, nrows, ncols)).flatten()
                axes = []

                for i, subfig in enumerate(subfigs):
                    plot_type = plot_types[i]
                    if plot_type == "Blank":
                        continue

                    log_prior = results.hmc.log_prior
                    mean_log_prior = np.mean(log_prior, axis=0)
                    log_like = results.hmc.log_like
                    mean_log_like = np.mean(log_like, axis=0)
                    log_prob = results.hmc.log_prob
                    mean_log_prob = np.mean(log_prob, axis=0)
                    valid_log_prob = input_sn_mask[:, 0, 0] & np.isfinite(mean_log_prob)

                    delta_m = results.hmc.samples[..., 0]
                    delta_p = results.hmc.samples[..., 1]

                    # delta_m = model.hmc.delta_m.numpy()[..., 0]
                    if o.reduce == "mean":
                        mean_delta_m = np.mean(delta_m, axis=0)
                    elif o.reduce == "median":
                        mean_delta_m = np.median(delta_m, axis=0)
                    else:
                        mean_delta_m = np.array([
                            max_central(delta_m[:, sn], weight=log_prob[:, sn])[1]
                            for sn in range(delta_m.shape[-1])
                        ])
                    # delta_p = model.hmc.delta_p.numpy()[..., 0]
                    if o.reduce == "mean":
                        mean_delta_p = np.mean(delta_p, axis=0)
                    if o.reduce == "median":
                        mean_delta_p = np.median(delta_p, axis=0)
                    else:
                        mean_delta_p = np.array([
                            max_central(delta_p[:, sn], weight=log_prob[:, sn])[1]
                            for sn in range(delta_p.shape[-1])
                        ])
                    names = data.sn_name[..., :1, :]

                    if plot_type == "Best":
                        mean_log_prob[~valid_log_prob] = -np.inf
                        sn_mask = (mean_log_prob == mean_log_prob.max())[:, None, None]
                    elif plot_type == "Worst":
                        mean_log_prob[~valid_log_prob] = np.inf
                        sn_mask = (mean_log_prob == mean_log_prob.min())[:, None, None]
                    elif plot_type == "Mean":
                        mean_log_prob[~valid_log_prob] = np.nan
                        mean_log_prob_dist = np.abs(
                            mean_log_prob - np.nanmean(mean_log_prob)
                        )
                        mean_log_prob_dist[~valid_log_prob] = np.inf
                        sn_mask = (mean_log_prob_dist == mean_log_prob_dist.min())[
                            :, None, None
                        ]
                    elif plot_type == "Max DeltaM":
                        dm = np.abs(mean_delta_m)
                        dm[~valid_log_prob] = -np.inf
                        sn_mask = (dm == dm.max())[:, None, None]
                    elif plot_type == "Min DeltaM":
                        dm = np.abs(mean_delta_m)
                        dm[~valid_log_prob] = np.inf
                        sn_mask = (dm == dm.min())[:, None, None]
                    elif plot_type == "Median":
                        mean_log_prob[~valid_log_prob] = np.nan
                        median_log_prob_dist = np.abs(
                            mean_log_prob
                            - np.nanmedian(mean_log_prob[input_sn_mask[:, 0, 0]])
                        )
                        median_log_prob_dist[~valid_log_prob] = np.inf
                        sn_mask = (median_log_prob_dist == median_log_prob_dist.min())[
                            :, None, None
                        ]
                    elif plot_type == "Max DeltaP":
                        dp = np.abs(mean_delta_p)
                        dp[~valid_log_prob] = -np.inf
                        sn_mask = (dp == dp.max())[:, None, None]
                    elif plot_type == "Min DeltaP":
                        dp = np.abs(mean_delta_p)
                        dp[~valid_log_prob] = np.inf
                        sn_mask = (dp == dp.min())[:, None, None]
                    elif plot_type.split()[0] == "SN:":
                        sn_mask = names == plot_type.split()[-1]
                    else:
                        random_log_prob_dist = np.abs(
                            mean_log_prob
                            - self.rng.choice(mean_log_prob[valid_log_prob])
                        )
                        random_log_prob_dist[~valid_log_prob] = np.inf
                        sn_mask = (random_log_prob_dist == random_log_prob_dist.min())[
                            :, None, None
                        ]

                    fig, ax = self._plot_comparison(
                        subset,
                        seed,
                        results,
                        model,
                        data,
                        input_mask,
                        sn_mask,
                        input_spec_mask,
                        input_wl_mask,
                        figs=[subfig],
                        save=False,
                        force=True,
                        decorate=False,
                    )[-1]
                    subfigs[i] = fig
                    if np.any(sn_mask[:, 0, 0]):
                        lp = mean_log_prior[sn_mask[:, 0, 0]][0]
                        ll = mean_log_like[sn_mask[:, 0, 0]][0]
                        lP = mean_log_prob[sn_mask[:, 0, 0]][0]
                        dm = mean_delta_m[sn_mask[:, 0, 0]][0]
                        dp = mean_delta_p[sn_mask[:, 0, 0]][0]
                        ax[0].set_title(
                            plot_type
                            + f" ({lp:.2E}+{ll:.2E}={lP:.2E}, {dm:.2E}, {dp:.2E})"
                        )
                    else:
                        ax[0].set_title(plot_type)
                    axes.extend(ax)
                supfig.suptitle(o.plot_kwargs["title"])
                supfig = Plotter.save(supfig, o.savepath / f"{o.name}.{o.ext}")
                Plotter.close(supfig, axes)
            self.analysis.plot_comparison = plot_comparison

    def _plot_map(
        self,
        subset: str,
        seed: int,
        data: "LazySNPAEData",
        input_mask: "npt.NDArray[bool]",
        input_sn_mask: "npt.NDArray[bool]",
        input_spec_mask: "npt.NDArray[bool]",
        input_wl_mask: "npt.NDArray[bool]",
        map_results,
        map_labels,
        *,
        is_init: bool,
    ) -> None:
        analysis = None
        if is_init:
            if self.analysis.plot_map_init is not None:
                if not isinstance(self.analysis.plot_map_init, list):
                    self.analysis.plot_map_init = [self.analysis.plot_map_init]
                analysis = self.analysis.plot_map_init
        elif self.analysis.plot_map_best is not None:
            if not isinstance(self.analysis.plot_map_best, list):
                self.analysis.plot_map_best = [self.analysis.plot_map_best]
            analysis = self.analysis.plot_map_best

        if analysis is not None:
            for opts in analysis:
                o = opts.model_copy(deep=True)
                if o.name is None:
                    o.name = "map_" + ("init" if is_init else "best")

                if o.masked:
                    o.name += "_masked"

                if o.savepath is None:
                    o.savepath = (
                        self.paths.plots / str(self.seeds[0]) / subset / str(seed)
                    )
                o.savepath.mkdir(parents=True, exist_ok=True)
                if (o.savepath / f"{o.name}.{o.ext}").exists():
                    continue

                chain_data = {}
                samples = map_results

                if o.masked:
                    (
                        _wl,
                        _amplitude,
                        _sigma,
                        _sn_name,
                        _time,
                        mask,
                        _sn_mask,
                        _spec_mask,
                        _wl_mask,
                    ) = SpectraPlotter.prep(
                        data,
                        o,
                        mask=input_mask,
                        sn_mask=input_sn_mask,
                        spec_mask=input_spec_mask,
                        wl_mask=input_wl_mask,
                    )

                    # Determine which spectra to keep
                    # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
                    mask_spec = np.any(mask, axis=-1)

                    # Determine which SNe to keep
                    # Will mask out any SN with *no* unmasked spectra
                    mask_sn = np.any(mask_spec, axis=-1)

                    samples = samples[mask_sn, ...]

                self.log.debug(f"Plotting {o.name}")

                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": f"{subset}_{self.name}_{o.name}"}
                if o.labels is None:
                    o.labels = {}
                title = o.plot_kwargs["title"]

                chain_data[title] = samples
                o.labels[title] = map_labels

                DistributionPlotter.plot_corner(
                    chain_data,
                    o,
                    statistics="cumulative" if o.reduce == "median" else o.reduce,
                )

    def _plot_hmc(
        self,
        subset: str,
        seed: int,
        data: "LazySNPAEData",
        results: PosteriorStepResult,
        input_mask: "npt.NDArray[bool]",
        input_sn_mask: "npt.NDArray[bool]",
        input_spec_mask: "npt.NDArray[bool]",
        input_wl_mask: "npt.NDArray[bool]",
        hmc_labels,
    ) -> None:
        if self.analysis.plot_hmc is not None:
            if not isinstance(self.analysis.plot_hmc, list):
                self.analysis.plot_hmc = [self.analysis.plot_hmc]
            for opts in self.analysis.plot_hmc:
                o = opts.model_copy(deep=True)
                if o.name is None:
                    o.name = "hmc"

                if o.masked:
                    o.name += "_masked"
                if o.mean:
                    o.name += "_mean"

                if o.savepath is None:
                    o.savepath = (
                        self.paths.plots / str(self.seeds[0]) / subset / str(seed)
                    )
                o.savepath.mkdir(parents=True, exist_ok=True)
                if (o.savepath / f"{o.name}.{o.ext}").exists():
                    continue

                (
                    _wl,
                    _amplitude,
                    _sigma,
                    _sn_name,
                    _time,
                    mask,
                    _sn_mask,
                    _spec_mask,
                    _wl_mask,
                ) = SpectraPlotter.prep(
                    data,
                    o,
                )
                # Determine which spectra to keep
                # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
                mask_spec = np.any(mask, axis=-1)

                # Determine which SNe to keep
                # Will mask out any SN with *no* unmasked spectra
                sn_mask = np.any(mask_spec, axis=-1)

                if not np.any(sn_mask):
                    continue

                chain_data = {}
                samples = results.hmc.samples
                log_prob = results.hmc.log_prob
                self.log.debug(f"Plotting {o.name}")

                if o.masked:
                    (
                        _wl,
                        _amplitude,
                        _sigma,
                        _sn_name,
                        _time,
                        mask,
                        _sn_mask,
                        _spec_mask,
                        _wl_mask,
                    ) = SpectraPlotter.prep(
                        data,
                        o,
                        mask=input_mask,
                        sn_mask=input_sn_mask,
                        spec_mask=input_spec_mask,
                        wl_mask=input_wl_mask,
                    )

                    # Determine which spectra to keep
                    # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
                    mask_spec = np.any(mask, axis=-1)

                    # Determine which SNe to keep
                    # Will mask out any SN with *no* unmasked spectra
                    mask_sn = np.any(mask_spec, axis=-1)
                    sn_mask &= mask_sn

                samples = samples[..., sn_mask, :]
                log_prob = log_prob[..., sn_mask]

                weights = None
                if o.mean:
                    if o.reduce == "mean":
                        chains = samples.mean(axis=0)
                    elif o.reduce == "median":
                        chains = np.median(samples, axis=0)
                    elif o.reduce == "max_central":
                        chains = np.array([
                            np.array([
                                max_central(
                                    samples[..., sn, pos], weight=log_prob[:, sn]
                                )[1]
                                for pos in range(samples.shape[-1])
                            ])
                            for sn in range(samples.shape[-2])
                        ])
                elif samples.shape[-2] == 1:
                    chains = samples[..., 0, :]
                    weights = log_prob[..., 0]
                else:
                    chains = np.reshape(samples, (-1, samples.shape[-1]))
                o.mean = False

                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": f"{subset}_{self.name}_{o.name}"}
                if o.labels is None:
                    o.labels = {}
                title = o.plot_kwargs["title"]

                chain_data[title] = chains
                o.labels[title] = hmc_labels

                DistributionPlotter.plot_corner(
                    chain_data,
                    o,
                    statistics="cumulative" if o.reduce == "median" else o.reduce,
                    log_posterior=weights,
                    plot_point=weights is not None,
                    marker_style="P",
                    marker_size=100,
                )

    def _plot_dispersion(
        self,
        subset: str,
        seed: int,
        data: "LazySNPAEData",
        input_mask: "npt.NDArray[bool]",
        input_sn_mask: "npt.NDArray[bool]",
        input_spec_mask: "npt.NDArray[bool]",
        input_wl_mask: "npt.NDArray[bool]",
        *,
        force: bool = False,
    ) -> None:
        if self.analysis.plot_dispersion is not None:
            if not isinstance(self.analysis.plot_dispersion, list):
                self.analysis.plot_dispersion = [self.analysis.plot_dispersion]
            for opts in self.analysis.plot_dispersion:
                o = opts.model_copy(deep=True)
                if o.subset != subset:
                    continue
                if o.name is None:
                    o.name = f"dispersion_{o.reduce}"
                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": f"{subset}_{self.name}_{o.reduce}"}
                if o.savepath is None:
                    o.savepath = (
                        self.paths.plots / str(self.seeds[0]) / subset / str(seed)
                    )
                o.savepath.mkdir(parents=True, exist_ok=True)
                if (o.savepath / f"{o.name}.{o.ext}").exists() and not force:
                    continue
                self.log.debug(f"Plotting {o.name}")

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
                            else np.concatenate((legacy[k], legacy_data[k]), axis=axis)
                        )
                        for (k, axis) in legacy_keys
                    }

                if len(legacy) == 0:
                    legacy = None

                stats = DispersionPlotter.plot_dispersion(
                    data,
                    list(self.results[subset].values()),
                    o,
                    twins=twins,
                    legacy=legacy,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                    force=force,
                )
                if stats is not None:
                    statistics = {
                        "no_mask": stats[0],
                        "combined_mask": stats[-1],
                    }
                    if stats[1] is not None:
                        statistics["sn_mask"] = stats[1]
                    if stats[2] is not None:
                        statistics["twins_mask"] = stats[2]
                    if stats[3] is not None:
                        statistics["salt_mask"] = stats[3]
                    with (o.savepath / "stats.json").open("w") as io:
                        json.dump(statistics, io)

    @override
    def _analyse(self, *args: Any, **kwargs: Any) -> None:
        for subset in self.subsets:
            for seed in self.seeds:
                model = self.models[subset][str(seed)]
                results = self.results[subset][str(seed)]

                data = model.data
                input_mask = model.data_mask
                input_sn_mask = model.sn_mask
                input_spec_mask = model.spec_mask
                input_wl_mask = model.wl_mask

                map_init_results = []
                map_best_results = []
                labels = {}
                ind = 0
                if model.map.train_delta_m:
                    labels[ind] = "Δℳ"
                    map_init_results.append(results.map.init_delta_m)
                    map_best_results.append(results.map.best_delta_m)
                    ind += 1
                if model.map.train_delta_p:
                    labels[ind] = "Δp"
                    map_init_results.append(results.map.init_delta_p)
                    map_best_results.append(results.map.best_delta_p)
                    ind += 1
                if self.nflow.physical_latents:
                    labels[ind] = "μΔAᵥ"
                    map_init_results.append(results.map.init_u_delta_av)
                    map_best_results.append(results.map.best_u_delta_av)
                    ind += 1
                for i in range(model.map.n_u_latents):
                    labels[ind + i] = f"μ{i + 1}"
                map_init_results.append(results.map.init_u_latents)
                map_best_results.append(results.map.best_u_latents)
                map_init_results = np.concatenate(map_init_results, axis=-1)
                map_best_results = np.concatenate(map_best_results, axis=-1)

                self._plot_comparison(
                    subset,
                    seed,
                    results,
                    model,
                    data,
                    input_mask,
                    input_sn_mask,
                    input_spec_mask,
                    input_wl_mask,
                )

                self._plot_comparison_spectra(
                    subset,
                    seed,
                    results,
                    model,
                    data,
                    input_mask,
                    input_sn_mask,
                    input_spec_mask,
                    input_wl_mask,
                )

                self._plot_comparison_array(
                    subset,
                    seed,
                    model,
                    results,
                    data,
                    input_mask,
                    input_sn_mask,
                    input_spec_mask,
                    input_wl_mask,
                )

                self._plot_map(
                    subset,
                    seed,
                    data,
                    input_mask,
                    input_sn_mask,
                    input_spec_mask,
                    input_wl_mask,
                    map_init_results,
                    labels,
                    is_init=True,
                )

                self._plot_map(
                    subset,
                    seed,
                    data,
                    input_mask,
                    input_sn_mask,
                    input_spec_mask,
                    input_wl_mask,
                    map_best_results,
                    labels,
                    is_init=False,
                )

                self._plot_hmc(
                    subset,
                    seed,
                    data,
                    results,
                    input_mask,
                    input_sn_mask,
                    input_spec_mask,
                    input_wl_mask,
                    labels,
                )

                self._plot_dispersion(
                    subset,
                    seed,
                    data,
                    input_mask,
                    input_sn_mask,
                    input_spec_mask,
                    input_wl_mask,
                )

    @override
    def _is_cleaned(self, *args: Any, **kwargs: Any) -> bool:
        base_path = self.paths.results / self.model_name
        profile_path = base_path / "latest_logs"
        if profile_path.exists():
            self.log.debug(f"{self.name} is not cleaned as {profile_path} exists")
            return False
        for subset in self.subsets:
            for seed in self.seeds:
                subset_path = base_path / subset / str(seed)
                map_path = subset_path / "map"
                if map_path.exists():
                    self.log.debug(f"{self.name} is not cleaned as {map_path} exists")
                    return False
                hmc_path = subset_path / "hmc"
                if hmc_path.exists():
                    self.log.debug(f"{self.name} is not cleaned as {hmc_path} exists")
                    return False
        return True

    @override
    def _clean(self, *args: Any, **kwargs: Any) -> None:
        base_path = self.paths.results / self.model_name
        profile_path = base_path / "latest_logs"
        if profile_path.exists():
            self.log.warning(f"Removing {profile_path}!")
            shutil.rmtree(profile_path)
        for subset in self.subsets:
            for seed in self.seeds:
                subset_path = base_path / subset / str(seed)
                map_path = subset_path / "map"
                if map_path.exists():
                    self.log.warning(f"Removing {map_path}!")
                    shutil.rmtree(map_path)
                hmc_path = subset_path / "hmc"
                if hmc_path.exists():
                    self.log.warning(f"Removing {hmc_path}!")
                    shutil.rmtree(hmc_path)

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
            self.analysis = self.options.analysis or PosteriorStepAnalysis()

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

    def setup_data_masks(self) -> None:
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


class PosteriorStep(Model[PosteriorStepConfig, Posterior]):
    id: "ClassVar[str]" = "posterior"
    model_backend: "ClassVar[dict[str, Callable[[], type[PosteriorModel]]]]" = {
        "TensorFlow": lambda: (
            importlib.import_module(".tf", __package__).TFPosteriorModel
        ),
    }

    def __init__(self, config: "PosteriorStepConfig") -> None:
        super().__init__(config)

        self.plots: dict[str, dict[str, Any]] = {}

    @override
    def _model(
        self,
        *args: Any,
        force: bool = False,
        variants: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for name in variants:
            variant = self.variants[name]
            if force or not hasattr(variant, "model"):
                model = self.models_cls[name](variant, variant.subsets[0], variant.seed)
            else:
                model = variant.model
            variant.model = model

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

            super()._analyse(*args, **{**kwargs, "variants": [variant_name]})

            for subset in variant.subsets:
                for seed in variant.seeds:
                    model = variant.models[subset][str(seed)]
                    results = variant.results[subset][str(seed)]

                    self.r_hat[f"{subset}_{seed}_{model.name}"] = {}
                    self.r_hat[f"{subset}_{seed}_{model.name}"]["min"] = np.min(
                        results.hmc.r_hat, axis=0
                    ).tolist()
                    self.r_hat[f"{subset}_{seed}_{model.name}"]["median"] = np.median(
                        results.hmc.r_hat, axis=0
                    ).tolist()
                    self.r_hat[f"{subset}_{seed}_{model.name}"]["max"] = np.max(
                        results.hmc.r_hat, axis=0
                    ).tolist()

                    data = model.data
                    input_mask = model.data_mask
                    input_sn_mask = model.sn_mask
                    input_spec_mask = model.spec_mask
                    input_wl_mask = model.wl_mask

                    map_init_results = []
                    map_best_results = []
                    variant_labels = {}
                    ind = 0
                    if model.map.train_delta_m:
                        variant_labels[ind] = "Δℳ"
                        map_init_results.append(results.map.init_delta_m)
                        map_best_results.append(results.map.best_delta_m)
                        ind += 1
                    if model.map.train_delta_p:
                        variant_labels[ind] = "Δp"
                        map_init_results.append(results.map.init_delta_p)
                        map_best_results.append(results.map.best_delta_p)
                        ind += 1
                    if variant.nflow.physical_latents:
                        variant_labels[ind] = "μΔAᵥ"
                        map_init_results.append(results.map.init_u_delta_av)
                        map_best_results.append(results.map.best_u_delta_av)
                        ind += 1
                    for i in range(model.map.n_u_latents):
                        variant_labels[ind + i] = f"μ{i + 1}"
                    map_init_results.append(results.map.init_u_latents)
                    map_best_results.append(results.map.best_u_latents)
                    map_init_results = np.concatenate(map_init_results, axis=-1)
                    map_best_results = np.concatenate(map_best_results, axis=-1)

                    plot_chain_data, plot_labels, plot_weights, plot_opts = (
                        self._plot_hmc(
                            variant,
                            subset,
                            seed,
                            data,
                            results,
                            input_mask,
                            input_sn_mask,
                            input_spec_mask,
                            input_wl_mask,
                            variant_labels,
                        )
                    )
                    for n, v in plot_labels.items():
                        if n not in self.plot_chain_data:
                            self.plot_chain_data[n] = {}
                        if n not in self.plot_labels:
                            self.plot_labels[n] = {}
                        if n not in self.plot_weights and n in plot_weights:
                            self.plot_weights[n] = {}
                        for t in v:
                            self.plot_labels[n][t] = v[t]
                            self.plot_chain_data[n][t] = plot_chain_data[n][t]
                            if n in plot_weights:
                                self.plot_weights[n][t] = plot_weights[n][t]
                            self.plot_opts[n] = plot_opts[n]

    @override
    @callback
    def analyse(self, *args: "Any", **kwargs: "Any") -> None:
        self.r_hat = {}
        self.plots = {}
        self.plot_labels = {}
        self.plot_chain_data = {}
        self.plot_weights = {}
        self.plot_opts = {}
        super().analyse(*args, **kwargs)
        r_hat_path = self.paths.plots / "r_hat.json"
        if not r_hat_path.exists() or self.force:
            with r_hat_path.open("w") as io:
                json.dump(self.r_hat, io)
        if len(self.variants) > 1:
            for name in self.plot_labels:
                o = self.plot_opts[name].model_copy(deep=True)
                o.labels = self.plot_labels[name]
                self.plots[name] = self.plots.get(name, {"fig": None, "ax": None})
                fig = self.plots[name]["fig"]
                ax = self.plots[name]["ax"]
                fig, ax = DistributionPlotter.plot_corner(
                    self.plot_chain_data[name],
                    o,
                    statistics="cumulative" if o.reduce == "median" else o.reduce,
                    log_posterior=self.plot_weights.get(name),
                    plot_point=self.plot_weights.get(name) is not None,
                    fig=fig,
                    ax=ax,
                    marker_style="P",
                    marker_size=100,
                    save=False,
                    force=True,
                )
                self.plots[name]["fig"] = fig
                self.plots[name]["ax"] = ax
            for name, opts in self.plots.items():
                savepath = self.paths.plots / name
                savepath.parent.mkdir(parents=True, exist_ok=True)
                if savepath.exists():
                    continue
                fig = opts["fig"]
                ax = opts["ax"]
                if fig is None:
                    continue
                self.log.debug(f"plotting {name}")
                fig = Plotter.save(fig, savepath)
                Plotter.close(fig, ax)

    def _plot_hmc(
        self,
        variant: Posterior,
        subset: str,
        seed: int,
        data: "LazySNPAEData",
        results: PosteriorStepResult,
        input_mask: "npt.NDArray[bool]",
        input_sn_mask: "npt.NDArray[bool]",
        input_spec_mask: "npt.NDArray[bool]",
        input_wl_mask: "npt.NDArray[bool]",
        hmc_labels,
    ) -> tuple[dict, dict, dict, dict]:
        v_chain_data = {}
        v_labels = {}
        v_weights = {}
        v_opts = {}
        if variant.analysis.plot_hmc is not None:
            for opts in variant.analysis.plot_hmc:
                o = opts.model_copy(deep=True)
                if o.name is None:
                    o.name = "hmc"

                if o.masked:
                    o.name += "_masked"
                if o.mean:
                    o.name += "_mean"

                name = f"{variant.seeds[0]}/{subset}/{seed}/{o.name}.{o.ext}"

                self.log.debug(f"Plotting {o.name}")

                (
                    _wl,
                    _amplitude,
                    _sigma,
                    _sn_name,
                    _time,
                    mask,
                    _sn_mask,
                    _spec_mask,
                    _wl_mask,
                ) = SpectraPlotter.prep(
                    data,
                    o,
                )
                # Determine which spectra to keep
                # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
                mask_spec = np.any(mask, axis=-1)

                # Determine which SNe to keep
                # Will mask out any SN with *no* unmasked spectra
                sn_mask = np.any(mask_spec, axis=-1)

                if not np.any(sn_mask):
                    continue

                samples = results.hmc.samples
                log_prob = results.hmc.log_prob

                if o.masked:
                    (
                        _wl,
                        _amplitude,
                        _sigma,
                        _sn_name,
                        _time,
                        mask,
                        _sn_mask,
                        _spec_mask,
                        _wl_mask,
                    ) = SpectraPlotter.prep(
                        data,
                        o,
                        mask=input_mask,
                        sn_mask=input_sn_mask,
                        spec_mask=input_spec_mask,
                        wl_mask=input_wl_mask,
                    )

                    # Determine which spectra to keep
                    # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
                    mask_spec = np.any(mask, axis=-1)

                    # Determine which SNe to keep
                    # Will mask out any SN with *no* unmasked spectra
                    mask_sn = np.any(mask_spec, axis=-1)
                    sn_mask &= mask_sn

                samples = samples[..., sn_mask, :]
                log_prob = log_prob[..., sn_mask]

                weights = None
                if o.mean:
                    if o.reduce == "mean":
                        chains = samples.mean(axis=0)
                    elif o.reduce == "median":
                        chains = np.median(samples, axis=0)
                    elif o.reduce == "max_central":
                        chains = np.array([
                            np.array([
                                max_central(
                                    samples[..., sn, pos], weight=log_prob[:, sn]
                                )[1]
                                for pos in range(samples.shape[-1])
                            ])
                            for sn in range(samples.shape[-2])
                        ])
                elif samples.shape[-2] == 1:
                    chains = samples[..., 0, :]
                    weights = log_prob[..., 0]
                else:
                    chains = np.reshape(samples, (-1, samples.shape[-1]))
                o.mean = False

                if o.plot_kwargs is None:
                    o.plot_kwargs = {"title": f"{subset}_{self.name}_{o.name}"}
                if o.labels is None:
                    o.labels = {}
                title = variant.name
                if name not in v_chain_data:
                    v_chain_data[name] = {}
                if name not in v_labels:
                    v_labels[name] = {}
                if name not in v_weights and weights is not None:
                    v_weights[name] = {}

                v_chain_data[name][title] = chains
                v_labels[name][title] = hmc_labels
                if weights is not None:
                    v_weights[name][title] = weights
                v_opts[name] = o
        return v_chain_data, v_labels, v_weights, v_opts


PosteriorStep.register_step(Posterior)
