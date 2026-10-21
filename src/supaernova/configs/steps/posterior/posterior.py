# Copyright 2025 Patrick Armstrong

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from numpy import typing as npt
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from supaernova.analysis.dispersion import DispersionPlot
from supaernova.analysis.distribution import DistributionPlot
from supaernova.analysis.spectra import ComparisonArrayPlot, ComparisonPlot, SpectraPlot
from supaernova.configs.base import BaseConfig
from supaernova.configs.steps import StepAnalysis, StepResult
from supaernova.configs.steps.data import DataStepConfig
from supaernova.configs.steps.models import BackendConfig, ModelConfig
from supaernova.configs.steps.nflow import NFlowStepConfig
from supaernova.configs.steps.pae import PAEStepConfig

type InitPreset = Literal["initial", "current", "best", "legacy"]
type InitULatents = Literal["u_random", "u_constant"]
type InitZLatents = Literal["z_random", "z_data", "z_constant"]
type InitLatents = InitPreset | InitULatents | InitZLatents
type InitParams = InitPreset | Literal["random", "data", "constant", "scale"]


class PosteriorMAPStage(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    stage: NonNegativeInt
    name: str
    fname: str
    savepath: Path | None = None
    loadpath: Path | None = None
    n_chains: int
    init: bool = False
    setup: bool = False
    init_u_delta_av: InitParams
    init_latents: InitLatents
    init_delta_av: InitParams
    init_delta_m: InitParams
    init_delta_p: InitParams
    init_bias: InitParams

    # === Model Validators ===
    # --- Before ---
    @model_validator(mode="before")
    @classmethod
    def init_values(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("init", False):
            init_all = "initial"
            data["init_u_delta_av"] = init_all
            data["init_latents"] = init_all
            data["init_delta_av"] = init_all
            data["init_delta_m"] = init_all
            data["init_delta_p"] = init_all
            data["init_bias"] = init_all
        return data

    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PosteriorStepMAPResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    chain_min: npt.NDArray[np.int64]
    converged: npt.NDArray[np.bool_]
    num_evaluations: int
    negative_log_prob: npt.NDArray[np.float64]
    negative_log_like: npt.NDArray[np.float64]
    negative_log_prior: npt.NDArray[np.float64]
    init_u_delta_av: npt.NDArray[np.float64]
    init_u_latents: npt.NDArray[np.float64]
    init_delta_av: npt.NDArray[np.float64]
    init_delta_m: npt.NDArray[np.float64]
    init_delta_p: npt.NDArray[np.float64]
    init_z_latents: npt.NDArray[np.float64]
    best_u_delta_av: npt.NDArray[np.float64]
    best_u_latents: npt.NDArray[np.float64]
    best_delta_av: npt.NDArray[np.float64]
    best_delta_m: npt.NDArray[np.float64]
    best_delta_p: npt.NDArray[np.float64]
    best_z_latents: npt.NDArray[np.float64]
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PosteriorStepHMCResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    samples: npt.NDArray[np.float64]
    r_hat: npt.NDArray[np.float64]
    u_delta_av: npt.NDArray[np.float64]
    u_latents: npt.NDArray[np.float64]
    delta_av: npt.NDArray[np.float64]
    z_latents: npt.NDArray[np.float64]
    delta_m: npt.NDArray[np.float64]
    delta_p: npt.NDArray[np.float64]
    log_prior: npt.NDArray[np.float64]
    log_like: npt.NDArray[np.float64]
    log_prob: npt.NDArray[np.float64]
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PosteriorStepResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    ind: npt.NDArray[np.int64]
    sn_name: npt.NDArray[np.str_]
    spectra_id: npt.NDArray[np.str_]
    map: PosteriorStepMAPResult
    hmc: PosteriorStepHMCResult
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PosteriorStepAnalysis(StepAnalysis):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    plot_map_init: DistributionPlot | list[DistributionPlot] | None = None
    plot_map_best: DistributionPlot | list[DistributionPlot] | None = None
    plot_hmc: DistributionPlot | list[DistributionPlot] | None = None
    plot_dispersion: DispersionPlot | list[DispersionPlot] | None = None
    plot_comparison: ComparisonPlot | list[ComparisonPlot] | None = None
    plot_comparison_spectra: SpectraPlot | list[SpectraPlot] | None = None
    plot_comparison_array: ComparisonArrayPlot | list[ComparisonArrayPlot] | None = None
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PosteriorConfig(BackendConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Previous Stages ---
    data: str | int = 0
    pae: str | int = 0
    nflow: str | int = 0
    # --- Required ---
    iterations: int
    train_delta_m: bool
    train_delta_p: bool
    train_bias: bool
    validation_frac: Annotated[float, Field(ge=0, le=1)]
    # - MAP -
    n_random_chains: int
    n_delta_m_chains: int
    n_delta_av_chains: int
    # - HMC -
    n_leapfrog_adaption: PositiveInt
    n_leapfrog_burnin: PositiveInt
    n_leapfrog_run: PositiveInt
    n_run_steps: PositiveInt
    n_burnin_steps: NonNegativeInt | NonNegativeFloat
    n_adaption_steps: PositiveInt | PositiveFloat
    n_walkers: PositiveInt | PositiveFloat = 1.0
    n_thinning: NonNegativeInt = 0
    # HMC sampling is driven as a Python loop over chunks so TensorBoard
    # writes, the progress bar and (optional) checkpoints happen between
    # chunks in eager Python rather than inside the traced sample_chain loop.
    # 0 -> one chunk per phase (matches a single sample_chain call);
    # a float in (0, 1] -> that fraction of the phase's step count.
    n_chunk_steps: NonNegativeInt | NonNegativeFloat = 0.1
    n_adaption_chunk_steps: NonNegativeInt | NonNegativeFloat | None = None
    n_burnin_chunk_steps: NonNegativeInt | NonNegativeFloat | None = None
    # Write an intermediate checkpoint after every chunk so an interrupted
    # HMC run can resume without repeating completed burn-in / sampling.
    checkpoint_hmc: bool = True
    log_likelihood_scale: PositiveFloat = 1.0
    log_likelihood_spec_sum: bool = False
    log_likelihood_sum: bool = False
    step_size_scale: Literal["min", "max", "shift"] = "shift"
    fractional_error: bool = True
    weighted_error: bool = True
    measurement_error: bool = True
    reconstruction_error: Literal["train", "test", "combined", "match"] = "combined"
    bounded_u_latents: bool = False
    generalised_u_latents: float = 1
    # --- Optional ---
    debug: bool = False
    profile: bool = False
    analysis: PosteriorStepAnalysis | None = None
    kfold: int = 0
    train_subset: bool = True
    test_subset: bool = True
    subset: Literal["train", "test"] = "train"
    save_best: bool = False
    legacy: tuple[Path, ...] | None = None

    min_redshift: float | None = None
    max_redshift: float | None = None
    min_train_redshift: float | None = None
    max_train_redshift: float | None = None
    min_test_redshift: float | None = None
    max_test_redshift: float | None = None
    min_val_redshift: float | None = None
    max_val_redshift: float | None = None
    min_phase: float | None = None
    max_phase: float | None = None
    min_train_phase: float | None = None
    max_train_phase: float | None = None
    min_test_phase: float | None = None
    max_test_phase: float | None = None
    min_val_phase: float | None = None
    max_val_phase: float | None = None
    min_wavelength: float | None = None
    max_wavelength: float | None = None
    min_train_wavelength: float | None = None
    max_train_wavelength: float | None = None
    min_test_wavelength: float | None = None
    max_test_wavelength: float | None = None
    min_val_wavelength: float | None = None
    max_val_wavelength: float | None = None

    # - MAP -
    random_initial_positions: bool = False
    tolerance: PositiveFloat = 5e-3
    x_tolerance: NonNegativeFloat = 5e-3
    f_relative_tolerance: NonNegativeFloat = 0
    f_absolute_tolerance: NonNegativeFloat = 0
    max_iterations: PositiveInt = 2500
    max_line_search_iterations: PositiveInt | None = None
    num_correction_pairs: PositiveInt | None = None
    # Scale applied to the negative log-probability minimised by L-BFGS.
    # `tolerance` is compared against the gradient of this scaled loss.
    # If None, use the nearest power of 10 to 1 / median(|log_prob|) at the initial positions.
    # 0.01 reproduces the legacy behaviour.
    map_loss_scale: PositiveFloat | None = None
    # See [Betancourt et al (2014)](https://arxiv.org/abs/1411.6669).
    # Since we run MAP first, our initial guess is very good and we want small, precise steps.
    # So use a large acceptance rate.
    target_acceptance_rate: PositiveFloat = 0.85

    # u_delta_av_min: float | None = -10
    # u_delta_av_max: float | None = 10
    u_delta_av_min: float | None = -np.inf
    u_delta_av_max: float | None = np.inf
    u_delta_av_start: float = -0.5
    u_delta_av_end: float = 0.5
    u_delta_av_mean: float = 0
    u_delta_av_std: float = 1
    u_delta_av_prior: bool = True

    # u_latents_min: float | None = -10
    # u_latents_max: float | None = 10
    u_latents_min: float | None = -np.inf
    u_latents_max: float | None = np.inf
    u_latents_mean: float = 0
    u_latents_std: float = 1
    u_latents_prior: bool = True

    delta_av_start: float = -0.3
    delta_av_end: float = 0.3
    delta_av_mean: float = 0
    delta_av_std: float = 1

    # delta_m_min: float | None = -10
    # delta_m_max: float | None = 10
    delta_m_min: float | None = -np.inf
    delta_m_max: float | None = np.inf
    delta_m_start: float = -0.1
    delta_m_end: float = 0.1
    delta_m_mean: float = 0
    delta_m_std: float = 1
    delta_m_prior: bool = False

    # delta_p_min: float | None = -10
    # delta_p_max: float | None = 10
    delta_p_min: float | None = -np.inf
    delta_p_max: float | None = np.inf
    delta_p_start: float = -0.5
    delta_p_end: float = 0.5
    delta_p_mean: float = 0
    delta_p_std: float = 1
    delta_p_prior: bool = False

    # bias_min: float | None = -10
    # bias_max: float | None = 10
    bias_min: float | None = -np.inf
    bias_max: float | None = np.inf
    bias_start: float = -0.5
    bias_end: float = 0.5
    bias_mean: float = 0
    bias_std: float = 1
    bias_prior: bool = False

    # - HMC -

    # === Model Validators ===
    # --- Before ---
    @model_validator(mode="before")
    @classmethod
    def _validate_bounds(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for var in ["redshift", "phase", "wavelength"]:
                min_bound = data.get(f"min_{var}")
                if min_bound == "inf":
                    min_bound = np.inf
                elif min_bound == "-inf":
                    min_bound = -np.inf
                max_bound = data.get(f"max_{var}")
                if max_bound == "inf":
                    max_bound = np.inf
                elif max_bound == "-inf":
                    max_bound = -np.inf
                if (
                    (min_bound is not None)
                    and (max_bound is not None)
                    and (max_bound <= min_bound)
                ):
                    err = f"`max_{var}`: {max_bound} is not strictly greater than `min_{var}`: {min_bound}"
                    cls._raise(err)
                data[f"min_{var}"] = min_bound
                data[f"max_{var}"] = max_bound
        return data

    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PosteriorStepConfig(ModelConfig):
    # === Class Variables ===
    id: ClassVar[str] = "posterior"
    required_steps: ClassVar[list[str]] = [
        DataStepConfig.id,
        PAEStepConfig.id,
        NFlowStepConfig.id,
    ]
    model_backend: ClassVar[dict[str, Callable[[], type[PosteriorConfig]]]] = {
        "TensorFlow": lambda: (
            importlib.import_module(".tf", __package__).TFPosteriorConfig
        )
    }
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    base: PosteriorConfig
    # --- Optional ---
    variants: list[PosteriorConfig] | None = Field(None, validation_alias="variant")
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


PosteriorStepConfig.register_step(PosteriorConfig)
