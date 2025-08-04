# Copyright 2025 Patrick Armstrong

from typing import Any, Literal, ClassVar
from pathlib import Path
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt
from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
    NonNegativeFloat,
    model_validator,
)

from supaernova.configs.base import BaseConfig
from supaernova.configs.steps import StepResult, StepAnalysis
from supaernova.analysis.spectra import SpectraPlot
from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.data import DataStepConfig
from supaernova.analysis.dispersion import DispersionPlot
from supaernova.configs.steps.nflow import NFlowStepConfig
from supaernova.configs.steps.models import ModelConfig, BackendConfig
from supaernova.analysis.distribution import DistributionPlot

type InitPreset = Literal["initial", "current", "best"]
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
    chain_min: npt.NDArray[np.int32]
    converged: npt.NDArray[bool]
    num_evaluations: int
    negative_log_prob: npt.NDArray[np.float32]
    init_u_delta_av: npt.NDArray[np.float32]
    init_u_latents: npt.NDArray[np.float32]
    init_delta_av: npt.NDArray[np.float32]
    init_delta_m: npt.NDArray[np.float32]
    init_delta_p: npt.NDArray[np.float32]
    init_z_latents: npt.NDArray[np.float32]
    best_u_delta_av: npt.NDArray[np.float32]
    best_u_latents: npt.NDArray[np.float32]
    best_delta_av: npt.NDArray[np.float32]
    best_delta_m: npt.NDArray[np.float32]
    best_delta_p: npt.NDArray[np.float32]
    best_z_latents: npt.NDArray[np.float32]
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
    samples: npt.NDArray[np.float32]
    step_sizes_final: npt.NDArray[np.float32]
    is_accepted: npt.NDArray[np.float32]
    u_delta_av: npt.NDArray[np.float32]
    u_latents: npt.NDArray[np.float32]
    delta_av: npt.NDArray[np.float32]
    z_latents: npt.NDArray[np.float32]
    delta_m: npt.NDArray[np.float32]
    delta_p: npt.NDArray[np.float32]
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
    ind: npt.NDArray[np.int32]
    sn_name: npt.NDArray[str]
    spectra_id: npt.NDArray[str]
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
    plot_spectra: SpectraPlot | list[SpectraPlot] | None = None
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
    # --- Required ---
    iterations: int
    n_chains_early: int
    n_chains_mid: int
    n_chains_final: int
    n_burnin: PositiveInt
    n_samples: PositiveInt
    train_delta_m: bool
    train_delta_p: bool
    train_bias: bool
    # --- Optional ---
    analysis: PosteriorStepAnalysis | None = None
    data: str | int = 0
    kfold: int = 0
    pae: str | int = 0
    nflow: str | int = 0
    debug: bool = False
    profile: bool = False
    train_subset: bool = True
    test_subset: bool = True
    subset: Literal["train", "test"] = "train"
    save_best: bool = False
    n_leapfrog: PositiveInt = 2
    n_thinning: PositiveInt = 1
    tolerance: PositiveFloat = 1e-8
    x_tolerance: NonNegativeFloat = 1e-3
    f_relative_tolerance: NonNegativeFloat = 0
    f_absolute_tolerance: NonNegativeFloat = 0
    max_iterations: PositiveInt = 2500
    target_acceptance_rate: PositiveFloat = 0.651
    random_initial_positions: bool = False
    u_delta_av_min: float = -np.inf
    u_delta_av_max: float = np.inf
    u_delta_av_start: float = -1.0
    u_delta_av_end: float = 1.0
    u_delta_av_mean: float = 0.0
    u_delta_av_std: float = 1.0
    u_latents_min: float = -np.inf
    u_latents_max: float = np.inf
    u_latents_mean: float = 0.0
    u_latents_std: float = 1.0
    delta_av_min: float = -np.inf
    delta_av_max: float = np.inf
    delta_av_start: float = -0.5
    delta_av_end: float = 0.5
    delta_av_mean: float = 0.0
    delta_av_std: float = 0.5
    delta_m_min: float = -np.inf
    delta_m_max: float = np.inf
    delta_m_start: float = -1.5
    delta_m_end: float = 1.5
    delta_m_mean: float = 0.0
    delta_m_std: float = 0.1
    delta_p_min: float = -np.inf
    delta_p_max: float = np.inf
    delta_p_start: float = -1.0
    delta_p_end: float = 1.0
    delta_p_mean: float = 0.0
    delta_p_std: float = 0.01
    bias_min: float = -np.inf
    bias_max: float = np.inf
    bias_start: float = -1.0
    bias_end: float = 1.0
    bias_mean: float = 0.0
    bias_std: float = 1.0


class PosteriorStepConfig(ModelConfig):
    # === Class Variables ===
    id: ClassVar[str] = "posterior"
    required_steps: ClassVar[list[str]] = [
        DataStepConfig.id,
        PAEStepConfig.id,
        NFlowStepConfig.id,
    ]
    model_backend: ClassVar[dict[str, Callable[[], type[PosteriorConfig]]]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPosteriorConfig
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
