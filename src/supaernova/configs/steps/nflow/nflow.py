# Copyright 2025 Patrick Armstrong
from typing import ClassVar, Annotated
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt
from pydantic import Field, PositiveInt, PositiveFloat

from supaernova.configs.steps import StepResult, StepAnalysis
from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.models import ModelConfig, BackendConfig
from supaernova.analysis.distribution import DistributionPlot


class NFlowStepResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    ind: "npt.NDArray[np.int32]"
    sn_name: "npt.NDArray[np.str_]"
    spectra_id: "npt.NDArray[np.str_]"
    latents: "npt.NDArray[np.float32]"
    log_prob: "npt.NDArray[np.float32]"
    z_to_u: "npt.NDArray[np.float32]"
    u_to_z: "npt.NDArray[np.float32]"
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class NFlowStepAnalysis(StepAnalysis):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    plot_z_latents: DistributionPlot | list[DistributionPlot] | None = None
    plot_u_latents: DistributionPlot | list[DistributionPlot] | None = None
    plot_latents: DistributionPlot | list[DistributionPlot] | None = None
    plot_latent_steps: DistributionPlot | list[DistributionPlot] | None = None
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class NFlowConfig(BackendConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    physical_latents: bool
    validation_frac: Annotated[float, Field(ge=0, le=1)]
    n_batches: PositiveInt
    # --- Optional ---
    analysis: NFlowStepAnalysis | None = None
    pae: str | int = 0
    debug: bool = False
    profile: bool = False
    save_best: bool = False
    patience: PositiveFloat = 0.02

    lr: PositiveFloat = 0.0001
    lr_decay_steps: PositiveFloat = 300
    lr_decay_rate: PositiveFloat = 0.95
    lr_weight_decay_rate: PositiveFloat = 0.0001

    epochs: PositiveInt = 5000
    batch_normalisation: bool = False

    n_hidden_units: PositiveInt = 12
    n_layers: PositiveInt = 18


class NFlowStepConfig(ModelConfig):
    # === Class Variables ===
    id: ClassVar[str] = "nflow"
    required_steps: ClassVar[list[str]] = [PAEStepConfig.id]
    model_backend: ClassVar[dict[str, Callable[[], type[NFlowConfig]]]] = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFNFlowConfig
    }
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    base: NFlowConfig
    # --- Optional ---
    variants: list[NFlowConfig] | None = Field(None, validation_alias="variant")
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


NFlowStepConfig.register_step(NFlowConfig)
