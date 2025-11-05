# Copyright 2025 Patrick Armstrong
from typing import Any, ClassVar, Annotated
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt
from pydantic import Field, PositiveInt, PositiveFloat, NonNegativeInt, model_validator

from supaernova.configs.steps import StepResult, StepAnalysis
from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.data import DataStepConfig
from supaernova.configs.steps.models import ModelConfig, BackendConfig
from supaernova.analysis.distribution import DistributionPlot


class NFlowModelResult(StepResult):
    ind: "npt.NDArray[int]"
    sn_name: "npt.NDArray[str]"
    spectra_id: "npt.NDArray[str]"
    z_latents: "npt.NDArray[float]"
    u_latents: "npt.NDArray[float]"
    u_to_z_latents: "npt.NDArray[float]"
    log_prob: "npt.NDArray[float]"


class NFlowStepResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    models: dict[str, NFlowModelResult]
    model: Any
    min_redshift: float
    max_redshift: float
    min_phase: float
    max_phase: float
    min_wavelength: float
    max_wavelength: float
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
    # --- Previous Stages ---
    data: str | int = 0
    pae: str | int = 0
    # --- Required ---
    physical_latents: bool
    n_layers: PositiveInt
    n_hidden_units: PositiveInt
    validation_frac: Annotated[float, Field(ge=0, le=1)]
    n_batches: PositiveInt
    # --- Optional ---
    analysis: NFlowStepAnalysis | None = None
    debug: bool = False
    profile: bool = False
    kfold: int = 0
    save_best: bool = False
    repeats: PositiveInt = 1

    epochs: PositiveInt = 10000
    patience: PositiveFloat | PositiveInt = 0.25  # Run for 50%
    lr: PositiveFloat = 0.001
    lr_decay_steps: PositiveInt | PositiveFloat = 1.0
    lr_decay_rate: PositiveFloat = 0.1

    latent_offset_scale: PositiveFloat = 0.1

    ema_steps: NonNegativeInt = 0
    ema_momentum: PositiveFloat = 0.999

    batch_normalisation: bool = False

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

    # === Model Validators ===
    # --- Before ---
    @classmethod
    @model_validator(mode="before")
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


class NFlowStepConfig(ModelConfig):
    # === Class Variables ===
    id: ClassVar[str] = "nflow"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id, PAEStepConfig.id]
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
