# Copyright 2025 Patrick Armstrong
from typing import Any, Literal, ClassVar, Annotated
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt
from pydantic import Field, PositiveInt, PositiveFloat, model_validator

from supaernova.configs.steps import StepResult, StepAnalysis
from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.data import DataStepConfig
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
    z_latents: "npt.NDArray[np.float32]"
    u_latents: "npt.NDArray[np.float32]"
    u_to_z_latents: "npt.NDArray[np.float32]"
    log_prob: "npt.NDArray[np.float32]"
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
    n_hidden_units: PositiveInt
    n_layers: PositiveInt
    # --- Optional ---
    analysis: NFlowStepAnalysis | None = None
    data: str | int = 0
    kfold: int | None = None
    pae: str | int = 0
    debug: bool = False
    profile: bool = False
    save_best: bool = False
    patience: PositiveFloat = 0.05

    lr: PositiveFloat = 0.001
    lr_decay_steps: PositiveFloat = 300
    lr_decay_rate: PositiveFloat = 0.95
    lr_weight_decay_rate: PositiveFloat = 0.0001

    epochs: PositiveInt = 500
    batch_normalisation: bool = False

    min_redshift: float | Literal["inf", "-inf"] | None = None
    max_redshift: float | Literal["inf", "-inf"] | None = None
    min_train_redshift: float | Literal["inf", "-inf"] | None = None
    max_train_redshift: float | Literal["inf", "-inf"] | None = None
    min_test_redshift: float | Literal["inf", "-inf"] | None = None
    max_test_redshift: float | Literal["inf", "-inf"] | None = None
    min_val_redshift: float | Literal["inf", "-inf"] | None = None
    max_val_redshift: float | Literal["inf", "-inf"] | None = None
    min_phase: float | Literal["inf", "-inf"] | None = None
    max_phase: float | Literal["inf", "-inf"] | None = None
    min_train_phase: float | Literal["inf", "-inf"] | None = None
    max_train_phase: float | Literal["inf", "-inf"] | None = None
    min_test_phase: float | Literal["inf", "-inf"] | None = None
    max_test_phase: float | Literal["inf", "-inf"] | None = None
    min_val_phase: float | Literal["inf", "-inf"] | None = None
    max_val_phase: float | Literal["inf", "-inf"] | None = None
    min_wavelength: float | Literal["inf", "-inf"] | None = None
    max_wavelength: float | Literal["inf", "-inf"] | None = None
    min_train_wavelength: float | Literal["inf", "-inf"] | None = None
    max_train_wavelength: float | Literal["inf", "-inf"] | None = None
    min_test_wavelength: float | Literal["inf", "-inf"] | None = None
    max_test_wavelength: float | Literal["inf", "-inf"] | None = None
    min_val_wavelength: float | Literal["inf", "-inf"] | None = None
    max_val_wavelength: float | Literal["inf", "-inf"] | None = None

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
