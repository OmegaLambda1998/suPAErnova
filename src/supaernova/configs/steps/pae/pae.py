# Copyright 2025 Patrick Armstrong
from typing import Any, Self, Literal, ClassVar, Annotated
from pathlib import Path
import importlib
import itertools
from collections.abc import Callable

import numpy as np
from numpy import typing as npt
from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    AfterValidator,
    NonNegativeFloat,
    field_validator,
    model_validator,
)

from supaernova.utils.lazy import LazyObj, LazyLambda
from supaernova.configs.base import BaseConfig
from supaernova.analysis.spectra import ComparisonPlot
from supaernova.configs.steps.data import LazySNPAEData, DataStepConfig
from supaernova.configs.steps.steps import StepResult, StepAnalysis
from supaernova.configs.steps.models import ModelConfig, BackendConfig
from supaernova.analysis.distribution import DistributionPlot


def clear(value: LazyObj) -> LazyObj:
    value.clear()
    return value


class PAEStage(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    stage: PositiveInt
    prev_stage: PositiveInt | None
    name: str
    fname: str
    epochs: PositiveInt
    patience: PositiveFloat | PositiveInt
    debug: bool
    profile: bool
    learning_rate: PositiveFloat
    learning_rate_decay_steps: PositiveInt
    learning_rate_decay_rate: PositiveFloat
    learning_rate_weight_decay_rate: PositiveFloat

    data: Annotated[LazySNPAEData, AfterValidator(clear)]
    mask: npt.NDArray[bool]
    sn_mask: npt.NDArray[bool]
    spec_mask: npt.NDArray[bool]
    wl_mask: npt.NDArray[bool]

    train_data: Annotated[LazySNPAEData, AfterValidator(clear)]
    train_mask: npt.NDArray[bool]
    train_sn_mask: npt.NDArray[bool]
    train_spec_mask: npt.NDArray[bool]
    train_wl_mask: npt.NDArray[bool]

    test_data: Annotated[LazySNPAEData, AfterValidator(clear)]
    test_mask: npt.NDArray[bool]
    test_sn_mask: npt.NDArray[bool]
    test_spec_mask: npt.NDArray[bool]
    test_wl_mask: npt.NDArray[bool]

    val_data: Annotated[LazySNPAEData, AfterValidator(clear)]
    val_mask: npt.NDArray[bool]
    val_sn_mask: npt.NDArray[bool]
    val_spec_mask: npt.NDArray[bool]
    val_wl_mask: npt.NDArray[bool]

    # --- Optional ---
    savepath: Path | None = None
    loadpath: Path | None = None
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PAEStageResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    stage: int
    ind: npt.NDArray[int]
    sn_name: npt.NDArray[str]
    spectra_id: npt.NDArray[str]
    input_amp: npt.NDArray[float]
    input_d_amp: npt.NDArray[float]
    input_phase: npt.NDArray[float]
    input_mask: npt.NDArray[float]
    input_sn_mask: npt.NDArray[float]
    input_spec_mask: npt.NDArray[float]
    input_wl_mask: npt.NDArray[float]
    input_colourlaw: npt.NDArray[float] | None
    latents: npt.NDArray[float]
    output_amp: npt.NDArray[float]
    diff_amp: npt.NDArray[float]
    loss: float
    pred_loss: float
    model_loss: float
    resid_loss: float
    delta_loss: float
    cov_loss: float
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class LazyPAEStageResult(LazyLambda[PAEStageResult, PAEStageResult]):
    pass


class PAEStepResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    model: Any
    stages: dict[
        str,
        dict[str, Annotated[LazyPAEStageResult, AfterValidator(clear)]],
    ]
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


class PAEStepAnalysis(StepAnalysis):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    plot_comparison: ComparisonPlot | list[ComparisonPlot] | None = None
    plot_latents: DistributionPlot | list[DistributionPlot] | None = None
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PAEConfig(BackendConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Previous Stages ---
    data: str | int = 0
    # --- Required ---
    physical_latents: bool
    seperate_latent_training: bool
    seperate_z_latent_training: bool
    validation_frac: Annotated[float, Field(ge=0, le=1)]
    n_batches: PositiveInt

    # --- Optional ---
    analysis: PAEStepAnalysis | None = None
    debug: bool = False
    profile: bool = False
    kfold: int = 0
    architecture: Literal["dense", "convolutional"] = "dense"
    encode_dims: tuple[PositiveInt, ...] = (256, 128)
    decode_dims: tuple[PositiveInt, ...] = ()
    n_z_latents: PositiveInt = 3
    batch_normalisation: bool = False
    dropout: Annotated[float, Field(ge=0, le=1)] = 0
    save_best: bool = False

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

    phase_offset_scale: float = -0.02
    amplitude_offset_scale: NonNegativeFloat = 1.0
    mask_fraction: Annotated[float, Field(ge=0, le=1)] = 0.1
    loss_residual_penalty: NonNegativeFloat = 0
    loss_delta_av_penalty: NonNegativeFloat = 0
    loss_delta_m_penalty: NonNegativeFloat = 0
    loss_delta_p_penalty: NonNegativeFloat = 0
    loss_covariance_penalty: NonNegativeFloat = 50000
    loss_decorrelate_all: bool = True
    loss_decorrelate_dust: bool = True
    loss_clip_delta: PositiveFloat = 25

    delta_av_epochs: PositiveInt = 1000
    delta_av_patience: PositiveFloat | PositiveInt = 0.5  # Run for 100%
    delta_av_lr: PositiveFloat = 0.005
    delta_av_lr_decay_steps: PositiveInt | PositiveFloat = 300
    delta_av_lr_decay_rate: PositiveFloat = 0.95
    delta_av_lr_weight_decay_rate: PositiveFloat = 0.0001

    zs_epochs: PositiveInt = 1000
    zs_patience: PositiveFloat | PositiveInt = 0.5  # Run for 100%
    zs_lr: PositiveFloat = 0.005
    zs_lr_decay_steps: PositiveInt | PositiveFloat = 300
    zs_lr_decay_rate: PositiveFloat = 0.95
    zs_lr_weight_decay_rate: PositiveFloat = 0.0001

    delta_m_epochs: PositiveInt = 5000
    delta_m_patience: PositiveFloat | PositiveInt = 0.5  # Run for 100%
    delta_m_lr: PositiveFloat = 0.005
    delta_m_lr_decay_steps: PositiveInt | PositiveFloat = 300
    delta_m_lr_decay_rate: PositiveFloat = 0.95
    delta_m_lr_weight_decay_rate: PositiveFloat = 0.0001

    delta_p_epochs: PositiveInt = 5000
    delta_p_patience: PositiveFloat | PositiveInt = 0.5  # Run for 100%
    delta_p_lr: PositiveFloat = 0.001
    delta_p_lr_decay_steps: PositiveInt | PositiveFloat = 300
    delta_p_lr_decay_rate: PositiveFloat = 0.95
    delta_p_lr_weight_decay_rate: PositiveFloat = 0.0001

    final_epochs: PositiveFloat = 5000
    final_patience: PositiveFloat | PositiveInt = 0.5  # Run for 100%
    final_lr: PositiveFloat = 0.001
    final_lr_decay_steps: PositiveInt | PositiveFloat = 300
    final_lr_decay_rate: PositiveFloat = 0.95
    final_lr_weight_decay_rate: PositiveFloat = 0.0001

    # === Model Validators ===
    # --- Before ---
    @classmethod
    @model_validator(mode="before")
    def _validate_bounds(cls: type[Self], data: Any) -> Any:
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
    @model_validator(mode="after")
    def _validate_decode_dims(self: Self) -> Self:
        if len(self.decode_dims) == 0:
            self.decode_dims = tuple(reversed(self.encode_dims))
        if not all(x < y for x, y in itertools.pairwise(self.decode_dims)):
            err = f"`decode_dims`: {self.decode_dims} is not monotonically decreasing"
            self._raise(err)
        return self

    @model_validator(mode="after")
    def _validate_n_latents(self: Self) -> Self:
        if not self.physical_latents and self.n_z_latents == 0:
            err = "You must specify either non-zero `n_z_latents`, or `physical_latents=True`. With both `physical_latents=False` and `n_z_latents=0, there will be no latents to train at all!"
            self._raise(err)
        return self

    # === Field Validators ===
    # --- Before ---
    @field_validator("encode_dims", mode="before")
    @classmethod
    def _validate_encode_dims(
        cls: type[Self], value: tuple[PositiveInt, ...]
    ) -> tuple[PositiveInt, ...]:
        if len(value) == 0:
            err = "`encode_dims` can not be empty"
            cls._raise(err)
        if not all(x > y for x, y in itertools.pairwise(value)):
            err = f"`encode_dims`: {value} is not monotonically decreasing"
            cls._raise(err)
        return value

    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class PAEStepConfig(ModelConfig):
    # === Class Variables ===
    id: ClassVar[str] = "pae"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id]
    model_backend: ClassVar[dict[str, Callable[[], type[PAEConfig]]]] = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFPAEConfig
    }
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    base: PAEConfig
    # --- Optional ---
    variants: list[PAEConfig] | None = Field(None, validation_alias="variant")
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


PAEStepConfig.register_step(PAEConfig)
