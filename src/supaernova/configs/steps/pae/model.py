# Copyright 2025 Patrick Armstrong
from typing import Self, Literal, ClassVar, Annotated
from pathlib import Path
import itertools

from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
    field_validator,
    model_validator,
)

from supaernova.analysis.spectra import SpectraPlot
from supaernova.configs.steps.data import DataStepConfig
from supaernova.configs.steps.steps import AbstractStepAnalysis
from supaernova.analysis.distribution import DistributionPlot
from supaernova.configs.steps.backends import AbstractModelConfig


class PAEStepAnalysis(AbstractStepAnalysis):
    plot_residual: SpectraPlot | list[SpectraPlot] | None = None
    plot_latents: DistributionPlot | list[DistributionPlot] | None = None


class PAEModelConfig(AbstractModelConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "pae_model"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id]
    analysis: PAEStepAnalysis

    # === Required ===
    debug: bool = False
    profile: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        for var in ["redshift", "phase"]:
            for data in ["train", "test", "val"]:
                min_bound = getattr(self, f"min_{data}_{var}")
                max_bound = getattr(self, f"max_{data}_{var}")
                if min_bound >= max_bound:
                    err = f"`max_{data}_{var}`: {max_bound} is not strictly greater than `min_{data}_{var}`: {min_bound}"
                    self._raise(err)
        return self

    # --- Network Design ---
    architecture: Literal["dense", "convolutional"]
    encode_dims: list[PositiveInt]
    decode_dims: list[PositiveInt] = []

    @field_validator("encode_dims", mode="before")
    @classmethod
    def validate_encode_dims(cls, value: list[int]) -> list[int]:
        if len(value) == 0:
            err = "`encode_dims` can not be empty"
            cls._raise(err)
        if not all(x > y for x, y in itertools.pairwise(value)):
            err = f"`encode_dims`: {value} is not monotonically decreasing"
            cls._raise(err)
        return value

    @model_validator(mode="after")
    def validate_decode_dims(self) -> Self:
        if len(self.decode_dims) == 0:
            self.decode_dims = list(reversed(self.encode_dims))
        if not all(x < y for x, y in itertools.pairwise(self.decode_dims)):
            err = f"`decode_dims`: {self.decode_dims} is not monotonically decreasing"
            self._raise(err)
        return self

    physical_latents: bool
    n_z_latents: PositiveInt

    @model_validator(mode="after")
    def validate_n_latents(self) -> Self:
        if not self.physical_latents and self.n_z_latents == 0:
            err = "You must specify either non-zero `n_z_latents`, or `physical_latents=True`. With both `physical_latents=False` and `n_z_latents=0, there will be no latents to train at all!"
            self._raise(err)
        return self

    # --- Training ---
    # Overfitting
    batch_normalisation: bool
    dropout: Annotated[float, Field(ge=0, le=1)]

    # Latent training
    seperate_latent_training: bool
    seperate_z_latent_training: bool

    # === Optional ===
    seed: int
    batch_size: PositiveInt

    save_best: bool
    patience: float | int

    # --- Data ---
    min_train_redshift: float
    max_train_redshift: float
    min_test_redshift: float
    max_test_redshift: float
    min_val_redshift: float
    max_val_redshift: float
    min_redshift: float
    max_redshift: float

    min_train_phase: float
    max_train_phase: float
    min_test_phase: float
    max_test_phase: float
    min_val_phase: float
    max_val_phase: float
    min_phase: float
    max_phase: float

    # --- Data Offsets ---
    phase_offset_scale: float
    amplitude_offset_scale: NonNegativeFloat
    mask_fraction: Annotated[float, Field(ge=0, le=1)]

    # --- Loss ---
    loss_residual_penalty: NonNegativeFloat

    loss_delta_av_penalty: NonNegativeFloat
    loss_delta_m_penalty: NonNegativeFloat
    loss_delta_p_penalty: NonNegativeFloat

    loss_covariance_penalty: NonNegativeFloat
    loss_decorrelate_all: bool
    loss_decorrelate_dust: bool

    loss_clip_delta: PositiveFloat

    # --- Stages ---
    # ΔAᵥ
    delta_av_epochs: PositiveInt
    delta_av_lr: PositiveFloat
    delta_av_lr_decay_steps: PositiveInt
    delta_av_lr_decay_rate: PositiveFloat
    delta_av_lr_weight_decay_rate: PositiveFloat

    # Zs
    zs_epochs: PositiveInt
    zs_lr: PositiveFloat
    zs_lr_decay_steps: PositiveInt
    zs_lr_decay_rate: PositiveFloat
    zs_lr_weight_decay_rate: PositiveFloat

    # Δℳ
    delta_m_epochs: PositiveInt
    delta_m_lr: PositiveFloat
    delta_m_lr_decay_steps: PositiveInt
    delta_m_lr_decay_rate: PositiveFloat
    delta_m_lr_weight_decay_rate: PositiveFloat

    # Δ𝓅
    delta_p_epochs: PositiveInt
    delta_p_lr: PositiveFloat
    delta_p_lr_decay_steps: PositiveInt
    delta_p_lr_decay_rate: PositiveFloat
    delta_p_lr_weight_decay_rate: PositiveFloat

    # Final
    final_epochs: PositiveFloat
    final_lr: PositiveFloat
    final_lr_decay_steps: PositiveInt
    final_lr_decay_rate: PositiveFloat
    final_lr_weight_decay_rate: PositiveFloat
