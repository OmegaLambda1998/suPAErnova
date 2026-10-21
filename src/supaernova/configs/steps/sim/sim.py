from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from supaernova.configs.steps import StepConfig
from supaernova.configs.steps.data import (
    DataStepAnalysis,
    DataStepConfig,
    DataStepResult,
)
from supaernova.configs.steps.nflow import NFlowStepConfig
from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.variants import VariantConfig


class SimStepResult(DataStepResult):
    pass


class SimStepAnalysis(DataStepAnalysis):
    pass


class SimConfig(StepConfig):
    n_sn: PositiveInt | None = None
    analysis: SimStepAnalysis | None = None
    redshift: PositiveFloat = 0
    filters: list[Path] | None = None
    # Apply `filters` to the rest-frame spectra unshifted (e.g. rest-frame Landolt
    # bands, to match the real data), rather than blueshifting them by `redshift`.
    # `redshift` still sets the cadence's time dilation.
    rest_frame_filters: bool = False
    cadence: PositiveFloat
    n_spectra: NonNegativeInt | Literal[-1] = -1
    n_phot: NonNegativeInt | Literal[-1] = -1
    # Observer-frame cadences (days) at which spectra / each filter's photometry are
    # kept, starting from epoch index `*_offset`. Applied before spectra and
    # photometry are combined, so each can be dropped without affecting the other.
    spec_cadence: PositiveFloat | None = None
    spec_offset: NonNegativeInt = 0
    phot_cadence: dict[str, PositiveFloat] | None = None
    phot_offset: dict[str, NonNegativeInt] = Field(default_factory=dict)
    # Draw each SN's ΔM / Δp from a Gaussian fit to the PAE-encoded training SNe,
    # instead of 0. Requires a PAE with `physical_latents`.
    sample_delta_m: bool = False
    sample_delta_p: bool = False

    @model_validator(mode="after")
    def validate_phot_offset(self) -> "SimConfig":
        unknown = set(self.phot_offset) - set(self.phot_cadence or {})
        if unknown:
            self._raise(f"phot_offset has filters without a phot_cadence: {unknown}")
        return self


class SimStepConfig(VariantConfig):
    id: ClassVar[str] = "sim"
    required_steps: ClassVar[list[str]] = [
        DataStepConfig.id,
        PAEStepConfig.id,
        NFlowStepConfig.id,
    ]

    base: SimConfig
    variants: list[SimConfig] | None = Field(None, validation_alias="variant")


SimStepConfig.register_step(SimConfig)
DataStepConfig.register_proxy(SimStepConfig)
