from supaernova.utils import resolve_path
from pathlib import Path
from typing import ClassVar, Any, LiteralString, Literal
import numpy as np

from pydantic import Field, model_validator, PositiveInt, PositiveFloat, NonNegativeInt

from supaernova.configs.steps import StepConfig
from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.data import (
    DataStepConfig,
    DataStepResult,
    DataStepAnalysis,
)
from supaernova.configs.steps.nflow import NFlowStepConfig
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
