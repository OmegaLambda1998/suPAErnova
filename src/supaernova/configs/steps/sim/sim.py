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
