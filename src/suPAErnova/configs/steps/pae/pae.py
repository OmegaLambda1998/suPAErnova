# Copyright 2025 Patrick Armstrong
from typing import Any, ClassVar, Annotated
from pathlib import Path
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt
from pydantic import (
    Field,
    BaseModel,
    PositiveInt,
    PositiveFloat,
)

from suPAErnova.steps.data import DataStep
from suPAErnova.configs.steps.data import DataStepConfig, DataStepResult
from suPAErnova.configs.steps.model import AbstractModelStepConfig
from suPAErnova.configs.steps.steps import AbstractStepResult

from .model import PAEModelConfig


class PAEStage(BaseModel):
    stage: PositiveInt
    name: str
    fname: str
    savepath: Path | None = None
    loadpath: Path | None = None

    epochs: PositiveInt
    debug: bool

    learning_rate: PositiveFloat
    learning_rate_decay_steps: PositiveInt
    learning_rate_decay_rate: PositiveFloat
    learning_rate_weight_decay_rate: PositiveFloat

    train_data: DataStepResult
    test_data: DataStepResult
    val_data: DataStepResult

    moving_means: list[float]


class PAEStepResult(AbstractStepResult):
    stage: int

    ind: "npt.NDArray[np.int32]"
    sn_name: "npt.NDArray[np.str_]"
    spectra_id: "npt.NDArray[np.str_]"

    latents: "npt.NDArray[np.float32]"
    output_amp: "npt.NDArray[np.float32]"

    loss: float
    pred_loss: float
    model_loss: float
    resid_loss: float
    delta_loss: float
    cov_loss: float


class PAEStepConfig[Backend: str](AbstractModelStepConfig[Backend, PAEModelConfig]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, Callable[[], type[PAEModelConfig]]]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPAEModelConfig,
        "PyTorch": lambda: importlib.import_module(
            ".tch", __package__
        ).TCHPAEModelConfig,
    }
    id: ClassVar[str] = "pae"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id]

    # --- Previous Steps ---
    data: DataStep | None = None
    validation_frac: Annotated[float, Field(ge=0, le=1)]

    # --- Optional ---
    seed: int


PAEStepConfig.register_step()
