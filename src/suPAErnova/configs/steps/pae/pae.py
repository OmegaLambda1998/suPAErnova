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
    ConfigDict,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
)

from suPAErnova.steps.data import DataStep
from suPAErnova.configs.steps.data import DataStepConfig, DataStepResult
from suPAErnova.configs.steps.model import AbstractModelStepConfig
from suPAErnova.configs.steps.steps import AbstractStepResult

from .model import PAEModelConfig


class PAEStage(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    stage: PositiveInt
    prev_stage: PositiveInt | None
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
    train_sn_mask: npt.NDArray[np.bool_]
    train_spec_mask: npt.NDArray[np.bool_]
    test_data: DataStepResult
    test_sn_mask: npt.NDArray[np.bool_]
    test_spec_mask: npt.NDArray[np.bool_]
    val_data: DataStepResult
    val_sn_mask: npt.NDArray[np.bool_]
    val_spec_mask: npt.NDArray[np.bool_]
    all_data: DataStepResult
    all_sn_mask: npt.NDArray[np.bool_]
    all_spec_mask: npt.NDArray[np.bool_]


class PAEStepResult(AbstractStepResult):
    stage: int

    ind: "npt.NDArray[np.int32]"
    sn_name: "npt.NDArray[np.str_]"
    spectra_id: "npt.NDArray[np.str_]"

    input_amp: "npt.NDArray[np.float32]"
    input_d_amp: "npt.NDArray[np.float32]"
    input_phase: "npt.NDArray[np.float32]"
    input_mask: "npt.NDArray[np.float32]"
    input_colourlaw: "npt.NDArray[np.float32] | None"

    latents: "npt.NDArray[np.float32]"

    output_amp: "npt.NDArray[np.float32]"
    diff_amp: "npt.NDArray[np.float32]"

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
    }
    id: ClassVar[str] = "pae"
    required_steps: ClassVar[list[str]] = [DataStepConfig.id]

    # --- Previous Steps ---
    data: DataStep | None = None
    validation_frac: Annotated[float, Field(ge=0, le=1)]
    kfolds: list[NonNegativeInt] | None = None

    # --- Optional ---
    seed: int


PAEStepConfig.register_step()
