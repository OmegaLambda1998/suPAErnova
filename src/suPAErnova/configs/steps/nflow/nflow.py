# Copyright 2025 Patrick Armstrong

from typing import Any, ClassVar
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt

from suPAErnova.steps.pae import PAEStep
from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.model import AbstractModelStepConfig
from suPAErnova.configs.steps.steps import AbstractStepResult

from .model import NFlowModelConfig


class NFlowStepResult(AbstractStepResult):
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


class NFlowStepConfig[Backend: str](AbstractModelStepConfig[Backend, NFlowModelConfig]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, Callable[[], type[NFlowModelConfig]]]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFNFlowModelConfig,
        "PyTorch": lambda: importlib.import_module(
            ".tch", __package__
        ).TCHNFlowModelConfig,
    }
    id: ClassVar[str] = "nflow"
    required_steps: ClassVar[list[str]] = [PAEStepConfig.id]

    # --- Previous Steps ---
    pae: PAEStep[Any] | None = None

    # --- Optional ---
    seed: int = 12345


NFlowStepConfig.register_step()
