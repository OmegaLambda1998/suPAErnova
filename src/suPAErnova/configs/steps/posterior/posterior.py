# Copyright 2025 Patrick Armstrong

from typing import Any, ClassVar
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt

from suPAErnova.steps.nflow import NFlowStep
from suPAErnova.configs.steps.model import AbstractModelStepConfig
from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.steps import AbstractStepResult

from .model import PosteriorModelConfig


class PosteriorStepResult(AbstractStepResult):
    ind: "npt.NDArray[np.int32]"
    sn_name: "npt.NDArray[np.str_]"
    spectra_id: "npt.NDArray[np.str_]"


class PosteriorStepConfig[Backend: str](
    AbstractModelStepConfig[Backend, PosteriorModelConfig]
):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, Callable[[], type[PosteriorModelConfig]]]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPosteriorModelConfig,
        "PyTorch": lambda: importlib.import_module(
            ".tch", __package__
        ).TCHPosteriorModelConfig,
    }
    id: ClassVar[str] = "posterior"
    required_steps: ClassVar[list[str]] = [NFlowStepConfig.id]

    # --- Previous Steps ---
    nflow: NFlowStep[Any] | None = None

    # --- Optional ---
    seed: int = 12345


PosteriorStepConfig.register_step()
