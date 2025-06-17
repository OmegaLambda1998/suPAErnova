# Copyright 2025 Patrick Armstrong

from typing import Any, Literal, ClassVar
from pathlib import Path
import importlib
from collections.abc import Callable

import numpy as np
from numpy import typing as npt
from pydantic import BaseModel, NonNegativeInt, model_validator

from suPAErnova.steps.nflow import NFlowStep
from suPAErnova.configs.steps.model import AbstractModelStepConfig
from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.steps import AbstractStepResult

from .model import PosteriorModelConfig

type InitPreset = Literal["initial", "current", "best"]
type InitULatents = Literal["u_random", "u_constant"]
type InitZLatents = Literal["z_random", "z_data", "z_constant"]
type InitLatents = InitPreset | InitULatents | InitZLatents
type InitParams = InitPreset | Literal["random", "data", "constant", "scale"]


class PosteriorMapStage(BaseModel):
    stage: NonNegativeInt
    name: str
    fname: str
    savepath: Path | None = None
    loadpath: Path | None = None

    n_chains: int
    init: bool = False
    init_u_delta_av: InitParams
    init_latents: InitLatents
    init_delta_av: InitParams
    init_delta_m: InitParams
    init_delta_p: InitParams
    init_bias: InitParams

    @model_validator(mode="before")
    @classmethod
    def init_values(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("init", False):
            init_all = "initial"
            data["init_u_delta_av"] = init_all
            data["init_latents"] = init_all
            data["init_delta_av"] = init_all
            data["init_delta_m"] = init_all
            data["init_delta_p"] = init_all
            data["init_bias"] = init_all
        return data


class PosteriorStepMAPResult(AbstractStepResult):
    chain_min: npt.NDArray[np.int32]
    converged: npt.NDArray[np.bool]
    num_evaluations: int
    negative_log_prob: npt.NDArray[np.float32]

    init_u_delta_av: npt.NDArray[np.float32]
    init_u_latents: npt.NDArray[np.float32]
    init_delta_av: npt.NDArray[np.float32]
    init_delta_m: npt.NDArray[np.float32]
    init_delta_p: npt.NDArray[np.float32]
    init_z_latents: npt.NDArray[np.float32]

    best_u_delta_av: npt.NDArray[np.float32]
    best_u_latents: npt.NDArray[np.float32]
    best_delta_av: npt.NDArray[np.float32]
    best_delta_m: npt.NDArray[np.float32]
    best_delta_p: npt.NDArray[np.float32]
    best_z_latents: npt.NDArray[np.float32]


class PosteriorStepHMCResult(AbstractStepResult):
    samples: npt.NDArray[np.float32]
    step_sizes_final: npt.NDArray[np.float32]
    is_accepted: npt.NDArray[np.float32]

    u_delta_av: npt.NDArray[np.float32]
    u_latents: npt.NDArray[np.float32]
    delta_av: npt.NDArray[np.float32]
    z_latents: npt.NDArray[np.float32]
    delta_m: npt.NDArray[np.float32]
    delta_p: npt.NDArray[np.float32]


class PosteriorStepResult(AbstractStepResult):
    ind: npt.NDArray[np.int32]
    sn_name: npt.NDArray[np.str_]
    spectra_id: npt.NDArray[np.str_]

    map: PosteriorStepMAPResult
    hmc: PosteriorStepHMCResult


class PosteriorStepConfig[Backend: str](
    AbstractModelStepConfig[Backend, PosteriorModelConfig]
):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, Callable[[], type[PosteriorModelConfig]]]] = {
        "TensorFlow": lambda: (
            importlib.import_module(".tf", __package__)
        ).TFPosteriorModelConfig,
    }
    id: ClassVar[str] = "posterior"
    required_steps: ClassVar[list[str]] = [NFlowStepConfig.id]

    # --- Previous Steps ---
    nflow: NFlowStep[Any] | None = None

    # --- Optional ---
    seeds: list[int] = [12345]


PosteriorStepConfig.register_step()
