# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, Literal, ClassVar, override
from pathlib import Path  # noqa: TC003
import importlib

from pydantic import (
    BaseModel,
    PositiveInt,  # noqa: TC002
)

from suPAErnova.steps.backends import AbstractModel
from suPAErnova.configs.steps.data import DataStepResult  # noqa: TC001

if TYPE_CHECKING:
    from logging import Logger
    from collections.abc import Callable

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.nflow.model import NFlowModel
    from suPAErnova.configs.steps.posterior.model import PosteriorModelConfig

    from .tf import TFPosteriorModel
    from .tch import TCHPosteriorModel

    PosteriorModel = TFPosteriorModel | TCHPosteriorModel

InitVars = Literal["init", "random", "data", "scale", "trivial"]


class Stage(BaseModel):
    stage: "PositiveInt"
    name: str
    fname: str
    savepath: "Path | None" = None
    loadpath: "Path | None" = None

    debug: bool

    n_chains: int

    train_data: "DataStepResult"
    test_data: "DataStepResult"
    val_data: "DataStepResult"

    init_latents: "InitVars"
    init_delta_av: "InitVars"
    init_delta_m: "InitVars"
    init_delta_p: "InitVars"
    init_bias: "InitVars"


class PosteriorModelStep[Backend: str](AbstractModel[Backend]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[PosteriorModel]]"]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPosteriorModel,
        "PyTorch": lambda: importlib.import_module(
            ".tch", __package__
        ).TCHPosteriorModel,
    }
    id: ClassVar[str] = "posterior_model"

    def __init__(self, config: "PosteriorModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: PosteriorModelConfig
        self.config: "GlobalConfig"
        self.paths: "PathConfig"
        self.log: "Logger"
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool
        self.savepath: "Path"

        # --- Previous Step Variables ---
        self.nflow: NFlowModel
        self.train_data: DataStepResult
        self.test_data: DataStepResult
        self.val_data: DataStepResult

        # --- Setup Variables ---
        self.n_chains_early: int = self.options.n_chains_early
        self.n_chains_mid: int = self.options.n_chains_mid
        self.n_chains_final: int = self.options.n_chains_final

        self.stage_init: Stage
        self.stage_early: Stage
        self.stage_mid: Stage
        self.stage_final: Stage
        self.run_stages: list[Stage]

    @override
    def _setup(self, *, nflow: "NFlowModel") -> None:
        self.debug = self.options.debug

        self.nflow = nflow
        self.nflow.load()

        if self.nflow.model.physical_latents and self.options.train_delta_av:
            self.options.train_delta_av = False
            self.log.warning(
                f"ΔAᵥ is already included in the {self.nflow.name} Normalising Flow model, so will not be used as a free parameter."
            )

        self.train_data = self.nflow.pae.train_data
        self.test_data = self.nflow.pae.test_data
        self.val_data = self.nflow.pae.val_data

        self._model()
        self.savepath = self.paths.out / self.model.name

        # --- Stages ---
        stage_data = {
            "train_data": self.train_data,
            "test_data": self.test_data,
            "val_data": self.val_data,
            "debug": self.debug,
        }
        self.stage_init = Stage.model_validate({
            "stage": 1,
            "name": "init",
            "fname": "init",
            "n_chains": 1,
            "init_latents": "init",
            "init_delta_av": "init",
            "init_delta_m": "init",
            "init_delta_p": "init",
            "init_bias": "init",
            **stage_data,
        })
        self.stage_early = Stage.model_validate({
            "stage": 2,
            "name": "early",
            "fname": "early",
            "n_chains": self.n_chains_early,
            "init_latents": "random",
            "init_delta_av": "random",
            "init_delta_m": "random",
            "init_delta_p": "random",
            "init_bias": "random",
            **stage_data,
        })
        self.stage_mid = Stage.model_validate({
            "stage": 3,
            "name": "mid",
            "fname": "mid",
            "n_chains": self.n_chains_mid,
            "init_latents": "trivial",
            "init_delta_av": "data",
            "init_delta_m": "scale",
            "init_delta_p": "random",
            "init_bias": "random",
            **stage_data,
        })
        self.stage_final = Stage.model_validate({
            "stage": 4,
            "name": "final",
            "fname": "final",
            "n_chains": self.n_chains_final,
            "init_latents": "trivial",
            "init_delta_av": "scale",
            "init_delta_m": "random",
            "init_delta_p": "random",
            "init_bias": "random",
            **stage_data,
        })
        self.run_stages = [
            self.stage_init,
            self.stage_early,
            self.stage_mid,
            self.stage_final,
        ]

    @override
    def _completed(self) -> bool:
        self._model()

        final_stage = self.run_stages[-1]
        self.model.stage = final_stage
        final_savepath = self.savepath / final_stage.fname / self.model.model_path

        if not final_savepath.exists():
            self.log.debug(
                f"{self.name} has not completed as {final_savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self._model()

        final_stage = self.run_stages[-1]
        self.model.stage = final_stage
        final_savepath = self.savepath / final_stage.fname

        self.log.debug(f"Loading final Posterior model weights from {final_savepath}")
        self.model.load_checkpoint(final_savepath)

    @override
    def _run(self) -> None:
        self._model()
        savepath: Path | None = None
        for i, stage in enumerate(self.run_stages):
            self.log.debug(f"Starting Stage {i}: {stage.name}")
            if savepath is not None:
                stage.loadpath = savepath
            savepath = self.paths.out / self.model.name / stage.fname
            stage.savepath = savepath

            model_path = savepath / self.model.model_path
            weights_path = savepath / self.model.weights_path
            if model_path.exists() and not self.force:
                # Don't retrain stages if you don't need to
                self.log.debug(f"Loading stage weights from {weights_path}")
                self.model.stage = stage
                self.model.load_checkpoint(savepath)
            else:
                self.model.train_model(stage)
            self.model.save_checkpoint(savepath)

    @override
    def _result(self) -> None:
        self._model()

        final_stage = self.run_stages[-1]
        self.model.stage = final_stage
        final_savepath = self.savepath / final_stage.fname

        self.log.debug(f"Saving final Posterior model weights to {final_savepath}")
        self.model.save_checkpoint(final_savepath)

    @override
    def _analyse(self) -> None:
        pass

    #
    # === Posterior Specific Functions ===
    #
