# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, override
import importlib

from suPAErnova.steps.backends import AbstractModel
from suPAErnova.configs.steps.posterior import PosteriorStepResult
from suPAErnova.configs.steps.posterior.posterior import PosteriorMapStage

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.steps.nflow.model import NFlowModel
    from suPAErnova.configs.steps.posterior.model import PosteriorModelConfig

    from .tf import TFPosteriorModel

    PosteriorModel = TFPosteriorModel


class PosteriorModelStep[Backend: str](AbstractModel[Backend]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[PosteriorModel]]"]] = {
        "TensorFlow": lambda: importlib.import_module(
            ".tf", __package__
        ).TFPosteriorModel,
        "PyTorch": lambda: importlib.import_module(
            ".tch", __package__
        ).TCHposteriorModel,
    }
    id: ClassVar[str] = "posterior_model"

    def __init__(self, config: "PosteriorModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: PosteriorModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Config Variabls ---
        self.debug: bool
        self.savepath: Path

        self.nflow: NFlowModel

        self.results: PosteriorStepResult

        # --- Setup Variables ---
        self.n_chains_early: int = self.options.n_chains_early
        self.n_chains_mid: int = self.options.n_chains_mid
        self.n_chains_final: int = self.options.n_chains_final

        self.map_stage_init: PosteriorMapStage
        self.map_stage_early: PosteriorMapStage
        self.map_stage_mid: PosteriorMapStage
        self.map_stage_final: PosteriorMapStage
        self.map_stages: list[PosteriorMapStage]

    @override
    def _setup(self, *, nflow: "NFlowModel") -> None:
        self.debug = self.options.debug

        self.nflow = nflow
        self.nflow.load()

        self._model()
        self.savepath = self.paths.out / self.model.name

        # --- Stages ---
        self.map_stage_init = PosteriorMapStage.model_validate({
            "stage": 0,
            "name": "init",
            "fname": "init",
            "n_chains": 1,
            "init": True,
        })
        self.map_stage_early = PosteriorMapStage.model_validate({
            "stage": 1,
            "name": "random",
            "fname": "random",
            "n_chains": self.n_chains_early,
            "init_u_delta_av": "random",
            "init_latents": "u_random",
            "init_delta_av": "data",
            "init_delta_m": "random",
            "init_delta_p": "random",
            "init_bias": "current",
        })
        self.map_stage_mid = PosteriorMapStage.model_validate({
            "stage": 2,
            "name": "delta_m",
            "fname": "delta_m",
            "n_chains": self.n_chains_mid,
            "init_u_delta_av": "constant",
            "init_latents": "u_constant",
            "init_delta_av": "data",
            "init_delta_m": "scale",
            "init_delta_p": "random",
            "init_bias": "current",
        })
        self.map_stage_final = PosteriorMapStage.model_validate({
            "stage": 3,
            "name": "delta_av",
            "fname": "delta_av",
            "n_chains": self.n_chains_final,
            "init_u_delta_av": "data",
            "init_latents": "z_constant",
            "init_delta_av": "scale",
            "init_delta_m": "constant",
            "init_delta_p": "random",
            "init_bias": "current",
        })

        self.map_stages = [
            self.map_stage_init,
            self.map_stage_early,
            self.map_stage_mid,
            self.map_stage_final,
        ]

    @override
    def _completed(self) -> bool:
        self._model()
        savepath = self.savepath / self.model.model_path

        if not savepath.exists():
            self.log.debug(
                f"{self.name} has not completed as {savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self._model()

        self.log.debug(f"Loading final Posterior model weights from {self.savepath}")
        self.model.load_checkpoint(self.savepath)

        self._result()

    @override
    def _run(self) -> None:
        self._model()
        model_path = self.savepath / self.model.model_path
        weights_path = self.savepath / self.model.weights_path
        if model_path.exists() and not self.force:
            # Don't retrain stages if you don't need to
            self.log.debug(f"Loading weights from {weights_path}")
            self.model.load_checkpoint(self.savepath)
        else:
            self.model.train_model(self.map_stages, savepath=self.savepath)
        self.model.save_checkpoint(self.savepath)

    @override
    def _result(self) -> None:
        self._model()
        self.log.debug(f"Saving final Posterior model weights to {self.savepath}")
        self.model.save_checkpoint(self.savepath)

        data = self.nflow.pae.data.data

        map_results = {
            "chain_min": self.model.map.chain_min.numpy(),
            "converged": self.model.map.converged.numpy(),
            "num_evaluations": self.model.map.num_evaluations.numpy(),
            "negative_log_prob": self.model.map.negative_log_prob.numpy(),
            "init_u_delta_av": self.model.map.u_delta_av.initial.numpy(),
            "init_u_latents": self.model.map.u_latents.initial.numpy(),
            "init_delta_av": self.model.map.delta_av.initial.numpy(),
            "init_delta_m": self.model.map.delta_m.initial.numpy(),
            "init_delta_p": self.model.map.delta_p.initial.numpy(),
            "init_z_latents": self.model.map.z_latents.initial.numpy(),
            "best_u_delta_av": self.model.map.u_delta_av.best.numpy(),
            "best_u_latents": self.model.map.u_latents.best.numpy(),
            "best_delta_av": self.model.map.delta_av.best.numpy(),
            "best_delta_m": self.model.map.delta_m.best.numpy(),
            "best_delta_p": self.model.map.delta_p.best.numpy(),
            "best_z_latents": self.model.map.z_latents.best.numpy(),
        }

        hmc_results = {
            "samples": self.model.hmc.samples.numpy(),
            "step_sizes_final": self.model.hmc.step_sizes_final.numpy(),
            "is_accepted": self.model.hmc.is_accepted.numpy(),
        }

        model_results = {
            "ind": data.ind,
            "sn_name": data.sn_name,
            "spectra_id": data.spectra_id,
            "map": map_results,
            "hmc": hmc_results,
        }

        self.results = PosteriorStepResult.model_validate(model_results)

    @override
    def _analyse(self) -> None:
        pass

    #
    # === PosteriorModel Specific Functions ===
    #
