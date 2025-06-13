from typing import TYPE_CHECKING, Any, ClassVar, override

from suPAErnova.steps.model import AbstractModelStep

from .model import PosteriorModelStep

if TYPE_CHECKING:
    from logging import Logger

    from suPAErnova.steps.nflow import NFlowStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.configs.steps.posterior import PosteriorStepConfig


class PosteriorStep[Backend: str](
    AbstractModelStep[Backend, PosteriorModelStep[Backend]]
):
    # Class Variables
    id: ClassVar[str] = "posterior"

    def __init__(self, config: "PosteriorStepConfig[Backend]") -> None:
        # --- Superclass Variables ---
        self.options: PosteriorStepConfig[Backend]
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Previous Step Variables ---
        self.nflow: NFlowStep[Any]

    @override
    def _setup(self, *, nflow: "NFlowStep[Any]") -> None:
        super()._setup()

        # --- Previous Step Variables ---
        self.nflow = nflow

        # --- Models ---
        for i, model in enumerate(self.models):
            model.setup(nflow=self.nflow.models[i])


PosteriorStep.register_step()
