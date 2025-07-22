from typing import TYPE_CHECKING, Any, ClassVar, override

from supaernova.steps.model import AbstractModelStep

from .model import NFlowModelStep

if TYPE_CHECKING:
    from logging import Logger

    from supaernova.steps.pae import PAEStep
    from supaernova.configs.paths import PathConfig
    from supaernova.configs.globals import GlobalConfig
    from supaernova.configs.steps.nflow import NFlowStepConfig


class NFlowStep[Backend: str](AbstractModelStep[Backend, NFlowModelStep[Backend]]):
    # Class Variables
    id: ClassVar[str] = "nflow"

    def __init__(self, config: "NFlowStepConfig[Backend]") -> None:
        # --- Superclass Variables ---
        self.options: NFlowStepConfig[Backend]
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        # --- Previous Step Variables ---
        self.pae: PAEStep[Any]

    @override
    def _setup(self, *, pae: "PAEStep[Any]") -> None:
        super()._setup()

        # --- Previous Step Variables ---
        self.pae = pae

        # --- Models ---
        for i, model in enumerate(self.models):
            model.setup(pae=self.pae.models[i])


NFlowStep.register_step()
