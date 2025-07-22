from typing import TYPE_CHECKING, Any, ClassVar, get_args

from supaernova.configs.steps.backends import BACKENDS

from .steps import SNPAEStep

if TYPE_CHECKING:
    from logging import Logger
    from collections.abc import Callable

    from supaernova.configs.paths import PathConfig
    from supaernova.configs.globals import GlobalConfig
    from supaernova.configs.steps.steps import AbstractStepResult
    from supaernova.configs.steps.backends import AbstractModelConfig


class AbstractModel[Backend: str](SNPAEStep):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[Any]]"]]

    def __init__(self, config: "AbstractModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: AbstractModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        self.model: Any
        self.results: AbstractStepResult
        self.model_cls: type[Any]
        for backend_name in BACKENDS:
            if self.options.backend in get_args(BACKENDS[backend_name]):
                self.model_cls = self.model_backend[backend_name]()

    def _model(self, *, force: bool = False) -> Any:
        if not force and hasattr(self, "model"):
            return self.model
        self.model = self.model_cls(self)
        return self.model
