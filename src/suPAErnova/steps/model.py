from typing import TYPE_CHECKING, get_args, override

from .steps import SNPAEStep

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.configs.steps.model import AbstractModelStepConfig
    from suPAErnova.configs.steps.steps import AbstractStepResult
    from suPAErnova.configs.steps.backends import AbstractModelConfig

    from .backends import AbstractModel


class AbstractModelStep[Backend: str, Model: AbstractModel[Backend]](SNPAEStep):
    def __init__(
        self, config: "AbstractModelStepConfig[Backend, AbstractModelConfig]"
    ) -> None:
        # --- Superclass Variables ---
        self.options: AbstractModelStepConfig[Backend, AbstractModelConfig]
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        super().__init__(config)

        self.models: list[Model]
        self.n_models: int
        self.results: list[AbstractStepResult]

    @override
    def _setup(self, *args: "Any", **kwargs: "Any") -> None:
        model_step: type[Model] = get_args(self.__orig_bases__[0])[1]
        self.models = [model_step(model) for model in self.options.models or []]
        self.n_models = len(self.models)

    @override
    def _completed(self) -> bool:
        return all(model.completed() for model in self.models)

    @override
    def _load(self) -> None:
        for model in self.models:
            model.load()

    @override
    def _run(self) -> None:
        for model in self.models:
            model.run()

    @override
    def _result(self) -> None:
        for model in self.models:
            model.result()
        self.results = [model.results for model in self.models]

    @override
    def _analyse(self) -> None:
        for model in self.models:
            model.analyse()
