from typing import TYPE_CHECKING, Any, ClassVar, get_args, override

from supaernova.configs.steps.models import BACKENDS

from .variants import Variant

if TYPE_CHECKING:
    from collections.abc import Callable

    from supaernova.configs.steps.models import ModelConfig


class Model(Variant):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[Any]]"]]

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        self.models: dict[str, Any] = {}
        self.models_cls: dict[str, type[Any]] = {}

        for backend_name, backend_args in BACKENDS.items():
            for name, variant in self.variants.items():
                if variant.options.backend in get_args(backend_args):
                    self.models_cls[name] = self.model_backend[backend_name]()

    @override
    def _setup(self, *args: "Any", **kwargs: "Any") -> None:
        super()._setup(*args, **kwargs)
        self._model()

    @override
    def _completed(self) -> bool:
        self._model()
        return super()._completed()

    @override
    def _load(self) -> None:
        self._model()
        super()._load()

    @override
    def _run(self) -> None:
        self._model()
        super()._run()

    @override
    def _result(self) -> None:
        self._model()
        super()._result()

    @override
    def _analyse(self) -> None:
        self._model()
        super()._analyse()

    def _model(self, *, force: bool = False) -> None:
        for name, variant in self.variants.items():
            if hasattr(variant, "model"):
                self.models[name] = variant.model
            if force or not self.models.get(name):
                self.models[name] = self.models_cls[name](variant)
            variant.model = self.models[name]
