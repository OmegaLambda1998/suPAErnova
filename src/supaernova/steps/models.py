from typing import TYPE_CHECKING, Any, Self, ClassVar, get_args, override

from supaernova.configs.steps.models import BACKENDS, ModelConfig, BackendConfig

from .steps import Step
from .variants import Variant

if TYPE_CHECKING:
    from collections.abc import Callable

    from supaernova.configs.steps import StepResult


class ModelStep[C: BackendConfig](Step[C]):
    def __init__(self: Self, config: C) -> None:
        super().__init__(config)
        self.model: Any


class Model[C: ModelConfig, S: ModelStep](Variant[C, S]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[Any]]"]]

    class ModelResult(Variant.VariantResult):
        def __init__(self: Self, instance: "Model") -> None:
            self.instance: Model[C, S] = instance
            super().__init__(instance)

        @override
        def __getitem__(self: Self, key: str) -> "Any":
            if key in self.instance.variants:
                variant = self.instance.variants[key]
                kwargs: dict[str, StepResult] = {}
                for k, (v, n) in self.instance.variant_required_steps[key].items():
                    kwargs[k] = v.results[n]
                    v.clear(**{**kwargs, "variants": [n]})
                variant.setup(**kwargs)
                self.instance._model(variants=[key])
                variant.result(**kwargs)
                return variant.results
            return super(Variant.VariantResult, self).__getitem__(key)

    def __init__(self: Self, config: C) -> None:
        super().__init__(config)

        self.models_cls: dict[str, type[Any]] = {}

        self.results: Model.ModelResult[str, StepResult] = Model.ModelResult(self)

        for backend_name, backend_args in BACKENDS.items():
            for name, variant in self.variants.items():
                if variant.options.backend in get_args(backend_args):
                    self.models_cls[name] = self.model_backend[backend_name]()

    @override
    def _setup(self: Self, *args: "Any", **kwargs: "Any") -> None:
        super()._setup(*args, **kwargs)
        self._model(*args, **kwargs)

    @override
    def _run(self: Self, *args: Any, **kwargs: Any) -> None:
        self._model(*args, **kwargs)
        super()._run(*args, **kwargs)

    @override
    def _save(self: Self, *args: Any, **kwargs: Any) -> None:
        self._model(*args, **kwargs)
        super()._save(*args, **kwargs)

    @override
    def _load(self: Self, *args: Any, **kwargs: Any) -> None:
        self._model(*args, **kwargs)
        super()._load(*args, **kwargs)

    @override
    def _result(self: Self, *args: Any, **kwargs: Any) -> None:
        self._model(*args, **kwargs)
        super()._result(*args, **kwargs)

    @override
    def _analyse(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for name in variants:
            self._model(*args, **{**kwargs, "variants": [name]})
            super()._analyse(*args, **{**kwargs, "variants": [name]})

    def _model(
        self: Self,
        *args: Any,
        force: bool = False,
        variants: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for name in variants:
            variant = self.variants[name]
            if force or not hasattr(variant, "model"):
                model = self.models_cls[name](variant)
            else:
                model = variant.model
            variant.model = model
