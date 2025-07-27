from typing import TYPE_CHECKING, ClassVar, override

from .steps import Step

if TYPE_CHECKING:
    from typing import Any

    from supaernova.configs.steps import StepResult
    from supaernova.configs.steps.variants import VariantConfig


class Variant(Step):
    # === Class Variables ===
    variant_steps: ClassVar[dict[str, type["Step"]]] = {}

    # === Class Methods ===
    @classmethod
    def register_step(cls, variant_cls: type[Step]) -> None:
        cls.variant_steps[cls.id] = variant_cls
        super().register_step()

    # === Instance Methods ===
    def __init__(self, config: "VariantConfig") -> None:
        super().__init__(config)
        self.results: dict[str, StepResult]

        variant_step: type[Step] = self.variant_steps[self.id]
        self.variants: dict[str, Step] = {}
        for variant in self.options.variants:
            step = variant_step(variant)
            self.variants[step.name] = step

    @override
    def _setup(self, *args: "Any", **kwargs: "Any") -> None:
        for variant in self.variants.values():
            kwargs_ = {}
            for key, val in kwargs.items():
                k = variant.options.get(key, 0)
                v = (
                    list(val.variants.values())[k]
                    if isinstance(k, int)
                    else val.variants[k]
                )
                kwargs_[key] = v
            variant.setup(*args, **kwargs_)

    @override
    def _completed(self) -> bool:
        return all(variant.completed() for variant in self.variants.values())

    @override
    def _load(self) -> None:
        for variant in self.variants.values():
            variant.load()

    @override
    def _run(self) -> None:
        for variant in self.variants.values():
            variant.run()

    @override
    def _result(self) -> None:
        for variant in self.variants.values():
            variant.result()
        self.results = {
            variant.name: variant.results for variant in self.variants.values()
        }

    @override
    def _analyse(self) -> None:
        for variant in self.variants.values():
            variant.analyse()

    # === Static Methods ===
