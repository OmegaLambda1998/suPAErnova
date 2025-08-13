from typing import TYPE_CHECKING, ClassVar, override
from collections import UserDict

from supaernova.configs.callbacks import callback

from .steps import Step

if TYPE_CHECKING:
    from typing import Any

    from supaernova.configs.steps import StepConfig, StepResult
    from supaernova.configs.steps.variants import VariantConfig


class Variant(Step):
    # === Class Variables ===
    variant_steps: ClassVar[dict[str, type["Step"]]] = {}

    # === Class Methods ===
    @classmethod
    def register_step(cls, variant_cls: type[Step]) -> None:
        cls.variant_steps[cls.id] = variant_cls
        super().register_step()

    class VariantResult(UserDict):
        def __init__(self, instance: "Variant") -> None:
            self.instance = instance
            super().__init__()

        @property
        def results(self):
            return {
                variant.name: None
                if not hasattr(variant, "results")
                else variant.results
                for variant in self.instance.variants.values()
            }

        @override
        def __getattribute__(self, key: str):
            if key == "data":
                return self.results
            return super().__getattribute__(key)

    # === Instance Methods ===
    def __init__(self, config: "VariantConfig") -> None:
        super().__init__(config)

        self.variant_step: type[Step] = self.variant_steps[self.id]
        self.variant_configs: dict[str, StepConfig] = {}
        self.variants: dict[str, Step] = {}
        self._init_variants()

        self.results: Variant.VariantResult[str, StepResult] = Variant.VariantResult(
            self
        )

    def _init_variants(
        self,
        *,
        variants: "StepConfig | list[StepConfig] | None" = None,
        overrides: dict[str, dict[str, "Any"]] | None = None,
    ) -> None:
        if variants is None:
            variants = self.options.variants
        if not isinstance(variants, list):
            variants = [variants]
        if overrides is None:
            overrides = {}
        for variant in variants:
            step = self.variant_step(variant)
            for k, v in overrides.get(step.name, {}).items():
                setattr(step, k, v)
            self.variant_configs[step.name] = variant
            self.variants[step.name] = step

    @override
    def _setup(
        self, *args: "Any", variants: str | list[str] | None = None, **kwargs: "Any"
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant_name in variants:
            variant = self.variants[variant_name]
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
        self.is_setup = all(variant.is_setup for variant in self.variants.values())

    @override
    @callback
    def setup(self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_setup:
            self.set_seed()
            variants = kwargs.get("variants", self.variants)
            for variant in variants:
                self._setup(*args, **{**kwargs, "variants": [variant]})

    @override
    def _completed(
        self, *args: "Any", variants: str | list[str] | None = None, **kwargs: "Any"
    ) -> bool:
        if variants is None:
            return False
        if not isinstance(variants, list):
            variants = [variants]
        return all(self.variants[variant].completed() for variant in variants)

    @override
    @callback
    def completed(self, *args: "Any", **kwargs: "Any") -> bool:
        self.set_seed()
        variants = kwargs.get("variants", self.variants)
        for variant in variants:
            self.setup(*args, **{**kwargs, "variants": [variant]})
        self.log.debug(f"Checking if {self.name} has completed")
        is_completed = self._completed(*args, **kwargs)
        self.log.debug(
            f"{self.name} has {'' if self.is_completed else 'not '}completed"
        )
        return is_completed

    @override
    def _load(
        self, *args: "Any", variants: str | list[str] | None = None, **kwargs: "Any"
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant in variants:
            self.variants[variant].load(*args, **kwargs)
        self.is_loaded = all(variant.is_loaded for variant in self.variants.values())

    @override
    @callback
    def load(self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_loaded:
            self.set_seed()
            if self.completed(*args, **kwargs):
                variants = kwargs.get("variants", self.variants)
                for variant in variants:
                    self.setup(*args, **{**kwargs, "variants": [variant]})
                    self._load(*args, **{**kwargs, "variants": [variant]})
            else:
                self.run(*args, **kwargs)

    @override
    def _run(
        self, *args: "Any", variants: str | list[str] | None = None, **kwargs: "Any"
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant in variants:
            self.variants[variant].run(*args, **kwargs)
        self.is_loaded = all(variant.is_loaded for variant in self.variants.values())

    @override
    @callback
    def run(self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_loaded and (self.force or not self.completed(*args, **kwargs)):
            self.set_seed()
            variants = kwargs.get("variants", self.variants)
            for variant in variants:
                self.setup(*args, **{**kwargs, "variants": [variant]})
                self._run(*args, **{**kwargs, "variants": [variant]})

    @override
    def _result(
        self, *args: "Any", variants: str | list[str] | None = None, **kwargs: "Any"
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant_name in variants:
            variant = self.variants[variant_name]
            variant.result(*args, **kwargs)
        self.has_results = all(
            variant.has_results for variant in self.variants.values()
        )

    @override
    @callback
    def result(self, *args: "Any", **kwargs: "Any") -> None:
        if not self.has_results:
            self.set_seed()
            variants = kwargs.get("variants", self.variants)
            for variant in variants:
                self.load(*args, **{**kwargs, "variants": [variant]})
                self._result(*args, **{**kwargs, "variants": [variant]})

    @override
    def _analyse(
        self, *args: "Any", variants: str | list[str] | None = None, **kwargs: "Any"
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant_name in variants:
            variant = self.variants[variant_name]
            variant.analyse(*args, **kwargs)
        self.was_analysed = all(
            variant.was_analysed for variant in self.variants.values()
        )

    @override
    @callback
    def analyse(self, *args: "Any", **kwargs: "Any") -> None:
        if not self.was_analysed:
            self.set_seed()
            variants = kwargs.get("variants", self.variants)
            for variant in variants:
                self.result(*args, **{**kwargs, "variants": [variant]})
                self._analyse(*args, **{**kwargs, "variants": [variant]})
                self._init_variants(
                    variants=self.variant_configs[variant],
                )

    # === Static Methods ===
