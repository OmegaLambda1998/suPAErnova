from typing import TYPE_CHECKING, ClassVar, override
from collections import UserDict

from supaernova.utils import pp
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

        @override
        def __getitem__(self, key: str) -> "Any":
            if key in self.instance.variants:
                variant = self.instance.variants[key]
                kwargs = {}
                for k, (v, n) in self.instance.variant_required_steps[key].items():
                    kwargs[k] = v.results[n]
                    v.clear(**{**kwargs, "variants": [n]})
                variant.result(**kwargs)
                return variant.results
            return super().__getitem__(key)

    # === Instance Methods ===
    def __init__(self, config: "VariantConfig") -> None:
        super().__init__(config)

        self.variant_step: type[Step] = self.variant_steps[self.id]
        self.variant_configs: dict[str, StepConfig] = {}
        self.variant_required_steps: dict[str, dict[str, Step]] = {}
        self.variants: dict[str, Step] = {}
        for variant in self.options.variants:
            step = self.variant_step(variant)
            self.variant_configs[step.name] = variant
            self.variants[step.name] = step
        self.results: Variant.VariantResult[str, StepResult] = Variant.VariantResult(
            self
        )

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
            self.variant_required_steps[variant_name] = {}
            for key, val in kwargs.items():
                k = variant.options.get(key, 0)
                n = list(val.results.keys())[k] if isinstance(k, int) else k
                v = (
                    list(val.results.values())[k]
                    if isinstance(k, int)
                    else val.results[k]
                )
                kwargs_[key] = v
                val.clear(*args, **{**kwargs, "variants": [n]})
                self.variant_required_steps[variant_name][key] = (val, n)
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
        return all(
            self.variants[variant].completed(*args, **{**kwargs, "variants": [variant]})
            for variant in variants
        )

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
            for variant_name in variants:
                self.result(*args, **{**kwargs, "variants": [variant_name]})
                self._analyse(*args, **{**kwargs, "variants": [variant_name]})
                self._clear(variants=[variant_name])

    @override
    def _clear(
        self,
        *args: "Any",
        setup: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        complete: bool = False,
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant_name in variants:
            variant = self.variants[variant_name]
            variant.clear(
                *args,
                setup=setup,
                load=load,
                result=result,
                analyse=analyse,
                complete=complete,
                **kwargs,
            )
            # self.variant_required_steps[variant_name] = {}
        super()._clear(
            *args,
            setup=setup,
            load=load,
            result=result,
            analyse=analyse,
            complete=complete,
            **kwargs,
        )

    @override
    @callback
    def clear(
        self,
        *args: "Any",
        setup: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        complete: bool = False,
        **kwargs: "Any",
    ) -> None:
        self.set_seed()
        variants = kwargs.get("variants", self.variants)
        for variant_name in variants:
            self._clear(
                *args,
                setup=setup,
                load=load,
                result=result,
                analyse=analyse,
                complete=complete,
                **{**kwargs, "variants": [variant_name]},
            )

    # === Static Methods ===
