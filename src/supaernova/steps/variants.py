from typing import TYPE_CHECKING, Self, ClassVar, cast, override
from collections import UserDict

from supaernova.configs.callbacks import callback

from .steps import Step

if TYPE_CHECKING:
    from typing import Any

    from supaernova.configs.steps import StepResult
    from supaernova.configs.steps.variants import VariantConfig


class Variant[C: VariantConfig, S: Step](Step[C]):
    # === Class Variables ===
    variant_steps: ClassVar[dict[str, type[S]]] = {}

    # === Class Methods ===
    @classmethod
    def register_step(cls: type[Self], variant_cls: type[S]) -> None:
        cls.variant_steps[cls.id] = variant_cls
        super().register_step()

    class VariantResult(UserDict):
        def __init__(self: Self, instance: "Variant[C, S]") -> None:
            self.instance: Variant[C, S] = instance
            super().__init__()

        @override
        def __getitem__(self: Self, key: str) -> "Any":
            if key in self.instance.variants:
                variant = self.instance.variants[key]
                variant_kwargs: dict[str, StepResult] = {}
                for k, (v, n) in self.instance.variant_required_steps[key].items():
                    variant_kwargs[k] = v.results[n]
                    v.clear(**{**variant_kwargs, "variants": [n]})
                variant.result(**variant_kwargs)
                return variant.results
            return super().__getitem__(key)

    # === Instance Methods ===
    def __init__(self: Self, config: C) -> None:
        super().__init__(config)

        self.variant_step: type[S] = self.variant_steps[self.id]
        self.variant_required_steps: dict[
            str, dict[str, tuple[Variant[VariantConfig, Step], str]]
        ] = {}
        self.variants: dict[str, S] = {}
        for variant in self.options.variants:
            step = self.variant_step(variant)
            self.variants[step.name] = step
        self.results: Variant[C, S].VariantResult[str, StepResult] = Variant[
            C, S
        ].VariantResult(self)

    @override
    def _is_setup(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> bool:
        if variants is None:
            variants = list(self.variants.keys())
        if not isinstance(variants, list):
            variants = [variants]
        return all(
            self.variants[variant].is_setup(*args, **{**kwargs, "variants": [variant]})
            for variant in variants
        )

    @override
    def _setup(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Variant[VariantConfig, Step]",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for name in variants:
            variant = self.variants[name]
            variant_kwargs = {}
            self.variant_required_steps[name] = {}
            for key, val in kwargs.items():
                k = cast("int | str", variant.options.get(key, 0))
                n = list(val.results.keys())[k] if isinstance(k, int) else k
                v = (
                    list(val.results.values())[k]
                    if isinstance(k, int)
                    else val.results[k]
                )
                variant_kwargs[key] = v
                val.clear(*args, **{**kwargs, "variants": [n]})
                self.variant_required_steps[name][key] = (val, n)
            variant.setup(*args, **variant_kwargs)

    @override
    @callback
    def setup(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_setup(*args, **kwargs):
            variants = kwargs.get("variants", self.variants)
            self.set_seed()
            self.log.info(f"Setting up {self.name}")
            for variant in variants:
                if not self._is_setup(*args, **{**kwargs, "variants": [variant]}):
                    self._setup(*args, **{**kwargs, "variants": [variant]})
            self.log.info(f"Finished setting up {self.name}")

    @override
    def _has_run(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> bool:
        if variants is None:
            variants = list(self.variants.keys())
        if not isinstance(variants, list):
            variants = [variants]
        return all(
            self.variants[variant].has_run(*args, **{**kwargs, "variants": [variant]})
            for variant in variants
        )

    @override
    def _run(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant in variants:
            self.variants[variant].run(*args, **kwargs)

    @override
    @callback
    def run(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.has_run(*args, **kwargs):
            variants = kwargs.get("variants", self.variants)
            self.set_seed()
            self.log.info(f"Running {self.name}")
            for variant in variants:
                if not self._has_run(*args, **{**kwargs, "variants": [variant]}):
                    self._run(*args, **{**kwargs, "variants": [variant]})
            self.log.info(f"Finished running {self.name}")

    @override
    def _is_saved(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> bool:
        if variants is None:
            variants = list(self.variants.keys())
        if not isinstance(variants, list):
            variants = [variants]
        return all(
            self.variants[variant].is_saved(*args, **{**kwargs, "variants": [variant]})
            for variant in variants
        )

    @override
    def _save(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant in variants:
            self.variants[variant].save(*args, **kwargs)

    @override
    @callback
    def save(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_saved(*args, **kwargs):
            variants = kwargs.get("variants", self.variants)
            self.set_seed()
            self.log.info(f"Saving {self.name}")
            for variant in variants:
                if not self._is_saved(*args, **{**kwargs, "variants": [variant]}):
                    self._save(*args, **{**kwargs, "variants": [variant]})
            self.log.info(f"Finished saving {self.name}")

    @override
    def _is_loaded(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> bool:
        if variants is None:
            variants = list(self.variants.keys())
        if not isinstance(variants, list):
            variants = [variants]
        return all(
            self.variants[variant].is_loaded(*args, **{**kwargs, "variants": [variant]})
            for variant in variants
        )

    @override
    def _load(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant in variants:
            self.variants[variant].load(*args, **kwargs)

    @override
    @callback
    def load(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_loaded(*args, **kwargs):
            variants = kwargs.get("variants", self.variants)
            self.set_seed()
            if self.is_saved(*args, **kwargs):
                self.log.info(f"Loading {self.name}")
                for variant in variants:
                    if not self._is_loaded(*args, **{**kwargs, "variants": [variant]}):
                        self._load(*args, **{**kwargs, "variants": [variant]})
                self.log.info(f"Finished loading {self.name}")
            else:
                for variant in variants:
                    if not self._is_saved(*args, **{**kwargs, "variants": [variant]}):
                        self._save(*args, **{**kwargs, "variants": [variant]})

    @override
    def _has_results(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> bool:
        if variants is None:
            variants = list(self.variants.keys())
        if not isinstance(variants, list):
            variants = [variants]
        return all(
            self.variants[variant].has_results(
                *args, **{**kwargs, "variants": [variant]}
            )
            for variant in variants
        )

    @override
    def _result(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]
        for variant_name in variants:
            variant = self.variants[variant_name]
            variant.result(*args, **kwargs)

    @override
    @callback
    def result(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.has_results(*args, **kwargs):
            variants = kwargs.get("variants", self.variants)
            self.set_seed()
            self.log.info(f"Gathering {self.name} results")
            for variant in variants:
                if not self._has_results(*args, **{**kwargs, "variants": [variant]}):
                    self._result(*args, **{**kwargs, "variants": [variant]})
            self.log.info(f"Finished gathering {self.name} results")

    @override
    def _was_analysed(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> bool:
        if variants is None:
            variants = list(self.variants.keys())
        if not isinstance(variants, list):
            variants = [variants]
        return all(
            self.variants[variant].was_analysed(
                *args, **{**kwargs, "variants": [variant]}
            )
            for variant in variants
        )

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
        for variant_name in variants:
            variant = self.variants[variant_name]
            variant.analyse(*args, **kwargs)

    @override
    @callback
    def analyse(self: Self, *args: "Any", **kwargs: "Any") -> None:
        self.setup(*args, **kwargs)
        if self.force or not self.was_analysed(*args, **kwargs):
            variants = kwargs.get("variants", self.variants)
            self.set_seed()
            self.log.info(f"Analysing {self.name}")
            for variant_name in variants:
                if self.force or not self._was_analysed(
                    *args, **{**kwargs, "variants": [variant_name]}
                ):
                    self._analyse(*args, **{**kwargs, "variants": [variant_name]})
                    self._clear(variants=[variant_name])
            self.log.info(f"Finished analysing {self.name}")

    @override
    def _clear(
        self: Self,
        *args: "Any",
        setup: bool = False,
        run: bool = False,
        save: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
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
                run=run,
                save=save,
                load=load,
                result=result,
                analyse=analyse,
                **kwargs,
            )
            # self.variant_required_steps[variant_name] = {}
        super()._clear(
            *args,
            setup=setup,
            run=run,
            save=save,
            load=load,
            result=result,
            analyse=analyse,
            **kwargs,
        )

    @override
    @callback
    def clear(
        self: Self,
        *args: "Any",
        setup: bool = False,
        run: bool = False,
        save: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        **kwargs: "Any",
    ) -> None:
        variants = kwargs.get("variants", self.variants)
        self.set_seed()
        self.log.info(f"Clearing {self.name}")
        for variant_name in variants:
            self._clear(
                *args,
                setup=setup,
                run=run,
                save=save,
                load=load,
                result=result,
                analyse=analyse,
                **{**kwargs, "variants": [variant_name]},
            )
        self.log.info(f"Finished clearing {self.name}")

    # === Static Methods ===
