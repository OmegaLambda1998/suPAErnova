# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, Any, Self, ClassVar
from pathlib import Path
import pkgutil
import importlib
from collections.abc import Iterable

import numpy as np

from supaernova.configs.callbacks import callback

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger
    from collections.abc import Callable

    from supaernova.configs.paths import PathConfig
    from supaernova.configs.steps import StepConfig, StepResult, StepAnalysis
    from supaernova.configs.globals import GlobalConfig
    from supaernova.configs.steps.variants import VariantConfig


class Step[C: StepConfig]:
    # === Class Variables ===
    steps: ClassVar[dict[str, type["Step[C]"]]] = {}
    id: ClassVar[str]

    # === Class Methods ===
    @classmethod
    def register_step(cls: type[Self]) -> None:
        cls.steps[cls.id] = cls

    @staticmethod
    def register_steps() -> None:
        base_name = ".".join(
            __name__.split(".")[:-1]
        )  # Remove the last duplicated part
        for _, module, is_pkg in pkgutil.iter_modules([str(Path(__file__).parent)]):
            if is_pkg and module != "posterior":
                importlib.import_module(f"{base_name}.{module}")

    # === Instance Methods ===
    def __init__(self: Self, config: C) -> None:
        self.__class__.id = config.__class__.id
        self.name: str = (
            config.name
            if config.name != config.__class__.__name__
            else self.__class__.__name__
        ).replace("Config", "")

        self.options: C = config
        self.config: GlobalConfig = config.config
        self.paths: PathConfig = config.paths
        self.log: Logger = config.log
        self.force: bool = self.config.force
        self.verbose: bool = self.config.verbose
        self.callbacks: dict[str, str | dict[str, Callable[[Any], None]]] = (
            config.callbacks
        )
        self.skip: bool = self.options.skip

        self.seed: int = config.seed
        self.rng = np.random.default_rng(self.seed)
        self.set_seed()

        self.results: StepResult
        self.analysis: StepAnalysis

    def _is_setup(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return False

    @callback
    def is_setup(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        self.log.debug(f"Checking if {self.name} is setup")
        is_setup = self._is_setup(*args, **kwargs)
        self.log.debug(f"{self.name} is {'' if is_setup else 'not '}setup")
        return is_setup

    def _setup(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def setup(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_setup(*args, **kwargs):
            self.set_seed()
            self.log.info(f"Setting up {self.name}")
            self._setup(*args, **kwargs)
            self.log.info(f"Finished setting up {self.name}")

    def _has_run(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return False

    @callback
    def has_run(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        self.log.debug(f"Checking if {self.name} has run")
        has_run = self._has_run(*args, **kwargs)
        self.log.debug(f"{self.name} has {'' if has_run else 'not '}run")
        return has_run

    def _run(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def run(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.has_run(*args, **kwargs):
            self.set_seed()
            self.log.info(f"Running {self.name}")
            self.setup(*args, **kwargs)
            self._run(*args, **kwargs)
            self.log.info(f"Finished running {self.name}")

    def _is_saved(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return False

    @callback
    def is_saved(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        self.log.debug(f"Checking if {self.name} is saved")
        is_saved = self._is_saved(*args, **kwargs)
        self.log.debug(f"{self.name} is {'' if is_saved else 'not '}saved")
        return is_saved

    def _save(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def save(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_saved(*args, **kwargs):
            self.set_seed()
            self.run(*args, **kwargs)
            self.log.info(f"Saving {self.name}")
            self._save(*args, **kwargs)
            self.log.info(f"Finished saving {self.name}")

    def _is_loaded(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return self._has_run(*args, **kwargs)

    @callback
    def is_loaded(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        self.log.debug(f"Checking if {self.name} is loaded")
        is_loaded = self._is_loaded(*args, **kwargs)
        self.log.debug(f"{self.name} is {'' if is_loaded else 'not '}loaded")
        return is_loaded

    def _load(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def load(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_loaded(*args, **kwargs):
            self.set_seed()
            if self.is_saved(*args, **kwargs):
                self.log.info(f"Loading {self.name}")
                self.setup(*args, **kwargs)
                self._load(*args, **kwargs)
                self.log.info(f"Finished loading {self.name}")
            else:
                self.save(*args, **kwargs)

    def _has_results(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return False

    @callback
    def has_results(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        self.log.debug(f"Checking if {self.name} has all results")
        has_results = self._has_results(*args, **kwargs)
        self.log.debug(f"{self.name} is {'not ' if has_results else ''}missing results")
        return has_results

    def _result(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def result(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.has_results(*args, **kwargs):
            self.set_seed()
            self.log.info(f"Gathering {self.name} results")
            self.load(*args, **kwargs)
            self._result(*args, **kwargs)
            self.log.info(f"Finished gathering {self.name} results")

    def _was_analysed(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return False

    @callback
    def was_analysed(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        self.log.debug(f"Checking if {self.name} has all analyses")
        was_analysed = self._was_analysed(*args, **kwargs)
        self.log.debug(
            f"{self.name} is {'not ' if was_analysed else ''}missing analyses"
        )
        return was_analysed

    def _analyse(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def analyse(self: Self, *args: "Any", **kwargs: "Any") -> None:
        self.setup(*args, **kwargs)
        if self.force or not self.was_analysed(*args, **kwargs):
            self.set_seed()
            self.log.info(f"Analysing {self.name}")
            self.result(*args, **kwargs)
            self._analyse(*args, **kwargs)
            self.log.info(f"Finished analysing {self.name}")

    def has_attributes(self: Self, attributes: str | Iterable[str]) -> bool:
        if not isinstance(attributes, Iterable):
            attributes = [attributes]
        return all(hasattr(self, attr) for attr in attributes)

    def clear_attributes(self: Self, attributes: str | Iterable[str]) -> None:
        if not isinstance(attributes, Iterable):
            attributes = [attributes]
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

    def _clear(
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
        pass

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
        if not any((setup, run, save, load, result, analyse)):
            setup = True
            run = True
            save = True
            load = True
            result = True
            analyse = True

        self.set_seed()
        self.log.info(f"Clearing {self.name}")
        self._clear(
            *args,
            setup=setup,
            run=run,
            save=save,
            load=load,
            result=result,
            analyse=analyse,
            **kwargs,
        )
        self.log.info(f"Finished clearing {self.name}")

    def set_seed(self: Self, seed: int = 0) -> None:
        seed = self.seed + seed
        self.rng.bit_generator.state = type(self.rng.bit_generator)(seed).state

    # === Static Methods ===
