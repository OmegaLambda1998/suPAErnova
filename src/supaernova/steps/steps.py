# Copyright 2025 Patrick Armstrong
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self, ClassVar
from pathlib import Path
import pkgutil
import importlib

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

        self.is_setup = False
        self.is_completed = False
        self.is_loaded = False
        self.has_results = False
        self.was_analysed = False
        self.results: StepResult
        self.analysis: StepAnalysis

    @abstractmethod
    def _setup(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def setup(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_setup:
            self.set_seed()
            self.log.info(f"Setting up {self.name}")
            self._setup(*args, **kwargs)
            self.log.info(f"Finished setting up {self.name}")

    @abstractmethod
    def _completed(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        pass

    @callback
    def completed(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        if not self.is_completed:
            self.set_seed()
            self.setup(*args, **kwargs)
            self.log.debug(f"Checking if {self.name} has completed")
            self.is_completed = self._completed(*args, **kwargs)
            self.log.debug(
                f"{self.name} has {'' if self.is_completed else 'not '}completed"
            )
        return self.is_completed

    @abstractmethod
    def _load(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def load(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_loaded:
            if self.completed(*args, **kwargs):
                self.set_seed()
                self.setup(*args, **kwargs)
                self.log.info(f"Loading {self.name}")
                self._load(*args, **kwargs)
                self.log.info(f"Finished loading {self.name}")
            else:
                self.set_seed()
                self.run(*args, *kwargs)

    @abstractmethod
    def _run(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def run(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.is_loaded and (self.force or not self.completed(*args, **kwargs)):
            self.set_seed()
            self.setup(*args, **kwargs)
            self.log.info(f"Running {self.name}")
            self._run(*args, **kwargs)
            self.log.info(f"Finished running {self.name}")

    @abstractmethod
    def _result(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def result(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.has_results:
            self.set_seed()
            self.load(*args, **kwargs)
            self.log.info(f"Gathering {self.name} results")
            self._result(*args, **kwargs)
            self.log.info(f"Finished gathering {self.name} results")

    @abstractmethod
    def _analyse(self: Self, *args: "Any", **kwargs: "Any") -> None:
        pass

    @callback
    def analyse(self: Self, *args: "Any", **kwargs: "Any") -> None:
        if not self.was_analysed:
            self.set_seed()
            self.result(*args, **kwargs)
            self.log.info(f"Analysing {self.name}")
            self._analyse(*args, **kwargs)
            self.log.info(f"Finished analysing {self.name}")

    def clear_attributes(self: Self, attributes: str | list[str]) -> None:
        if not isinstance(attributes, list):
            attributes = [attributes]
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

    @abstractmethod
    def _clear(
        self: Self,
        *args: "Any",
        setup: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        complete: bool = False,
        **kwargs: "Any",
    ) -> None:
        if complete:
            setup = True
            load = True
            result = True
            analyse = True
        if setup:
            self.is_setup = False
        if load:
            self.is_loaded = False
        if result:
            self.has_results = False
        if analyse:
            self.was_analysed = False

    @callback
    def clear(
        self: Self,
        *args: "Any",
        setup: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        complete: bool = False,
        **kwargs: "Any",
    ) -> None:
        self.set_seed()
        self.log.info(f"Clearing {self.name}")
        self._clear(
            *args,
            setup=setup,
            load=load,
            result=result,
            analyse=analyse,
            complete=complete,
            **kwargs,
        )
        self.log.info(f"Finished clearing {self.name}")

    def set_seed(self: Self, seed: int = 0) -> None:
        seed = self.seed + seed
        self.rng.bit_generator.state = type(self.rng.bit_generator)(seed).state

    # === Static Methods ===
