# Copyright 2025 Patrick Armstrong
import os
from abc import abstractmethod
import random as rn
from typing import TYPE_CHECKING, ClassVar
from pathlib import Path
import pkgutil
import importlib

import numpy as np

from suPAErnova.configs import callback

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger

    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.steps import StepConfig
    from suPAErnova.configs.globals import GlobalConfig


class SNPAEStep:
    # Class Variables
    steps: ClassVar[dict[str, type["SNPAEStep"]]] = {}
    id: ClassVar[str]

    @classmethod
    def register_step(cls) -> None:
        cls.steps[cls.id] = cls

    @staticmethod
    def register_steps() -> None:
        base_name = ".".join(
            __name__.split(".")[:-1]
        )  # Remove the last duplicated part
        for _, module, is_pkg in pkgutil.iter_modules([str(Path(__file__).parent)]):
            if is_pkg:
                importlib.import_module(f"{base_name}.{module}")

    def __init__(self, config: "StepConfig") -> None:
        # Class Variables
        self.__class__.id = config.__class__.id
        self.name: str = (
            config.name
            if config.name != config.__class__.__name__
            else self.__class__.__name__
        ).replace("Config", "")

        # Init Variables
        self.options: StepConfig = config
        self.config: GlobalConfig = config.config
        self.paths: PathConfig = config.paths
        self.log: Logger = config.log
        self.force: bool = self.config.force
        self.verbose: bool = self.config.verbose

        self.seed: int = 0
        self.set_seed()
        self.is_setup: bool = False
        self.is_loaded: bool = False
        self.is_saved: bool = False

    @abstractmethod
    def _setup(self, *_args: "Any", **_kwargs: "Any") -> None:
        pass

    @callback
    def setup(self, *args: "Any", **kwargs: "Any") -> None:
        self.set_seed()
        self.log.info(f"Setting up {self.name}")
        self._setup(*args, **kwargs)
        self.is_setup = True
        self.log.info(f"Finished setting up {self.name}")

    @abstractmethod
    def _completed(self) -> bool:
        pass

    @callback
    def completed(self, *args: "Any", **kwargs: "Any") -> bool:
        self.set_seed()
        if not self.is_setup:
            self.setup(*args, **kwargs)
        self.log.debug(f"Checking if {self.name} has completed")
        completed = self._completed()
        self.log.debug(f"{self.name} has {'' if completed else 'not '}completed")
        self.is_completed = completed
        return completed

    @abstractmethod
    def _load(self) -> None:
        pass

    @callback
    def load(self, *args: "Any", **kwargs: "Any") -> None:
        self.set_seed()
        if not self.is_setup:
            self.setup(*args, **kwargs)
        if self.completed(*args, **kwargs):
            self.log.info(f"Loading {self.name}")
            self._load()
            self.is_loaded = True
            self.log.info(f"Finished loading {self.name}")
        else:
            self.run(*args, *kwargs)

    @abstractmethod
    def _run(self) -> None:
        pass

    @callback
    def run(self, *args: "Any", **kwargs: "Any") -> None:
        self.set_seed()
        if not self.is_setup:
            self.setup(*args, **kwargs)
        if self.force or not self.completed(*args, **kwargs):
            self.log.info(f"Running {self.name}")
            self._run()
            self.is_loaded = True
            self.log.info(f"Finished running {self.name}")
        else:
            self.load(*args, **kwargs)

    @abstractmethod
    def _result(self) -> None:
        pass

    @callback
    def result(self, *args: "Any", **kwargs: "Any") -> None:
        self.set_seed()
        if not self.is_loaded:
            self.load(*args, **kwargs)
        self.log.info(f"Saving {self.name} results")
        self._result()
        self.is_saved = True
        self.log.info(f"Finished saving {self.name} results")

    @abstractmethod
    def _analyse(self) -> None:
        pass

    @callback
    def analyse(self, *args: "Any", **kwargs: "Any") -> None:
        self.set_seed()
        if not self.is_saved:
            self.result(*args, **kwargs)
        self.log.info(f"Analysing {self.name}")
        self._analyse()
        self.log.info(f"Finished analysing {self.name}")

    def set_seed(self, seed: int = 0) -> None:
        seed = self.seed + seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        rn.seed(seed)
