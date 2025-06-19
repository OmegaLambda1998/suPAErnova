# Copyright 2025 Patrick Armstrong

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar
from pathlib import Path
import pkgutil
import importlib

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

    @abstractmethod
    def _setup(self, *_args: "Any", **_kwargs: "Any") -> None:
        pass

    @callback
    def setup(self, *args: "Any", **kwargs: "Any") -> None:
        self.log.info(f"Setting up {self.name}")
        self._setup(*args, **kwargs)
        self.log.info(f"Finished setting up {self.name}")

    @abstractmethod
    def _completed(self) -> bool:
        pass

    @callback
    def completed(self) -> bool:
        self.log.debug(f"Checking if {self.name} has completed")
        completed = self._completed()
        self.log.debug(f"{self.name} has {'' if completed else 'not '}completed")
        return completed

    @abstractmethod
    def _load(self) -> None:
        pass

    @callback
    def load(self) -> None:
        self.log.info(f"Loading {self.name}")
        self._load()
        self.log.info(f"Finished loading {self.name}")

    @abstractmethod
    def _run(self) -> None:
        pass

    @callback
    def run(self) -> None:
        if self.force or not self.completed():
            self.log.info(f"Running {self.name}")
            self._run()
            self.log.info(f"Finished running {self.name}")
        else:
            self.load()

    @abstractmethod
    def _result(self) -> None:
        pass

    @callback
    def result(self) -> None:
        self.log.info(f"Saving {self.name} results")
        self._result()
        self.log.info(f"Finished saving {self.name} results")

    @abstractmethod
    def _analyse(self) -> None:
        pass

    @callback
    def analyse(self) -> None:
        self.log.info(f"Analysing {self.name}")
        self._analyse()
        self.log.info(f"Finished analysing {self.name}")
