# Copyright 2025 Patrick Armstrong
from copy import deepcopy
from typing import ClassVar, cast
from pathlib import Path

from pydantic import ConfigDict, PositiveInt

from supaernova.utils import resolve_path
from supaernova.typing import T
from supaernova.configs.base import BaseConfig
from supaernova.configs.paths import PathConfig
from supaernova.configs.callbacks import CallbackConfig


class StepResult(BaseConfig):
    pass
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class StepAnalysis(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    force: bool = False
    skip: bool = True
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class StepConfig(CallbackConfig):
    # === Class Variables ===
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    steps: ClassVar[dict[str, type["StepConfig"]]] = {}
    proxies: ClassVar[dict[str, list[type["StepConfig"]]]] = {}
    required_steps: ClassVar[list[str]] = []
    id: ClassVar[str] = ""

    # === Class Methods ===
    @classmethod
    def register_step(cls) -> None:
        cls.steps[cls.id] = cls

    @classmethod
    def register_proxy(cls, proxy: type["StepConfig"]) -> None:
        if cls.id not in cls.proxies:
            cls.proxies[cls.id] = []
        cls.proxies[cls.id].append(proxy)

    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    skip: bool = False
    seed: PositiveInt = 12345
    analysis: StepAnalysis | None = None

    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    def __init__(self, **input_config: T) -> None:
        input_config = deepcopy(input_config)
        paths = input_config.get("paths")
        if paths is None:
            self._raise(f'Input Config: {input_config} missing "paths"')
        paths = cast("PathConfig", paths)

        paths.results = resolve_path(
            paths.results / input_config.get("name", self.__class__.__name__),
            relative_path=paths.base,
        )

        paths.plots = resolve_path(
            paths.plots / input_config.get("name", self.__class__.__name__),
            relative_path=paths.base,
            mkdir=True,
        )

        paths.log = resolve_path(
            paths.log / input_config.get("name", self.__class__.__name__),
            relative_path=paths.base,
            mkdir=True,
        )

        if input_config.get("external"):
            external_path = resolve_path(
                Path(input_config["external"]), relative_path=paths.base
            )
            input_config["external"] = external_path
            if not external_path.exists():
                external_path.mkdir(parents=True, exist_ok=True)
            if not paths.results.exists():
                rel_path = external_path.relative_to(paths.results.parent, walk_up=True)
                paths.results.symlink_to(rel_path, target_is_directory=True)
        else:
            paths.results.mkdir(parents=True, exist_ok=True)
        input_config["paths"] = paths
        super().__init__(**input_config)

    # === Static Methods ===
