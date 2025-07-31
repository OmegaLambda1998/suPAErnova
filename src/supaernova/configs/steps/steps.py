# Copyright 2025 Patrick Armstrong
from copy import deepcopy
from typing import Any, ClassVar
from pathlib import Path

from pydantic import ConfigDict, PositiveInt

from supaernova.utils import resolve_path
from supaernova.configs.base import BaseConfig
from supaernova.configs.callbacks import CallbackConfig


class StepResult(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    metadata: dict[str, Any] | None = None
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class StepAnalysis(BaseConfig):
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


class StepConfig(CallbackConfig):
    # === Class Variables ===
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    steps: ClassVar[dict[str, type["StepConfig"]]] = {}
    required_steps: ClassVar[list[str]] = []
    id: ClassVar[str] = ""

    # === Class Methods ===
    @classmethod
    def register_step(cls) -> None:
        cls.steps[cls.id] = cls

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
    def __init__(self, **input_config) -> None:
        input_config = deepcopy(input_config)
        input_config["paths"].results = resolve_path(
            input_config["paths"].results
            / input_config.get("name", self.__class__.__name__),
            relative_path=input_config["paths"].base,
        )

        input_config["paths"].plots = resolve_path(
            input_config["paths"].plots
            / input_config.get("name", self.__class__.__name__),
            relative_path=input_config["paths"].base,
            mkdir=True,
        )

        input_config["paths"].log = resolve_path(
            input_config["paths"].log
            / input_config.get("name", self.__class__.__name__),
            relative_path=input_config["paths"].base,
            mkdir=True,
        )

        if input_config.get("external"):
            external_path = resolve_path(
                Path(input_config["external"]), relative_path=input_config["paths"].base
            )
            input_config["external"] = external_path
            if not input_config["paths"].results.exists():
                input_config["paths"].results.symlink_to(
                    external_path, target_is_directory=True
                )
        else:
            input_config["paths"].results.mkdir(parents=True, exist_ok=True)
        super().__init__(**input_config)

    # === Static Methods ===
