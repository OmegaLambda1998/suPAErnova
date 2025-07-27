# Copyright 2025 Patrick Armstrong
from typing import Any
from logging import Logger
from pathlib import Path

import toml
from pydantic import FilePath, DirectoryPath

from supaernova.utils import deepmerge, resolve_path
from supaernova.logging import setup_logging

from .base import BaseConfig
from .paths import PathConfig
from .globals import GlobalConfig


class InputConfig(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    @classmethod
    def _default_config(cls, **input_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "log": setup_logging(
                input_config.get("name", cls.__name__),
                log_path=input_config["paths"].log,
                verbose=input_config["config"].verbose,
            ),
        }

    @classmethod
    def _extend_config(
        cls,
        input_config: dict[str, Any],
        base_path: Path | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        base_path = base_path or input_config.get("paths", {}).get("base")
        if "extends" in input_config and input_config["extends"] is not None:
            extends_path = resolve_path(
                input_config["extends"], relative_path=base_path
            )
            if not extends_path.exists():
                err = f"Extension Path: {extends_path} does not exist."
                cls._raise(err)
            if extends_path.suffix != ".toml":
                err = f"Extension Path: {extends_path} is not a `.toml` file."
                cls._raise(err)
            extension = toml.load(extends_path)
            extends_base_path = (
                extension.get("paths", {}).get("base") or extends_path.parent
            )
            if key is not None:
                extension = extension.get(key, {})
            for k, v in extension.items():
                if isinstance(v, dict):
                    extension[k] = cls._extend_config(
                        v,
                        base_path=extends_base_path,
                        key=k,
                    )
            input_config = deepmerge(extension, input_config)
            input_config["extends"] = extends_path
        if (
            "external" in input_config
            and not Path(input_config["external"]).is_absolute()
        ):
            input_config["external"] = base_path / input_config["external"]
        for k, v in input_config.items():
            if isinstance(v, dict):
                input_config[k] = cls._extend_config(
                    v,
                    base_path=base_path,
                    key=k,
                )
        return input_config

    # === Field Variables ===
    # --- Required ---
    config: GlobalConfig
    paths: PathConfig
    log: Logger
    # --- Optional ---
    extends: FilePath | None = None
    external: DirectoryPath | None = None

    # === Model Validators ===
    # --- Before ---
    # --- After ---

    # === Field Validators ===
    # --- Before ---
    # --- After ---

    # === Instance Methods ===
    def __init__(self, **input_config) -> None:
        config = self._extend_config(input_config)
        super().__init__(**config)
        self.save()

    def save(self) -> None:
        save_file = self.paths.log / f"{self.name}.toml"
        with save_file.open(
            "w",
            encoding="utf-8",
        ) as io:
            toml.dump(self.model_dump(exclude={"log"}), io)

    # === Static Methods ===
