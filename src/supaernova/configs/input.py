# Copyright 2025 Patrick Armstrong
from typing import Self, Literal, cast
from logging import Logger
from pathlib import Path

import toml
from pydantic import FilePath, DirectoryPath

from supaernova.utils import deepmerge, resolve_path
from supaernova.typing import T, Config
from supaernova.logging import setup_logging

from .base import BaseConfig
from .paths import PathConfig
from .globals import GlobalConfig


class InputConfig(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    @classmethod
    def _default_config(cls: type[Self], **input_config: T) -> Config[T]:
        paths = input_config.get("paths")
        if paths is None:
            cls._raise(f'Input Config: {input_config} missing "paths"')
        paths = cast("PathConfig", paths)

        config = input_config.get("config")
        if config is None:
            cls._raise(f'Input Config: {input_config} missing "config"')
        config = cast("GlobalConfig", config)

        return {
            "log": setup_logging(
                input_config.get("name", cls.__name__),
                log_path=paths.log,
                verbose=config.verbose,
            ),
        }

    @classmethod
    def _extend_config(
        cls: type[Self],
        input_config: Config[T],
        base_path: Path | None = None,
        key: str | None = None,
    ) -> Config[T]:
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
            input_config.get("external")
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
    external: DirectoryPath | Literal[False] = False

    # === Model Validators ===
    # --- Before ---
    # --- After ---

    # === Field Validators ===
    # --- Before ---
    # --- After ---

    # === Instance Methods ===
    def __init__(self: Self, **input_config: T) -> None:
        config = self._extend_config(input_config)
        super().__init__(**config)
        self.save()

    def save(self: Self) -> None:
        save_file = self.paths.log / f"{self.name}.toml"
        with save_file.open(
            "w",
            encoding="utf-8",
        ) as io:
            toml.dump(self.model_dump(exclude={"log"}), io)

    # === Static Methods ===
