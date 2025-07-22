# Copyright 2025 Patrick Armstrong

from typing import Any, Self
from pathlib import Path

from pydantic import (
    BaseModel,
    JsonValue,
    DirectoryPath,
)


class PathConfig(BaseModel):
    base: DirectoryPath = Path.cwd()
    out: DirectoryPath
    log: DirectoryPath

    @classmethod
    def from_config(
        cls,
        input_config: dict[str, JsonValue],
        *,  # Force keyword-only arguments
        base_path: DirectoryPath,
        out_path: DirectoryPath,
        log_path: DirectoryPath,
    ) -> Self:
        config = {
            **cls.default_config(
                base_path=base_path, out_path=out_path, log_path=log_path
            ),
            **input_config,
        }
        return cls.model_validate(config)

    @classmethod
    def default_config(
        cls,
        *,  # Force keyword-only arguments
        base_path: DirectoryPath,
        out_path: DirectoryPath,
        log_path: DirectoryPath,
    ) -> dict[str, Any]:
        return {"base": base_path, "out": out_path, "log": log_path}

    @staticmethod
    def resolve_path(
        input_path: Path | None = None,
        *,
        default_path: Path | None = None,
        relative_path: Path,
        mkdir: bool = False,
    ) -> Path:
        if input_path is None:
            if default_path is None:
                err = "Cannot resolve `input_path=None` with `default_path=None"
                raise ValueError(err)
            input_path = default_path
        if not input_path.is_absolute():
            input_path = relative_path / input_path
        final_path = input_path.resolve()
        if mkdir:
            final_path.mkdir(parents=True, exist_ok=True)
        return final_path
