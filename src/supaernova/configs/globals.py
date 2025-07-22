# Copyright 2025 Patrick Armstrong

from typing import Any, Self

from pydantic import (
    BaseModel,
    JsonValue,
    StrictBool,
)


class GlobalConfig(BaseModel):
    verbose: StrictBool
    force: StrictBool

    @classmethod
    def from_config(
        cls,
        input_config: dict[str, JsonValue],
        *,  # Force keyword-only arguments
        verbose: StrictBool = False,
        force: StrictBool = False,
    ) -> Self:
        config = {**cls.default_config(verbose=verbose, force=force), **input_config}
        return cls.model_validate(config)

    @classmethod
    def default_config(
        cls,
        *,  # Force keyword-only arguments
        verbose: StrictBool = False,
        force: StrictBool = False,
    ) -> dict[str, Any]:
        return {"verbose": verbose, "force": force}
