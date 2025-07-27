# Copyright 2025 Patrick Armstrong
from typing import Any, Self, Never

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


class BaseConfig(BaseModel):
    # === Class Variables ===
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # === Class Methods ===
    @classmethod
    def _normalise_config(cls, input_config: dict[str, Any]) -> dict[str, Any]:
        rtn: dict[str, Any] = {}
        for k, v in input_config.items():
            val = v
            if isinstance(v, dict):
                val = cls._normalise_config(v)
            rtn[k.lower()] = val
        return rtn

    @classmethod
    def _default_config(cls, **input_config: dict[str, Any]) -> dict[str, Any]:
        return {}

    @classmethod
    def _raise(cls, err: str, error: type[Exception] = ValueError) -> Never:
        name = cls.name if hasattr(cls, "name") else cls.__name__
        err = f"{name}:\n{err}\n"
        raise error(err)

    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    name: str = "MISSING"

    # === Model Validators ===
    # --- Before ---
    @model_validator(mode="before")
    @classmethod
    def _get_name(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["name"] = data.get("name", cls.__name__)
        return data

    # --- After ---
    @model_validator(mode="after")
    def _set_name(self) -> Self:
        self.__class__.name = self.name
        return self

    # === Field Validators ===
    # --- Before ---
    # --- After ---

    # === Instance Methods ===
    def __init__(self, **input_config) -> None:
        config = self._normalise_config({
            **self._default_config(**input_config),
            **input_config,
        })
        super().__init__(**config)

    def get(self, key, value=None):
        if hasattr(self, key):
            return getattr(self, key)
        return value

    # === Static Methods ===
