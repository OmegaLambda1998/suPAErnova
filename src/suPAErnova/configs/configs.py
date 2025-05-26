# Copyright 2025 Patrick Armstrong

from typing import Any, Self, Never, Protocol
from logging import Logger
from pathlib import Path
from collections.abc import Callable

import toml
from pydantic import BaseModel, ConfigDict, model_validator

from suPAErnova.logging import setup_logging

from .paths import PathConfig
from .globals import GlobalConfig


class CallbackFunc[Instance: Any, Returns](Protocol):
    def __call__(_self, self: Instance, *args: Any, **kwargs: Any) -> Returns: ...

    __name__: str


def callback[Instance: Any, Returns](
    fn: CallbackFunc[Instance, Returns],
) -> Callable[..., Returns]:
    def wrapper(self: Instance, *args: Any, **kwargs: Any) -> Returns:
        callbacks: dict[str, Callable[[Instance], None]] = self.options.callbacks.get(
            fn.__name__.lower(), {}
        )
        pre_callback = callbacks.get("pre")
        if pre_callback is not None:
            pre_callback(self)
        rtn = fn(self, *args, **kwargs)
        post_callback = callbacks.get("post")
        if post_callback is not None:
            post_callback(self)
        return rtn

    return wrapper


class SNPAEConfig(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # pyright: ignore[reportIncompatibleVariableOverride]
    name: str

    @model_validator(mode="before")
    @classmethod
    def get_name(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["name"] = data.get("name", cls.__name__)
        return data

    @model_validator(mode="after")
    def set_name(self) -> Self:
        if self.__class__.__name__ not in self.name:
            self.name = f"{self.__class__.__name__} - {self.name}"
        self.__class__.name = self.name
        return self

    # Required
    config: GlobalConfig
    paths: PathConfig
    log: Logger

    # Optional
    callbacks: dict[str, str | dict[str, Callable[[Any], None]]] = {}

    @model_validator(mode="after")
    def validate_callbacks(self) -> Self:
        for fn, callback in self.callbacks.items():
            if isinstance(callback, str):
                fn_callbacks = {}
                callback_path = self.paths.resolve_path(
                    Path(callback), relative_path=self.paths.base
                )
                if not callback_path.exists():
                    err = f"`{fn}` callback: `{callback}` resolved to `{callback_path}`, which does not exist."
                    self._raise(err)
                with callback_path.open("r") as io:
                    script_code = io.read()
                # Create an isolated namespace
                local_scope = {}
                exec(script_code, globals(), local_scope)  # noqa: S102
                if "pre" in local_scope:
                    if not isinstance(local_scope["pre"], Callable):
                        err = f"`pre-{fn}` callback is not callable"
                        self._raise(err)
                    fn_callbacks["pre"] = local_scope["pre"]
                if "post" in local_scope:
                    if not isinstance(local_scope["post"], Callable):
                        err = f"`pre-{fn}` callback is not callable"
                        self._raise(err)
                    fn_callbacks["post"] = local_scope["post"]
                self.callbacks[fn] = fn_callbacks
        self.callbacks = self.normalise_input(self.callbacks)
        return self

    @classmethod
    def from_config(
        cls,
        input_config: dict[str, Any],
    ) -> Self:
        config = {**cls.default_config(input_config), **input_config}
        cfg = cls.model_validate(config)
        cfg.save()
        return cfg

    @classmethod
    def default_config(cls, input_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "log": setup_logging(
                input_config.get("name", cls.__name__),
                log_path=input_config["paths"].log,
                verbose=input_config["config"].verbose,
            ),
        }

    @classmethod
    def _raise(cls, err: str, error: type[Exception] = ValueError) -> Never:
        name = cls.name if hasattr(cls, "name") else cls.__name__
        err = f"{name}:\n{err}\n"
        raise error(err)

    def save(self) -> None:
        save_file = self.paths.log / f"{self.name}.toml"
        with save_file.open(
            "w",
            encoding="utf-8",
        ) as io:
            toml.dump(self.model_dump(exclude={"log"}), io)

    @staticmethod
    def normalise_input(input_config: dict[str, Any]) -> dict[str, Any]:
        rtn: dict[str, Any] = {}
        for k, v in input_config.items():
            val = v
            if isinstance(v, dict):
                val = SNPAEConfig.normalise_input(v)
            rtn[k.lower()] = val
        return rtn
