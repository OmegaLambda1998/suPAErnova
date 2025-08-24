from typing import Any, Self, Literal, TypeVar, Protocol, ParamSpec
from collections.abc import Callable

from pydantic import BaseModel, model_validator

from supaernova.utils import resolve_path

from .input import InputConfig

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


class Callbackable(Protocol):
    def __call__(
        _self: Self, self: "CallbackConfig", *args: Param.args, **kwargs: Param.kwargs
    ) -> RetType: ...

    __name__: str


def callback(fn: Callbackable) -> Callbackable:
    def wrapper(
        self: "CallbackConfig", *args: Param.args, **kwargs: Param.kwargs
    ) -> RetType:
        callbacks_spec: list[CallbackSpec] | None = (self.callbacks or {}).get(
            fn.__name__.lower()
        )
        if callbacks_spec is None:
            return fn(self, *args, **kwargs)

        pre_cbs = []
        post_cbs = []

        for cb_spec in callbacks_spec:
            cb = cb_spec.callback
            cb_args = cb_spec.args or []
            cb_kwargs = cb_spec.kwargs or {}
            if isinstance(cb, dict):
                if "pre" in cb:
                    pre_cbs.append((cb["pre"], cb_args, cb_kwargs))
                if "post" in cb:
                    post_cbs.append((cb["post"], cb_args, cb_kwargs))

        for pre_callback, cb_args, cb_kwargs in pre_cbs:
            self.log.debug(
                f"Executing {self.name} {fn.__name__.lower()} pre(*{cb_args = }, **{cb_kwargs = })"
            )
            pre_callback(self, *cb_args, **cb_kwargs)
        rtn = fn(self, *args, **kwargs)
        for post_callback, cb_args, cb_kwargs in post_cbs:
            self.log.debug(
                f"Executing {self.name} {fn.__name__.lower()} post(*{cb_args = }, **{cb_kwargs = })"
            )
            post_callback(self, *cb_args, **cb_kwargs)
        return rtn

    return wrapper


class CallbackSpec(BaseModel):
    callback: str | dict[Literal["pre", "post"], Callable[[Any], None]]
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None


class CallbackConfig(InputConfig):
    # === Class Variables ===
    # === Class Methods ===

    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    callbacks: dict[str, list[CallbackSpec]] | None = None

    # === Model Validators ===
    # --- Before ---
    # --- After ---
    @model_validator(mode="after")
    def _set_callbacks(self: Self) -> Self:
        if not isinstance(self.callbacks, dict):
            return self
        for fn, callbacks_spec in self.callbacks.items():
            cb_specs = []
            for cb in callbacks_spec:
                callbacks = cb.callback
                if isinstance(callbacks, str):
                    fn_callbacks = {}
                    callbacks_path = resolve_path(
                        callbacks, relative_path=self.paths.base
                    )
                    if not callbacks_path.exists():
                        err = f"`{fn}` callbacks: `{callbacks}` resolved to `{callbacks_path}`, which does not exist."
                        self._raise(err)
                    with callbacks_path.open("r") as io:
                        script_code = io.read()
                    # Create an isolated namespace
                    local_scope = {}
                    exec(script_code, globals(), local_scope)  # noqa: S102
                    if "pre" in local_scope:
                        if not isinstance(local_scope["pre"], Callable):
                            err = f"`pre-{fn}` callbacks is not callable"
                            self._raise(err)
                        fn_callbacks["pre"] = local_scope["pre"]
                    if "post" in local_scope:
                        if not isinstance(local_scope["post"], Callable):
                            err = f"`pre-{fn}` callbacks is not callable"
                            self._raise(err)
                        fn_callbacks["post"] = local_scope["post"]
                    cb.callback = fn_callbacks
                    cb_specs.append(cb)
                else:
                    cb_specs.append(cb)
            self.callbacks[fn] = cb_specs
        self.callbacks = self._normalise_config(self.callbacks)
        return self

    # === Field Validators ===
    # --- Before ---
    # --- After ---

    # === Instance Methods ===

    # === Static Methods ===
