from typing import Any, Self, Protocol
from collections.abc import Callable

from pydantic import model_validator

from supaernova.utils import resolve_path

from .input import InputConfig


class CallbackFunc[Instance: Any, Returns](Protocol):
    def __call__(_self, self: Instance, *args: Any, **kwargs: Any) -> Returns: ...

    __name__: str


def callback[Instance: Any, Returns](
    fn: CallbackFunc[Instance, Returns],
) -> Callable[..., Returns]:
    def wrapper(self: Instance, *args: Any, **kwargs: Any) -> Returns:
        callbacks: dict[str, Callable[[Instance], None]] = self.callbacks.get(
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


class CallbackConfig(InputConfig):
    # === Class Variables ===
    # === Class Methods ===

    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    callbacks: dict[str, str | dict[str, Callable[[Any], None]]] = {}

    # === Model Validators ===
    # --- Before ---
    # --- After ---
    @model_validator(mode="after")
    def _set_callbacks(self) -> Self:
        for fn, callback in self.callbacks.items():
            if isinstance(callback, str):
                fn_callbacks = {}
                callback_path = resolve_path(callback, relative_path=self.paths.base)
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
        self.callbacks = self._normalise_config(self.callbacks)
        return self

    # === Field Validators ===
    # --- Before ---
    # --- After ---

    # === Instance Methods ===

    # === Static Methods ===
