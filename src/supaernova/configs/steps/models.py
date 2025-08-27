from typing import Any, Self, Literal, ClassVar, get_args
from collections.abc import Callable

from pydantic import model_validator

from .steps import StepConfig
from .variants import VariantConfig

TFBackend = Literal["tf"]
Backend = TFBackend

BACKENDS: dict[str, Backend] = {"TensorFlow": TFBackend}
BACKENDS_STR = ", ".join(
    f'`"{get_args(B)[0]}" for {backend}`' for backend, B in BACKENDS.items()
)


class BackendConfig(StepConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    backend: Backend
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class ModelConfig(VariantConfig):
    # === Class Variables ===
    model_backend: ClassVar[dict[str, Callable[[], type["ModelConfig"]]]]

    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    @model_validator(mode="before")
    @classmethod
    def _prepare_configs(cls: type[Self], data: Any, *, validate: bool = True) -> Any:
        data = super()._prepare_configs(data, validate=False)
        if isinstance(data, dict):
            variant_configs = []
            for i, variant_config in enumerate(data["variant"]):
                backend = variant_config.get("backend")
                if backend is None:
                    key = "Base" if i == 0 else f"Variant {i}"
                    loc = (
                        f"[{cls.id}.base.backend] = {BACKENDS_STR}"
                        if i == 0
                        else f"the {i}th [[{cls.id}.variant.backed]] = {BACKENDS_STR}"
                    )
                    err = f"{key} config is missing a backend key. Please define it in {loc}."
                    cls._raise(err)
                config_cls = None
                for backend_name, backend_args in BACKENDS.items():
                    if backend in get_args(backend_args):
                        config_cls = cls.model_backend[backend_name]()
                if config_cls is None:
                    err = f"Unknown backend: {backend}. Please choose one from {BACKENDS_STR}"
                    cls._raise(err)
                elif validate:
                    variant_configs.append(config_cls(**variant_config))
                else:
                    variant_configs.append(variant_config)
            data["variant"] = variant_configs
        return data

    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===
