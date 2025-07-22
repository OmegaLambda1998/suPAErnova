from typing import Any, TypeVar, ClassVar, get_args
from collections.abc import Callable

from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    model_validator,
)

from .steps import StepConfig
from .backends import BACKENDS, BACKENDS_STR, AbstractModelConfig


class AbstractModelStepConfig[Backend: str, ModelConfig: AbstractModelConfig](
    StepConfig
):
    model_backend: ClassVar[dict[str, Callable[[], type[ModelConfig]]]]

    # --- Models ---
    model: ModelConfig
    models: list[ModelConfig] | None = Field(None, validation_alias="variant")

    @model_validator(mode="before")
    @classmethod
    def prep_model_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "model" not in data:
                err = f"No Base Model has been defined. Please define one in [{cls.id}.model]"
                raise ValueError(err)

            if isinstance(data["model"], AbstractModelConfig):
                data["variant"] = data["models"]
                data.pop("models", None)
            else:
                default_model_config = {
                    "paths": data.get("paths"),
                    "config": data.get("config"),
                    "log": data.get("log"),
                }
                base_model_config = {**default_model_config, **data.get("model", {})}
                data["model"] = base_model_config

                model_configs = [
                    data["model"],
                    *[
                        {**base_model_config, **model_config}
                        for model_config in data.get("variant") or []
                    ],
                ]
                data.pop("variant", None)
                data["variant"] = []
                for i, model_config in enumerate(model_configs):
                    backend = model_config.get("backend")
                    if backend is None:
                        err = f"{'Base' if i == 0 else f'Variant {i}'} Model is missing a backend key. Please choose from {BACKENDS_STR}"
                        raise ValueError(err)
                    model_config_cls = None
                    for backend_name in BACKENDS:
                        if backend in get_args(BACKENDS[backend_name]):
                            model_config_cls = cls.model_backend[backend_name]
                    if model_config_cls is None:
                        err = f"Unknown backend: {backend}. Please choose from {BACKENDS_STR}"
                        raise ValueError(err)
                    model_variant = model_config_cls().from_config(model_config)
                    data["variant"].append(model_variant)
        return data
