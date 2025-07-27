from typing import Any, ClassVar, get_args

from pydantic import (
    Field,
    model_validator,
)

from .steps import StepConfig, StepResult, StepAnalysis


class VariantConfig(StepConfig):
    # === Class Variables ===
    variant_steps: ClassVar[dict[str, type["StepConfig"]]] = {}

    # === Class Methods ===
    @classmethod
    def register_step(cls, variant_cls: type[StepConfig]) -> None:
        cls.variant_steps[cls.id] = variant_cls
        super().register_step()

    # === Field Variables ===
    # --- Required ---
    base: StepConfig

    # --- Optional ---
    variants: list[StepConfig] | None = Field(None, validation_alias="variant")

    # === Model Validators ===
    # --- Before ---
    @model_validator(mode="before")
    @classmethod
    def _prepare_configs(cls, data: Any, *, validate: bool = True) -> Any:
        if isinstance(data, dict):
            if "base" not in data:
                err = f"No base {cls.id} config has been defined. Please define one in [{cls.id}.base]"
                cls._raise(err)
            if isinstance(data["base"], VariantConfig):
                data["variant"] = data["variants"]
                data.pop("variants", None)
            else:
                default_config = {
                    "paths": data.get("paths"),
                    "config": data.get("config"),
                    "log": data.get("log"),
                }
                base_config = {**default_config, **data.get("base", {})}
                data["base"] = base_config

                variant_configs = [
                    data["base"],
                    *[
                        {**base_config, **variant_config}
                        for variant_config in data.get("variant") or []
                    ],
                ]
                if validate:
                    data["variant"] = [
                        cls.variant_steps[cls.id](**variant_config)
                        for variant_config in variant_configs
                    ]
                else:
                    data["variant"] = variant_configs
        return data

    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===
