# Copyright 2025 Patrick Armstrong

from pydantic import StrictBool

from supaernova.typing import T, Config

from .base import BaseConfig


class GlobalConfig(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    @classmethod
    def _default_config(
        cls,
        **input_config: T,
    ) -> Config[T]:
        return {
            "verbose": input_config.get("verbose", False),
            "force": input_config.get("force", False),
        }

    # === Field Variables ===
    # --- Required ---
    verbose: StrictBool
    clean: StrictBool
    debug: StrictBool
    force: StrictBool

    # --- Optional ---

    # === Model Validators ===
    # --- Before ---
    # --- After ---

    # === Field Validators ===
    # --- Before ---
    # --- After ---

    # === Instance Methods ===

    # === Static Methods ===
