# Copyright 2025 Patrick Armstrong

from typing import Any

from pydantic import StrictBool

from .base import BaseConfig


class GlobalConfig(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    @classmethod
    def _default_config(
        cls,
        **input_config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "verbose": input_config.get("verbose", False),
            "force": input_config.get("force", False),
        }

    # === Field Variables ===
    # --- Required ---
    verbose: StrictBool
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
