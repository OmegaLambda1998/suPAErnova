# Copyright 2025 Patrick Armstrong

from typing import Any

from pydantic import DirectoryPath

from .base import BaseConfig


class PathConfig(BaseConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    base: DirectoryPath
    out: DirectoryPath
    results: DirectoryPath
    plots: DirectoryPath
    log: DirectoryPath
    # --- Optional ---

    # === Model Validators ===
    # --- Before ---
    # --- After ---

    # === Field Validators ===
    # --- Before ---
    # --- After ---

    # === Instance Methods ===
