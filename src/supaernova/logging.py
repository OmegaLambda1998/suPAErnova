# Copyright 2025 Patrick Armstrong

import sys
from typing import TYPE_CHECKING
import logging

import coloredlogs

if TYPE_CHECKING:
    from pathlib import Path


def setup_logging(
    module: str,
    *,  # Force keyword-only arguments
    log_path: "Path | None" = None,
    verbose: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear existing handlers
    while logger.handlers:
        logger.handlers.pop()

    # --- Formatting ---
    debug_fmt = "[%(levelname)8s] %(filename)10s | %(message)s"
    info_fmt = "%(message)s"
    level_styles = coloredlogs.parse_encoded_styles(
        "debug=8;info=green;warning=yellow;error=red,bold;critical=red,inverse"
    )

    # --- Stream Handler ---
    # Set level and formatting
    stream_level = logging.DEBUG if verbose else logging.INFO
    stream_fmt = coloredlogs.ColoredFormatter(
        debug_fmt if verbose else info_fmt, level_styles=level_styles
    )

    # Initialise file handler
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(stream_fmt)
    logger.addHandler(stream_handler)

    # --- File Handler ---
    if log_path is not None:
        # Determine output log file
        if log_path.is_dir():
            log_path /= f"{module}.log"

        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set level and formatting
        file_level = logging.DEBUG
        file_fmt = logging.Formatter(debug_fmt)

        # Initialise file handler
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger
