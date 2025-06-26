# Copyright 2025 Patrick Armstrong
import sys
from typing import TYPE_CHECKING, cast
from pathlib import Path
import traceback
import contextlib

from pydantic import ValidationError
from tqdm.contrib.logging import logging_redirect_tqdm

from .steps import SNPAEStep
from .configs import SNPAEConfig
from .logging import setup_logging
from .configs.input import InputConfig
from .configs.paths import PathConfig
from .configs.steps import StepConfig
from .configs.globals import GlobalConfig

if TYPE_CHECKING:
    from pydantic import JsonValue

def prepare_config(
    input_config: dict[str, "JsonValue"],
    *,  # Force keyword-only arguments
    verbose: bool = False,
    force: bool = False,
    base_path: Path | None = None,
    out_path: Path | None = None,
) -> InputConfig:
    # Normalise input_config
    user_config = SNPAEConfig.normalise_input(input_config)

    # Setup global config
    user_config["config"] = GlobalConfig.from_config(
        cast("dict[str, JsonValue]", user_config.get("config", {})),
        verbose=verbose,
        force=force,
    )

    # Setup  paths config
    base_path = PathConfig.resolve_path(
        base_path,
        default_path=Path.cwd(),
        relative_path=Path.cwd(),
    )
    out_path = PathConfig.resolve_path(
        out_path,
        default_path=base_path / "output",
        relative_path=base_path,
        mkdir=True,
    )
    log_path = PathConfig.resolve_path(
        out_path / "logs",
        default_path=out_path / "logs",
        relative_path=base_path,
        mkdir=True,
    )
    user_config["paths"] = PathConfig.from_config(
        cast("dict[str, JsonValue]", user_config.get("paths", {})),
        base_path=base_path,
        out_path=out_path,
        log_path=log_path,
    )

    # Propagate global and paths to steps
    SNPAEStep.register_steps()
    for step, step_config in StepConfig.steps.items():
        if step in user_config:
            user_config[step] = step_config.from_config({
                "config": user_config["config"],
                "paths": user_config["paths"],
                **user_config[step],
            })

    return InputConfig.from_config(user_config)


def main(
    input_config: dict[str, "JsonValue"],
    *,  # Force keyword-only arguments
    verbose: bool = False,
    force: bool = False,
    base_path: Path | None = None,
    out_path: Path | None = None,
) -> None:
    log = setup_logging(__name__, verbose=verbose)
    log.info("Started SuPAErnova")

    try:
        # Setup context based on logging verbosity
        #   If verbose, redirect logging stdout through tqdm
        #   Otherwise use a nullcontext (which does nothing)
        cm = logging_redirect_tqdm() if verbose else contextlib.nullcontext()
        with cm:
            config = prepare_config(
                input_config,
                verbose=verbose,
                force=force,
                base_path=base_path,
                out_path=out_path,
            )
            config.run()
    except ValidationError as e:
        log.error(e)  # noqa: TRY400
        sys.exit(1)
    except Exception:
        log.exception(traceback.format_exc())
        sys.exit(1)
