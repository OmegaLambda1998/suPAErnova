# Copyright 2025 Patrick Armstrong

import sys
from time import time
from typing import TYPE_CHECKING
from pathlib import Path
import traceback

from pydantic import ValidationError
from tqdm.contrib.logging import logging_redirect_tqdm

from .steps import Step
from .utils import resolve_path
from .logging import setup_logging
from .configs.run import RunConfig
from .configs.input import InputConfig
from .configs.paths import PathConfig
from .configs.steps import StepConfig
from .configs.globals import GlobalConfig

if TYPE_CHECKING:
    from pydantic import JsonValue

    from .typing import Config


def prepare_config(
    config: "Config[JsonValue]",
    *,  # Force keyword-only arguments
    verbose: bool = False,
    force: bool = False,
    base_path: Path | None = None,
    out_path: Path | None = None,
    result_path: Path | None = None,
    plot_path: Path | None = None,
    log_path: Path | None = None,
) -> "RunConfig":
    config = InputConfig._extend_config(config, base_path=base_path)

    # Setup global config
    config["config"] = GlobalConfig(
        verbose=config.get("config", {}).get("verbose", verbose),
        force=config.get("config", {}).get("force", force),
    )

    # Setup paths config
    base_path = resolve_path(
        base_path or config.get("paths", {}).get("base"),
        default_path=Path.cwd(),
        relative_path=Path.cwd(),
    )
    out_path = resolve_path(
        out_path or config.get("paths", {}).get("out"),
        default_path=base_path / "output",
        relative_path=base_path,
        mkdir=True,
    )
    result_path = resolve_path(
        result_path or config.get("paths", {}).get("results"),
        default_path=out_path / "results",
        relative_path=out_path,
        mkdir=True,
    )
    plot_path = resolve_path(
        plot_path or config.get("paths", {}).get("plot"),
        default_path=out_path / "plots",
        relative_path=out_path,
        mkdir=True,
    )
    log_path = resolve_path(
        log_path or config.get("paths", {}).get("log"),
        default_path=out_path / "logs",
        relative_path=out_path,
        mkdir=True,
    )
    config["paths"] = PathConfig(
        base=base_path,
        out=out_path,
        results=result_path,
        plots=plot_path,
        log=log_path,
    )

    # Propagate global and paths to steps
    Step.register_steps()
    for step, step_config in StepConfig.steps.items():
        if step in config:
            config[step] = step_config(
                config=config["config"],
                paths=config["paths"],
                **config[step],
            )

    return RunConfig(**config)


def main(
    input_config: "Config[JsonValue]",
    *,  # Force keyword-only arguments
    verbose: bool = False,
    force: bool = False,
    base_path: Path | None = None,
    out_path: Path | None = None,
    result_path: Path | None = None,
    plot_path: Path | None = None,
    log_path: Path | None = None,
) -> None:
    log = setup_logging(__name__, verbose=verbose)
    log.info("Started SuPAErnova")
    try:
        with logging_redirect_tqdm():
            config = prepare_config(
                input_config,
                verbose=verbose,
                force=force,
                base_path=base_path,
                out_path=out_path,
                result_path=result_path,
                plot_path=plot_path,
                log_path=log_path,
            )
            start_time = time()
            config.run()
            end_time = time()
            exec_time = end_time - start_time
            unit = "s"
            t = 60
            if exec_time > t:
                exec_time /= t
                unit = "m"
            if exec_time > t:
                exec_time /= t
                unit = "h"
            log.info(f"SuPAErnovae took {exec_time:.2f}{unit}")
    except ValidationError as e:
        log.error(e)  # noqa: TRY400
        sys.exit(1)
    except Exception:
        log.exception(traceback.format_exc())
        sys.exit(1)
