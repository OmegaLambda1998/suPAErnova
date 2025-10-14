# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING
from pathlib import Path

import toml
import click

from . import main
from .utils import resolve_path

if TYPE_CHECKING:
    from pydantic import JsonValue

    from supaernova.typing import Config


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, type=bool, default=False)
@click.option("-c", "--clean", is_flag=True, type=bool, default=False)
@click.option("-d", "--debug", is_flag=True, type=bool, default=False)
@click.option("-f", "--force", is_flag=True, type=bool, default=False)
@click.option(
    "-b",
    "--base_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "-o",
    "--out_path",
    type=click.Path(exists=False, path_type=Path),
    default=None,
)
@click.option(
    "-r",
    "--result_path",
    type=click.Path(exists=False, path_type=Path),
    default=None,
)
@click.option(
    "-p",
    "--plot_path",
    type=click.Path(exists=False, path_type=Path),
    default=None,
)
@click.option(
    "-l",
    "--log_path",
    type=click.Path(exists=False, path_type=Path),
    default=None,
)
def cli(
    input_path: Path,
    *,  # Force keyword-only arguments
    verbose: bool = False,
    clean: bool = False,
    debug: bool = False,
    force: bool = False,
    base_path: Path | None = None,
    out_path: Path | None = None,
    result_path: Path | None = None,
    plot_path: Path | None = None,
    log_path: Path | None = None,
) -> None:
    input_config: Config[JsonValue] = toml.load(input_path)

    # Set base_path to input_path.parent if none provided
    base_path = resolve_path(
        base_path,
        default_path=input_path.parent,
        relative_path=Path.cwd(),
    )

    return main(
        input_config,
        verbose=verbose,
        clean=clean,
        debug=debug,
        force=force,
        base_path=base_path,
        out_path=out_path,
        result_path=result_path,
        plot_path=plot_path,
        log_path=log_path,
    )


if __name__ == "__main__":
    cli()
