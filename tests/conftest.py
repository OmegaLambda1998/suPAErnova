from typing import TYPE_CHECKING

import pytest

from .fixtures.paths import data_path, root_path, cache_path, tests_path, fixtures_path
from .fixtures.pae.snpae import snpae_pae_step_factory, snpae_pae_result_factory
from .fixtures.data.snpae import snpae_data_step_factory, snpae_data_result_factory
from .fixtures.pae.legacy import legacy_pae_step_factory, legacy_pae_result_factory
from .fixtures.data.legacy import legacy_data_step_factory, legacy_data_result_factory
from .fixtures.nflow.snpae import snpae_nflow_step_factory, snpae_nflow_result_factory
from .fixtures.nflow.legacy import (
    legacy_nflow_step_factory,
    legacy_nflow_result_factory,
)
from .fixtures.posterior.snpae import (
    snpae_posterior_step_factory,
    snpae_posterior_result_factory,
)
from .fixtures.posterior.legacy import (
    legacy_posterior_step_factory,
    legacy_posterior_result_factory,
)

if TYPE_CHECKING:
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_configure(config: "Config") -> None:
    config.addinivalue_line(
        "markers",
        'setup("snpae"|"legacy_snpae"): Setup cached data without running any tests',
    )

    config.addinivalue_line(
        "markers",
        "paper_parity: Paper parity tests",
    )

    config.addinivalue_line(
        "markers",
        "data: Data Step tests",
    )

    config.addinivalue_line(
        "markers",
        "pae: PAE Step tests",
    )

    config.addinivalue_line(
        "markers",
        "nflow: NFlow Step tests",
    )

    config.addinivalue_line(
        "markers",
        "posterior: Posterior Step tests",
    )


def pytest_addoption(parser: "Parser") -> None:
    parser.addoption(
        "--setup",
        action="store",
        help='--setup "snpae"|"legacy_snpae"|"all": Setup cached data without running any tests',
    )


def pytest_runtest_setup(item: "Item") -> None:
    # Handle `setup` tests
    setup = [mark.args[0] for mark in item.iter_markers(name="setup")]
    if setup:
        if (
            item.config.getoption("--setup") != "all"
            and item.config.getoption("--setup") not in setup
        ):
            pytest.skip("Setup skipped")
    elif item.config.getoption("--setup"):
        pytest.skip("Setting up, so not running tests")
