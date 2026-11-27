"""The current code, run through the `dev` uncertainty preset, reproduces the dev branch."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from golden import compute  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "dev_golden.npz"


def _dev_functions():
    from supaernova.steps.posterior.tf.likelihood import epoch_log_likelihood
    from supaernova.utils.observations import (
        band_weights,
        build_rows,
        epoch_bands,
        integrate_bands,
    )

    return integrate_bands, band_weights, build_rows, epoch_bands, epoch_log_likelihood


def test_dev_golden() -> None:
    expected = dict(np.load(GOLDEN))
    got = compute(*_dev_functions())
    assert set(got) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(
            np.asarray(got[key], dtype=np.float64),
            value.astype(np.float64),
            rtol=1e-6,
            atol=1e-5 if key.startswith("ll_") else 1e-12,
            err_msg=key,
        )
