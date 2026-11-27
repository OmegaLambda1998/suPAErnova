"""Write `tests/data/dev_golden.npz` from the `dev` branch's uncertainty path.

Run on the dev branch (94e5e83) from an environment with TensorFlow, e.g. from
suPAErnova_redux:

    uv run --no-sync python ../suPAErnova/tests/make_golden.py
"""

from pathlib import Path

import numpy as np

from golden import compute
from supaernova.steps.posterior.tf.likelihood import epoch_log_likelihood
from supaernova.utils.observations import (
    band_weights,
    build_rows,
    epoch_bands,
    integrate_bands,
)

if __name__ == "__main__":
    out = compute(
        integrate_bands, band_weights, build_rows, epoch_bands, epoch_log_likelihood
    )
    path = Path(__file__).resolve().parent / "data" / "dev_golden.npz"
    np.savez(path, **out)
    print(f"wrote {path} ({len(out)} arrays)")
