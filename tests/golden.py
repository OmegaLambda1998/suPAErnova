"""Deterministic synthetic fixtures and the dev-branch reference outputs built from them.

`compute()` runs the uncertainty-path functions (band integration, row building,
band collection and the per-epoch likelihood) on a small synthetic data set with
a fixed seed. `make_golden.py` saves its output from the `dev` branch
(94e5e83) to `tests/data/dev_golden.npz`; `test_dev_golden.py` recomputes it on
the current code, through the `dev` uncertainty preset, and requires agreement.

Only synthetic arrays are used, so no survey data enters the repository.
"""

from typing import Any

import numpy as np

N_SN, N_EP, N_WL, N_F = 3, 5, 40, 2
SEED = 20261127


def synthetic() -> dict[str, Any]:
    """A small data set: 3 SNe × 5 epochs × 40 pixels, 2 top-hat-ish filters."""
    rng = np.random.default_rng(SEED)
    wavelength = np.linspace(3500.0, 7500.0, N_WL)
    throughput = np.zeros((N_WL, N_F))
    throughput[6:16, 0] = np.hanning(12)[1:-1]
    throughput[22:34, 1] = np.hanning(14)[1:-1]
    effective_index = np.array([11, 28])
    flux = 1.0 + 0.3 * np.sin(wavelength / 300.0)[None, None] + 0.1 * rng.normal(
        size=(N_SN, N_EP, N_WL)
    )
    sigma = 0.02 + 0.01 * rng.random((N_SN, N_EP, N_WL))
    valid = rng.random((N_SN, N_EP, N_WL)) > 0.05
    spec_obs = np.array(
        [[1, 0, 1, 0, 1], [1, 1, 0, 0, 0], [0, 1, 1, 0, 1]], dtype=bool
    )
    band_obs = np.zeros((N_SN, N_EP, N_F), dtype=bool)
    band_obs[0, 1] = True
    band_obs[0, 3] = [True, False]
    band_obs[1, 2:4] = True
    band_obs[2, 0] = True
    band_obs[2, 2] = [False, True]  # shared with a spectrum epoch
    return {
        "wavelength": wavelength,
        "throughput": throughput,
        "effective_index": effective_index,
        "flux": flux,
        "sigma": sigma,
        "valid": valid,
        "spec_obs": spec_obs,
        "band_obs": band_obs,
        "rng_model": rng.normal(size=(2, N_SN, 1, 1)),
    }


def compute(
    integrate_bands,
    band_weights,
    build_rows,
    epoch_bands,
    epoch_log_likelihood,
    *,
    sigma_scale: float = 1.0,
    sigma_floor: float = 0.0,
) -> dict[str, np.ndarray]:
    """Run the dev uncertainty path on the synthetic data; return named outputs.

    The functions are passed in so the same driver works on dev and on later
    branches (which call them through the `dev` preset).
    """
    import tensorflow as tf

    s = synthetic()
    g = band_weights(s["wavelength"], s["throughput"])
    band_flux, band_sigma, band_valid = integrate_bands(
        s["flux"], s["sigma"], s["valid"], g
    )
    band_obs = s["band_obs"] & band_valid
    rows = build_rows(
        wavelength=s["wavelength"],
        throughput=s["throughput"],
        effective_index=s["effective_index"],
        spec_obs=s["spec_obs"],
        spec_flux=s["flux"],
        spec_sigma=s["sigma"],
        spec_valid=s["valid"],
        band_obs=band_obs,
        band_flux=band_flux,
        band_sigma=band_sigma,
        sigma_scale=sigma_scale,
        sigma_floor=sigma_floor,
        allow_shared_epochs=True,
    )
    n_rows = rows["amplitude"].shape[1]
    wl_rows = np.broadcast_to(s["wavelength"], (N_SN, n_rows, N_WL))
    bands = epoch_bands(
        amplitude=rows["amplitude"],
        sigma=rows["sigma"],
        datum_mask=rows["mask"].astype(bool),
        spectra_mask=rows["spectra_mask"],
        phot_mask=rows["phot_mask"],
        wavelength=wl_rows,
        throughput=rows["throughput"],
        effective_wavelength=rows["effective_wavelength"],
        epoch=rows["epoch"],
    )
    # A model: the noisy flux, smoothed and scaled per SN (two "walkers")
    base = np.where(rows["spectra_mask"].astype(bool), rows["amplitude"], 1.0)
    model = base[None] * (1.0 + 0.05 * s["rng_model"]) + 0.01
    model_error = 0.08 * np.abs(model)
    spec_pixels = rows["mask"].astype(bool) & rows["spectra_mask"].astype(bool)
    out = {f"rows_{k}": np.asarray(v) for k, v in rows.items()}
    out.update({f"bands_{k}": np.asarray(v) for k, v in bands.items()})
    out.update(
        band_flux=band_flux, band_sigma=band_sigma, band_valid=band_valid, weights=g
    )
    for rho in (0.0, 0.3, 1.0):
        for joint in (False, True):
            ll, n_data = epoch_log_likelihood(
                model=tf.constant(model, tf.float32),
                model_error=tf.constant(model_error, tf.float32),
                correlation=rho,
                spec_pixels=tf.constant(spec_pixels),
                spec_flux=tf.constant(rows["amplitude"], tf.float32),
                spec_variance=tf.constant(rows["sigma"] ** 2, tf.float32),
                band_active=tf.constant(bands["band_active"]),
                band_flux=tf.constant(bands["band_flux"], tf.float32),
                band_variance=tf.constant(bands["band_sigma"] ** 2, tf.float32),
                band_weights=tf.constant(bands["band_weights"], tf.float32),
                joint=joint,
            )
            out[f"ll_rho{rho}_joint{int(joint)}"] = ll.numpy()
            out[f"ndata_rho{rho}_joint{int(joint)}"] = np.asarray(n_data)
    return out
