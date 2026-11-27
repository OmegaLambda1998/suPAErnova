"""The `main` preset's likelihood reproduces the main branch's per-datum formula.

Reference (main, d891470, `steps/posterior/tf/tf.py` ~L460-529): every datum is an
independent Normal with σ_tot² = σ_model² + σ_in², where the model error passed
to `photometry` as `sigma_correlated` is r·√(m² + σ_in²) per pixel and is summed
linearly over a filter (Σ g r S) for a band. A band's own pixels sat in a
photometry row with σ_in = 0 off its effective bin, so there S = |m|.
"""

import math

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def _reference(m, d, s, r, spec, g, band_flux, band_sigma, band_active):
    """Main's log-likelihood per row, written out in numpy."""
    s_model = r * np.sqrt(m * m + s * s)
    var = s_model**2 + s**2
    ll_pix = -0.5 * ((d - m) ** 2 / var + np.log(var) + math.log(2 * math.pi))
    ll = np.sum(np.where(spec, ll_pix, 0.0), axis=-1)
    band_model = np.einsum("...i,...ib->...b", m, g)
    band_model_error = np.einsum("...i,...ib->...b", r * np.abs(m), g)  # coherent
    var_b = band_model_error**2 + band_sigma**2
    ll_b = -0.5 * ((band_flux - band_model) ** 2 / var_b + np.log(var_b) + math.log(2 * math.pi))
    return ll + np.sum(np.where(band_active, ll_b, 0.0), axis=-1)


def test_independent_matches_main_formula() -> None:
    from supaernova.steps.posterior.tf.likelihood import independent_log_likelihood

    rng = np.random.default_rng(0)
    n_sn, n_rows, n_wl, n_f = 3, 4, 30, 2
    m = 1 + 0.2 * rng.normal(size=(2, n_sn, n_rows, n_wl))
    d = m[0] + 0.05 * rng.normal(size=(n_sn, n_rows, n_wl))
    s = 0.02 + 0.02 * rng.random((n_sn, n_rows, n_wl))
    r = 0.05 + 0.05 * rng.random((n_sn, n_rows, n_wl))
    spec = rng.random((n_sn, n_rows, n_wl)) > 0.2
    spec[:, -1] = False  # a photometry-only row
    g = np.zeros((n_sn, n_rows, n_wl, n_f))
    g[..., 5:12, 0] = 1 / 7
    g[..., 15:25, 1] = 1 / 10
    band_active = rng.random((n_sn, n_rows, n_f)) > 0.4
    band_flux = 1 + 0.05 * rng.normal(size=(n_sn, n_rows, n_f))
    band_sigma = 0.01 + 0.01 * rng.random((n_sn, n_rows, n_f))

    expected = _reference(m, d, s, r, spec, g, band_flux, band_sigma, band_active)

    ll, n_data = independent_log_likelihood(
        model=tf.constant(m),
        model_error=tf.constant(r * np.sqrt(m * m + s * s)),
        band_model_error=tf.constant(r * np.abs(m)),
        spec_pixels=tf.constant(spec),
        spec_flux=tf.constant(np.where(spec, d, 0.0)),
        spec_variance=tf.constant(np.where(spec, s * s, 1.0)),
        band_active=tf.constant(band_active),
        band_flux=tf.constant(band_flux),
        band_variance=tf.constant(band_sigma**2),
        band_weights=tf.constant(g),
        coherent=True,
    )
    np.testing.assert_allclose(ll.numpy(), expected, rtol=1e-10)
    np.testing.assert_array_equal(
        n_data.numpy()[0], spec.sum(-1) + band_active.sum(-1)
    )


def test_calibration_round_trip() -> None:
    from supaernova.utils.observations import calibrate_sigma, uncalibrate_sigma

    raw = np.array([0.0, 1e-3, 0.02, 0.5])
    for mode in ("linear", "quadrature"):
        cal = calibrate_sigma(raw, 1.4, 4e-3, mode)
        np.testing.assert_allclose(uncalibrate_sigma(cal, 1.4, 4e-3, mode), raw, atol=1e-12)
    np.testing.assert_allclose(calibrate_sigma(raw, 1.4, 4e-10), 1.4 * raw + 4e-10)
    np.testing.assert_allclose(
        calibrate_sigma(raw, 1.0, 3e-3, "quadrature"), np.sqrt(raw**2 + 9e-6)
    )
