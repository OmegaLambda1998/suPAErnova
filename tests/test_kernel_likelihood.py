"""The kernel likelihood equals a dense Gaussian with the same covariance rules."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

tf = pytest.importorskip("tensorflow")

from supaernova.configs.uncertainty import UncertaintyModel  # noqa: E402
from supaernova.steps.posterior.tf.kernel_likelihood import (  # noqa: E402
    build_kernel_structure,
    sn_kernel_log_likelihood,
)
from supaernova.utils.kernels import ResidualKernel, k_lambda, k_time_rows  # noqa: E402

N_SN, N_ROWS, N_WL, N_F, W = 2, 4, 24, 2, 3
PARAMS = {
    "noise": {"wavelength": {"w0": 0.5, "w_s": 0.3, "ell_s": 0.8, "w_l": 0.2, "ell_l": 9.0, "long_form": "exp"}, "time": None},
    "floor": {"wavelength": {"w0": 0.2, "w_s": 0.2, "ell_s": 1.5, "w_l": 0.6, "ell_l": 12.0, "long_form": "exp"}, "time": {"w_t": 1.0, "tau": 8.0}},
    "model": {"wavelength": {"w0": 0.0, "w_s": 0.6, "ell_s": 1.2, "w_l": 0.4, "ell_l": 10.0, "long_form": "exp"}, "time": {"w_t": 0.8, "tau": 10.0}},
}


def _fixture():
    rng = np.random.default_rng(7)
    amp = 1 + 0.2 * rng.normal(size=(N_SN, N_ROWS, N_WL))
    sigma = 0.02 + 0.03 * rng.random((N_SN, N_ROWS, N_WL))
    spec = rng.random((N_SN, N_ROWS, N_WL)) > 0.1
    spec[:, 2] = False  # a band-only row
    spec[1, 3] = False  # an empty row (padding)
    g = np.zeros((N_SN, N_ROWS, N_WL, N_F))
    g[..., 3:11, 0] = 1 / 8
    g[..., 12:21, 1] = 1 / 9
    band_active = np.zeros((N_SN, N_ROWS, N_F), bool)
    band_active[:, 2] = True
    band_active[0, 0, 1] = True  # a band on a spectrum row (same epoch)
    band_flux = 1 + 0.05 * rng.normal(size=(N_SN, N_ROWS, N_F))
    band_sigma = 0.01 + 0.01 * rng.random((N_SN, N_ROWS, N_F))
    days = np.array([[0.0, 4.0, 9.0, 20.0], [0.0, 3.0, 3.0, -1e6]])
    row_valid = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], bool)
    model = amp[None] * (1 + 0.03 * rng.normal(size=(2, N_SN, 1, 1))) + 0.01
    r = 0.08
    e = r * np.abs(model)
    return amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, e


def _dense(amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, m, e, a, f, local_time=True):
    """Dense reference: every covariance entry from the documented rules."""
    grid = np.arange(N_WL, dtype=float)
    dl = grid[:, None] - grid[None]
    K = {t: k_lambda(dl, PARAMS[t]["wavelength"]) for t in PARAMS}
    K_loc = {t: k_lambda(dl, {**PARAMS[t]["wavelength"], "w_l": 0.0}) for t in PARAMS}
    K_long = {t: K[t] - K_loc[t] for t in PARAMS}
    n_sp = -(-N_WL // W)
    out = []
    for s in range(N_SN):
        items = []  # (kind, row, index, weight vector over pixels)
        for r in range(N_ROWS):
            if not row_valid[s, r]:
                continue
            for p in range(n_sp):
                idx = np.arange(p * W, min((p + 1) * W, N_WL))
                ok = spec[s, r, idx]
                if ok.any():
                    vec = np.zeros(N_WL)
                    vec[idx[ok]] = 1 / ok.sum()
                    items.append(("s", r, p, vec))
            for b in range(N_F):
                if band_active[s, r, b]:
                    items.append(("b", r, b, g[s, r, :, b]))
        n = len(items)
        d = np.array([(it[3] @ amp[s, it[1]] if it[0] == "s" else band_flux[s, it[1], it[2]]) for it in items])
        mu = np.array([it[3] @ m[s, it[1]] for it in items])
        C = np.zeros((n, n))
        for i, (ki, ri, xi, vi) in enumerate(items):
            for j, (kj, rj, xj, vj) in enumerate(items):
                kt_f = k_time_rows(days[s, [ri, rj]], PARAMS["floor"]["time"])[0, 1 if ri != rj else 0]
                kt_m = k_time_rows(days[s, [ri, rj]], PARAMS["model"]["time"])[0, 1 if ri != rj else 0]
                same_local = ki == kj and xi == xj and (local_time or ri == rj)
                c = 0.0
                if ki == "s" and kj == "s" and ri == rj:
                    vs_i, vs_j = vi * sigma[s, ri], vj * sigma[s, rj]
                    c += a * a * (vs_i @ K_long["noise"] @ vs_j + (vs_i @ K_loc["noise"] @ vs_j if same_local else 0))
                if ki == "b" and kj == "b" and ri == rj and xi == xj:
                    c += a * a * band_sigma[s, ri, xi] ** 2
                c += f * f * kt_f * (vi @ K_long["floor"] @ vj + (vi @ K_loc["floor"] @ vj if same_local else 0))
                ei, ej = vi * e[s, ri], vj * e[s, rj]
                c += kt_m * (ei @ K_long["model"] @ ej + (ei @ K_loc["model"] @ ej if same_local else 0))
                C[i, j] = c
        out.append(multivariate_normal(mu, C).logpdf(d))
    return np.array(out)


def test_kernel_likelihood_matches_dense() -> None:
    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, e = _fixture()
    a, f = 1.3, 0.01
    rk = ResidualKernel(a=a, f=f, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(
        preset="proposal_fixed",
        kernel={"bin_width": W, "n_modes": N_WL, "noise": {"n_modes": N_WL}, "floor": {"on_bands": True}},
    )
    struct = build_kernel_structure(
        rk=rk, uncertainty=u, amplitude=amp, sigma=sigma, spec_pixels=spec,
        band_active=band_active, band_flux=band_flux, band_sigma=band_sigma,
        band_weights=g, band_source_sigma=None,
    )
    ll, _ = sn_kernel_log_likelihood(
        model=tf.constant(model), model_error=tf.constant(e), days=tf.constant(days),
        row_valid=tf.constant(row_valid), struct=struct, jitter=0.0,
    )
    for w in range(model.shape[0]):
        want = _dense(amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model[w], e[w], a, f)
        np.testing.assert_allclose(ll.numpy()[w], want, rtol=1e-8, atol=1e-6)


def test_kernel_likelihood_gradient() -> None:
    """d lnL / dΔM-like scale matches a central finite difference."""
    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, _ = _fixture()
    rk = ResidualKernel(a=1.3, f=0.01, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(preset="proposal_fixed", kernel={"bin_width": W, "n_modes": 6})
    struct = build_kernel_structure(
        rk=rk, uncertainty=u, amplitude=amp, sigma=sigma, spec_pixels=spec,
        band_active=band_active, band_flux=band_flux, band_sigma=band_sigma,
        band_weights=g, band_source_sigma=None,
    )

    def ll(scale):
        m = tf.constant(model[:1]) * scale
        out, _ = sn_kernel_log_likelihood(
            model=m, model_error=0.08 * tf.abs(m), days=tf.constant(days),
            row_valid=tf.constant(row_valid), struct=struct,
        )
        return tf.reduce_sum(out)

    scale = tf.Variable(tf.constant(1.0, tf.float64))
    with tf.GradientTape() as tape:
        value = ll(scale)
    grad = float(tape.gradient(value, scale))
    h = 1e-5
    fd = (float(ll(tf.constant(1.0 + h, tf.float64))) - float(ll(tf.constant(1.0 - h, tf.float64)))) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_non_finite_model_is_isolated() -> None:
    """A non-finite SN gives NaN for that SN only, without failing factorisations."""
    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, _ = _fixture()
    rk = ResidualKernel(a=1.3, f=0.01, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(preset="proposal_fixed", kernel={"bin_width": W})
    struct = build_kernel_structure(
        rk=rk, uncertainty=u, amplitude=amp, sigma=sigma, spec_pixels=spec,
        band_active=band_active, band_flux=band_flux, band_sigma=band_sigma,
        band_weights=g, band_source_sigma=None,
    )
    m = model[:1].copy()
    good, _ = sn_kernel_log_likelihood(
        model=tf.constant(m), model_error=0.08 * tf.abs(tf.constant(m)),
        days=tf.constant(days), row_valid=tf.constant(row_valid), struct=struct,
    )
    m[:, 1] = np.nan
    bad, _ = sn_kernel_log_likelihood(
        model=tf.constant(m), model_error=0.08 * tf.abs(tf.constant(m)),
        days=tf.constant(days), row_valid=tf.constant(row_valid), struct=struct,
    )
    assert np.isnan(bad.numpy()[0, 1])
    np.testing.assert_allclose(bad.numpy()[0, 0], good.numpy()[0, 0])


def test_precomputed_constants_and_float32() -> None:
    """`add_constants` reproduces the on-the-fly path; float32 stays close to float64."""
    from supaernova.steps.posterior.tf.kernel_likelihood import add_constants

    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, _ = _fixture()
    rk = ResidualKernel(a=1.3, f=0.01, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(preset="proposal_fixed", kernel={"bin_width": W})
    struct = build_kernel_structure(
        rk=rk, uncertainty=u, amplitude=amp, sigma=sigma, spec_pixels=spec,
        band_active=band_active, band_flux=band_flux, band_sigma=band_sigma,
        band_weights=g, band_source_sigma=None,
    )
    kwargs = dict(
        model=tf.constant(model), model_error=0.08 * tf.abs(tf.constant(model)),
        days=tf.constant(days), row_valid=tf.constant(row_valid),
    )
    plain, _ = sn_kernel_log_likelihood(struct=struct, **kwargs)
    pre = add_constants(struct, days, row_valid)
    fast, _ = sn_kernel_log_likelihood(struct=pre, **kwargs)
    np.testing.assert_allclose(fast.numpy(), plain.numpy(), rtol=1e-10)
    f32, _ = sn_kernel_log_likelihood(struct={**pre, "dtype": tf.float32}, jitter=1e-7, **kwargs)
    np.testing.assert_allclose(f32.numpy(), plain.numpy(), rtol=1e-4)


def test_time_mode_truncation() -> None:
    """Keeping every time mode is exact; truncating changes lnL only modestly."""
    from supaernova.steps.posterior.tf.kernel_likelihood import add_constants

    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, _ = _fixture()
    rk = ResidualKernel(a=1.3, f=0.01, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(preset="proposal_fixed", kernel={"bin_width": W})
    struct = build_kernel_structure(
        rk=rk, uncertainty=u, amplitude=amp, sigma=sigma, spec_pixels=spec,
        band_active=band_active, band_flux=band_flux, band_sigma=band_sigma,
        band_weights=g, band_source_sigma=None,
    )
    kwargs = dict(
        model=tf.constant(model), model_error=0.08 * tf.abs(tf.constant(model)),
        days=tf.constant(days), row_valid=tf.constant(row_valid),
    )
    plain, _ = sn_kernel_log_likelihood(struct=struct, **kwargs)
    full, _ = sn_kernel_log_likelihood(struct=add_constants(struct, days, row_valid, time_modes=N_ROWS), **kwargs)
    np.testing.assert_allclose(full.numpy(), plain.numpy(), rtol=1e-10)
    trunc, _ = sn_kernel_log_likelihood(struct=add_constants(struct, days, row_valid, time_modes=2), **kwargs)
    # The fixture packs 4 epochs into 20 d (τ ≈ 10 d), a hard case for truncation
    assert np.all(np.abs(trunc.numpy() - plain.numpy()) < 0.1 * np.abs(plain.numpy()))


def test_kernel_likelihood_without_local_time_correlation() -> None:
    """`local_time = False`: local parts independent between epochs, still exact."""
    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, e = _fixture()
    a, f = 1.3, 0.01
    rk = ResidualKernel(a=a, f=f, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(
        preset="proposal_fixed",
        kernel={"bin_width": W, "n_modes": N_WL, "noise": {"n_modes": N_WL}, "floor": {"on_bands": True}},
    )
    struct = build_kernel_structure(
        rk=rk, uncertainty=u, amplitude=amp, sigma=sigma, spec_pixels=spec,
        band_active=band_active, band_flux=band_flux, band_sigma=band_sigma,
        band_weights=g, band_source_sigma=None,
    )
    ll, _ = sn_kernel_log_likelihood(
        model=tf.constant(model), model_error=tf.constant(e), days=tf.constant(days),
        row_valid=tf.constant(row_valid), struct={**struct, "local_time": False}, jitter=0.0,
    )
    for w in range(model.shape[0]):
        want = _dense(amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid,
                      model[w], e[w], a, f, local_time=False)
        np.testing.assert_allclose(ll.numpy()[w], want, rtol=1e-8, atol=1e-6)


def test_padding_rows_do_not_change_result() -> None:
    """Extra padding rows leave lnL unchanged at the default (relative) jitter."""
    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, _ = _fixture()
    rk = ResidualKernel(a=1.3, f=0.01, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(preset="proposal_fixed", kernel={"bin_width": W})

    def run(extra):
        def pad(x, value=0):
            widths = [(0, 0)] * x.ndim
            widths[1] = (0, extra)
            return np.pad(x, widths, constant_values=value)

        struct = build_kernel_structure(
            rk=rk, uncertainty=u, amplitude=pad(amp), sigma=pad(sigma), spec_pixels=pad(spec, False),
            band_active=pad(band_active, False), band_flux=pad(band_flux), band_sigma=pad(band_sigma),
            band_weights=pad(g), band_source_sigma=None,
        )
        m = np.pad(model, [(0, 0), (0, 0), (0, extra), (0, 0)])
        ll, _ = sn_kernel_log_likelihood(
            model=tf.constant(m), model_error=0.08 * tf.abs(tf.constant(m)),
            days=tf.constant(pad(days)), row_valid=tf.constant(pad(row_valid, False)),
            struct=struct, jitter=1e-6,
        )
        return ll.numpy()

    np.testing.assert_allclose(run(3), run(0), rtol=1e-9)


def test_jitter_is_relative_to_real_variances() -> None:
    """A 1e-6 relative jitter barely moves lnL, even with padding rows present."""
    amp, sigma, spec, g, band_active, band_flux, band_sigma, days, row_valid, model, _ = _fixture()
    rk = ResidualKernel(a=1.3, f=0.01, r_table=np.zeros((N_WL, 2)), centers=np.array([0.0, 1.0]), params=PARAMS)
    u = UncertaintyModel(preset="proposal_fixed", kernel={"bin_width": W})
    struct = build_kernel_structure(
        rk=rk, uncertainty=u, amplitude=amp, sigma=sigma, spec_pixels=spec,
        band_active=band_active, band_flux=band_flux, band_sigma=band_sigma,
        band_weights=g, band_source_sigma=None,
    )
    kwargs = dict(
        model=tf.constant(model), model_error=0.08 * tf.abs(tf.constant(model)),
        days=tf.constant(days), row_valid=tf.constant(row_valid), struct=struct,
    )
    exact, _ = sn_kernel_log_likelihood(jitter=0.0, **kwargs)
    jittered, _ = sn_kernel_log_likelihood(jitter=1e-6, **kwargs)
    assert np.max(np.abs(jittered.numpy() - exact.numpy())) < 1e-3
