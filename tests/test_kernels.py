"""Kernel draws and the residual-kernel estimator."""

import numpy as np
import pytest

from supaernova.configs.uncertainty import UncertaintyModel
from supaernova.utils.kernels import (
    draw_correlated,
    eigenmodes,
    estimate_residual_kernel,
    k_lambda,
    k_time,
    psd_sqrt,
)

WL = {"w0": 0.1, "w_s": 0.5, "ell_s": 3.0, "w_l": 0.4, "ell_l": 15.0, "long_form": "se"}
TIME = {"w_t": 0.8, "tau": 10.0}


def test_kernels_are_correlations() -> None:
    grid = np.arange(50.0)
    K = k_lambda(grid[:, None] - grid[None], WL)
    np.testing.assert_allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    t = np.array([0.0, 0.0, 5.0, 30.0])  # repeated times: singular, still PSD
    Kt = k_time(t[:, None] - t[None], {"w_t": 1.0, "tau": 8.0})
    L = psd_sqrt(Kt)
    np.testing.assert_allclose(L @ L.T, Kt, atol=1e-12)


def test_eigenmodes_capture_long_range() -> None:
    vals, vecs = eigenmodes(100, {"w_l": 0.4, "ell_l": 30.0, "long_form": "se"}, q=12)
    grid = np.arange(100.0)
    K = 0.4 * np.exp(-0.5 * ((grid[:, None] - grid[None]) / 30.0) ** 2)
    np.testing.assert_allclose((vecs * vals) @ vecs.T, K, atol=1e-6)


def test_draw_covariance() -> None:
    rng = np.random.default_rng(1)
    n_draw, n_rows, n_wl = 4000, 3, 12
    days = np.tile([0.0, 4.0, 20.0], (n_draw, 1))
    eps = draw_correlated(rng, np.ones((n_draw, n_rows, n_wl)), WL, TIME, days, np.ones((n_draw, n_rows), bool))
    x = eps.reshape(n_draw, -1)  # row-major: (row, wl)
    emp = x.T @ x / n_draw
    grid = np.arange(n_wl, dtype=float)
    want = np.kron(k_time(days[0][:, None] - days[0][None], TIME), k_lambda(grid[:, None] - grid[None], WL))
    assert np.max(np.abs(emp - want)) < 5 / np.sqrt(n_draw) * 3


@pytest.mark.slow
def test_estimator_recovers_parameters() -> None:
    rng = np.random.default_rng(2)
    n_sn, n_rows, n_wl = 150, 6, 80
    truth = {
        "a": 1.3,
        "f": 0.01,
        "r": 0.1,
        "noise": ({"w0": 0.6, "w_s": 0.3, "ell_s": 0.7, "w_l": 0.1, "ell_l": 10.0, "long_form": "se"}, None),
        "floor": ({"w0": 0.2, "w_s": 0.2, "ell_s": 3.0, "w_l": 0.6, "ell_l": 20.0, "long_form": "se"}, {"w_t": 1.0, "tau": 8.0}),
        "model": (WL, TIME),
    }
    time = np.sort(rng.random((n_sn, n_rows)), axis=1)
    days = time * 50.0
    rows = np.ones((n_sn, n_rows), bool)
    wl = np.linspace(0, 1, n_wl)
    m = (0.5 + 0.4 * np.sin(6 * wl)[None, None] + 0.2 * rng.random((n_sn, 1, 1))) * rng.lognormal(0, 0.8, (n_sn, 1, 1))
    sigma = rng.lognormal(np.log(0.02), 0.8, (n_sn, 1, 1)) * (0.5 + rng.random((n_sn, n_rows, n_wl)))
    res = (
        draw_correlated(rng, truth["a"] * sigma, *truth["noise"], days, rows)
        + draw_correlated(rng, np.full_like(sigma, truth["f"]), *truth["floor"], days, rows)
        + draw_correlated(rng, truth["r"] * np.abs(m) * np.ones_like(sigma), *truth["model"], days, rows)
    )
    cube = {"res": res, "sigma": sigma, "model": m * np.ones_like(sigma), "use": np.ones_like(res, bool), "time": time}
    rk = estimate_residual_kernel(cube, UncertaintyModel(preset="proposal"), phase_range=50.0)
    assert abs(rk.a - truth["a"]) < 0.05 * truth["a"]
    assert abs(rk.f - truth["f"]) < 0.15 * truth["f"]
    assert abs(np.median(rk.r_table) - truth["r"]) < 0.1 * truth["r"]
    got = rk.params["model"]["wavelength"]
    assert abs(got["w_s"] - WL["w_s"]) < 0.1 and abs(got["w_l"] - WL["w_l"]) < 0.1
    assert abs(got["ell_s"] - WL["ell_s"]) < 0.25 * WL["ell_s"]
    assert abs(got["ell_l"] - WL["ell_l"]) < 0.3 * WL["ell_l"]
    t = rk.params["model"]["time"]
    assert abs(t["w_t"] - TIME["w_t"]) < 0.15 and abs(t["tau"] - TIME["tau"]) < 0.35 * TIME["tau"]


def test_draw_without_time_kernel_is_independent_between_rows() -> None:
    rng = np.random.default_rng(3)
    n_draw = 4000
    days = np.zeros((n_draw, 3))  # equal times must still be independent rows
    eps = draw_correlated(rng, np.ones((n_draw, 3, 5)), WL, None, days, np.ones((n_draw, 3), bool))
    corr = np.corrcoef(eps[:, 0, 0], eps[:, 1, 0])[0, 1]
    assert abs(corr) < 0.06
