# Copyright 2025 Patrick Armstrong
"""Correlated noise, floor and model-error kernels, and their estimation.

Each held-out spectral pixel's PAE residual is modelled as Gaussian with

    Cov(res_i, res_j) = a² σ_i σ_j k_n(Δλ) δ_epoch
                      + f² k_f(Δλ) k_f^t(Δt)
                      + e_i e_j k_m(Δλ) k_m^t(Δt),       e_i = r(λ_i, t_i) |m_i|,

with, for every term, a wavelength kernel (Δλ in pixels)

    k(Δλ) = w0 δ + w_s Matérn-3/2(Δλ; ℓ_s) + w_l L(Δλ; ℓ_l),   L = SE or exp,

and a time kernel between epochs of one SN (Δt in days)

    k^t(Δt) = (1 − w_t) δ + w_t exp(−|Δt| / τ).

`estimate_residual_kernel` fits every parameter left as ``"auto"``:

1. a, f and r(λ-group, time-bin) jointly by maximum likelihood on the pixel
   variances, ``a²σ² + f² + r² m²``;
2. each term's wavelength kernel from the residual covariance at every pixel
   lag, by weighted least squares of res_i·res_j on the three variance terms;
3. the floor and model time kernels the same way, on pairs of epochs of one SN.

This is the procedure of the uncertainty study
(suPAErnova_redux/projects/uncertainty_regression).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares, minimize

if TYPE_CHECKING:
    from numpy import typing as npt

    from supaernova.configs.uncertainty import UncertaintyModel

TERMS = ("noise", "floor", "model")
WL_GROUP = 12  # pixels per r(λ) group in the variance fit
LAGS = np.r_[0:16, 16:48:2, 48:128:8, 128:288:16]
DT_EDGES = np.array([0.0, 3.0, 6.0, 10.0, 15.0, 25.0, 40.0, 60.0])  # days


# === Kernels ===
def matern32(x: "npt.ArrayLike", ell: float) -> "npt.NDArray[np.float64]":
    s = np.sqrt(3.0) * np.abs(np.asarray(x, dtype=np.float64)) / ell
    return (1.0 + s) * np.exp(-s)


def long_range(x: "npt.ArrayLike", ell: float, form: str) -> "npt.NDArray[np.float64]":
    x = np.abs(np.asarray(x, dtype=np.float64))
    if form == "exp":
        return np.exp(-x / ell)
    return np.exp(-0.5 * (x / ell) ** 2)


def k_lambda(dl: "npt.ArrayLike", p: dict[str, Any]) -> "npt.NDArray[np.float64]":
    """Wavelength kernel at pixel separation ``dl`` (parameters all numeric)."""
    dl = np.abs(np.asarray(dl, dtype=np.float64))
    return (
        p["w0"] * (dl == 0)
        + p["w_s"] * matern32(dl, p["ell_s"])
        + p["w_l"] * long_range(dl, p["ell_l"], p.get("long_form", "se"))
    )


def k_time(dt: "npt.ArrayLike", p: dict[str, Any] | None) -> "npt.NDArray[np.float64]":
    """Time kernel at epoch separation ``dt`` days.

    ``None`` gives δ(Δt): only use it for separations of distinct epochs. For
    rows that are independent even at equal times (measurement noise), use an
    identity matrix instead.
    """
    dt = np.abs(np.asarray(dt, dtype=np.float64))
    if p is None:
        return (dt == 0).astype(np.float64)
    return (1.0 - p["w_t"]) * (dt == 0) + p["w_t"] * np.exp(-dt / p["tau"])


def k_time_rows(
    t: "npt.ArrayLike", p: dict[str, Any] | None
) -> "npt.NDArray[np.float64]":
    """Time kernel matrix over a set of distinct epochs at days ``t``.

    The white part couples an epoch only with itself (δ over epochs, not over
    Δt = 0), so distinct epochs at equal times are still independent there.
    """
    t = np.asarray(t, dtype=np.float64)
    eye = np.eye(t.size)
    if p is None:
        return eye
    return (1.0 - p["w_t"]) * eye + p["w_t"] * np.exp(
        -np.abs(t[:, None] - t[None]) / p["tau"]
    )


def eigenmodes(
    n: int, p: dict[str, Any], q: int, *, rtol: float = 1e-10
) -> tuple["npt.NDArray[np.float64]", "npt.NDArray[np.float64]"]:
    """Top-q eigenpairs of the long-range part ``w_l L(Δλ; ℓ_l)`` on an n-pixel grid.

    Returns:
        ``(values (q',), vectors (n, q'))`` with q' ≤ q, dropping modes below
        ``rtol`` times the largest eigenvalue.
    """
    grid = np.arange(n, dtype=np.float64)
    K = p["w_l"] * long_range(
        grid[:, None] - grid[None], p["ell_l"], p.get("long_form", "se")
    )
    vals, vecs = np.linalg.eigh(K)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = (
        vals > rtol * max(vals[0], 0.0)
        if vals.size and vals[0] > 0
        else np.zeros_like(vals, bool)
    )
    keep[q:] = False
    return np.clip(vals[keep], 0.0, None), vecs[:, keep]


def psd_sqrt(K: "npt.NDArray[np.float64]") -> "npt.NDArray[np.float64]":
    """A square root L with L Lᵀ = K for a (possibly singular) PSD matrix."""
    vals, vecs = np.linalg.eigh(np.asarray(K, dtype=np.float64))
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


# === Time bins (identical to `TFPAEModel.recon_error`) ===
def time_bins(
    width: float = 0.1,
) -> tuple["npt.NDArray[np.float64]", "npt.NDArray[np.float64]"]:
    n_bins = int((1.0 - 0.0) // width) + 2
    edges = np.linspace(-width / 2, 1.0 + width / 2, n_bins + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


# === Estimation ===
@dataclass
class ResidualKernel:
    """Everything the posterior and sim need: amplitudes, r table, kernel parameters."""

    a: float
    f: float
    r_table: "npt.NDArray[np.float64]"  # (n_wl, n_time_bins), like `recon_error`
    centers: "npt.NDArray[np.float64]"  # time-bin centres (normalised time)
    params: dict[
        str, dict[str, Any]
    ]  # per term: {"wavelength": {...}, "time": {...} | None}
    lag_curves: dict[str, Any] = field(default_factory=dict)  # diagnostics
    n_pixels: int = 0

    def term(self, name: str) -> dict[str, Any]:
        return self.params[name]


def _value(v: Any) -> float | None:
    return None if isinstance(v, str) else float(v)


def _fit_variances(
    res, sig, m, b, n_bins, a_fixed, f_fixed, r0=0.08
) -> tuple[float, float, "npt.NDArray[np.float64]", "npt.NDArray[np.bool_]"]:
    """ML fit of a, f and r_b in Var = a²σ² + f² + r_b² m² (free unless fixed)."""
    r2, s2, m2 = res * res, sig * sig, m * m
    free_a, free_f = a_fixed is None, f_fixed is None

    def nll(theta):
        a2 = np.exp(theta[0]) if free_a else a_fixed**2
        f2 = np.exp(theta[1]) if free_f else f_fixed**2
        rb2 = np.exp(theta[2:])
        v = a2 * s2 + f2 + rb2[b] * m2
        g = 0.5 * (1.0 / v - r2 / v**2)
        grad = np.r_[
            np.sum(g * s2) * a2 if free_a else 0.0,
            np.sum(g) * f2 if free_f else 0.0,
            np.bincount(b, g * m2, n_bins) * rb2,
        ]
        return 0.5 * np.sum(np.log(v) + r2 / v), grad

    f0 = np.median(sig) if free_f else max(f_fixed, 1e-12)
    x0 = np.r_[0.0, np.log(f0**2 * 0.1 + 1e-30), np.full(n_bins, np.log(r0**2))]
    out = minimize(nll, x0, jac=True, method="L-BFGS-B")
    a = float(np.exp(out.x[0] / 2)) if free_a else float(a_fixed)
    f = float(np.exp(out.x[1] / 2)) if free_f else float(f_fixed)
    seen = np.bincount(b, minlength=n_bins) > 0
    return a, f, np.exp(out.x[2:] / 2), seen


def _lag_regression(y, regs, weights):
    X = np.stack(regs, -1)
    XtW = X.T * weights
    A = XtW @ X
    coef = np.linalg.lstsq(A, XtW @ y, rcond=None)[0]
    resid = y - X @ coef
    scale = np.sum(weights * resid**2) / max(y.size - X.shape[1], 1)
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(A)) * scale, 0, None))
    return coef, se


def _fit_parametric(x, y, se, spec: dict[str, Any], keys, model_fn, bounds, x0s):
    """Least squares of the free parameters in `keys` (others fixed by `spec`)."""
    free = [k for k in keys if _value(spec.get(k)) is None]
    fixed = {k: _value(spec[k]) for k in keys if k not in free}
    if not free:
        return dict(fixed)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    lo = [bounds[k][0] for k in free]
    hi = [bounds[k][1] for k in free]
    best = None
    for x0 in x0s:
        start = np.clip([x0[k] for k in free], lo, hi)
        res = least_squares(
            lambda th: (
                (model_fn({**fixed, **dict(zip(free, th, strict=True))}, x[ok]) - y[ok])
                / se[ok]
            ),
            start,
            bounds=(lo, hi),
        )
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    return {**fixed, **dict(zip(free, (float(v) for v in best.x), strict=True))}


def _resolve_wavelength(lags, est, err, spec: dict[str, Any]) -> dict[str, Any]:
    form = spec.get("long_form", "se")
    keys = ("w_s", "ell_s", "w_l", "ell_l")

    def model_fn(p, x):
        return p["w_s"] * matern32(x, p["ell_s"]) + p["w_l"] * long_range(
            x, p["ell_l"], form
        )

    bounds = {
        "w_s": (0.0, 1.0),
        "ell_s": (0.2, 80.0),
        "w_l": (0.0, 1.0),
        "ell_l": (3.0, 3000.0),
    }
    x0s = [
        {"w_s": 0.5, "ell_s": 3.0, "w_l": 0.4, "ell_l": 100.0},
        {"w_s": 0.2, "ell_s": 1.0, "w_l": 0.6, "ell_l": 60.0},
        {"w_s": 0.8, "ell_s": 5.0, "w_l": 0.2, "ell_l": 200.0},
    ]
    sel = lags >= 1
    p = _fit_parametric(
        lags[sel].astype(float), est[sel], err[sel], spec, keys, model_fn, bounds, x0s
    )
    w0 = _value(spec.get("w0"))
    if w0 is None:
        total = p["w_s"] + p["w_l"]
        if total > 1:  # renormalise so the white part is non-negative
            p["w_s"], p["w_l"] = p["w_s"] / total, p["w_l"] / total
        w0 = 1.0 - p["w_s"] - p["w_l"]
    return {"w0": float(w0), **p, "long_form": form}


def _resolve_time(mid, est, err, spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    bounds = {"w_t": (1e-3, 1.0), "tau": (0.5, 500.0)}
    x0s = [{"w_t": 0.9, "tau": 10.0}, {"w_t": 0.5, "tau": 30.0}]
    return _fit_parametric(
        mid,
        est,
        err,
        spec,
        ("w_t", "tau"),
        lambda p, x: p["w_t"] * np.exp(-x / p["tau"]),
        bounds,
        x0s,
    )


def estimate_residual_kernel(
    cube: dict[str, "npt.NDArray"],
    uncertainty: "UncertaintyModel",
    *,
    phase_range: float,
    log: Any = None,
) -> ResidualKernel:
    """Fit every ``"auto"`` kernel parameter to a residual cube.

    Args:
        cube: From `TFPAEModel.residual_cube`: ``res``, ``sigma``, ``model``,
            ``use`` (n_sn, n_rows, n_wl), ``time`` (n_sn, n_rows) normalised time.
        uncertainty: The uncertainty model; its numeric kernel values are kept fixed.
        phase_range: Days spanned by normalised time 0–1 (for Δt).
        log: Optional logger.
    """
    kernel = uncertainty.kernel
    res, sig, m = (
        np.asarray(cube[k], dtype=np.float64) for k in ("res", "sigma", "model")
    )
    use = np.asarray(cube["use"], dtype=bool) & (sig > 0) & np.isfinite(res)
    time = np.asarray(cube["time"], dtype=np.float64)
    n_sn, _, n_wl = res.shape
    edges, centers = time_bins(uncertainty.model_error.time_bin)
    n_tb = centers.size
    t_bin = np.clip(np.searchsorted(edges, time, side="right") - 1, 0, n_tb - 1)
    n_groups = int(np.ceil(n_wl / WL_GROUP))
    b_all = (np.arange(n_wl) // WL_GROUP)[None, None, :] * n_tb + t_bin[..., None]
    b_all = np.broadcast_to(b_all, res.shape)

    # 1. Variances: a, f, r(λ-group, time-bin)
    a, f, r, seen = _fit_variances(
        res[use],
        sig[use],
        m[use],
        b_all[use],
        n_groups * n_tb,
        _value(kernel.noise.amplitude),
        _value(kernel.floor.amplitude),
    )
    r = np.where(seen, r, np.median(r[seen]) if seen.any() else 0.08)
    r_groups = r.reshape(n_groups, n_tb)
    r_table = r_groups[np.arange(n_wl) // WL_GROUP]  # (n_wl, n_tb)
    if log is not None:
        log.info(
            f"Residual kernel: a = {a:.3f}, f = {f:.3g}, median r = {np.median(r[seen]):.3f}"
        )

    e = r_table[np.arange(n_wl)[None, None, :], t_bin[..., None]] * m
    var = a * a * sig * sig + f * f + e * e

    # 2. Wavelength kernels from the covariance at each pixel lag
    lags = LAGS[LAGS < n_wl]
    est = np.full((lags.size, 3), np.nan)
    err = np.full((lags.size, 3), np.nan)
    for i, lag in enumerate(lags):
        ok = use[..., : n_wl - lag] & use[..., lag:]
        if ok.sum() < 200:
            continue
        pick_i = lambda x, ok=ok, lag=lag: x[..., : n_wl - lag][ok]
        pick_j = lambda x, ok=ok, lag=lag: x[..., lag:][ok]
        regs = [
            a * a * pick_i(sig) * pick_j(sig),
            np.full(ok.sum(), f * f),
            pick_i(e) * pick_j(e),
        ]
        est[i], err[i] = _lag_regression(
            pick_i(res) * pick_j(res), regs, 1.0 / (pick_i(var) * pick_j(var))
        )

    params: dict[str, dict[str, Any]] = {}
    for k, name in enumerate(TERMS):
        term = getattr(kernel, name)
        spec = term.wavelength.model_dump(exclude={"name"})
        params[name] = {
            "wavelength": _resolve_wavelength(lags, est[:, k], err[:, k], spec)
        }

    # 3. Time kernels from pairs of epochs of one SN, at the same pixel
    row_ok = use.any(-1)
    pair_p, pair_q, pair_dt = [], [], []
    for s in range(n_sn):
        rows = np.flatnonzero(row_ok[s])
        for ii, p in enumerate(rows):
            for q in rows[ii + 1 :]:
                dt = abs(time[s, p] - time[s, q]) * phase_range
                if dt > 0:
                    pair_p.append((s, p))
                    pair_q.append((s, q))
                    pair_dt.append(dt)
    t_mid, t_est, t_err = [], [], []
    if pair_dt:
        pair_p, pair_q, pair_dt = np.array(pair_p), np.array(pair_q), np.array(pair_dt)
        which = np.searchsorted(DT_EDGES, pair_dt, side="right") - 1
        for j in range(DT_EDGES.size - 1):
            sel = which == j
            if not sel.any():
                continue
            P, Q = pair_p[sel], pair_q[sel]
            ok = use[P[:, 0], P[:, 1]] & use[Q[:, 0], Q[:, 1]]
            if ok.sum() < 200:
                continue
            rp, rq = res[P[:, 0], P[:, 1]][ok], res[Q[:, 0], Q[:, 1]][ok]
            ep, eq = e[P[:, 0], P[:, 1]][ok], e[Q[:, 0], Q[:, 1]][ok]
            vp, vq = var[P[:, 0], P[:, 1]][ok], var[Q[:, 0], Q[:, 1]][ok]
            coef, se = _lag_regression(
                rp * rq, [np.full(ok.sum(), f * f), ep * eq], 1.0 / (vp * vq)
            )
            t_mid.append(0.5 * (DT_EDGES[j] + DT_EDGES[j + 1]))
            t_est.append(coef)
            t_err.append(se)
    t_mid, t_est, t_err = np.array(t_mid), np.array(t_est), np.array(t_err)
    for k, name in ((0, "floor"), (1, "model")):
        term = getattr(kernel, name)
        spec = None if term.time is None else term.time.model_dump(exclude={"name"})
        if (
            spec is not None
            and t_mid.size
            and any(isinstance(v, str) for v in spec.values())
        ):
            params[name]["time"] = _resolve_time(t_mid, t_est[:, k], t_err[:, k], spec)
        elif spec is not None and not any(isinstance(v, str) for v in spec.values()):
            params[name]["time"] = {k_: float(v) for k_, v in spec.items()}
        else:
            params[name]["time"] = None if spec is None else {"w_t": 0.86, "tau": 12.0}
    params["noise"]["time"] = None

    if log is not None:
        for name in TERMS:
            w, t = params[name]["wavelength"], params[name]["time"]
            log.info(
                f"  {name}: λ white {w['w0']:.2f}, short {w['w_s']:.2f} (ℓ {w['ell_s']:.1f} px), "
                f"long {w['w_l']:.2f} (ℓ {w['ell_l']:.0f} px); "
                + (
                    "time white"
                    if t is None
                    else f"time {t['w_t']:.2f} × exp(−Δt/{t['tau']:.1f} d)"
                )
            )
    return ResidualKernel(
        a=a,
        f=f,
        r_table=r_table,
        centers=centers,
        params=params,
        lag_curves={
            "lags": lags,
            "est": est,
            "err": err,
            "t_mid": t_mid,
            "t_est": t_est,
            "t_err": t_err,
        },
        n_pixels=int(use.sum()),
    )


# === Draws ===
def draw_correlated(
    rng: np.random.Generator,
    scale: "npt.NDArray[np.float64]",
    wavelength: dict[str, Any],
    time: dict[str, Any] | None,
    days: "npt.NDArray[np.float64]",
    rows: "npt.NDArray[np.bool_]",
) -> "npt.NDArray[np.float64]":
    """Exact draw of ε = s ⊙ η with Cov(η) = K_λ ⊗ K_t, independently per SN.

    Args:
        rng: Random generator.
        scale: ``(n_sn, n_rows, n_wl)`` per-pixel amplitude s (e.g. a·σ, f, r|m|).
        wavelength: Numeric wavelength kernel parameters (see `k_lambda`).
        time: Numeric time kernel parameters, or None for independent epochs.
        days: ``(n_sn, n_rows)`` epoch times in days.
        rows: ``(n_sn, n_rows)`` which rows are real epochs.

    Returns:
        ε with the shape of `scale`; zero on rows that aren't epochs.
    """
    scale = np.asarray(scale, dtype=np.float64)
    n_sn, _, n_wl = scale.shape
    grid = np.arange(n_wl, dtype=np.float64)
    L_lam = psd_sqrt(k_lambda(grid[:, None] - grid[None], wavelength))
    eps = np.zeros_like(scale)
    for s in range(n_sn):
        idx = np.flatnonzero(rows[s])
        if not idx.size:
            continue
        t = np.asarray(days[s, idx], dtype=np.float64)
        L_t = psd_sqrt(k_time_rows(t, time))
        z = rng.normal(size=(n_wl, idx.size))
        eps[s, idx] = (L_lam @ z @ L_t.T).T
    return eps * scale


def band_noise_sigma(
    sigma: "npt.NDArray[np.floating]",
    weights: "npt.NDArray[np.floating]",
    wavelength: dict[str, Any],
) -> "npt.NDArray[np.float64]":
    """Band 1σ from pixel σ with correlated pixel noise: √(hᵀ K_λ h), h = g ∘ σ.

    Args:
        sigma: ``(..., n_wl)`` pixel 1σ (zero where invalid).
        weights: ``(n_wl, n_f)`` or ``(..., n_wl, n_f)`` band weights.
        wavelength: Numeric noise wavelength kernel parameters.

    Returns:
        ``(..., n_f)``.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    n_wl = sigma.shape[-1]
    grid = np.arange(n_wl, dtype=np.float64)
    K = k_lambda(grid[:, None] - grid[None], wavelength)
    h = np.asarray(weights, dtype=np.float64) * sigma[..., None]  # (..., n_wl, n_f)
    return np.sqrt(
        np.clip(np.einsum("...ib,...ib->...b", h, np.matmul(K, h)), 0.0, None)
    )
