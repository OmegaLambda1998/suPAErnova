"""Per-SN Gaussian likelihood with correlated noise, floor and model error.

One Gaussian per SN over all its epochs. Spectra are first averaged into
super-pixels of ``w`` pixels (``A``), so the data are each spectrum epoch's
super-pixels ``d̄ = A d`` and each band ``g·d``. With the model ``m`` and model
error ``e = r(λ, t)|m|``, the covariance is

    C = A (N + F + E) Aᵀ  (bands through g instead of A),
    N = a² D_σ K_n D_σ ⊗ I_t,   F = f² K_f ⊗ K_f^t,   E = D_e (K_m ⊗ K_m^t) D_e,

with each wavelength kernel split into a local part (white + Matérn-3/2) and a
long-range part (smooth, ``q`` eigenmodes Φ, values λ). It is evaluated as

    C = B + Ũ Ũᵀ,

* ``B``: block-diagonal. One block per super-pixel and one per filter, each over
  the SN's epochs (≤ R × R). It holds the measurement noise and each term's local
  part within the super-pixel (or band) times its time kernel. Local
  correlations *between* super-pixels, and between a band and a spectrum, are
  dropped: the documented approximation, within ~10% of the exact σ(ΔM) for
  ``w ≈ 3`` (uncertainty study).
* ``Ũ = U (L_t ⊗ Λ^½)``: each term's long-range part, U holding the modes binned
  through ``A`` (or ``g``) and scaled by the term's amplitude, ``L_t L_tᵀ = K^t``.

Then, with ``M = I + Ũᵀ B⁻¹ Ũ`` and ``v = Ũᵀ B⁻¹ ρ`` (all in float64),

    ρᵀ C⁻¹ ρ = ρᵀ B⁻¹ ρ − vᵀ M⁻¹ v,      log|C| = log|B| + log|M|.

`build_kernel_structure` precomputes everything that doesn't depend on the
model (numpy, once); `sn_kernel_log_likelihood` evaluates the rest per call.
"""

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from supaernova._tf import tf
from supaernova.utils.kernels import eigenmodes, k_lambda, matern32

if TYPE_CHECKING:
    from numpy import typing as npt

    from supaernova.configs.uncertainty import UncertaintyModel
    from supaernova.utils.kernels import ResidualKernel

DT = tf.float64

# Which axes of each per-SN structure array index the SN's (compacted) rows
ROW_AXES: dict[str, tuple[int, ...]] = {
    **dict.fromkeys(
        (
            "A",
            "sp_valid",
            "data_bar",
            "noise_var",
            "U_noise",
            "U_floor",
            "A_binned",
            "band_active",
            "band_flux",
            "band_noise_var",
            "band_weights",
            "U_floor_band",
            "floor_band_local",
            "floor_band_g",
            "days",
            "row_valid",
            "Ltres_floor",
            "Ltres_model",
        ),
        (1,),
    ),
    **dict.fromkeys(("Kt_floor", "Kt_model"), (1, 2)),
    # L_t: rows on axis 1; axis 2 holds time modes, which aren't row-indexed
    # unless every mode is kept (then they are the rows themselves)
    **dict.fromkeys(("Lt_floor", "Lt_model"), (1,)),
    **dict.fromkeys(("B_floor", "Bb_floor"), (2, 3)),
}
STATIC_KEYS = (
    "modes_model",
    "stencils",
    "local",
    "time",
    "n_sp",
    "bin_width",
    "dtype",
    "local_time",
)


def _local(p: dict[str, Any]) -> dict[str, Any]:
    """The local part (white + Matérn) of a wavelength kernel, as kernel params."""
    return {**p, "w_l": 0.0}


def _local_matrix(p: dict[str, Any], n: int) -> "npt.NDArray[np.float64]":
    offs = np.arange(n, dtype=np.float64)
    return k_lambda(offs[:, None] - offs[None], _local(p))


def _stencil(p: dict[str, Any]) -> "npt.NDArray[np.float64]":
    half = int(np.ceil(12 * max(p["ell_s"], 0.5))) if p["w_s"] > 0 else 0
    offs = np.arange(-half, half + 1, dtype=np.float64)
    return p["w0"] * (offs == 0) + p["w_s"] * matern32(offs, p["ell_s"])


def build_kernel_structure(
    *,
    rk: "ResidualKernel",
    uncertainty: "UncertaintyModel",
    amplitude: "npt.NDArray[np.floating]",
    sigma: "npt.NDArray[np.floating]",
    spec_pixels: "npt.NDArray[np.bool_]",
    band_active: "npt.NDArray[np.bool_] | None",
    band_flux: "npt.NDArray[np.floating] | None",
    band_sigma: "npt.NDArray[np.floating] | None",
    band_weights: "npt.NDArray[np.floating] | None",
    band_source_sigma: "npt.NDArray[np.floating] | None",
) -> dict[str, "npt.NDArray[np.float64]"]:
    """Model-independent parts of the kernel likelihood, per (SN, row).

    Every returned array has leading shape ``(n_sn, n_rows)``, so it can be
    gathered with the posterior's other per-row data.
    """
    kernel = uncertainty.kernel
    w = kernel.bin_width
    n_sn, n_rows, n_wl = amplitude.shape
    n_sp = -(-n_wl // w)
    pad = n_sp * w - n_wl

    valid = np.asarray(spec_pixels, dtype=bool)
    counts = np.add.reduceat(
        np.pad(valid, ((0, 0), (0, 0), (0, pad))), np.arange(0, n_sp * w, w), axis=-1
    )
    bin_of = np.arange(n_wl) // w
    A = np.where(valid, 1.0 / np.maximum(counts[..., bin_of], 1), 0.0)
    sp_valid = counts > 0
    amp = np.where(valid, np.asarray(amplitude, dtype=np.float64), 0.0)
    sig = np.where(valid, np.asarray(sigma, dtype=np.float64), 0.0)

    def binned(x):
        return np.add.reduceat(
            np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)]),
            np.arange(0, n_sp * w, w),
            axis=-1,
        )

    params = {name: dict(rk.term(name)) for name in ("noise", "floor", "model")}
    q = {
        name: (
            getattr(kernel, name).n_modes
            if getattr(kernel, name).n_modes is not None
            else kernel.n_modes
        )
        for name in params
    }

    # A term without long-range modes keeps its long-range variance as white
    # noise, so each pixel's variance is unchanged (only that correlation is lost)
    for name, p in params.items():
        if q[name] == 0 and p["wavelength"]["w_l"] > 0:
            wl = dict(p["wavelength"])
            wl["w0"], wl["w_l"] = wl["w0"] + wl["w_l"], 0.0
            p["wavelength"] = wl

    # Noise: local variance within each super-pixel; long range as modes
    a = rk.a
    Kn_loc = _local_matrix(params["noise"]["wavelength"], w)
    As = np.pad(A * sig, ((0, 0), (0, 0), (0, pad))).reshape(n_sn, n_rows, n_sp, w)
    noise_var = a * a * np.einsum("srpi,ij,srpj->srp", As, Kn_loc, As)

    out: dict[str, Any] = {
        "A": A,
        "sp_valid": sp_valid,
        "data_bar": binned(A * amp),
        "noise_var": noise_var,
    }
    modes = {}
    for name, p in params.items():
        n_q = q[name] if p["wavelength"]["w_l"] > 0 else 0
        vals, vecs = (
            eigenmodes(n_wl, p["wavelength"], n_q)
            if n_q
            else (np.zeros(0), np.zeros((n_wl, 0)))
        )
        modes[name] = vecs * np.sqrt(vals)  # (n_wl, q)
    out["modes_model"] = modes["model"]
    out["U_noise"] = a * np.einsum(
        "srpi,pik->srpk",
        As,
        np.pad(modes["noise"], ((0, pad), (0, 0))).reshape(n_sp, w, -1),
    )
    Af = np.pad(A, ((0, 0), (0, 0), (0, pad))).reshape(n_sn, n_rows, n_sp, w)
    out["U_floor"] = rk.f * np.einsum(
        "srpi,pik->srpk",
        Af,
        np.pad(modes["floor"], ((0, pad), (0, 0))).reshape(n_sp, w, -1),
    )
    out["A_binned"] = Af  # (n_sn, n_rows, n_sp, w), for the floor's local blocks

    n_f = 0 if band_weights is None else band_weights.shape[-1]
    if n_f:
        g = np.asarray(band_weights, dtype=np.float64)
        act = np.asarray(band_active, dtype=bool)
        # Band noise is entirely in B, so it keeps the full noise kernel
        noise_p = rk.term("noise")["wavelength"]
        grid = np.arange(n_wl, dtype=np.float64)
        Kn = k_lambda(grid[:, None] - grid[None], noise_p)
        src = (
            None
            if band_source_sigma is None
            else np.asarray(band_source_sigma, dtype=np.float64)
        )
        bsig = np.asarray(band_sigma, dtype=np.float64)
        if src is not None and np.any(src > 0):
            h = g * src[..., None]
            rebuilt = np.sqrt(
                np.clip(
                    np.einsum("srib,srib->srb", h, np.einsum("ij,srjb->srib", Kn, h)),
                    0,
                    None,
                )
            )
            has_src = np.any(src > 0, axis=-1, keepdims=True)
            bsig = np.where(has_src, rebuilt, bsig)
        out["band_active"] = act
        out["band_flux"] = np.where(act, np.asarray(band_flux, dtype=np.float64), 0.0)
        out["band_noise_var"] = np.where(act, a * a * bsig * bsig, 1.0)
        out["band_weights"] = g
        on_bands = kernel.floor.on_bands
        out["U_floor_band"] = (rk.f if on_bands else 0.0) * np.einsum(
            "srib,ik->srbk", g, modes["floor"]
        )
        floor_stencil = _stencil(params["floor"]["wavelength"])
        Gf = (rk.f if on_bands else 0.0) * g
        out["floor_band_local"] = _conv_np(Gf, floor_stencil)  # K_loc (f g)
        out["floor_band_g"] = Gf

    out["stencils"] = {
        "model": _stencil(params["model"]["wavelength"]),
    }
    out["local"] = {
        "floor": rk.f * rk.f * _local_matrix(params["floor"]["wavelength"], w),
        "model": _local_matrix(params["model"]["wavelength"], w),
    }
    out["time"] = {"floor": params["floor"]["time"], "model": params["model"]["time"]}
    out["n_sp"] = n_sp
    out["bin_width"] = w
    return out


def add_constants(
    struct: dict[str, Any],
    days: "npt.NDArray[np.floating]",
    row_valid: "npt.NDArray[np.bool_]",
    time_modes: int | None = None,
) -> dict[str, Any]:
    """Precompute the model-independent time kernels, their roots and floor blocks.

    ``days`` / ``row_valid`` are ``(n_sn, R)`` for the rows `struct` holds.
    """
    valid = np.asarray(row_valid, dtype=bool)
    days = np.where(valid, np.asarray(days, dtype=np.float64), 0.0)
    eye = np.eye(valid.shape[-1])
    pair = valid[..., :, None] & valid[..., None, :]
    out = dict(struct)
    for name in ("floor", "model"):
        p = struct["time"][name]
        if p is None:
            K = np.broadcast_to(eye, pair.shape).copy()
        else:
            dt = np.abs(days[..., :, None] - days[..., None, :])
            K = (1.0 - p["w_t"]) * eye + p["w_t"] * np.exp(-dt / p["tau"])
        # Padding rows get no time modes (eigenvalue 0), so truncation keeps real ones
        K = np.where(pair, K, 0.0)
        vals, vecs = np.linalg.eigh(K)  # ascending
        L = vecs * np.sqrt(np.clip(vals, 0.0, None))[..., None, :]
        if time_modes is not None and time_modes < L.shape[-1]:
            L = L[..., -time_modes:]  # leading time modes
            res = np.einsum("...rr->...r", K) - np.sum(L * L, axis=-1)
            out[f"Ltres_{name}"] = np.where(valid, np.clip(res, 0.0, None), 0.0)
        out[f"Kt_{name}"] = K
        out[f"Lt_{name}"] = L
    A_b = np.asarray(struct["A_binned"], dtype=np.float64)
    KAf = np.einsum("srpi,ij->srpj", A_b, struct["local"]["floor"])
    out["B_floor"] = np.einsum("srpi,stpi->sprt", A_b, KAf) * out["Kt_floor"][:, None]
    if "floor_band_g" in struct:
        out["Bb_floor"] = (
            np.einsum(
                "srib,stib->sbrt", struct["floor_band_g"], struct["floor_band_local"]
            )
            * out["Kt_floor"][:, None]
        )
    return out


def _conv_np(
    h: "npt.NDArray[np.float64]", stencil: "npt.NDArray[np.float64]"
) -> "npt.NDArray[np.float64]":
    """Convolve (…, n_wl, n_f) with a symmetric stencil along n_wl (zero-padded)."""
    half = stencil.size // 2
    padded = np.pad(h, [(0, 0)] * (h.ndim - 2) + [(half, half), (0, 0)])
    out = np.zeros_like(h)
    n = h.shape[-2]
    for k, c in enumerate(stencil):
        if c:
            out += c * padded[..., k : k + n, :]
    return out


def _conv_tf(h: tf.Tensor, stencil: "npt.NDArray[np.float64]") -> tf.Tensor:
    half = stencil.size // 2
    if half == 0:
        return float(stencil[0]) * h
    n = h.shape[-2]
    paddings = [[0, 0]] * (len(h.shape) - 2) + [[half, half], [0, 0]]
    padded = tf.pad(h, paddings)
    out = tf.zeros_like(h)
    for k, c in enumerate(stencil):
        if c:
            out += float(c) * padded[..., k : k + n, :]
    return out


def _time_kernel(
    days: tf.Tensor, valid: tf.Tensor, p: dict[str, Any] | None, dtype: tf.DType = DT
) -> tf.Tensor:
    """K^t over an SN's rows (S, R, R), identity on padding and for ``p = None``."""
    n = days.shape[-1]
    eye = tf.eye(n, dtype=dtype)
    if p is None:
        return eye + tf.zeros_like(days[..., None])
    sep = tf.abs(days[..., :, None] - days[..., None, :])
    k = (1.0 - p["w_t"]) * eye + p["w_t"] * tf.exp(-sep / p["tau"])
    pair = tf.cast(valid[..., :, None] & valid[..., None, :], dtype)
    return k * pair + eye * (1.0 - pair)


def _psd_sqrt(K: tf.Tensor) -> tf.Tensor:
    vals, vecs = tf.linalg.eigh(K)
    return vecs * tf.sqrt(tf.maximum(vals, 0.0))[..., None, :]


def _block_cholesky(
    B: tf.Tensor, valid: tf.Tensor, jitter: float, dtype: tf.DType = DT
) -> tf.Tensor:
    """Cholesky of (…, R, R) blocks, with invalid rows/cols replaced by identity."""
    n = B.shape[-1]
    eye = tf.eye(n, dtype=dtype)
    pair = tf.cast(valid[..., :, None] & valid[..., None, :], dtype)
    # Jitter relative to the block's real variances (padding is set to identity
    # below, and must not set the scale)
    real = tf.abs(tf.linalg.diag_part(B)) * tf.cast(valid, dtype)
    scale = tf.reduce_max(real, axis=-1, keepdims=True)[..., None]
    B = B * pair + eye * (1.0 - pair)
    return tf.linalg.cholesky(B + jitter * scale * eye * pair)


def sn_kernel_log_likelihood(
    *,
    model: tf.Tensor,
    model_error: tf.Tensor,
    days: tf.Tensor,
    row_valid: tf.Tensor,
    struct: dict[str, Any],
    jitter: float = 1e-10,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Per-SN log-likelihood.

    Args:
        model: ``(..., n_sn, R, n_wl)`` model spectra (rows already compacted).
        model_error: Same shape, e = r(λ, t)·|model| (or another scale).
        days: ``(n_sn, R)`` epoch times in days.
        row_valid: ``(n_sn, R)`` rows that are real epochs.
        struct: From `build_kernel_structure`, gathered to the same rows, with
            its arrays as tensors.
        jitter: Relative diagonal jitter for the Cholesky factorisations.

    Returns:
        ``(log_likelihood (..., n_sn), n_data (n_sn,))``.
    """
    dt = struct.get("dtype", DT)
    local_time = struct.get("local_time", True)
    m = tf.cast(model, dt)
    e = tf.cast(model_error, dt)
    # An SN whose model is non-finite (e.g. a wild MAP start) is evaluated on
    # zeros, so no factorisation fails, and its log-likelihood is set to NaN
    # (mapped to -inf by the caller), as in `epoch_log_likelihood`
    finite = tf.reduce_all(tf.math.is_finite(m) & tf.math.is_finite(e), axis=[-2, -1])
    m = tf.where(finite[..., None, None], m, 0.0)
    e = tf.where(finite[..., None, None], e, 0.0)
    w, n_sp = struct["bin_width"], struct["n_sp"]
    n_wl = m.shape[-1]
    pad = n_sp * w - n_wl
    days = tf.cast(days, dt)
    row_valid = tf.cast(row_valid, tf.bool)

    def to_bins(x):  # (…, n_wl) → (…, n_sp, w)
        x = tf.pad(x, [[0, 0]] * (len(x.shape) - 1) + [[0, pad]])
        return tf.reshape(x, [*x.shape[:-1], n_sp, w])

    A = tf.cast(struct["A"], dt)
    sp_valid = tf.cast(struct["sp_valid"], tf.bool) & row_valid[..., None]
    A_b = to_bins(A)
    m_bar = tf.reduce_sum(to_bins(A * m), axis=-1)  # (…, S, R, P)
    rho = tf.where(sp_valid, tf.cast(struct["data_bar"], dt) - m_bar, 0.0)

    Kt = {
        k: tf.cast(struct[f"Kt_{k}"], dt)
        if f"Kt_{k}" in struct
        else _time_kernel(days, row_valid, struct["time"][k], dt)
        for k in ("floor", "model")
    }

    # --- Spectral blocks over epochs, per super-pixel: (…, S, P, R, R) ---
    V = to_bins(A * e)  # (…, S, R, P, w)

    # --- Long-range modes, per term: loadings W[r, p, k] and time factors L_t ---
    # Ũ[(r, p), (term, r', k)] = W[r, p, k] L_t[r, r'], so Ũᵀ B⁻¹ Ũ is assembled
    # from G₀[r, k, s, j] = Σ_p W[r, p, k] (B_p⁻¹)[r, s] W'[s, p, j] without ever
    # forming Ũ (whose size grows as R² q per super-pixel). L_t may keep only the
    # leading time modes; each row's dropped time variance ("Ltres") is put back
    # on B's diagonal, so every datum's variance stays exact.
    n_rows = row_valid.shape[-1]
    eye_r = tf.eye(n_rows, dtype=dt) + tf.zeros_like(days[..., None])
    modes_m = tf.constant(struct["modes_model"], dt)
    W_m = tf.einsum("...rpi,kpi->...rpk", V, to_bins(tf.transpose(modes_m)))
    L_tm = (
        tf.cast(struct["Lt_model"], dt)
        if "Lt_model" in struct
        else _psd_sqrt(Kt["model"])
    )
    L_tf = (
        tf.cast(struct["Lt_floor"], dt)
        if "Lt_floor" in struct
        else _psd_sqrt(Kt["floor"])
    )
    U_floor = tf.cast(struct["U_floor"], dt)
    res_f = tf.cast(struct["Ltres_floor"], dt) if "Ltres_floor" in struct else None
    res_m = tf.cast(struct["Ltres_model"], dt) if "Ltres_model" in struct else None
    dropped = 0.0
    if res_f is not None:
        dropped += res_f[..., None] * tf.reduce_sum(U_floor * U_floor, axis=-1)
    if res_m is not None:
        dropped += res_m[..., None] * tf.reduce_sum(W_m * W_m, axis=-1)
    KV = tf.einsum("...rpi,ij->...rpj", V, tf.constant(struct["local"]["model"], dt))
    if local_time:
        B_m = tf.einsum("...rpi,...tpi->...prt", V, KV) * Kt["model"][..., None, :, :]
    if "B_floor" in struct:
        B_f = tf.cast(struct["B_floor"], dt)
    else:
        KAf = tf.einsum("srpi,ij->srpj", A_b, tf.constant(struct["local"]["floor"], dt))
        B_f = tf.einsum("srpi,stpi->sprt", A_b, KAf) * Kt["floor"][:, None, :, :]
    noise = tf.cast(struct["noise_var"], dt) + dropped  # (…, S, R, P)
    sp_t = (
        tf.transpose(sp_valid, [0, 2, 1])
        if len(sp_valid.shape) == 3
        else tf.linalg.matrix_transpose(sp_valid)
    )
    rho_s = tf.linalg.matrix_transpose(rho)[..., None]  # (…, S, P, R, 1)
    if local_time:
        B_s = B_m + B_f + tf.linalg.diag(tf.linalg.matrix_transpose(noise))
        L_s = _block_cholesky(B_s, sp_t, jitter, dt)
        logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L_s)) * tf.cast(sp_t, dt), axis=[-2, -1]
        )
    else:
        # Local parts independent between epochs: B is diagonal
        D_s = (
            noise
            + tf.einsum("...rpi,...rpi->...rp", V, KV)
            + tf.linalg.matrix_transpose(tf.linalg.diag_part(B_f))
        )
        D_s = tf.where(sp_valid, D_s, 1.0)  # (…, S, R, P)
        logdet = tf.reduce_sum(tf.math.log(D_s) * tf.cast(sp_valid, dt), axis=[-2, -1])
        L_s = tf.sqrt(D_s)  # only its batch shape is used below
    n_data = tf.reduce_sum(tf.cast(sp_valid, dt), axis=[-2, -1])

    # --- Band blocks over epochs, per filter: (…, S, F, R, R) ---
    n_f = 0 if "band_weights" not in struct else int(struct["band_weights"].shape[-1])
    if n_f:
        g = tf.cast(struct["band_weights"], dt)  # (S, R, n_wl, F)
        act = tf.cast(struct["band_active"], tf.bool) & row_valid[..., None]
        band_model = tf.einsum("...sri,srib->...srb", m, g)
        rho_b = tf.where(act, tf.cast(struct["band_flux"], dt) - band_model, 0.0)
        H = g * e[..., None]  # (…, S, R, n_wl, F)
        KH = _conv_tf(H, struct["stencils"]["model"])
        if local_time:
            Bb_m = (
                tf.einsum("...rib,...tib->...brt", H, KH) * Kt["model"][..., None, :, :]
            )
        if "Bb_floor" in struct:
            Bb_f = tf.cast(struct["Bb_floor"], dt)
        else:
            Gf = tf.cast(struct["floor_band_g"], dt)
            KGf = tf.cast(struct["floor_band_local"], dt)
            Bb_f = tf.einsum("srib,stib->sbrt", Gf, KGf) * Kt["floor"][:, None, :, :]
        H_m_modes = tf.einsum("...rib,ik->...rbk", H, modes_m)
        U_floor_band = tf.cast(struct["U_floor_band"], dt)
        bnoise = tf.cast(struct["band_noise_var"], dt)
        if res_f is not None:
            bnoise += res_f[..., None] * tf.reduce_sum(U_floor_band**2, axis=-1)
        if res_m is not None:
            bnoise += res_m[..., None] * tf.reduce_sum(H_m_modes**2, axis=-1)
        act_t = tf.linalg.matrix_transpose(act)  # (S, F, R)
        if local_time:
            B_b = Bb_m + Bb_f + tf.linalg.diag(tf.linalg.matrix_transpose(bnoise))
            L_b = _block_cholesky(B_b, act_t, jitter, dt)
            logdet += 2.0 * tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(L_b)) * tf.cast(act_t, dt),
                axis=[-2, -1],
            )
        else:
            D_b = (
                bnoise
                + tf.einsum("...rib,...rib->...rb", H, KH)
                + tf.linalg.matrix_transpose(tf.linalg.diag_part(Bb_f))
            )
            D_b = tf.where(act, D_b, 1.0)  # (…, S, R, F)
            logdet += tf.reduce_sum(tf.math.log(D_b) * tf.cast(act, dt), axis=[-2, -1])
            L_b = tf.sqrt(D_b)
        n_data += tf.reduce_sum(tf.cast(act, dt), axis=[-2, -1])

    spec_terms = [
        (tf.cast(struct["U_noise"], dt), eye_r),
        (U_floor, L_tf),
        (W_m, L_tm),
    ]
    spec_mask_w = tf.cast(sp_valid, dt)[..., None]  # (S, R, P, 1)
    spec_terms = [(W * spec_mask_w, L) for W, L in spec_terms if W.shape[-1] > 0]

    eye_rr = tf.eye(n_rows, dtype=dt)
    if local_time:
        Binv_s = tf.linalg.cholesky_solve(
            L_s, eye_rr + tf.zeros_like(L_s)
        )  # (…, S, P, R, R)
        y_s = tf.linalg.cholesky_solve(L_s, rho_s)[..., 0]  # B⁻¹ρ, (…, S, P, R)
    else:
        Binv_s = None
        y_s = tf.linalg.matrix_transpose(rho / D_s)  # (…, S, P, R)
    quad = tf.reduce_sum(rho_s[..., 0] * y_s, axis=[-2, -1])

    band_terms = []
    if n_f:
        act_w = tf.cast(act, dt)[..., None]  # (S, R, F, 1)
        band_terms = [
            (U_floor_band * act_w, L_tf),
            (H_m_modes * act_w, L_tm),
        ]
        if local_time:
            Binv_b = tf.linalg.cholesky_solve(L_b, eye_rr + tf.zeros_like(L_b))
            y_b = tf.linalg.cholesky_solve(
                L_b, tf.linalg.matrix_transpose(rho_b)[..., None]
            )[..., 0]
        else:
            Binv_b = None
            y_b = tf.linalg.matrix_transpose(rho_b / D_b)  # (…, S, F, R)
        quad += tf.reduce_sum(tf.linalg.matrix_transpose(rho_b) * y_b, axis=[-2, -1])

    # Stack every term's modes into one loading matrix W[r, p, K] (bands: W_b[r, b, K];
    # noise has no band loadings) and a per-column time factor L[r, a, K]
    if spec_terms:
        n_noise = 1 if tf.cast(struct["U_noise"], dt).shape[-1] > 0 else 0
        lead = tf.shape(L_s)[:-3] if local_time else tf.shape(L_s)[:-2]  # (…, S)
        n_cols = max(int(L.shape[-1]) for _, L in spec_terms)
        W_cols, Wb_cols, L_cols = [], [], []
        for i, (W, L) in enumerate(spec_terms):
            q_i = W.shape[-1]
            W_cols.append(tf.broadcast_to(W, tf.concat([lead, tf.shape(W)[-3:]], 0)))
            j = i - n_noise
            if band_terms and j >= 0:
                Wb = band_terms[j][0]
                Wb_cols.append(
                    tf.broadcast_to(Wb, tf.concat([lead, tf.shape(Wb)[-3:]], 0))
                )
            elif band_terms:
                Wb_cols.append(tf.zeros(tf.concat([lead, [n_rows, n_f, q_i]], 0), dt))
            L = tf.pad(
                L, [[0, 0]] * (len(L.shape) - 1) + [[0, n_cols - int(L.shape[-1])]]
            )
            L_cols.append(tf.repeat(L[..., None], q_i, axis=-1))  # (S, R, C, q_i)
        W_all = tf.concat(W_cols, axis=-1)  # (…, S, R, P, Q)
        L_all = tf.concat(L_cols, axis=-1)  # (S, R, C, Q)

        def gram(W, Binv):
            """G₀[r, K, s, J] = Σ_p W[r, p, K] Binv[p, r, s] W[s, p, J] (two steps)."""
            W_p = tf.einsum("...rpk->...prk", W)  # (…, S, P, R, Q)
            T = Binv[..., :, :, :, None] * W_p[..., :, None, :, :]  # (…, S, P, R, R, Q)
            n_q = tf.shape(W)[-1]
            T = tf.reshape(T, tf.concat([tf.shape(T)[:-2], [n_rows * n_q]], 0))
            Wt = tf.einsum("...prk->...rkp", W_p)  # (…, S, R, Q, P)
            T = tf.einsum("...prx->...rpx", T)  # (…, S, R, P, R·Q)
            G0 = tf.linalg.matmul(Wt, T)  # (…, S, R, Q, R·Q)
            return tf.reshape(G0, tf.concat([tf.shape(G0)[:-1], [n_rows, n_q]], 0))

        z = tf.einsum("...rpk,...pr->...rk", W_all, y_s)  # (…, S, R, Q)
        if band_terms:
            Wb_all = tf.concat(Wb_cols, axis=-1)  # (…, S, R, F, Q)
            z += tf.einsum("...rbk,...br->...rk", Wb_all, y_b)
        if local_time:
            G0 = gram(W_all, Binv_s)
            if band_terms:
                G0 += gram(Wb_all, Binv_b)
            # Time factors: G[a, K, c, J] = Σ_rs L[r, a, K] G₀[r, K, s, J] L[s, c, J]
            X = tf.einsum("...rksj,...rak->...aksj", G0, L_all)
            G = tf.einsum("...aksj,...scj->...akcj", X, L_all)
        else:
            # B diagonal: G₀ is block-diagonal over epochs, G_d[r, K, J]
            Gd = tf.einsum("...rpk,...rpj->...rkj", W_all / D_s[..., None], W_all)
            if band_terms:
                Gd += tf.einsum(
                    "...rbk,...rbj->...rkj", Wb_all / D_b[..., None], Wb_all
                )
            Y = L_all[..., :, :, :, None] * Gd[..., :, None, :, :]  # (…, S, R, C, Q, Q)
            G = tf.einsum("...rakj,...rcj->...akcj", Y, L_all)
        v = tf.einsum("...rk,...rak->...ak", z, L_all)
        size = n_cols * int(W_all.shape[-1])
        G = tf.reshape(G, tf.concat([tf.shape(G)[:-4], [size, size]], 0))
        v = tf.reshape(v, tf.concat([tf.shape(v)[:-2], [size]], 0))
        M = tf.eye(size, dtype=dt) + G
        L_M = tf.linalg.cholesky(M)
        u = tf.linalg.triangular_solve(L_M, v[..., None], lower=True)[..., 0]
        quad -= tf.reduce_sum(u * u, axis=-1)
        logdet += 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_M)), axis=-1)

    ll = -0.5 * (quad + logdet + n_data * math.log(2 * math.pi))
    ll = tf.where(finite, ll, tf.constant(np.nan, dt))
    return tf.cast(ll, model.dtype), tf.cast(n_data, model.dtype)
