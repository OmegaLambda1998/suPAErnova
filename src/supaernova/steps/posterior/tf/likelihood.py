"""Per-epoch Gaussian likelihood with a shared measurement + model-error covariance.

For one epoch, the data are its spectrum pixels ``S`` (if observed) and its
bands ``b`` (if observed). With the model spectrum ``m``, per-pixel model-error
amplitude ``e = σ_recon · scale`` and band weights ``g`` (so band model is
``g·m``), the pixel-space model error is

    Σ = (1 - f) diag(e²) + f e eᵀ,

and the data covariance is ``C = N + P Σ Pᵀ``, where ``N`` is the measurement
covariance and ``P`` picks spectrum pixels and integrates bands. Every variant
(spectrum only, photometry only, both) uses this one covariance.

Written as ``C = Λ + Ũ Q̃ Ũᵀ`` with

    Λ = diag([σ_S² + (1 - f) e_S², σ_b²])
    Ũ = [X | I | w],  X[:, b] = [(1 - f) e_S² g_Sb; 0],  I[:, b] = [0; δ_b],
                      w = [e_S; u],  u_b = Σ_i g_ib e_i
    Q̃ = [[0, I, 0], [I, B, 0], [0, 0, f]],  B = (1 - f) Σ_i g_i g_iᵀ e_i²,

it is evaluated exactly with the Woodbury identity and the matrix determinant
lemma, whose inner system has size at most ``2 n_f + 1``:

    ρᵀ C⁻¹ ρ = ρᵀ Λ⁻¹ ρ - vᵀ (I + Q̃ A)⁻¹ Q̃ v,   log|C| = log|Λ| + log|I + Q̃ A|,

with ``A = ŨᵀΛ⁻¹Ũ`` and ``v = ŨᵀΛ⁻¹ρ``. The ``X`` columns only exist for rows
holding both a spectrum and bands.
"""

import math

from supaernova._tf import tf


def _sum_wl(x: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(x, axis=-1)


def _project(x: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    """``Σ_i x_i g_ib``: ``x`` ``(..., n_sn, n_rows, n_wl)``, ``g`` ``(n_sn, n_rows, n_wl, n_f)``.

    Returns:
        ``(..., n_sn, n_rows, n_f)``.
    """
    return tf.reduce_sum(x[..., None] * weights, axis=-2)


def _gram(x: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    """``Σ_i x_i² g_ib g_ic`` as a batched matrix product.

    Returns:
        ``(..., n_sn, n_rows, n_f, n_f)``.
    """
    h = weights * x[..., None]
    return tf.linalg.matmul(h, h, transpose_a=True)


def epoch_log_likelihood(
    *,
    model: tf.Tensor,
    model_error: tf.Tensor,
    correlation: float,
    spec_pixels: tf.Tensor,
    spec_flux: tf.Tensor,
    spec_variance: tf.Tensor,
    band_active: tf.Tensor | None,
    band_flux: tf.Tensor | None,
    band_variance: tf.Tensor | None,
    band_weights: tf.Tensor | None,
    joint: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Log-likelihood of every row (each row holds one epoch's data).

    Args:
        model: ``(..., n_sn, n_rows, n_wl)`` model spectrum.
        model_error: ``(..., n_sn, n_rows, n_wl)`` per-pixel model-error amplitude ``e``.
        correlation: ``f``, the coherent fraction of the model-error variance.
        spec_pixels: ``(n_sn, n_rows, n_wl)`` which pixels are spectrum data.
        spec_flux: ``(n_sn, n_rows, n_wl)`` spectrum data.
        spec_variance: ``(n_sn, n_rows, n_wl)`` spectrum measurement variance.
        band_active: ``(n_sn, n_rows, n_f)`` which bands are data, or None.
        band_flux: ``(n_sn, n_rows, n_f)`` band data.
        band_variance: ``(n_sn, n_rows, n_f)`` band measurement variance.
        band_weights: ``(n_sn, n_rows, n_wl, n_f)`` band integration weights ``g``.
        joint: Whether any row holds both spectrum pixels and bands.

    Returns:
        ``(log_likelihood, n_data)``, each ``(..., n_sn, n_rows)``.
    """
    dt = model.dtype
    f = tf.constant(correlation, dtype=dt)
    one_f = 1.0 - f
    spec = tf.cast(spec_pixels, dt)
    # Broadcast so every term carries the model's batch (walker) dimensions
    e = model_error + tf.zeros_like(model)
    e2 = e * e

    lam = tf.where(spec_pixels, spec_variance + one_f * e2, tf.ones_like(e2))
    a = spec / lam
    rho = spec * (spec_flux - model)
    quad = _sum_wl(a * rho * rho)
    logdet = _sum_wl(spec * tf.math.log(lam))
    n_data = _sum_wl(spec) + tf.zeros_like(quad)

    a_ww = _sum_wl(a * e2)
    v_w = _sum_wl(a * e * rho)

    n_f = 0 if band_weights is None else int(band_weights.shape[-1])
    if n_f == 0:
        if correlation > 0:
            denom = 1.0 + f * a_ww
            quad -= f * v_w * v_w / denom
            logdet += tf.math.log(denom)
        ll = -0.5 * (quad + logdet + n_data * math.log(2 * math.pi))
        return ll, n_data

    act = tf.cast(band_active, dt)
    s2 = tf.where(band_active, band_variance, tf.ones_like(band_variance))
    rho_b = act * (band_flux - _project(model, band_weights))
    u = act * _project(e, band_weights)
    quad += tf.reduce_sum(rho_b * rho_b / s2, axis=-1)
    logdet += tf.reduce_sum(act * tf.math.log(s2), axis=-1)
    n_data += tf.reduce_sum(act, axis=-1)

    pair = act[..., :, None] * act[..., None, :]
    big_b = one_f * _gram(e, band_weights) * pair

    a_ii = tf.linalg.diag(act / s2)
    a_iw = u / s2
    a_ww += tf.reduce_sum(u * u / s2, axis=-1)
    v_i = rho_b / s2
    v_w += tf.reduce_sum(u * rho_b / s2, axis=-1)

    shape = tf.shape(big_b)
    zeros_ff = tf.zeros(shape, dtype=dt)
    zeros_f1 = tf.zeros(tf.concat([shape[:-1], [1]], 0), dtype=dt)
    eye_act = tf.linalg.diag(act) + zeros_ff

    if joint:
        a_xx = one_f * one_f * _gram(e2 * tf.sqrt(a), band_weights) * pair
        a_xw = one_f * act * _project(a * e2 * e, band_weights)
        v_x = one_f * act * _project(a * e2 * rho, band_weights)
        mat_a = tf.concat(
            [
                tf.concat([a_xx, zeros_ff, a_xw[..., None]], -1),
                tf.concat([zeros_ff, a_ii + zeros_ff, a_iw[..., None]], -1),
                tf.concat(
                    [a_xw[..., None, :], a_iw[..., None, :], a_ww[..., None, None]], -1
                ),
            ],
            -2,
        )
        vec_v = tf.concat([v_x, v_i, v_w[..., None]], -1)
        mat_q = tf.concat(
            [
                tf.concat([zeros_ff, eye_act, zeros_f1], -1),
                tf.concat([eye_act, big_b, zeros_f1], -1),
                tf.concat(
                    [
                        tf.linalg.matrix_transpose(zeros_f1),
                        tf.linalg.matrix_transpose(zeros_f1),
                        f + tf.zeros_like(a_ww)[..., None, None],
                    ],
                    -1,
                ),
            ],
            -2,
        )
    else:
        mat_a = tf.concat(
            [
                tf.concat([a_ii + zeros_ff, a_iw[..., None]], -1),
                tf.concat([a_iw[..., None, :], a_ww[..., None, None]], -1),
            ],
            -2,
        )
        vec_v = tf.concat([v_i, v_w[..., None]], -1)
        mat_q = tf.concat(
            [
                tf.concat([big_b, zeros_f1], -1),
                tf.concat(
                    [
                        tf.linalg.matrix_transpose(zeros_f1),
                        f + tf.zeros_like(a_ww)[..., None, None],
                    ],
                    -1,
                ),
            ],
            -2,
        )

    # The inner system mixes 1/σ² of high-S/N bands with small model-error terms,
    # so solve it in double precision
    mat_a = tf.cast(mat_a, tf.float64)
    mat_q = tf.cast(mat_q, tf.float64)
    vec_v = tf.cast(vec_v, tf.float64)
    # Samplers can step to where the model is non-finite. Solve a harmless identity
    # system there (inverting a non-finite matrix raises, rather than giving nan) and
    # return a nan log-likelihood, which callers turn into -inf.
    finite = (
        tf.reduce_all(tf.math.is_finite(mat_a), axis=[-2, -1])
        & tf.reduce_all(tf.math.is_finite(mat_q), axis=[-2, -1])
        & tf.reduce_all(tf.math.is_finite(vec_v), axis=-1)
    )
    mat_a = tf.where(finite[..., None, None], mat_a, tf.zeros_like(mat_a))
    mat_q = tf.where(finite[..., None, None], mat_q, tf.zeros_like(mat_q))
    vec_v = tf.where(finite[..., None], vec_v, tf.zeros_like(vec_v))
    k = int(mat_a.shape[-1])
    mat_m = tf.eye(k, dtype=tf.float64) + tf.linalg.matmul(mat_q, mat_a)
    _, logabsdet = tf.linalg.slogdet(mat_m)
    qv = tf.linalg.matvec(mat_q, vec_v)
    z = tf.linalg.solve(mat_m, qv[..., None])[..., 0]
    correction = tf.reduce_sum(vec_v * z, axis=-1)

    quad -= tf.cast(correction, dt)
    logdet += tf.cast(logabsdet, dt)
    ll = -0.5 * (quad + logdet + n_data * math.log(2 * math.pi))
    ll = tf.where(finite, ll, tf.constant(math.nan, dtype=dt))
    return ll, n_data
