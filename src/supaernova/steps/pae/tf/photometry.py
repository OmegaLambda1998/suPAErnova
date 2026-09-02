from typing import TYPE_CHECKING

from supaernova._tf import tf, tfp, ks, JIT_COMPILE
from supaernova.utils.tf import db, pp


@tf.function(jit_compile=JIT_COMPILE)
def _trapz_weights(wavelength: tf.Tensor) -> tf.Tensor:
    """Per-point quadrature weights of the trapezoidal rule along the wavelength axis (-2).

    trapz(f, x) = sum_i weights_i * f_i, so propagating independent per-point
    variance through the integral requires weights_i**2 (Var[c*X] = c**2 *
    Var[X]) applied to each squared term, not a second unweighted trapz call
    over the already-squared integrand (which reapplies weights_i once more,
    diluting the result by an extra, wavelength-spacing-dependent factor).

    Returns:
        The trapezoidal rule's per-point quadrature weights, same shape as `wavelength`.
    """
    dx = wavelength[..., 1:, :] - wavelength[..., :-1, :]
    return (
        tf.concat(
            [
                dx[..., :1, :],
                dx[..., :-1, :] + dx[..., 1:, :],
                dx[..., -1:, :],
            ],
            axis=-2,
        )
        * 0.5
    )


@tf.function(jit_compile=JIT_COMPILE)
def photometry_amplitude_setup(
    wavelength: tf.Tensor,
    throughput: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    wavelength = tf.cast(wavelength, tf.float32)
    throughput = tf.cast(throughput, tf.float32)
    # (n_sn, n_spec, 1, n_filters)
    denom = tfp.math.trapz(throughput * wavelength, wavelength, axis=-2)[..., None, :]
    amp_mask = tf.cast(
        tf.where(denom == 0, tf.zeros_like(denom), tf.ones_like(denom)), tf.bool
    )
    denom = tf.where(denom == 0, tf.ones_like(denom), denom)

    return denom, amp_mask


@tf.function(jit_compile=JIT_COMPILE)
def photometry_amplitude(
    wavelength: tf.Tensor,
    amplitude: tf.Tensor,
    throughput: tf.Tensor,
    effective_wavelength: tf.Tensor,
    spec_mask: tf.Tensor,
    phot_mask: tf.Tensor,
    cached: tuple[tf.Tensor, tf.Tensor] | None = None,
) -> tf.Tensor:

    if cached is None:
        denom, amp_mask = photometry_amplitude_setup(wavelength, throughput)
    else:
        denom, amp_mask = cached

    # (n_sn, n_spec, 1, n_filters)
    numer = tfp.math.trapz(
        amplitude * throughput * wavelength,
        wavelength,
        axis=-2,
    )[..., None, :]

    # (n_sn, n_spec, n_wl, n_filters)
    phot_amp = numer / denom
    phot_amp = tf.where(amp_mask, phot_amp, tf.zeros_like(phot_amp))

    # (n_sn, n_spec, n_wl, n_filters)
    phot_amp = spec_mask * amplitude + phot_mask * effective_wavelength * phot_amp

    # (n_sn, n_spec, n_wl)
    phot_amp = tf.reduce_sum(phot_amp, axis=-1)

    return phot_amp


@tf.function(jit_compile=JIT_COMPILE)
def photometry_sigma_setup(
    wavelength: tf.Tensor,
    throughput: tf.Tensor,
    cached_amp: tuple[tf.Tensor, tf.Tensor] | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    # phot_amp is a weighted mean with weight w = throughput * wavelength, so
    # the variance of that mean is sum(w_i^2 sigma_i^2) / sum(w_i)^2 - the
    # denominator is the *square of the amplitude's denom*, not a separate
    # integral of the squared weight (which would instead give a weighted
    # average of sigma^2 that never shrinks with more integrated pixels).
    if cached_amp is None:
        denom, amp_mask = photometry_amplitude_setup(wavelength, throughput)
    else:
        denom, amp_mask = cached_amp
    denom_sigma = denom**2
    sigma_mask = amp_mask

    return denom_sigma, sigma_mask


@tf.function(jit_compile=JIT_COMPILE)
def photometry_sigma(
    wavelength: tf.Tensor,
    sigma: tf.Tensor,
    throughput: tf.Tensor,
    effective_wavelength: tf.Tensor,
    spec_mask: tf.Tensor,
    phot_mask: tf.Tensor,
    cached_amp: tuple[tf.Tensor, tf.Tensor] | None = None,
    cached: tuple[tf.Tensor, tf.Tensor] | None = None,
) -> tf.Tensor:

    if cached is None:
        denom_sigma, sigma_mask = photometry_sigma_setup(
            wavelength, throughput, cached_amp
        )
    else:
        denom_sigma, sigma_mask = cached

    # Variance propagation through the filter integration.
    # phot_amp is `sum_i c_i * amp_i * w_i / denom` for trapezoidal weights
    # c_i, so its variance is `sum_i c_i**2 * sigma_i**2 * w_i**2 / denom**2`.
    # A plain trapz() call on the squared integrand would apply c_i again
    # (not c_i**2), so the quadrature weights are squared explicitly here.
    # (n_sn, n_spec, 1, n_filters)
    numer_sigma = tf.reduce_sum(
        tf.square(_trapz_weights(wavelength) * sigma * throughput * wavelength),
        axis=-2,
    )[..., None, :]

    # (n_sn, n_spec, 1, n_filters)
    phot_sigma = tf.sqrt(numer_sigma / denom_sigma)
    phot_sigma = tf.where(
        sigma_mask,
        phot_sigma,
        tf.ones_like(phot_sigma),
    )

    # Put the uncertainty only at the effective wavelength bin,
    # matching the phot_amp construction
    # (n_sn, n_spec, n_wl, n_filters)
    phot_sigma = tf.sqrt(
        tf.square(spec_mask * sigma)
        + tf.square(phot_mask * effective_wavelength * phot_sigma)
    )

    # Same reduction as phot_amp
    # (n_sn, n_spec, n_wl)
    phot_sigma = tf.sqrt(tf.reduce_sum(tf.square(phot_sigma), axis=-1))  # / phot_mask

    return phot_sigma


@tf.function(jit_compile=JIT_COMPILE)
def photometry_sigma_correlated(
    wavelength: tf.Tensor,
    sigma: tf.Tensor,
    throughput: tf.Tensor,
    effective_wavelength: tf.Tensor,
    spec_mask: tf.Tensor,
    phot_mask: tf.Tensor,
    cached_amp: tuple[tf.Tensor, tf.Tensor] | None = None,
) -> tf.Tensor:
    """Propagate a *fully wavelength-correlated* uncertainty through the filter integral.

    `photometry_sigma` treats `sigma` as independent per-pixel noise, so the band
    uncertainty shrinks like `1 / sqrt(N_band)`. A coherent SED-shape error (e.g. the
    autoencoder reconstruction error) does not average down across a passband: with
    `phot_amp = sum_i g_i * amp_i` and `g_i = c_i * w_i / denom` (`c_i` the trapezoidal
    weights, `w_i = throughput_i * wavelength_i`), a same-sign perturbation propagates
    *linearly*, `sigma[phot_amp] = sum_i g_i * sigma_i`. For a roughly flat fractional
    error this collapses to the same fractional error on `phot_amp`.

    Uses the amplitude denominator (`sum_i c_i w_i`), not its square, so it reuses
    `cached_amp` and needs no separate cache.

    Returns:
        The correlated component of the uncertainty, same shape/placement as
        `photometry_sigma` (at the effective-wavelength bin for photometric rows,
        `sigma` passed through for spectrum rows).
    """
    if cached_amp is None:
        denom, sigma_mask = photometry_amplitude_setup(wavelength, throughput)
    else:
        denom, sigma_mask = cached_amp

    # (n_sn, n_spec, 1, n_filters) -- linear weighted sum, NOT sum of squares
    numer = tf.reduce_sum(
        _trapz_weights(wavelength) * sigma * throughput * wavelength,
        axis=-2,
    )[..., None, :]

    # (n_sn, n_spec, 1, n_filters)
    phot_sigma = numer / denom
    phot_sigma = tf.where(
        sigma_mask,
        phot_sigma,
        tf.ones_like(phot_sigma),
    )

    # Put the uncertainty only at the effective wavelength bin,
    # matching the phot_amp construction
    # (n_sn, n_spec, n_wl, n_filters)
    phot_sigma = tf.sqrt(
        tf.square(spec_mask * sigma)
        + tf.square(phot_mask * effective_wavelength * phot_sigma)
    )

    # Same reduction as phot_amp
    # (n_sn, n_spec, n_wl)
    return tf.sqrt(tf.reduce_sum(tf.square(phot_sigma), axis=-1))


@tf.function(jit_compile=JIT_COMPILE)
def photometry(
    wavelength: tf.Tensor,
    amplitude: tf.Tensor,
    sigma: tf.Tensor,
    throughput: tf.Tensor,
    effective_wavelength: tf.Tensor,
    spec_mask: tf.Tensor | None = None,
    phot_mask: tf.Tensor | None = None,
    cached_amp: tuple[tf.Tensor, tf.Tensor] | None = None,
    cached_sigma: tuple[tf.Tensor, tf.Tensor] | None = None,
    sigma_correlated: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Integrate spectroscopy through filters to produce pseudo-photometry.

    Args:
        wavelength: Tensor of shape (n_sn, n_spec, n_wl). The wavelength grid in Angstroms.
        amplitude: Tensor of shape (n_sn, n_spec, n_wl). The amplitude of each realisation of each spectrum of each SN.
        sigma: Tensor of shape (n_sn, n_spec, n_wl). The uncertainty in the amplitude, treated as independent per-pixel noise (band uncertainty shrinks like 1/sqrt(N_band)).
        throughput: Tensor of shape (n_sn, n_spec, n_wl, n_filters). The filter throughput curves to integrate over.
        effective_wavelength: Tensor of shape (n_sn, n_spec, n_wl, n_filters). Should be equal to 1 at the effective wavelength of a given filter, and 0 otherwise.
        spec_mask: Tensor of shape (n_sn, n_spec, n_wl, n_filters). Should be 1 when the original spectra should be included in the final amplitude, or 0 otherwise. Defaults to all ones (i.e. keep original spectra everywhere)
        phot_mask: Tensor of shape (n_sn, n_spec, n_wl, n_filters). Should be 1 when photometry should be included in the final amplitude, or 0 otherwise. Defaults to all ones (i.e. keep original spectra everywhere)
        cached_amp: Optional cached `(denom, mask)` from `photometry_amplitude_setup`.
        cached_sigma: Optional cached `(denom_sigma, mask)` from `photometry_sigma_setup`.
        sigma_correlated: Optional tensor, same shape as `sigma`, of a *fully wavelength-correlated* uncertainty component (e.g. a coherent SED-shape / reconstruction error). Propagated linearly through the band integral (no 1/sqrt(N_band) shrink) and added in quadrature to the `sigma` component. Defaults to None, in which case only `sigma` is propagated and the result is identical to the previous behaviour.

    Returns:
        The amplitude and sigma with photometry calculated.
    """
    wavelength = tf.cast(tf.convert_to_tensor(wavelength), dtype=tf.float32)
    amplitude = tf.cast(tf.convert_to_tensor(amplitude), dtype=tf.float32)
    sigma = tf.cast(tf.convert_to_tensor(sigma), dtype=tf.float32)
    if sigma_correlated is not None:
        sigma_correlated = tf.cast(
            tf.convert_to_tensor(sigma_correlated), dtype=tf.float32
        )
    throughput = tf.cast(tf.convert_to_tensor(throughput), dtype=tf.float32)
    effective_wavelength = tf.cast(
        tf.convert_to_tensor(effective_wavelength), dtype=tf.float32
    )
    if spec_mask is None:
        spec_mask = tf.ones_like(amplitude)
    blank_mask = tf.zeros_like(throughput)[..., :1, :-1]
    spec_mask = tf.cast(tf.convert_to_tensor(spec_mask), dtype=tf.float32)[..., None]
    if throughput.shape[-1] > 1:
        spec_mask = tf.concat((blank_mask, spec_mask), axis=-1)
    if phot_mask is None:
        phot_mask = tf.ones_like(amplitude)
    if throughput.shape[-1] == 1:
        phot_mask = tf.zeros_like(spec_mask)
    else:
        phot_mask = tf.cast(tf.convert_to_tensor(phot_mask), dtype=tf.float32)[
            ..., None
        ]
        phot_mask = tf.repeat(phot_mask, throughput.shape[-1] - 1, axis=-1)
        phot_mask = tf.concat((phot_mask, blank_mask[..., :1]), axis=-1)
    wavelength = tf.repeat(wavelength[..., None], throughput.shape[-1], axis=-1)
    amplitude = amplitude[..., None]
    sigma = sigma[..., None]
    if sigma_correlated is not None:
        sigma_correlated = sigma_correlated[..., None]
    # amplitude = tf.repeat(amplitude[..., None], throughput.shape[-1], axis=-1)
    # sigma = tf.repeat(sigma[..., None], throughput.shape[-1], axis=-1)

    phot_amp = photometry_amplitude(
        wavelength,
        amplitude,
        throughput,
        effective_wavelength,
        spec_mask,
        phot_mask,
        cached_amp,
    )

    phot_sigma = photometry_sigma(
        wavelength,
        sigma,
        throughput,
        effective_wavelength,
        spec_mask,
        phot_mask,
        cached_amp,
        cached_sigma,
    )

    if sigma_correlated is not None:
        # Independent (per-pixel) and fully-correlated (SED-shape) components
        # propagate through the band integral differently; combine in quadrature.
        phot_sigma_correlated = photometry_sigma_correlated(
            wavelength,
            sigma_correlated,
            throughput,
            effective_wavelength,
            spec_mask,
            phot_mask,
            cached_amp,
        )
        phot_sigma = tf.sqrt(tf.square(phot_sigma) + tf.square(phot_sigma_correlated))

    return phot_amp, phot_sigma
