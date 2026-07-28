from typing import TYPE_CHECKING

from supaernova._tf import tf, tfp, ks
from supaernova.utils.tf import db, pp


@tf.function
def photometry(
    wavelength: tf.Tensor,
    amplitude: tf.Tensor,
    sigma: tf.Tensor,
    throughput: tf.Tensor,
    effective_wavelength: tf.Tensor,
    spec_mask: tf.Tensor | None = None,
    phot_mask: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Integrate spectroscopy through filters to produce pseudo-photometry.

    Args:
        wavelength: Tensor of shape (n_sn, n_spec, n_wl). The wavelength grid in Angstroms.
        amplitude: Tensor of shape (n_sn, n_spec, n_wl). The amplitude of each realisation of each spectrum of each SN.
        sigma: Tensor of shape (n_sn, n_spec, n_wl). The uncertainty in the amplitude.
        throughput: Tensor of shape (n_sn, n_spec, n_wl, n_filters). The filter throughput curves to integrate over.
        effective_wavelength: Tensor of shape (n_sn, n_spec, n_wl, n_filters). Should be equal to 1 at the effective wavelength of a given filter, and 0 otherwise.
        spec_mask: Tensor of shape (n_sn, n_spec, n_wl, n_filters). Should be 1 when the original spectra should be included in the final amplitude, or 0 otherwise. Defaults to all ones (i.e. keep original spectra everywhere)
        phot_mask: Tensor of shape (n_sn, n_spec, n_wl, n_filters). Should be 1 when photometry should be included in the final amplitude, or 0 otherwise. Defaults to all ones (i.e. keep original spectra everywhere)

    Returns:
        The amplitude and sigma with photometry calculated.
    """
    wavelength = tf.cast(tf.convert_to_tensor(wavelength), dtype=tf.float32)
    amplitude = tf.cast(tf.convert_to_tensor(amplitude), dtype=tf.float32)
    sigma = tf.cast(tf.convert_to_tensor(sigma), dtype=tf.float32)
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
    amplitude = tf.repeat(amplitude[..., None], throughput.shape[-1], axis=-1)
    wavelength = tf.repeat(wavelength[..., None], throughput.shape[-1], axis=-1)
    sigma = tf.repeat(sigma[..., None], throughput.shape[-1], axis=-1)

    # (n_sn, n_spec, 1, n_filters)
    numer = tfp.math.trapz(
        amplitude * throughput * wavelength,
        wavelength,
        axis=-2,
    )[..., None, :]

    # (n_sn, n_spec, 1, n_filters)
    denom = tfp.math.trapz(throughput * wavelength, wavelength, axis=-2)[..., None, :]
    amp_mask = tf.cast(
        tf.where(denom == 0, tf.zeros_like(denom), tf.ones_like(denom)), tf.bool
    )
    denom = tf.where(denom == 0, tf.ones_like(denom), denom)

    # (n_sn, n_spec, n_wl, n_filters)
    phot_amp = numer / denom
    phot_amp = tf.where(amp_mask, phot_amp, tf.zeros_like(phot_amp))

    # (n_sn, n_spec, n_wl, n_filters)
    phot_amp = spec_mask * amplitude + phot_mask * effective_wavelength * phot_amp

    # (n_sn, n_spec, n_wl)
    phot_amp = tf.reduce_sum(phot_amp, axis=-1)
    # phot_amp = tf.where(
    #     tf.abs(phot_amp) > ks.backend.epsilon(), phot_amp, tf.zeros_like(phot_amp)
    # )

    # Variance propagation through the filter integration
    # (n_sn, n_spec, 1, n_filters)
    numer_sigma = tfp.math.trapz(
        (sigma * throughput * wavelength) ** 2,
        wavelength,
        axis=-2,
    )[..., None, :]
    denom_sigma = tfp.math.trapz((throughput * wavelength) ** 2, wavelength, axis=-2)[
        ..., None, :
    ]
    sigma_mask = tf.cast(
        tf.where(
            denom_sigma == 0, tf.zeros_like(denom_sigma), tf.ones_like(denom_sigma)
        ),
        tf.bool,
    )
    denom_sigma = tf.where(denom_sigma == 0, tf.ones_like(denom_sigma), denom_sigma)

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
    # phot_sigma = tf.where(
    #     tf.abs(phot_sigma) > ks.backend.epsilon(),
    #     phot_sigma,
    #     tf.ones_like(phot_sigma),
    # )

    return phot_amp, phot_sigma
