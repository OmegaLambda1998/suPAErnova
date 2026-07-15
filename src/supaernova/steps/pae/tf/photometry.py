from typing import TYPE_CHECKING

from supaernova._tf import tf, tfp
from supaernova.utils.tf import db, pp

if TYPE_CHECKING:
    from supaernova._tf import ks


def photometry(
    wavelength: tf.Tensor,
    amplitude: tf.Tensor,
    sigma: tf.Tensor,
    throughput: tf.Tensor,
    effective_wavelength: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Integrate spectroscopy through filters to produce pseudo-photometry.

    Args:
        wavelength: Tensor of shape (n_sn, n_spec, n_wl). The wavelength grid in Angstroms.
        amplitude: Tensor of shape (n_sn, n_spec, n_wl). The amplitude of each realisation of each spectrum of each SN.
        sigma: Tensor of shape (n_sn, n_spec, n_wl). The uncertainty in the amplitude.
        throughput: Tensor of shape (n_sn, n_spec, n_wl, n_filters). The filter throughput curves to integrate over.
        effective_wavelength: Tensor of shape (n_sn, n_spec, n_wl, n_filters). Should be equal to 1 at the effective wavelength of a given filter, and 0 otherwise.

    Returns:
        The amplitude and sigma with photometry calculated.
    """
    wavelength = tf.convert_to_tensor(wavelength, dtype=tf.float32)
    amplitude = tf.convert_to_tensor(amplitude, dtype=tf.float32)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    throughput = tf.convert_to_tensor(throughput, dtype=tf.float32)
    effective_wavelength = tf.convert_to_tensor(effective_wavelength, dtype=tf.float32)
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
    denom = tfp.math.trapz(throughput / wavelength, wavelength, axis=-2)[..., None, :]

    # (n_sn, n_spec, n_wl, n_filters)
    phot = effective_wavelength * numer / denom
    phot = tf.where(tf.math.is_finite(phot), phot, tf.zeros_like(phot))

    # (n_sn, n_spec, 1, n_filters)
    is_phot = tf.cast(
        tf.reduce_sum(effective_wavelength, axis=-2, keepdims=True) > 0, tf.float32
    )

    # (n_sn, n_spec, n_wl, n_filters)
    # is_phot is 1 if there is an effective wavelength, and 0 otherwise
    # so this will set the amplitude to the photometric amplitude if there is an effective wavelength
    # otherwise it will use the orignal spectroscopic amplitude
    phot_amp = is_phot * phot + (1 - is_phot) * amplitude

    # (n_sn, n_spec, n_wl)
    phot_amp = tf.reduce_sum(phot_amp, axis=-1)
    return phot_amp, sigma
