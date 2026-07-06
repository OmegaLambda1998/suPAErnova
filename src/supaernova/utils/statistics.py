import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supaernova.configs.steps.data import SNPAEData, LazySNPAEData
    from numpy import typing as npt
    from collections.abc import Callable


def SNR(
    data: "SNPAEData | LazySNPAEData",
    mask: "npt.NDArray | None" = None,
    normalise: "bool" = False,
):
    amplitude = np.clip(data.amplitude, 0, np.inf)
    sigma = data.sigma
    if mask is None:
        mask = np.ones_like(data.mask)
    mask &= data.mask & data.sn_mask & data.spec_mask & data.wl_mask
    if np.count_nonzero(mask) == 0:
        return 0
    signal = np.where(mask, amplitude * amplitude, np.zeros_like(amplitude))
    noise = np.where(mask, sigma * sigma, np.zeros_like(sigma))
    snr = signal / noise
    snr = np.where(np.isfinite(snr), snr, np.zeros_like(snr))
    snr_sum = np.sum(snr, axis=(-2, -1))
    if normalise:
        snr_sum /= np.count_nonzero(mask, axis=(-2, -1))
    snr_mask = np.isfinite(snr_sum)
    snr_sum = np.where(snr_mask, snr_sum, np.zeros_like(snr_sum))
    snr_coadd = np.sqrt(snr_sum)
    return np.sum(snr_coadd) / np.count_nonzero(snr_mask)
