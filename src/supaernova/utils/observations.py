"""Shared construction of observation rows for real and simulated data.

Every data source (real spectra, synthetic photometry integrated from real
spectra, real photometry from files, and simulated spectra / photometry) goes
through the same functions here, so amplitudes and uncertainties are assembled
and calibrated identically regardless of where they came from.

Layout conventions (all arrays are ``(n_sn, n_rows, ...)``):

* An *epoch* is one (SN, phase). It can be observed as a spectrum and/or in
  any number of photometric bands.
* A spectrum is stored in its own row (``spectra_mask = 1``). Its datum mask is
  the per-pixel validity of the observed spectrum.
* All bands of an epoch are stored together in one photometry row
  (``phot_mask = 1``). Each band's flux and uncertainty sit at that filter's
  effective-wavelength bin, and the datum mask is 1 *only* at those bins.
* A spectrum and a photometry row of the same epoch share an ``epoch`` id, so
  consumers can treat their (shared) model error jointly.
* Measurement uncertainties are calibrated once, after assembly, with
  ``sigma = sigma_scale * sigma_raw + sigma_floor`` for every datum.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import typing as npt


def trapz_weights(wavelength: "npt.NDArray[np.floating]") -> "npt.NDArray[np.floating]":
    """Per-point trapezoidal quadrature weights along the last axis.

    Returns:
        Weights ``c`` such that ``trapz(f, wavelength) == sum(c * f)``.
    """
    dx = np.diff(wavelength, axis=-1)
    weights = np.zeros_like(wavelength, dtype=np.float64)
    weights[..., :-1] += 0.5 * dx
    weights[..., 1:] += 0.5 * dx
    return weights


def band_weights(
    wavelength: "npt.NDArray[np.floating]", throughput: "npt.NDArray[np.floating]"
) -> "npt.NDArray[np.float64]":
    """Linear band-integration weights.

    A band flux is the photon-weighted mean ``sum_i g[i, b] * flux[i]`` with
    ``g[i, b] = c_i T_b(λ_i) λ_i / sum_j c_j T_b(λ_j) λ_j``, which is exactly
    ``trapz(flux T λ) / trapz(T λ)``.

    Args:
        wavelength: Shape ``(..., n_wl)``.
        throughput: Shape ``(..., n_wl, n_filters)``.

    Returns:
        ``g`` with the shape of ``throughput``; zero for filters with no throughput.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    numer = (trapz_weights(wavelength) * wavelength)[..., None] * np.asarray(
        throughput, dtype=np.float64
    )
    denom = numer.sum(axis=-2, keepdims=True)
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)


def integrate_bands(
    flux: "npt.NDArray[np.floating]",
    sigma: "npt.NDArray[np.floating]",
    valid: "npt.NDArray[np.bool_]",
    weights: "npt.NDArray[np.floating]",
) -> tuple[
    "npt.NDArray[np.float64]", "npt.NDArray[np.float64]", "npt.NDArray[np.bool_]"
]:
    """Integrate spectra through filters.

    A band is only defined when *every* pixel its filter covers is valid, so a
    band never mixes observed and missing flux (the same rule is used for data,
    sims and the model). Uncertainties are propagated as independent per pixel.

    Args:
        flux: Shape ``(..., n_wl)``.
        sigma: Shape ``(..., n_wl)``, per-pixel 1σ uncertainty.
        valid: Shape ``(..., n_wl)``, per-pixel validity.
        weights: Band weights from `band_weights`, ``(n_wl, n_f)`` or ``(..., n_wl, n_f)``.

    Returns:
        ``(band_flux, band_sigma, band_valid)``, each of shape ``(..., n_f)``.
    """
    valid = np.asarray(valid, dtype=bool)
    weights = np.broadcast_to(weights, (*valid.shape, weights.shape[-1]))
    support = weights > 0
    band_valid = support.any(axis=-2) & ~np.any(support & ~valid[..., None], axis=-2)
    flux0 = np.where(valid, flux, 0.0)
    sigma0 = np.where(valid, sigma, 0.0)
    band_flux = np.einsum("...i,...ib->...b", flux0, weights)
    band_sigma = np.sqrt(np.einsum("...i,...ib->...b", sigma0**2, weights**2))
    return (
        np.where(band_valid, band_flux, 0.0),
        np.where(band_valid, band_sigma, 0.0),
        band_valid,
    )


def effective_indices(
    wavelength: "npt.NDArray[np.floating]", effective_wavelengths: "npt.ArrayLike"
) -> "npt.NDArray[np.int64]":
    """Index of the grid point nearest each filter's effective wavelength.

    Returns:
        One index per filter.
    """
    wavelength = np.asarray(wavelength)
    return np.array(
        [
            int(np.argmin(np.abs(wavelength - ew)))
            for ew in np.atleast_1d(effective_wavelengths)
        ],
        dtype=np.int64,
    )


def calibrate_sigma(
    sigma_raw: "npt.NDArray[np.floating]",
    sigma_scale: float,
    sigma_floor: float,
    floor_mode: str = "linear",
) -> "npt.NDArray[np.float64]":
    """The single measurement-uncertainty calibration applied to every datum.

    Returns:
        ``sigma_scale * sigma_raw + sigma_floor`` (``floor_mode = "linear"``), or
        ``sqrt((sigma_scale * sigma_raw)² + sigma_floor²)`` (``"quadrature"``).
    """
    scaled = sigma_scale * np.asarray(sigma_raw, dtype=np.float64)
    if floor_mode == "quadrature":
        return np.sqrt(scaled * scaled + sigma_floor * sigma_floor)
    return scaled + sigma_floor


def uncalibrate_sigma(
    sigma: "npt.NDArray[np.floating]",
    sigma_scale: float,
    sigma_floor: float,
    floor_mode: str = "linear",
) -> "npt.NDArray[np.float64]":
    """Invert `calibrate_sigma`, recovering the reported σ (clipped at 0)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    if floor_mode == "quadrature":
        return (
            np.sqrt(np.maximum(sigma * sigma - sigma_floor * sigma_floor, 0.0))
            / sigma_scale
        )
    return np.maximum(sigma - sigma_floor, 0.0) / sigma_scale


def build_rows(
    *,
    wavelength: "npt.NDArray[np.floating]",
    throughput: "npt.NDArray[np.floating]",
    effective_index: "npt.NDArray[np.integer]",
    spec_obs: "npt.NDArray[np.bool_]",
    spec_flux: "npt.NDArray[np.floating]",
    spec_sigma: "npt.NDArray[np.floating]",
    spec_valid: "npt.NDArray[np.bool_]",
    band_obs: "npt.NDArray[np.bool_]",
    band_flux: "npt.NDArray[np.floating]",
    band_sigma: "npt.NDArray[np.floating]",
    sigma_scale: float = 1.0,
    sigma_floor: float = 0.0,
    floor_mode: str = "linear",
    allow_shared_epochs: bool = True,
    min_rows: int = 1,
) -> dict[str, "npt.NDArray"]:
    """Lay observed epochs out as spectrum and photometry rows.

    Args:
        wavelength: Master wavelength grid, ``(n_wl,)``.
        throughput: Filter throughput on that grid, ``(n_wl, n_f)``.
        effective_index: Effective-wavelength bin of each filter, ``(n_f,)``.
        spec_obs: ``(n_sn, n_ep)``, epoch observed as a spectrum.
        spec_flux: ``(n_sn, n_ep, n_wl)``, spectrum flux.
        spec_sigma: ``(n_sn, n_ep, n_wl)``, raw (uncalibrated) spectrum 1σ.
        spec_valid: ``(n_sn, n_ep, n_wl)``, per-pixel validity of the spectrum datum.
        band_obs: ``(n_sn, n_ep, n_f)``, epoch observed in each band.
        band_flux: ``(n_sn, n_ep, n_f)``, band flux.
        band_sigma: ``(n_sn, n_ep, n_f)``, raw (uncalibrated) band 1σ.
        sigma_scale: Multiplicative uncertainty calibration.
        sigma_floor: Additive uncertainty calibration.
        floor_mode: How the floor combines with the scaled σ (see `calibrate_sigma`).
        allow_shared_epochs: If False, an epoch observed both as a spectrum and
            in any band raises, because its bands would be derived from (and
            share the noise of) that spectrum.
        min_rows: Pad to at least this many rows per SN.

    Returns:
        Row arrays: ``row_epoch`` ``(n_sn, n_rows)`` (index into epochs, -1 for
        padding), ``epoch`` ``(n_sn, n_rows, 1)`` (per-SN epoch id, -1 for
        padding), ``amplitude``, ``sigma``, ``mask`` ``(n_sn, n_rows, n_wl)``,
        ``spectra_mask``, ``phot_mask`` ``(n_sn, n_rows, 1)``, and
        ``throughput``, ``effective_wavelength`` ``(n_sn, n_rows, n_wl, n_f + 1)``
        (the last channel is the always-zero spectrum channel).

    Raises:
        ValueError: On shared epochs when not allowed, or if two observed bands
            of one epoch share an effective-wavelength bin.
    """
    spec_obs = np.asarray(spec_obs, dtype=bool)
    band_obs = np.asarray(band_obs, dtype=bool)
    spec_valid = np.asarray(spec_valid, dtype=bool) & spec_obs[..., None]
    spec_obs &= spec_valid.any(axis=-1)
    n_sn, n_ep = spec_obs.shape
    n_wl = len(wavelength)
    n_f = band_obs.shape[-1]
    phot_obs = band_obs.any(axis=-1)

    shared = spec_obs & phot_obs
    if not allow_shared_epochs and np.any(shared):
        n_shared = int(shared.sum())
        msg = (
            f"{n_shared} epochs are observed both as a spectrum and as photometry "
            "synthesised from that same spectrum, so the photometry would repeat the "
            "spectrum's noise. Select disjoint epochs (e.g. `spec_every` / "
            "`phot_every` offsets) instead."
        )
        raise ValueError(msg)

    if n_f > 0:
        for b1 in range(n_f):
            for b2 in range(b1 + 1, n_f):
                if effective_index[b1] == effective_index[b2] and np.any(
                    band_obs[..., b1] & band_obs[..., b2]
                ):
                    msg = (
                        f"Filters {b1} and {b2} share the effective-wavelength bin "
                        f"{effective_index[b1]} and are observed at the same epoch."
                    )
                    raise ValueError(msg)

    # Row order per SN: epochs in order, spectrum row before photometry row
    rows_per_sn = (spec_obs.astype(int) + phot_obs.astype(int)).sum(axis=-1)
    n_rows = max(int(rows_per_sn.max(initial=0)), min_rows, 1)

    row_epoch = np.full((n_sn, n_rows), -1, dtype=np.int64)
    row_is_spec = np.zeros((n_sn, n_rows), dtype=bool)
    row_is_phot = np.zeros((n_sn, n_rows), dtype=bool)
    for i in range(n_sn):
        r = 0
        for e in range(n_ep):
            if spec_obs[i, e]:
                row_epoch[i, r] = e
                row_is_spec[i, r] = True
                r += 1
            if phot_obs[i, e]:
                row_epoch[i, r] = e
                row_is_phot[i, r] = True
                r += 1

    has_row = row_epoch >= 0
    ep = np.where(has_row, row_epoch, 0)
    sn_idx = np.arange(n_sn)[:, None]

    amplitude = np.zeros((n_sn, n_rows, n_wl))
    sigma_raw = np.zeros((n_sn, n_rows, n_wl))
    mask = np.zeros((n_sn, n_rows, n_wl), dtype=bool)

    # --- Spectrum rows ---
    s = row_is_spec
    amplitude[s] = np.asarray(spec_flux)[sn_idx, ep][s]
    sigma_raw[s] = np.asarray(spec_sigma)[sn_idx, ep][s]
    mask[s] = spec_valid[sn_idx, ep][s]

    # --- Photometry rows ---
    tp = np.zeros((n_sn, n_rows, n_wl, n_f + 1))
    ew = np.zeros((n_sn, n_rows, n_wl, n_f + 1))
    p = row_is_phot
    if n_f > 0 and np.any(p):
        obs = band_obs[sn_idx, ep] & p[..., None]  # (n_sn, n_rows, n_f)
        flux = np.asarray(band_flux)[sn_idx, ep]
        sig = np.asarray(band_sigma)[sn_idx, ep]
        for b in range(n_f):
            k = int(effective_index[b])
            ob = obs[..., b]
            amplitude[..., k] = np.where(ob, flux[..., b], amplitude[..., k])
            sigma_raw[..., k] = np.where(ob, sig[..., b], sigma_raw[..., k])
            mask[..., k] |= ob
            tp[..., b] = np.where(
                ob[..., None], np.asarray(throughput)[None, None, :, b], 0.0
            )
            ew[..., k, b] = ob.astype(np.float64)

    sigma = np.where(
        mask, calibrate_sigma(sigma_raw, sigma_scale, sigma_floor, floor_mode), 0.0
    )
    amplitude = np.where(mask, amplitude, 0.0)

    # Per-SN epoch id, shared by a spectrum row and photometry row of one epoch
    epoch = np.where(has_row, row_epoch, -1)[..., None]

    return {
        "row_epoch": row_epoch,
        "epoch": epoch,
        "amplitude": amplitude,
        "sigma": sigma,
        "mask": mask,
        "spectra_mask": row_is_spec[..., None],
        "phot_mask": row_is_phot[..., None],
        "throughput": tp,
        "effective_wavelength": ew,
    }


def gather_rows(
    values: "npt.NDArray", row_epoch: "npt.NDArray[np.integer]", fill: object
) -> "npt.NDArray":
    """Copy per-epoch values ``(n_sn, n_ep, ...)`` onto rows ``(n_sn, n_rows, ...)``.

    Returns:
        The gathered array, with padding rows set to ``fill``.
    """
    values = np.asarray(values)
    ep = np.where(row_epoch >= 0, row_epoch, 0)
    out = values[np.arange(values.shape[0])[:, None], ep]
    pad = (row_epoch < 0).reshape(*row_epoch.shape, *([1] * (out.ndim - 2)))
    return np.where(pad, np.asarray(fill, dtype=out.dtype), out)


def epoch_bands(
    *,
    amplitude: "npt.NDArray[np.floating]",
    sigma: "npt.NDArray[np.floating]",
    datum_mask: "npt.NDArray[np.bool_]",
    spectra_mask: "npt.NDArray",
    phot_mask: "npt.NDArray",
    wavelength: "npt.NDArray[np.floating]",
    throughput: "npt.NDArray[np.floating]",
    effective_wavelength: "npt.NDArray[np.floating]",
    epoch: "npt.NDArray[np.integer] | None",
) -> dict[str, "npt.NDArray"]:
    """Collect each row's band data, moving bands onto their epoch's spectrum row.

    A likelihood that treats an epoch's spectrum and bands jointly needs them in
    one place. Each photometry row's bands are moved onto the spectrum row of the
    same epoch when there is one, so every row is self-contained: it holds its
    own spectrum pixels (if any) and every band of its epoch (if any).

    Args:
        amplitude: ``(n_sn, n_rows, n_wl)`` row amplitudes (bands at effective bins).
        sigma: ``(n_sn, n_rows, n_wl)`` row uncertainties.
        datum_mask: ``(n_sn, n_rows, n_wl)`` which entries are data.
        spectra_mask: ``(n_sn, n_rows, 1)`` spectrum rows.
        phot_mask: ``(n_sn, n_rows, 1)`` photometry rows.
        wavelength: ``(n_sn, n_rows, n_wl)``.
        throughput: ``(n_sn, n_rows, n_wl, n_f + 1)``.
        effective_wavelength: ``(n_sn, n_rows, n_wl, n_f + 1)``.
        epoch: ``(n_sn, n_rows, 1)`` epoch ids, or None for one epoch per row.

    Returns:
        ``band_flux``, ``band_sigma``, ``band_active`` ``(n_sn, n_rows, n_f)`` and
        ``band_weights`` ``(n_sn, n_rows, n_wl, n_f)``.
    """
    datum_mask = np.asarray(datum_mask, dtype=bool)
    n_sn = datum_mask.shape[0]
    n_f = throughput.shape[-1] - 1
    spec_row = np.asarray(spectra_mask)[..., 0].astype(bool)
    phot_row = np.asarray(phot_mask)[..., 0].astype(bool)
    eff = np.asarray(effective_wavelength)[..., :n_f] > 0
    on = eff & datum_mask[..., None] & phot_row[..., None, None]
    active = on.any(axis=-2)
    flux = np.where(on, np.asarray(amplitude)[..., None], 0.0).sum(axis=-2)
    sig = np.where(on, np.asarray(sigma)[..., None], 0.0).sum(axis=-2)
    weights = band_weights(wavelength, np.asarray(throughput)[..., :n_f])
    weights = np.where(active[..., None, :], weights, 0.0)

    if epoch is None or n_f == 0:
        return {
            "band_flux": flux,
            "band_sigma": sig,
            "band_active": active,
            "band_weights": weights,
        }

    epoch = np.asarray(epoch)[..., 0]
    spec_ok = spec_row & datum_mask.any(axis=-1)
    for i in range(n_sn):
        for p in np.nonzero(active[i].any(axis=-1))[0]:
            same = np.nonzero(spec_ok[i] & (epoch[i] == epoch[i, p]) & (epoch[i] >= 0))[
                0
            ]
            if len(same) == 0:
                continue
            s = same[0]
            flux[i, s], sig[i, s] = flux[i, p], sig[i, p]
            active[i, s], weights[i, s] = active[i, p], weights[i, p]
            flux[i, p] = sig[i, p] = 0.0
            active[i, p] = False
            weights[i, p] = 0.0
    return {
        "band_flux": flux,
        "band_sigma": sig,
        "band_active": active,
        "band_weights": weights,
    }
