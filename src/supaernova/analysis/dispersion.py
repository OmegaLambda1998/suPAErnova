from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np

from .spectra import SpectraPlot, SpectraPlotter
from .analysis import Plotter

if TYPE_CHECKING:
    import pandas as pd

    from supaernova.configs.steps.data import DataStepResult
    from supaernova.configs.steps.posterior import PosteriorStepResult

    from .analysis import Axis, Figure


class DispersionPlot(SpectraPlot):
    subset: Literal["train", "test"]
    legacy: tuple[Path, ...] | None = None
    twins: str | None = None


class DispersionPlotter(Plotter):
    @staticmethod
    def plot_dispersion(
        data: "DataStepResult",
        hmcs: "list[PosteriorStepResult]",
        config: "DispersionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        twins: "pd.DataFrame | None" = None,
        legacy: "dict[str, np.ndarray] | None" = None,
        force: bool = False,
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return

        _wl, _amplitude, _sigma, sn_mask, _spec_mask, _wl_mask = SpectraPlotter.prep(
            data, config
        )

        sn_mask = sn_mask[:, 0, 0]
        names = data.sn_name[:, 0, 0]

        twins_mask = np.ones_like(sn_mask)
        if twins is not None:
            twins_mask = np.zeros_like(sn_mask)
            twins_names = twins.name
            intersection = set(names) & set(twins_names)
            for name in intersection:
                ind = np.argwhere(names == name)[0]
                df = twins[twins.name == name]
                twins_mask[ind] = df.mask_twins

        redshift = data.redshift[:, 0, 0]
        order = np.argsort(redshift)
        redshift = redshift[order]
        redshift_error = (redshift * 3e5 + 300.0) / 3e5
        magshift_error = abs(-5 * np.log10(redshift / redshift_error))

        amplitudes = np.concatenate(
            [np.mean(hmc.hmc.delta_m, axis=0, keepdims=True) for hmc in hmcs], axis=0
        )[..., 0]
        amplitude_stds = np.concatenate(
            [
                np.sqrt(np.square(np.std(hmc.hmc.delta_m, axis=0, keepdims=True)))
                for hmc in hmcs
            ],
            axis=0,
        )[..., 0]

        weights = 1 / (amplitude_stds * amplitude_stds)
        weights /= weights.sum(axis=0)

        weighted_amplitudes = np.sum(weights * amplitudes, axis=0)
        weighted_amplitudes = weighted_amplitudes[order]

        weighted_variances = np.sum(
            weights * np.square(amplitudes - np.mean(amplitudes)), axis=0
        )
        weighted_variances = weighted_variances[order]

        weighted_stds = np.sqrt(
            (weighted_variances * weighted_variances)
            + (magshift_error * magshift_error)
        )

        def _plot(x, y, yerr, fig, ax, color, marker, alpha, title):
            k = 1.4826

            w_rms = np.sqrt(np.sum(y * y) / np.size(y))
            w_mad = np.std(y) / k

            print(title, x.shape, y.shape, yerr.shape)
            print("RMS: ", w_rms)
            print("NMAD: ", w_mad)

            fig, ax = Plotter.errorbar(
                x,
                y,
                yerr=yerr,
                fig=fig,
                ax=ax,
                color=color,
                marker=marker,
                alpha=alpha,
                label=f"{title} ({np.size(y)} SN) - RMS: {w_rms:.2f}, MAD: {w_mad:.2f}",
            )

            return fig, ax

        # === No Mask ===
        x = redshift
        y = weighted_amplitudes
        yerr = weighted_stds
        fig, ax = _plot(x, y, yerr, fig, ax, "black", "o", 0.25, "No Mask")

        # === SN Mask ===
        mask = sn_mask[order].astype(bool)
        x = redshift[mask]
        y = weighted_amplitudes[mask]
        yerr = weighted_stds[mask]
        fig, ax = _plot(x, y, yerr, fig, ax, "brown", "o", 0.25, "SN Mask")

        # === Twins Mask ===
        mask = twins_mask[order].astype(bool)
        x = redshift[mask]
        y = weighted_amplitudes[mask]
        yerr = weighted_stds[mask]
        fig, ax = _plot(x, y, yerr, fig, ax, "blue", "o", 0.25, "Twins Mask")

        # === Combined Mask ===
        mask = (sn_mask[order] * twins_mask[order]).astype(bool)
        x = redshift[mask]
        y = weighted_amplitudes[mask]
        yerr = weighted_stds[mask]
        fig, ax = _plot(x, y, yerr, fig, ax, "green", "o", 1, "Final")

        if legacy is not None:
            legacy_names = legacy["names"]
            intersection = set(names) & set(legacy_names)
            legacy_mask = np.zeros(legacy_names.shape, dtype=np.int32)
            for name in intersection:
                ind = np.argwhere(legacy_names == name)[0]
                legacy_mask[ind] = 1
            legacy_mask = legacy_mask.astype(bool)

            redshift = legacy["redshift"][legacy_mask]
            order = np.argsort(redshift)
            redshift = redshift[order]
            redshift_error = (redshift * 3e5 + 300.0) / 3e5
            magshift_error = abs(-5 * np.log10(redshift / redshift_error))

            amplitudes = legacy["amplitude_mcmc"][legacy_mask][None, ...]
            amplitude_stds = legacy["amplitude_mcmc_err"][legacy_mask][None, ...]

            weights = 1 / (amplitude_stds * amplitude_stds)
            weights /= weights.sum(axis=0)

            weighted_amplitudes = np.sum(weights * amplitudes, axis=0)
            weighted_amplitudes = amplitudes[0, ...][order]

            weighted_variances = np.sum(
                weights * np.square(amplitudes - np.mean(amplitudes)), axis=0
            )
            weighted_variances = weighted_variances[order]

            weighted_stds = np.sqrt(
                (weighted_variances * weighted_variances)
                + (magshift_error * magshift_error)
            )

            redshift_mask = ((redshift > 0.02) & (redshift < 1.0)).astype(int)
            sn_mask = redshift_mask * np.max(legacy["mask"][legacy_mask], axis=(-2, -1))

            twins_mask = np.ones_like(sn_mask)
            if twins is not None:
                twins_mask = np.zeros_like(sn_mask)
                names = legacy["names"][legacy_mask]
                twins_names = twins.name
                intersection = set(names) & set(twins_names)
                for name in intersection:
                    ind = np.argwhere(names == name)[0]
                    df = twins[twins.name == name]
                    twins_mask[ind] = df.mask_twins

            # === No Mask ===
            x = redshift
            y = weighted_amplitudes
            yerr = weighted_stds
            fig, ax = _plot(x, y, yerr, fig, ax, "black", "s", 0.25, "Legacy No Mask")

            # === SN Mask ===
            mask = sn_mask[order].astype(bool)
            x = redshift[mask]
            y = weighted_amplitudes[mask]
            yerr = weighted_stds[mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "brown", "s", 0.25, "Legacy SN Mask")

            # === Twins Mask ===
            mask = twins_mask[order].astype(bool)
            x = redshift[mask]
            y = weighted_amplitudes[mask]
            yerr = weighted_stds[mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "blue", "s", 0.25, "Legacy Twins Mask")

            # === Combined Mask ===
            mask = (sn_mask[order] * twins_mask[order]).astype(bool)
            x = redshift[mask]
            y = weighted_amplitudes[mask]
            yerr = weighted_stds[mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "green", "s", 1, "Legacy Final")

        fig, ax = Plotter.axhline(0, fig=fig, ax=ax, color="black")
        ax.set_ylim(-0.75, 0.75)
        # ax.legend()
        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
