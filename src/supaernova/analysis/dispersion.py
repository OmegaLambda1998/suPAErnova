from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np

from .spectra import SpectraPlot, SpectraPlotter
from .analysis import Plotter

if TYPE_CHECKING:
    from numpy import typing as npt
    import pandas as pd

    from supaernova.configs.steps.data import SNPAEData
    from supaernova.configs.steps.posterior import PosteriorStepResult

    from .analysis import Axis, Figure


class DispersionPlot(SpectraPlot):
    subset: Literal["train", "test"]
    legacy: tuple[Path, ...] | None = None
    twins: str | None = None


class DispersionPlotter(Plotter):
    @staticmethod
    def plot_dispersion(
        data: "SNPAEData",
        hmcs: "list[PosteriorStepResult]",
        config: "DispersionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        twins: "pd.DataFrame | None" = None,
        legacy: "dict[str, np.ndarray] | None" = None,
        force: bool = False,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
    ) -> None:
        fig, ax, _hline = Plotter.axhline(0, fig=fig, ax=ax, color="black")

        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return

        (
            _wl,
            _amplitude,
            _sigma,
            _input_mask,
            input_sn_mask,
            _input_spec_mask,
            _input_wl_mask,
        ) = SpectraPlotter.prep(
            data,
            config,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
        )

        input_sn_mask = input_sn_mask[:, 0, 0]
        names = data.sn_name[:, 0, 0]

        redshift = data.redshift[:, 0, 0]
        order = np.argsort(redshift)
        redshift = redshift[order]
        redshift_error = (redshift * 3e5 + 300.0) / 3e5
        magshift_error = abs(-5 * np.log10(redshift / redshift_error))

        twins_mask = np.ones_like(input_sn_mask)
        if twins is not None:
            twins_mask = np.zeros_like(input_sn_mask)
            twins_names = twins.name
            pae_names = names[order]
            intersection = set(pae_names) & set(twins_names)
            for name in intersection:
                ind = np.argwhere(pae_names == name)[0]
                df = twins[twins.name == name]
                twins_mask[ind] = df.mask_twins

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

        n_hmc = len(hmcs)
        n_scale = 1 if n_hmc == 1 else n_hmc / (n_hmc - 1)
        weighted_stds = np.sqrt(
            n_scale * (weighted_variances * weighted_variances)
            + (magshift_error * magshift_error)
        )

        def _plot(x, y, yerr, fig, ax, color, marker, alpha, title):
            k = 1.4826

            w_rms = np.sqrt(np.sum(y * y) / np.size(y))
            w_mad = np.std(y) / k

            fig, ax, _ebar = Plotter.errorbar(
                x,
                y,
                yerr=yerr,
                fig=fig,
                ax=ax,
                color=color,
                marker=marker,
                alpha=alpha,
                label=f"{title}\n{np.size(y)} SN\nRMS: {w_rms:.3f}\nNMAD: {w_mad:.3f}",
            )

            return fig, ax

        # === No Mask ===
        x = redshift
        y = weighted_amplitudes
        yerr = weighted_stds
        fig, ax = _plot(x, y, yerr, fig, ax, "black", "o", 0.25, "No Mask")

        if twins is not None:
            # === SN Mask ===
            plot_mask = input_sn_mask[order].astype(bool)
            x = redshift[plot_mask]
            y = weighted_amplitudes[plot_mask]
            yerr = weighted_stds[plot_mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "brown", "o", 0.25, "SN Mask")

            # === Twins Mask ===
            plot_mask = twins_mask.astype(bool)
            x = redshift[plot_mask]
            y = weighted_amplitudes[plot_mask]
            yerr = weighted_stds[plot_mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "blue", "o", 0.25, "Twins Mask")

        # === Combined Mask ===
        plot_mask = (input_sn_mask[order] * twins_mask).astype(bool)
        x = redshift[plot_mask]
        y = weighted_amplitudes[plot_mask]
        yerr = weighted_stds[plot_mask]
        fig, ax = _plot(x, y, yerr, fig, ax, "green", "o", 1, "Final")

        if legacy is not None:
            legacy_names = legacy["names"]
            intersection = set(names) & set(legacy_names)
            legacy_mask = np.zeros(legacy_names.shape, dtype=np.int32)
            for name in intersection:
                ind = np.argwhere(legacy_names == name)[0]
                legacy_mask[ind] = 1
            legacy_mask = legacy_mask.astype(bool)
            legacy_names = legacy["names"][legacy_mask]

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

            redshift_mask = ((redshift >= 0.02) & (redshift <= 1.0)).astype(np.int32)

            legacy_sn_mask = redshift_mask * legacy["mask"][legacy_mask].min(
                axis=-1
            ).max(axis=-1)

            twins_mask = np.ones_like(legacy_sn_mask)
            if twins is not None:
                twins_mask = np.zeros_like(legacy_sn_mask)
                names = legacy["names"][legacy_mask][order]
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
            plot_mask = legacy_sn_mask[order].astype(bool)
            x = redshift[plot_mask]
            y = weighted_amplitudes[plot_mask]
            yerr = weighted_stds[plot_mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "brown", "s", 0.25, "Legacy SN Mask")

            # === Twins Mask ===
            plot_mask = twins_mask.astype(bool)
            x = redshift[plot_mask]
            y = weighted_amplitudes[plot_mask]
            yerr = weighted_stds[plot_mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "blue", "s", 0.25, "Legacy Twins Mask")

            # === Combined Mask ===
            plot_mask = (legacy_sn_mask[order] * twins_mask).astype(bool)
            x = redshift[plot_mask]
            y = weighted_amplitudes[plot_mask]
            yerr = weighted_stds[plot_mask]
            fig, ax = _plot(x, y, yerr, fig, ax, "green", "s", 1, "Legacy Final")

        ax.set_xlabel("z")
        ax.set_ylabel("ΔM")
        ax.legend(bbox_to_anchor=(1.0, 1.0), ncols=1 if legacy is None else 2)
        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
