from typing import TYPE_CHECKING, Any, Literal
from pathlib import Path

import numpy as np

from .spectra import SpectraPlot, SpectraPlotter
from .analysis import Plotter

if TYPE_CHECKING:
    from numpy import typing as npt
    import pandas as pd

    from supaernova.configs.steps.data import LazySNPAEData
    from supaernova.configs.steps.posterior import PosteriorStepResult

    from .analysis import Axis, Figure


class DispersionPlot(SpectraPlot):
    subset: Literal["train", "test"]
    legacy: tuple[Path, ...] | None = None
    twins: str | None = None


class DispersionPlotter(Plotter):
    @staticmethod
    def plot_dispersion(
        data: "LazySNPAEData",
        hmcs: "list[PosteriorStepResult]",
        config: "DispersionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        twins: "pd.DataFrame | None" = None,
        legacy: "dict[str, npt.NDArray[Any]] | None" = None,
        force: bool = False,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
    ) -> None:
        if fig is None:
            fig_1 = Plotter.figure()
            fig_2 = None if legacy is None else Plotter.figure()
            fig = [fig_1, fig_2]
        else:
            fig_1, fig_2 = fig
        if ax is None:
            n_rows = 2
            n_cols = 1
            spectra_ax = Plotter.axis(fig_1, n_rows, n_cols, 1)
            spectra_ax.tick_params("x", labelbottom=False)

            pull_ax_1 = Plotter.axis(fig_1, n_rows, n_cols, 2, sharex=spectra_ax)
            if legacy is not None:
                residual_ax = Plotter.axis(fig_2, n_rows, n_cols, 1)
                residual_ax.tick_params("x", labelbottom=False)

                pull_ax_2 = Plotter.axis(fig_2, n_rows, n_cols, 2, sharex=residual_ax)
                fig_2.subplots_adjust(hspace=0)
            else:
                residual_ax = None
                pull_ax_2 = None
            fig_1.subplots_adjust(hspace=0)
            ax = [spectra_ax, pull_ax_1, residual_ax, pull_ax_2]
        else:
            spectra_ax, pull_ax_1, residual_ax, pull_ax_2 = ax

        fig_1, spectra_ax, _hline = Plotter.axhline(
            0, fig=fig_1, ax=spectra_ax, color="black"
        )

        if legacy is not None:
            fig_2, residual_ax, _hline = Plotter.axhline(
                0, fig=fig_2, ax=residual_ax, color="black"
            )

        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return
        if legacy:
            legacy_savepath = (
                config.savepath or Path()
            ) / f"{config.name}_residual.{config.ext}"

        pae_redshift = data.redshift[:, 0, 0]

        (
            _wl,
            _amplitude,
            _sigma,
            sn_name,
            _time,
            input_mask,
            _input_sn_mask,
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

        # Determine which spectra to keep
        # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
        mask_spec = input_mask.max(axis=-1)

        # Determine which SNe to keep
        # Will mask out any SN with *no* unmasked spectra
        mask_sn = mask_spec.max(axis=-1)

        pae_order = np.argsort(pae_redshift)
        pae_redshift = pae_redshift[pae_order]
        pae_redshift_error = (pae_redshift * 3e5 + 300.0) / 3e5
        pae_magshift_error = abs(-5 * np.log10(pae_redshift / pae_redshift_error))

        pae_mask = mask_sn[pae_order]

        pae_names = sn_name[:, 0, 0][pae_order]
        pae_amplitudes = np.concatenate(
            [np.mean(hmc.hmc.delta_m, axis=0, keepdims=True) for hmc in hmcs],
            axis=0,
        )[..., 0][..., pae_order]
        pae_amplitude_stds = np.concatenate(
            [
                np.sqrt(np.square(np.std(hmc.hmc.delta_m, axis=0, keepdims=True)))
                for hmc in hmcs
            ],
            axis=0,
        )[..., 0][..., pae_order]

        pae_weights = 1 / (pae_amplitude_stds * pae_amplitude_stds)
        pae_weighted_sum = pae_weights.sum(axis=0)
        pae_weighted_amplitudes = (pae_weights * pae_amplitudes).sum(
            axis=0
        ) / pae_weighted_sum

        pae_n_iter = len(hmcs)
        pae_n_eff = 1 if pae_n_iter == 1 else pae_n_iter / (pae_n_iter - 1)

        pae_weighted_variance = (
            (pae_weights * pae_amplitudes * pae_amplitudes).sum(axis=0)
            / pae_weighted_sum
        ) - (pae_weighted_amplitudes * pae_weighted_amplitudes)

        pae_weighted_deviations = np.sqrt(pae_n_eff * np.abs(pae_weighted_variance))

        pae_weighted_stds = np.sqrt(
            pae_weighted_deviations * pae_weighted_deviations
            + pae_magshift_error * pae_magshift_error
        )

        pae_twins_mask = np.ones_like(pae_mask)
        if twins is not None:
            pae_twins_mask = np.zeros_like(pae_mask)
            twins_names = twins.name
            pae_intersection = set(pae_names) & set(twins_names)
            for name in pae_intersection:
                ind = np.argwhere(pae_names == name)[0]
                df = twins[twins.name == name]
                pae_twins_mask[ind] = df.mask_twins

        def _plot(
            x: "npt.NDArray[Any]",
            y: "npt.NDArray[Any]",
            yerr: "npt.NDArray[Any]",
            mask: "npt.NDArray[Any] | None",
            fig: "tuple(Figure, Figure | None)",
            ax: "tuple[Axis, Axis, Axis | None, Axis | None]",
            color: str,
            marker: str,
            alpha: float,
            title: str,
            y_prime: "npt.NDArray[Any] | None" = None,
            yerr_prime: "npt.NDArray[Any] | None" = None,
        ) -> tuple[
            tuple["Figure", "Figure | None"],
            tuple["Axis", "Axis", "Axis | None", "Axis | None"],
        ]:
            fig_1, fig_2 = fig
            s_ax, p_ax_1, r_ax, p_ax_2 = ax

            if mask is not None:
                x = x[mask]
                y = y[mask]
                yerr = yerr[mask]
                y_prime = y_prime[mask] if y_prime is not None else None
                yerr_prime = yerr_prime[mask] if yerr_prime is not None else None

            k = 1.4826

            w_rms = np.std(y, axis=0)
            w_nmad = k * np.median(np.abs(y) - np.median(y))

            fig_1, s_ax, _ebar = Plotter.errorbar(
                x,
                y,
                yerr=yerr,
                fig=fig_1,
                ax=s_ax,
                color=color,
                marker=marker,
                alpha=alpha,
                label=f"{title}\n{np.size(y)} SN\nRMS: {w_rms:.3f}\nNMAD: {w_nmad:.3f}",
            )

            fig_1, p_ax_1, _ebar = Plotter.errorbar(
                x,
                np.abs(y) / yerr,
                fig=fig_1,
                ax=p_ax_1,
                color=color,
                marker=marker,
                alpha=alpha,
            )

            if (y_prime is not None) and (yerr_prime is not None):
                y_residual = y - y_prime
                err = np.sqrt(yerr * yerr + yerr_prime * yerr_prime)
                y_pull = np.abs(y_residual) / err

                fig_2, r_ax, _ebar = Plotter.errorbar(
                    x,
                    y_residual,
                    yerr=err,
                    fig=fig_2,
                    ax=r_ax,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                )

                fig_2, p_ax_2, _ebar = Plotter.errorbar(
                    x,
                    y_pull,
                    fig=fig_2,
                    ax=p_ax_2,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                )

            return (fig_1, fig_2), (s_ax, p_ax_1, r_ax, p_ax_2)

        pae_x = pae_redshift
        pae_y = pae_weighted_amplitudes
        pae_yerr = pae_weighted_stds
        no_plot_mask = None
        sn_plot_mask = pae_mask.astype(bool)
        twins_plot_mask = pae_twins_mask.astype(bool)
        combined_plot_mask = (pae_mask * pae_twins_mask).astype(bool)

        legacy_y = None
        legacy_yerr = None

        if legacy is not None:
            legacy_names = legacy["names"]
            legacy_intersection = set(pae_names) & set(legacy_names)
            legacy_mask = np.zeros(legacy_names.shape, dtype=np.int32)
            for name in legacy_intersection:
                ind = np.argwhere(legacy_names == name)[0]
                legacy_mask[ind] = 1
            legacy_mask = legacy_mask.astype(bool)

            legacy_redshift = legacy["redshift"][legacy_mask]
            legacy_order = np.argsort(legacy_redshift)
            legacy_redshift = legacy_redshift[legacy_order]
            legacy_redshift_error = (legacy_redshift * 3e5 + 300.0) / 3e5
            legacy_magshift_error = abs(
                -5 * np.log10(legacy_redshift / legacy_redshift_error)
            )

            legacy_names = legacy["names"][legacy_mask][legacy_order]
            legacy_amplitudes = legacy["amplitude_mcmc"][legacy_mask][None, ...][
                ..., legacy_order
            ]
            legacy_amplitude_stds = legacy["amplitude_mcmc_err"][legacy_mask][
                None, ...
            ][..., legacy_order]

            legacy_weights = 1 / (legacy_amplitude_stds * legacy_amplitude_stds)
            legacy_weighted_sum = legacy_weights.sum(axis=0)
            legacy_weighted_amplitudes = (legacy_weights * legacy_amplitudes).sum(
                axis=0
            ) / legacy_weighted_sum

            legacy_n_eff = 1

            legacy_weighted_variance = (
                (legacy_weights * legacy_amplitudes * legacy_amplitudes).sum(axis=0)
                / legacy_weighted_sum
            ) - (legacy_weighted_amplitudes * legacy_weighted_amplitudes)

            legacy_weighted_deviations = np.sqrt(
                legacy_n_eff * np.abs(legacy_weighted_variance)
            )

            legacy_weighted_stds = np.sqrt(
                legacy_weighted_deviations * legacy_weighted_deviations
                + legacy_magshift_error * legacy_magshift_error
            )
            # pae_mask *= np.isfinite(legacy_weighted_stds)

            # === No Mask ===
            legacy_x = legacy_redshift
            legacy_y = legacy_weighted_amplitudes
            legacy_yerr = legacy_weighted_stds
            fig, ax = _plot(
                legacy_x,
                legacy_y,
                legacy_yerr,
                no_plot_mask,
                fig,
                ax,
                "black",
                "s",
                0.25,
                "Legacy No Mask",
            )

            if twins is not None:
                # === SN Mask ===
                fig, ax = _plot(
                    legacy_x,
                    legacy_y,
                    legacy_yerr,
                    sn_plot_mask,
                    fig,
                    ax,
                    "brown",
                    "s",
                    0.25,
                    "Legacy SN Mask",
                )

                # === Twins Mask ===
                fig, ax = _plot(
                    legacy_x,
                    legacy_y,
                    legacy_yerr,
                    twins_plot_mask,
                    fig,
                    ax,
                    "blue",
                    "s",
                    0.25,
                    "Legacy Twins Mask",
                )

            # === Combined Mask ===
            fig, ax = _plot(
                legacy_x,
                legacy_y,
                legacy_yerr,
                combined_plot_mask,
                fig,
                ax,
                "green",
                "s",
                1,
                "Legacy Final",
            )

        # === No Mask ===
        fig, ax = _plot(
            pae_x,
            pae_y,
            pae_yerr,
            no_plot_mask,
            fig,
            ax,
            "black",
            "o",
            0.25,
            "No Mask",
            y_prime=legacy_y,
            yerr_prime=legacy_yerr,
        )

        if twins is not None:
            # === SN Mask ===
            fig, ax = _plot(
                pae_x,
                pae_y,
                pae_yerr,
                sn_plot_mask,
                fig,
                ax,
                "brown",
                "o",
                0.25,
                "SN Mask",
                y_prime=legacy_y,
                yerr_prime=legacy_yerr,
            )

            # === Twins Mask ===
            fig, ax = _plot(
                pae_x,
                pae_y,
                pae_yerr,
                twins_plot_mask,
                fig,
                ax,
                "blue",
                "o",
                0.25,
                "Twins Mask",
                y_prime=legacy_y,
                yerr_prime=legacy_yerr,
            )

        # === Combined Mask ===
        fig, ax = _plot(
            pae_x,
            pae_y,
            pae_yerr,
            combined_plot_mask,
            fig,
            ax,
            "green",
            "o",
            1,
            "Final",
            y_prime=legacy_y,
            yerr_prime=legacy_yerr,
        )
        spectra_ax.set_ylim(
            -1.1 * np.abs((pae_y - pae_yerr)[combined_plot_mask].min()),
            1.1 * (pae_y + pae_yerr)[combined_plot_mask].max(),
        )

        fig_1.align_ylabels([spectra_ax, pull_ax_1])
        spectra_ax.legend(bbox_to_anchor=(1.0, 1.0), ncols=1 if legacy is None else 2)
        spectra_ax.set_title(
            (config.plot_kwargs or {}).get("title", config.name.capitalize())
        )
        spectra_ax.set_ylabel("ΔM")

        pull_ax_1.set_ylabel("Pull")
        pull_ax_1.set_xlabel("z")

        fig_1 = Plotter.save(fig_1, savepath)
        Plotter.close(fig_1, [spectra_ax, pull_ax_1])

        if legacy is not None:
            fig_2.align_ylabels([residual_ax, pull_ax_2])
            residual_ax.set_title(
                (config.plot_kwargs or {}).get("title", config.name.capitalize())
                + " vs Legacy"
            )
            residual_ax.set_ylabel("Residual")
            residual_ax.set_ylim(
                -1.1
                * np.abs(((pae_y - legacy_y) - pae_yerr)[combined_plot_mask].min()),
                1.1 * ((pae_y - legacy_y) + pae_yerr)[combined_plot_mask].max(),
            )

            pull_ax_2.set_ylabel("Pull")
            pull_ax_2.set_xlabel("z")
            fig_2 = Plotter.save(fig_2, legacy_savepath)
            Plotter.close(fig_2, [residual_ax, pull_ax_2])
