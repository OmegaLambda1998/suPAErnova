from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np

from .spectra import SpectraPlot, SpectraPlotter
from .analysis import Plotter

if TYPE_CHECKING:
    import pandas as pd

    from suPAErnova.configs.steps.data import DataStepResult
    from suPAErnova.configs.steps.posterior import PosteriorStepResult

    from .analysis import Axis, Figure


class DispersionPlot(SpectraPlot):
    subset: Literal["train", "test"]
    legacy: Path | None = None
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

        x = data.redshift[:, 0, 0]
        order = x.argsort()
        x = x[order]
        z = (x * 3e5 + 300.0) / 3e5
        mag_err = abs(-5 * np.log10(x / z))

        amp = np.concat(
            [hmc.hmc.delta_m.mean(axis=0)[None, ...] for hmc in hmcs],
            axis=0,
        )
        amp_std = np.concat(
            [np.sqrt(hmc.hmc.delta_m.std(axis=0) ** 2)[None, ...] for hmc in hmcs],
            axis=0,
        )

        weight = 1 / (amp_std * amp_std)
        weighted_mean = np.sum(weight * amp, axis=0) / np.sum(weight, axis=0)

        weighted_dev = np.sqrt(
            (amp.shape[0] / (amp.shape[0] - 1))
            * (
                (np.sum(weight * amp * amp, axis=0) / np.sum(weight, axis=0))
                - (weighted_mean * weighted_mean)
            )
        )

        weighted_err = np.sqrt(weighted_dev * weighted_dev + mag_err * mag_err)

        weighted_mean = amp[0, ...]
        weighted_err *= 0

        weighted_mean = weighted_mean[order]
        weighted_err = weighted_err[order]

        _wl, _amplitude, _sigma, sn_mask, _spec_mask, _wl_mask = SpectraPlotter.prep(
            data, config
        )

        twins_mask = np.ones_like(sn_mask)
        if twins is not None:
            twins_mask = np.zeros_like(sn_mask)
            names = set(data.sn_name.flatten())
            twins_names = set(twins.name)
            intersection = names & twins_names
            for name in intersection:
                ind = np.argwhere(data.sn_name == name)[0][0]
                df = twins[twins.name == name]
                twins_mask[ind] = df.mask_twins

        sn_mask = np.sum(data.mask * sn_mask, axis=(-2, -1), keepdims=True)

        if legacy is not None:
            l_x = legacy["redshift"]
            l_order = l_x.argsort()
            l_x = l_x[l_order]
            l_z = (l_x * 3e5 + 300.0) / 3e5
            l_mag_err = abs(-5 * np.log10(l_x / l_z))

            l_amp = legacy["amplitude_mcmc"]
            l_amp_std = np.sqrt(legacy["amplitude_mcmc_err"] ** 2.0)

            l_weight = 1 / (l_amp_std * l_amp_std)
            l_weighted_mean = np.sum(l_weight * l_amp, axis=0) / np.sum(
                l_weight, axis=0
            )

            l_weighted_dev = np.sqrt(
                (l_amp.shape[0] / (l_amp.shape[0] - 1))
                * (
                    (
                        np.sum(l_weight * l_amp * l_amp, axis=0)
                        / np.sum(l_weight, axis=0)
                    )
                    - (l_weighted_mean * l_weighted_mean)
                )
            )

            l_weighted_err = np.sqrt(
                l_weighted_dev * l_weighted_dev + l_mag_err * l_mag_err
            )

            l_weighted_mean = l_amp
            l_weighted_err *= 0

            l_weighted_mean = l_weighted_mean[l_order]
            l_weighted_err = l_weighted_err[l_order]

            l_sn_mask = legacy["mask_sn"]

            l_twins_mask = np.ones_like(l_sn_mask)
            if twins is not None:
                l_twins_mask = np.zeros_like(l_sn_mask)
                l_names = set(legacy["names"])
                l_twins_names = set(twins.name)
                l_intersection = l_names & l_twins_names
                for name in l_intersection:
                    l_ind = np.argwhere(legacy["names"] == name)[0]
                    l_df = twins[twins.name == name]
                    l_twins_mask[l_ind] = l_df.mask_twins.astype(np.int32)

            l_sn_mask = np.sum(
                legacy["mask"] * l_sn_mask[:, None, None], axis=(-2, -1), keepdims=True
            )

            print(
                set(data.sn_name.flatten()).symmetric_difference(set(legacy["names"]))
            )

        k_nmad = 1.4826

        # No mask
        if legacy is not None:
            l_x = legacy["redshift"]
            l_y = l_weighted_mean
            l_yerr = l_weighted_err
            print(l_x.shape, l_y.shape, l_yerr.shape)
            print("l_n_sn", np.ones_like(l_sn_mask[:, 0, 0]).sum())
            mad_hmc = k_nmad * np.median(np.abs(l_y - np.median(l_y)))
            print("l_NMAD: ", mad_hmc)
            wrms_hmc = np.std(l_y, axis=0)
            print("l_RMS: ", wrms_hmc)
            fig, ax = Plotter.errorbar(
                l_x,
                l_y,
                yerr=l_yerr,
                fig=fig,
                ax=ax,
                marker="s",
                color="black",
                alpha=0.25,
            )

        x = data.redshift[:, 0, 0]
        y = weighted_mean[:, 0]
        yerr = weighted_err[:, 0]
        print(x.shape, y.shape, yerr.shape)
        print("n_sn", np.ones_like(sn_mask[:, 0, 0]).sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)
        fig, ax = Plotter.errorbar(
            x, y, yerr=yerr, fig=fig, ax=ax, color="black", alpha=0.25
        )

        # SN mask
        if legacy is not None:
            l_mask = l_sn_mask.astype(bool)[:, 0, 0]
            l_x = legacy["redshift"][l_mask]
            l_y = l_weighted_mean[l_mask]
            l_yerr = l_weighted_err[l_mask]
            print(l_x.shape, l_y.shape, l_yerr.shape, l_mask.shape)
            print("l_sn_mask", l_mask.sum())
            mad_hmc = k_nmad * np.median(np.abs(l_y - np.median(l_y)))
            print("l_NMAD: ", mad_hmc)
            wrms_hmc = np.std(l_y, axis=0)
            print("l_RMS: ", wrms_hmc)
            fig, ax = Plotter.errorbar(
                l_x,
                l_y,
                yerr=l_yerr,
                fig=fig,
                ax=ax,
                marker="s",
                color="red",
                alpha=0.25,
            )

        mask = sn_mask.astype(bool)[:, 0, 0]
        x = data.redshift[mask][:, 0, 0]
        y = weighted_mean[mask][:, 0]
        yerr = weighted_err[mask][:, 0]
        print(x.shape, y.shape, yerr.shape, mask.shape)
        print("sn_mask", mask.sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)
        fig, ax = Plotter.errorbar(
            x, y, yerr=yerr, fig=fig, ax=ax, color="red", alpha=0.25
        )

        # Twins mask
        if legacy is not None:
            l_mask = l_twins_mask.astype(bool)
            l_x = legacy["redshift"][l_mask]
            l_y = l_weighted_mean[l_mask]
            l_yerr = l_weighted_err[l_mask]
            print(l_x.shape, l_y.shape, l_yerr.shape, l_mask.shape)
            print("l_twins_mask", l_mask.sum())
            mad_hmc = k_nmad * np.median(np.abs(l_y - np.median(l_y)))
            print("l_NMAD: ", mad_hmc)
            wrms_hmc = np.std(l_y, axis=0)
            print("l_RMS: ", wrms_hmc)
            fig, ax = Plotter.errorbar(
                l_x,
                l_y,
                yerr=l_yerr,
                fig=fig,
                ax=ax,
                marker="s",
                color="blue",
                alpha=0.25,
            )

        mask = twins_mask.astype(bool)[:, 0, 0]
        x = data.redshift[mask][:, 0, 0]
        y = weighted_mean[mask][:, 0]
        yerr = weighted_err[mask][:, 0]
        print(x.shape, y.shape, yerr.shape, mask.shape)
        print("twins_mask", mask.sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)
        fig, ax = Plotter.errorbar(
            x, y, yerr=yerr, fig=fig, ax=ax, color="blue", alpha=0.25
        )

        # Final

        if legacy is not None:
            l_mask = (l_twins_mask * l_sn_mask[:, 0, 0]).astype(bool)
            l_x = legacy["redshift"][l_mask]
            l_y = l_weighted_mean[l_mask]
            l_yerr = l_weighted_err[l_mask]
            print(l_x.shape, l_y.shape, l_yerr.shape, l_mask.shape)
            print("l_final", l_mask.sum())
            mad_hmc = k_nmad * np.median(np.abs(l_y - np.median(l_y)))
            print("l_NMAD: ", mad_hmc)
            wrms_hmc = np.std(l_y, axis=0)
            print("l_RMS: ", wrms_hmc)
            fig, ax = Plotter.errorbar(
                l_x,
                l_y,
                yerr=l_yerr,
                fig=fig,
                ax=ax,
                marker="s",
                color="green",
            )

        mask = (twins_mask * sn_mask).astype(bool)[:, 0, 0]
        x = data.redshift[mask][:, 0, 0]
        y = weighted_mean[mask][:, 0]
        yerr = weighted_err[mask][:, 0]
        print(x.shape, y.shape, yerr.shape, mask.shape)
        print("final", mask.sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)

        fig, ax = Plotter.errorbar(x, y, yerr=yerr, fig=fig, ax=ax, color="green")

        ax.set_ylim(-0.75, 0.75)

        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
