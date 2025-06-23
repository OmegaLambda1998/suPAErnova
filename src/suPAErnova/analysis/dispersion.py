from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np

from .spectra import SpectraPlotter
from .analysis import Plotter, AbstractPlot

if TYPE_CHECKING:
    from numpy import typing as npt
    import pandas as pd

    from suPAErnova.configs.steps.data import DataStepResult
    from suPAErnova.configs.steps.steps import AbstractStepResult
    from suPAErnova.configs.steps.Posterior import PosteriorStepResult

    from .analysis import Axis, Figure


class DispersionPlot(AbstractPlot):
    subset: Literal["train", "test"]
    twins: str | None = None
    filter: (
        dict[str, dict["Literal['min', 'max', 'equals', 'contains']", str | float]]
        | None
    ) = None


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
        force: bool = False,
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return

        x = data.redshift[:, 0, 0]
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

        k_nmad = 1.4826

        # No mask
        x = data.redshift[:, 0, :]
        y = weighted_mean[:, 0]
        yerr = weighted_err[:, 0]
        fig, ax = Plotter.errorbar(
            x, y, yerr=yerr, fig=fig, ax=ax, color="black", alpha=0.25
        )
        print("n_sn", np.ones_like(sn_mask[:, 0, 0]).sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)

        # SN mask
        mask = sn_mask.astype(bool)[:, 0, 0]
        x = data.redshift[mask][:, 0, :]
        y = weighted_mean[mask][:, 0]
        yerr = weighted_err[mask][:, 0]
        fig, ax = Plotter.errorbar(x, y, yerr=yerr, fig=fig, ax=ax, alpha=0.25)
        print("sn_mask", mask.sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)

        # Twins mask
        mask = twins_mask.astype(bool)[:, 0, 0]
        x = data.redshift[mask][:, 0, :]
        y = weighted_mean[mask][:, 0]
        yerr = weighted_err[mask][:, 0]
        fig, ax = Plotter.errorbar(x, y, yerr=yerr, fig=fig, ax=ax, alpha=0.25)
        print("twins_mask", mask.sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)

        mask = (twins_mask * sn_mask).astype(bool)[:, 0, 0]
        x = data.redshift[mask][:, 0, :]
        y = weighted_mean[mask][:, 0]
        yerr = weighted_err[mask][:, 0]
        fig, ax = Plotter.errorbar(x, y, yerr=yerr, fig=fig, ax=ax)
        print("final", mask.sum())
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)
        print("RMS: ", wrms_hmc)

        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
