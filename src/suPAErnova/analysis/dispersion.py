from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np

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
        min_redshift: float = 0,
        max_redshift: float = np.inf,
        force: bool = False,
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return

        x = data.redshift[:, 0, :]
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

        twins_mask = np.ones_like(data.mask)
        if twins is not None:
            twins_mask = np.zeros_like(data.mask)
            names = set(data.sn_name.flatten())
            twins_names = set(twins.name)
            intersection = names & twins_names
            for name in intersection:
                ind = np.argwhere(data.sn_name == name)
                df = twins[twins.name == name]
                twins_mask[ind] = df.mask_twins
        redshift_mask = (
            (data.redshift >= min_redshift) & (data.redshift <= max_redshift)
        )[:, 0, 0]

        print("twins", np.sum(twins_mask[:, 0, 0]))
        print("redshift", np.sum(redshift_mask.astype(np.int32)))

        mask = twins_mask.astype(bool)[:, 0, 0] & redshift_mask
        x = data.redshift[mask][:, 0, :]
        y = weighted_mean[mask]
        yerr = weighted_err[mask]

        print(x.shape, y.shape, yerr.shape)

        k_nmad = 1.4826
        mad_hmc = k_nmad * np.median(np.abs(y - np.median(y)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(y, axis=0)[0]
        print("RMS: ", wrms_hmc)

        fig, ax = Plotter.errorbar(
            x, y, yerr=yerr, fig=fig, ax=ax, ms=8, marker="o", ls="none", lw=2
        )
        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
