from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import Plotter, AbstractPlot

if TYPE_CHECKING:
    from suPAErnova.configs.steps.data import DataStepResult
    from suPAErnova.configs.steps.steps import AbstractStepResult
    from suPAErnova.configs.steps.Posterior import PosteriorStepResult

    from .analysis import Axis, Figure


class DispersionPlot(AbstractPlot):
    subset: Literal["train", "test"]


class DispersionPlotter(Plotter):
    @staticmethod
    def plot_dispersion(
        data: "DataStepResult",
        hmcs: "list[PosteriorStepResult]",
        config: "DispersionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return

        mask_sn = np.logical_not(
            np.max(data.mask, axis=(-2, -1), keepdims=True).astype(bool)
        )

        x = np.ma.array(data.redshift[:, 0:1, 0:1], mask=mask_sn)[:, 0, 0:1]
        z = (x * 3e5 + 300.0) / 3e5
        mag_err = abs(-5 * np.log10(x / z))

        amp = np.concat(
            [
                np.ma.array(hmc.hmc.delta_m.mean(axis=0), mask=mask_sn)[None, ...]
                for hmc in hmcs
            ],
            axis=0,
        )
        amp_std = np.concat(
            [
                np.ma.array(np.sqrt(hmc.hmc.delta_m.std(axis=0) ** 2), mask=mask_sn)[
                    None, ...
                ]
                for hmc in hmcs
            ],
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

        k_nmad = 1.4826
        mad_hmc = k_nmad * np.median(np.abs(weighted_mean - np.median(weighted_mean)))
        print("NMAD: ", mad_hmc)
        wrms_hmc = np.std(weighted_mean, axis=0)[0]
        print("RMS: ", wrms_hmc)

        y = weighted_mean
        yerr = weighted_err

        fig, ax = Plotter.errorbar(
            x, y, yerr=yerr, fig=fig, ax=ax, ms=8, marker="o", ls="none", lw=2
        )
        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
