from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import Plotter, AbstractPlot

if TYPE_CHECKING:
    from suPAErnova.configs.steps.steps import AbstractStepResult
    from suPAErnova.configs.steps.posterior import (
        PosteriorStepResult,
    )

    from .analysis import Axis, Figure


class DistributionPlot(AbstractPlot):
    labels: "dict[str | int, str] | None" = None


class DistributionPlotter(Plotter):
    @staticmethod
    def prep_from_result(
        data: "AbstractStepResult", config: DistributionPlot
    ) -> pd.DataFrame:
        return pd.DataFrame({
            label: getattr(data, key) for (key, label) in (config.labels or {}).items()
        })

    @staticmethod
    def prep_from_array(data: "np.ndarray", config: DistributionPlot) -> pd.DataFrame:
        return pd.DataFrame({
            label: data[:, ind] for (ind, label) in (config.labels or {}).items()
        })

    @staticmethod
    def plot_corner(
        data: "AbstractStepResult | np.ndarray",
        config: "DistributionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return
        if isinstance(data, np.ndarray):
            chains = DistributionPlotter.prep_from_array(data, config)
        else:
            chains = DistributionPlotter.prep_from_result(data, config)
        fig, ax = Plotter.corner(chains, fig=fig, ax=ax)
        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
