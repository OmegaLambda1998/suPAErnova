from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import Plotter, AbstractPlot

if TYPE_CHECKING:
    from supaernova.configs.steps.steps import AbstractStepResult
    from supaernova.configs.steps.posterior import (
        PosteriorStepResult,
    )

    from .analysis import Axis, Figure


class DistributionPlot(AbstractPlot):
    labels: "dict[str | int, str | dict[str | int, str]] | None" = None
    mean: bool = False


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
        data: "AbstractStepResult | np.ndarray | list[AbstractStepResult] | list[np.ndarray] | dict[str, Any]",
        config: "DistributionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        **chain_kwargs: Any,
    ) -> tuple["Figure", "Axis"] | None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None

        labels = None
        if isinstance(data, dict):
            labels = list(data.keys())
            data = list(data.values())

        if not isinstance(data, list):
            data = [data]
        if config.mean:
            chains = {
                "mean": DistributionPlotter.prep_from_array(
                    np.mean(data, axis=0), config
                )
            }
        else:
            chains = []
            config_labels = config.labels
            for i, d in enumerate(data):
                if labels is not None:
                    config.labels = config_labels[labels[i]]
                if isinstance(d, np.ndarray):
                    chain = DistributionPlotter.prep_from_array(d, config)
                else:
                    chain = DistributionPlotter.prep_from_result(d, config)
                chains.append(chain)
            if labels is None:
                labels = range(len(chains))
            chains = {labels[i]: chain for (i, chain) in enumerate(chains)}

        try:
            fig, ax = Plotter.corner(chains, fig=fig, ax=ax, chain_kwargs=chain_kwargs)
        except:
            return None
        fig.suptitle((config.plot_kwargs or {}).get("title", config.name.capitalize()))

        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None
        return fig, ax
