from typing import TYPE_CHECKING, ClassVar
from pathlib import Path  # noqa: TC003

from pydantic import BaseModel, ConfigDict
import matplotlib as mpl

mpl.use("Cairo")


import chainconsumer as cc
import matplotlib.pyplot as plt

plt.style.use("fast")

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence

    import pandas as pd

    type Figure = mpl.figure.Figure
    type Axis = mpl.axis.Axes


class AbstractPlot(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, allow="extra"
    )

    name: str | None = None
    savepath: Path | None = None
    ext: str = "svg"


class Plotter:
    @staticmethod
    def figure(*args: "Any", **kwargs: "Any") -> "Figure":
        return plt.figure(*args, **kwargs)

    @staticmethod
    def axis(fig: "Figure", *args: "Any", **kwargs: "Any") -> "Axis":
        return fig.add_subplot(*args, **kwargs)

    @staticmethod
    def init(
        fig: "Figure | None",
        ax: "Axis | None",
        *args: "Any",
        fig_args: list["Any"] | None = None,
        fig_kwargs: dict[str, "Any"] | None = None,
        ax_args: list["Any"] | None = None,
        ax_kwargs: dict[str, "Any"] | None = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis]":
        if fig is None:
            if fig_args is None:
                fig_args = []
            if fig_kwargs is None:
                fig_kwargs = {}
            fig = Plotter.figure(*fig_args, *args, **fig_kwargs, **kwargs)
        if ax is None:
            if ax_args is None:
                ax_args = []
            if ax_kwargs is None:
                ax_kwargs = {}
            ax = Plotter.axis(fig, *ax_args, *args, **ax_kwargs, **kwargs)
        return fig, ax

    @staticmethod
    def _plot(
        plot_fn: str,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis]":
        fig, ax = Plotter.init(fig, ax, **kwargs)
        getattr(ax, plot_fn)(*args, **kwargs)
        return fig, ax

    @staticmethod
    def scatter(
        x,
        y,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis]":
        return Plotter._plot("scatter", x, y, *args, fig=fig, ax=ax, **kwargs)

    @staticmethod
    def lines(
        x,
        y,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis]":
        return Plotter._plot("plot", x, y, *args, fig=fig, ax=ax, **kwargs)

    @staticmethod
    def fill_between(
        x,
        low,
        high=0,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis]":
        return Plotter._plot(
            "fill_between", x, low, high, *args, fig=fig, ax=ax, **kwargs
        )

    @staticmethod
    def errorbar(
        x,
        y,
        *args: "Any",
        xerr=None,
        yerr=None,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis]":
        return Plotter._plot("errorbar", x, y, *args, fig=fig, ax=ax, **kwargs)

    @staticmethod
    def corner(
        chains: "dict[str, pd.DataFrame] | list[pd.DataFrame] | pd.DataFrame",
        *args: "Any",
        chain_args: list["Any"] | None = None,
        chain_kwargs: dict[str, "Any"] | None = None,
        plot_args: list["Any"] | None = None,
        plot_kwargs: dict[str, "Any"] | None = None,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis]":
        if chain_args is None:
            chain_args = []
        if chain_kwargs is None:
            chain_kwargs = {}
        if plot_args is None:
            plot_args = []
        if plot_kwargs is None:
            plot_kwargs = {}
        c = cc.ChainConsumer()
        if isinstance(chains, list):
            chains = {str(i): chain for (i, chain) in enumerate(chains)}
        if not isinstance(chains, dict):
            chains = {"0": chains}

        for name, chain in chains.items():
            c.add_chain(
                cc.Chain(
                    *chain_args,
                    *args,
                    samples=chain,
                    name=name,
                    **chain_kwargs,
                    **kwargs,
                )
            )
        fig = c.plotter.plot(*plot_args, *args, **plot_kwargs, **kwargs)
        ax = fig.gca()
        return fig, ax

    @staticmethod
    def save(fig: "Figure", savepath: "Path", *, clear: bool = True) -> "Figure":
        fig.savefig(savepath)
        if clear:
            fig, _ = Plotter.clear(fig=fig)
        return fig

    @staticmethod
    def clear(
        *, fig: "Figure | None" = None, ax: "Axis | None" = None
    ) -> "tuple[Figure, Axis]":
        if fig is not None:
            fig.clf()
        if ax is not None:
            ax.cla()
        return fig, ax

    @staticmethod
    def close(fig: "Figure", ax: "Axis | list[Axis]") -> None:
        if not isinstance(ax, list):
            ax = [ax]
        for a in ax:
            Plotter.clear(fig=fig, ax=a)
        plt.close(fig)
