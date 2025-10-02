from typing import TYPE_CHECKING, Any, TypeVar, ClassVar
from pathlib import Path  # noqa: TC003
import colorsys
from collections.abc import Iterable

import numpy as np
from cycler import cycler
from pydantic import BaseModel, ConfigDict
import matplotlib as mpl
import chainconsumer as cc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt

COLOURS = (
    np.array([
        [166, 206, 227, 255],
        [31, 120, 180, 255],
        [178, 223, 138, 255],
        [51, 160, 44, 255],
        [251, 154, 153, 255],
        [227, 26, 28, 255],
        [253, 191, 111, 255],
        [255, 127, 0, 255],
        [202, 178, 214, 255],
        [106, 61, 154, 255],
        [255, 255, 153, 255],
        [177, 89, 40, 255],
    ])
    / 255
)

mpl.use("Cairo")
mpl.rcParams["axes.prop_cycle"] = cycler(color=COLOURS)
width, height = mpl.rcParams["figure.figsize"]
dpi = mpl.rcParams["figure.dpi"]

if TYPE_CHECKING:
    from typing import Literal
    from collections.abc import Sequence

    from numpy import typing as npt
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.colorbar import Colorbar
    from matplotlib.container import Container, BarContainer, ErrorbarContainer
    from matplotlib.collections import (
        Collection,
        PathCollection,
        FillBetweenPolyCollection,
    )

    type Figure = plt.Figure
    type Axis = plt.Axes
    Plot = TypeVar("Plot", bound=Collection | Container)


class AbstractPlot(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="allow"
    )

    name: str | None = None
    savepath: Path | None = None
    ext: str = "png"
    plot_args: list[Any] | None = None
    plot_kwargs: dict[str, Any] | None = None


def scale_lightness(rgba, scale_l):
    # convert rgb to hls
    rgb = rgba[:-1]
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    rgb = colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)
    return (*rgb, rgba[-1])


class Plotter:
    colour_sequence = ListedColormap(COLOURS)
    colour_maps: ClassVar[list[LinearSegmentedColormap]] = [
        LinearSegmentedColormap.from_list(
            f"cmap_{i}",
            (scale_lightness(rgba, 0.75), scale_lightness(rgba, 1.25)),
        )
        for i, rgba in enumerate(COLOURS)
    ]

    @staticmethod
    def figure(*args: "Any", scale: int = 1, **kwargs: "Any") -> "Figure":
        return plt.figure(
            *args,
            figsize=(width * scale, height * scale),
            layout="constrained",
            **kwargs,
        )

    @staticmethod
    def subfig(
        fig: "Figure", nrows: int, ncols: int, *args: "Any", **kwargs: "Any"
    ) -> "npt.NDArray[Figure]":
        return fig.subfigures(nrows, ncols, *args, **kwargs)

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
    ) -> "tuple[Figure, Axis, Plot]":
        fig, ax = Plotter.init(fig, ax)
        ctr = getattr(ax, plot_fn)(*args, **kwargs)
        return fig, ax, ctr

    @staticmethod
    def colourbar(
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        cmap: "Colormap | None" = None,
        norm: "Normalize | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis, Colorbar]":
        fig, ax = Plotter.init(fig, ax)
        if cmap is None:
            cmap = Plotter.colour_sequence
        return fig, ax, fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)

    @staticmethod
    def scatter(
        x,
        y,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis, PathCollection]":
        return Plotter._plot(
            "scatter",
            x,
            y,
            *args,
            fig=fig,
            ax=ax,
            **kwargs,
        )

    @staticmethod
    def lines(
        x,
        y,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis, list[Line2D]]":
        return Plotter._plot("plot", x, y, *args, fig=fig, ax=ax, **kwargs)

    @staticmethod
    def hist(
        x,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        bins: "int | Sequence | str" = "fd",
        density: bool = True,
        norm: bool = False,
        xerr=None,
        orientation: "Literal['vertical', 'horizontal']" = "horizontal",
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis, tuple[Sequence[int] | list[Sequence[int]], Sequence[float], BarContainer]]":
        weights = np.ones_like(x) if xerr is None else xerr
        if norm:
            weights /= len(x)
            density = False

        return Plotter._plot(
            "hist",
            x,
            *args,
            fig=fig,
            ax=ax,
            bins=bins,
            density=density,
            weights=weights,
            orientation=orientation,
            **kwargs,
        )

    @staticmethod
    def axvline(
        x,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis, Line2D]":
        return Plotter._plot("axvline", x, *args, fig=fig, ax=ax, **kwargs)

    @staticmethod
    def axhline(
        y,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis, Line2D]":
        return Plotter._plot("axhline", y, *args, fig=fig, ax=ax, **kwargs)

    @staticmethod
    def fill_between(
        x,
        low,
        high=0,
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        **kwargs: "Any",
    ) -> "tuple[Figure, Axis, FillBetweenPolyCollection]":
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
    ) -> "tuple[Figure, Axis, ErrorbarContainer]":
        return Plotter._plot(
            "errorbar",
            x,
            y,
            *args,
            fig=fig,
            ax=ax,
            xerr=xerr,
            yerr=yerr,
            **{
                "linestyle": "none",
                "linewidth": 1,
                "elinewidth": 0.5,
                "marker": "o",
                "markersize": 1,
                **kwargs,
            },
        )

    @staticmethod
    def corner(
        chains: "dict[str, pd.DataFrame]",
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

        for name, chain in chains.items():
            chain_opts = {}
            for k, v in chain_kwargs.items():
                if isinstance(v, dict):
                    if name in v:
                        chain_opts[k] = v[name]
                else:
                    chain_opts[k] = v
            c.add_chain(
                cc.Chain(
                    *chain_args,
                    *args,
                    samples=chain,
                    name=str(name),
                    **chain_opts,
                    **kwargs,
                )
            )

        fig = c.plotter.plot(*plot_args, *args, **plot_kwargs, **kwargs)
        ax = fig.gca()
        return fig, ax

    @staticmethod
    def save(
        fig: "Figure", savepath: "Path", *, clear: bool = True, scale: int = 3
    ) -> "Figure":
        fig.savefig(savepath, transparent=True, bbox_inches="tight", dpi=dpi * scale)
        if clear:
            fig, _ = Plotter.clear(fig=fig)
        return fig

    @staticmethod
    def clear(
        *, fig: "Figure | None" = None, ax: "Axis | None" = None
    ) -> "tuple[Figure | None, Axis | None]":
        if fig is not None:
            fig.clf()
        if ax is not None:
            ax.cla()
        return fig, ax

    @staticmethod
    def close(fig: "Figure | None", ax: "Axis | list[Axis] | None") -> None:
        if fig is not None:
            Plotter.clear(fig=fig)
        if ax is not None:
            axes = [ax] if not isinstance(ax, Iterable) else ax
            for a in axes:
                Plotter.clear(ax=a)
        if fig is not None:
            plt.close(fig)
