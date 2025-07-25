from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np

from .analysis import Plotter, AbstractPlot

if TYPE_CHECKING:
    from typing import Any

    from numpy import typing as npt

    from supaernova.configs.steps.data import DataStepResult

    from .analysis import Axis, Figure


class SpectraPlot(AbstractPlot):
    mask: bool = True
    filter: (
        dict[
            str,
            dict["Literal['min', 'max', 'equals', 'contains']", str | float],
        ]
        | None
    ) = None


CONSTRAINTS = {
    "min": np.greater_equal,
    "max": np.less_equal,
    "equals": np.equal,
    "contains": lambda x, y: np.strings.find(x, y) != -1,
}


class SpectraPlotter(Plotter):
    @staticmethod
    def prep(
        data: "DataStepResult",
        config: "SpectraPlot",
    ) -> tuple[
        "npt.NDArray[np.float32]",
        "npt.NDArray[np.float32]",
        "npt.NDArray[np.float32]",
        "npt.NDArray[np.int32]",
        "npt.NDArray[np.int32]",
        "npt.NDArray[np.int32]",
    ]:
        wl = data.wavelength.copy()
        amplitude = data.amplitude.copy()
        sigma = data.sigma.copy()
        wl_mask = data.mask.copy()
        if not config.mask:
            wl_mask = np.ones_like(wl_mask)

        if config.filter is not None:
            for key, constraints in config.filter.items():
                value = getattr(data, key)
                for comparison, constraint in constraints.items():
                    compare = CONSTRAINTS[comparison]
                    wl_mask *= compare(value, constraint).astype(np.int32)

        spec_mask = wl_mask.max(axis=-1, keepdims=True)
        sn_mask = spec_mask.max(axis=-2, keepdims=True)

        return wl, amplitude, sigma, sn_mask, spec_mask, wl_mask

    # TODO: per plot-type args and kwargs
    @staticmethod
    def plot_spectra(
        data: "DataStepResult",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None
        wl, amplitude, sigma, sn_mask, spec_mask, wl_mask = SpectraPlotter.prep(
            data, config
        )

        n_sn, n_spec, _n_wl = wl_mask.shape

        for sn in range(n_sn):
            if sn_mask[sn, 0, 0]:
                for spec in range(n_spec):
                    if spec_mask[sn, spec, 0]:
                        ma = wl_mask[sn, spec, :].astype(np.bool_)
                        x = wl[sn, spec, :][ma]
                        y = amplitude[sn, spec, :][ma]
                        yerr = sigma[sn, spec, :][ma]

                        order = np.argsort(x)
                        x = x[order]
                        y = y[order]
                        yerr = yerr[order]

                        fig, ax = Plotter.errorbar(
                            x, y, *args, fig=fig, ax=ax, yerr=yerr, **kwargs
                        )
        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None
        return fig, ax

    # TODO: per plot-type args and kwargs
    @staticmethod
    def plot_summary(
        data: "DataStepResult",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None
        wl, amplitude, sigma, sn_mask, spec_mask, wl_mask = SpectraPlotter.prep(
            data, config
        )

        mask = np.logical_not(sn_mask * spec_mask * wl_mask)

        x = np.ma.masked_array(wl, mask).mean(axis=(0, 1))
        y = np.ma.masked_array(amplitude, mask)
        yerr = np.ma.masked_array(sigma, mask)

        # Mean
        y_mean = y.mean(axis=(0, 1))
        y_std = y.std(axis=(0, 1))
        yerr_mean = np.sqrt(np.sum(yerr * yerr, axis=(0, 1))) / yerr.count(axis=(0, 1))

        order = np.argsort(x)
        x = x[order]
        y_mean = y_mean[order]
        y_std = y_std[order]
        yerr_mean = yerr_mean[order]

        fig, ax = Plotter.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            *args,
            fig=fig,
            ax=ax,
            alpha=0.2,
            **kwargs,
        )
        fig, ax = Plotter.errorbar(
            x,
            y_mean,
            *args,
            fig=fig,
            ax=ax,
            yerr=yerr_mean,
            **kwargs,
        )

        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None
        return fig, ax

    @staticmethod
    def plot_residual(
        data: "DataStepResult",
        amplitude_prime: "npt.NDArray[np.float32]",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None
        wl, amplitude, sigma, sn_mask, spec_mask, wl_mask = SpectraPlotter.prep(
            data, config
        )

        mask = np.logical_not(sn_mask * spec_mask * wl_mask)

        x = np.ma.masked_array(wl, mask).mean(axis=(0, 1))
        y = np.ma.masked_array(amplitude, mask)
        yerr = np.ma.masked_array(sigma, mask)
        y_prime = np.ma.masked_array(amplitude_prime, mask)

        if fig is None:
            fig = Plotter.figure(fig)
        if ax is None:
            _ = Plotter.axis(fig, 211)
            _ = Plotter.axis(fig, 212)
            ax = fig.get_axes()
        spectra_ax, residual_ax = ax

        order = np.argsort(x)
        x = x[order]

        # Mean
        y_mean = y.mean(axis=(0, 1))[order]
        y_std = y.std(axis=(0, 1))[order]
        yerr_mean = (
            np.sqrt(np.sum(yerr * yerr, axis=(0, 1))) / yerr.count(axis=(0, 1))
        )[order]

        y_prime_mean = y_prime.mean(axis=(0, 1))[order]
        y_prime_std = y_prime.std(axis=(0, 1))[order]

        y_residual = y - y_prime
        y_residual_mean = y_residual.mean(axis=(0, 1))[order]
        y_residual_std = y_residual.std(axis=(0, 1))[order]

        # y
        fig, spectra_ax = Plotter.fill_between(
            x, y_mean - y_std, y_mean + y_std, fig=fig, ax=spectra_ax, alpha=0.2
        )
        fig, spectra_ax = Plotter.errorbar(
            x, y_mean, yerr=yerr_mean, fig=fig, ax=spectra_ax
        )

        # y_prime
        fig, spectra_ax = Plotter.fill_between(
            x,
            y_prime_mean - y_prime_std,
            y_prime_mean + y_prime_std,
            fig=fig,
            ax=spectra_ax,
            alpha=0.2,
        )
        fig, spectra_ax = Plotter.scatter(x, y_prime_mean, fig=fig, ax=spectra_ax)

        # residual
        fig, residual_ax = Plotter.fill_between(
            x,
            y_residual_mean - y_residual_std,
            y_residual_mean + y_residual_std,
            fig=fig,
            ax=residual_ax,
            alpha=0.2,
        )
        fig, residual_ax = Plotter.errorbar(
            x, y_residual_mean, yerr=yerr_mean, fig=fig, ax=residual_ax
        )

        fig, residual_ax = Plotter.axhline(0, color="black", fig=fig, ax=residual_ax)

        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None
        return fig, ax
