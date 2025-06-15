from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np

from .analysis import Plotter, AbstractPlot

if TYPE_CHECKING:
    from typing import Any

    from numpy import typing as npt

    from suPAErnova.configs.steps.data import DataStepResult

    from .analysis import Axis, Figure


class SpectraPlot(AbstractPlot):
    filter: (
        dict[str, dict["Literal['min', 'max', 'equals', 'contains']", str | float]]
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
        "npt.NDArray[np.bool]",
    ]:
        wl = data.wavelength.copy()
        amplitude = data.amplitude.copy()
        sigma = data.sigma.copy()
        mask = data.mask.copy()

        if config.filter is not None:
            for key, constraints in config.filter.items():
                value = getattr(data, key)
                for comparison, constraint in constraints.items():
                    compare = CONSTRAINTS[comparison]
                    mask *= compare(value, constraint).astype(np.int32)
        mask = mask.astype(np.bool)

        _n_sn, _n_spec, _n_wl = mask.shape

        return wl, amplitude, sigma, mask

    # TODO: per plot-type args and kwargs
    @staticmethod
    def plot_spectra(
        data: "DataStepResult",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        **kwargs: "Any",
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return
        wl, amplitude, sigma, mask = SpectraPlotter.prep(data, config)

        n_sn, n_spec, _n_wl = mask.shape

        for sn in range(n_sn):
            if np.any(mask[sn, :, :]):
                for spec in range(n_spec):
                    if np.any(mask[sn, spec, :]):
                        ma = mask[sn, spec, :]
                        x = wl[sn, spec, :][ma]
                        y = amplitude[sn, spec, :][ma]
                        yerr = sigma[sn, spec, :][ma]
                        fig, ax = Plotter.scatter(x, y, *args, fig=fig, ax=ax, **kwargs)
                        fig, ax = Plotter.fill_between(
                            x,
                            y - yerr,
                            y + yerr,
                            *args,
                            fig=fig,
                            ax=ax,
                            alpha=0.2,
                            **kwargs,
                        )
        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)

    # TODO: per plot-type args and kwargs
    @staticmethod
    def plot_summary(
        data: "DataStepResult",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        **kwargs: "Any",
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return
        wl, amplitude, sigma, mask = SpectraPlotter.prep(data, config)
        mask = np.logical_not(mask)

        _n_sn, _n_spec, _n_wl = mask.shape

        x = np.ma.masked_array(wl, mask)[0, 0, :]
        y = np.ma.masked_array(amplitude, mask)
        yerr = np.ma.masked_array(sigma, mask)

        # Mean
        y_mean = y.mean(axis=(0, 1))
        y_std = y.std(axis=(0, 1))
        yerr_mean = yerr.mean(axis=(0, 1))
        fig, ax = Plotter.scatter(x, y_mean, *args, fig=fig, ax=ax, **kwargs)
        fig, ax = Plotter.fill_between(
            x,
            y_mean - yerr_mean,
            y_mean + yerr_mean,
            *args,
            fig=fig,
            ax=ax,
            alpha=0.2,
            **kwargs,
        )
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

        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)

    @staticmethod
    def plot_residual(
        data: "DataStepResult",
        amplitude_prime: "npt.NDArray[np.float32]",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        **kwargs: "Any",
    ) -> None:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return
        wl, amplitude, sigma, mask = SpectraPlotter.prep(data, config)
        mask = np.logical_not(mask)

        _n_sn, _n_spec, _n_wl = mask.shape

        x = np.ma.masked_array(wl, mask)[0, 0, :]
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

        # Mean
        y_mean = y.mean(axis=(0, 1))
        y_std = y.std(axis=(0, 1))
        yerr_mean = yerr.mean(axis=(0, 1))

        y_prime_mean = y_prime.mean(axis=(0, 1))
        y_prime_std = y_prime.std(axis=(0, 1))

        y_residual = y - y_prime
        y_residual_mean = y_residual.mean(axis=(0, 1))
        y_residual_std = y_residual.std(axis=(0, 1))

        # y
        fig, spectra_ax = Plotter.scatter(x, y_mean, fig=fig, ax=spectra_ax)
        fig, spectra_ax = Plotter.fill_between(
            x, y_mean - yerr_mean, y_mean + yerr_mean, fig=fig, ax=spectra_ax, alpha=0.2
        )
        fig, spectra_ax = Plotter.fill_between(
            x, y_mean - y_std, y_mean + y_std, fig=fig, ax=spectra_ax, alpha=0.2
        )

        # y_prime
        fig, spectra_ax = Plotter.scatter(x, y_prime_mean, fig=fig, ax=spectra_ax)
        fig, spectra_ax = Plotter.fill_between(
            x,
            y_prime_mean - yerr_mean,
            y_prime_mean + yerr_mean,
            fig=fig,
            ax=spectra_ax,
            alpha=0.2,
        )
        fig, spectra_ax = Plotter.fill_between(
            x,
            y_prime_mean - y_prime_std,
            y_prime_mean + y_prime_std,
            fig=fig,
            ax=spectra_ax,
            alpha=0.2,
        )

        # residual
        fig, residual_ax = Plotter.scatter(x, y_residual_mean, fig=fig, ax=residual_ax)
        fig, residual_ax = Plotter.fill_between(
            x,
            y_residual_mean - yerr_mean,
            y_residual_mean + yerr_mean,
            fig=fig,
            ax=residual_ax,
            alpha=0.2,
        )
        fig, residual_ax = Plotter.fill_between(
            x,
            y_residual_mean - y_residual_std,
            y_residual_mean + y_residual_std,
            fig=fig,
            ax=residual_ax,
            alpha=0.2,
        )

        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, ax)
