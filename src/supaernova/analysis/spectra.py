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
    filter: (
        dict[
            str,
            dict["Literal['min', 'max', 'equals', 'contains']", str | float],
        ]
        | None
    ) = None


class ResidualPlot(SpectraPlot):
    base: str = ""
    base_wl: np.ndarray | None = None
    base_amp: np.ndarray | None = None
    base_sigma: np.ndarray | None = None
    base_mask: np.ndarray | None = None
    plot_base: bool = True


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
        *,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
    ) -> tuple[
        "npt.NDArray[float]",
        "npt.NDArray[float]",
        "npt.NDArray[float]",
        "npt.NDArray[int]",
        "npt.NDArray[int]",
        "npt.NDArray[int]",
        "npt.NDArray[int]",
    ]:
        wl = data.wavelength.copy()
        amplitude = data.amplitude.copy()
        sigma = data.sigma.copy()

        input_mask = data.mask.copy() if mask is None else mask
        # Wavelength Range Mask
        input_wl_mask = np.ones_like(input_mask) if wl_mask is None else wl_mask
        # Phase Range Mask
        input_spec_mask = (
            input_wl_mask.max(axis=-1, keepdims=True)
            if spec_mask is None
            else spec_mask
        )
        # Redshift Range Mask
        input_sn_mask = (
            input_spec_mask.max(axis=-2, keepdims=True) if sn_mask is None else sn_mask
        )

        if config.filter is not None:
            for key, constraints in config.filter.items():
                value = getattr(data, key)
                for comparison, constraint in constraints.items():
                    compare = CONSTRAINTS[comparison]
                    input_wl_mask *= compare(value, constraint).astype(np.int32)

        input_spec_mask *= input_wl_mask.max(axis=-1, keepdims=True)
        input_sn_mask *= input_spec_mask.max(axis=-2, keepdims=True)
        input_mask *= input_sn_mask * input_spec_mask * input_wl_mask

        return (
            wl,
            amplitude,
            sigma,
            input_mask,
            input_sn_mask,
            input_spec_mask,
            input_wl_mask,
        )

    @staticmethod
    def plot_spectra(
        data: "DataStepResult",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | tuple[None, None]:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None, None
        (
            wl,
            amplitude,
            sigma,
            input_mask,
            input_sn_mask,
            input_spec_mask,
            _input_wl_mask,
        ) = SpectraPlotter.prep(
            data,
            config,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
        )

        n_sn, n_spec, _n_wl = input_mask.shape

        i = 0
        for sn in range(n_sn):
            if input_sn_mask[sn, 0, 0]:
                colours = Plotter.colour_maps[i % len(Plotter.colour_maps)]
                i += 1
                fig, ax, _lines = Plotter.lines(
                    [],
                    [],
                    *args,
                    fig=fig,
                    ax=ax,
                    linestyle="-",
                    c=colours(0.5),
                    label=data.sn_name[sn, 0, 0],
                    **kwargs,
                )
                for spec in range(n_spec):
                    if input_spec_mask[sn, spec, 0]:
                        c = colours(data.time[sn, spec, 0])
                        ma = input_mask[sn, spec, :].astype(bool)
                        x = wl[sn, spec, :][ma]
                        y = amplitude[sn, spec, :][ma]
                        yerr = sigma[sn, spec, :][ma]

                        order = np.argsort(x)
                        x = x[order]
                        y = y[order]
                        yerr = yerr[order]

                        (
                            fig,
                            ax,
                            _ebar,
                        ) = Plotter.errorbar(
                            x,
                            y,
                            *args,
                            fig=fig,
                            ax=ax,
                            yerr=yerr,
                            linestyle="-",
                            c=c,
                        )

        fig, ax, cbar = Plotter.colourbar(
            fig=fig,
            ax=ax,
            cmap=colours,
        )
        cbar.set_label("Normalised Phase")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.set_title((config.plot_kwargs or {}).get("title", config.name.capitalize()))
        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None
        return fig, ax

    @staticmethod
    def plot_summary(
        data: "DataStepResult",
        config: "SpectraPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | tuple[None, None]:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None, None
        (
            wl,
            amplitude,
            sigma,
            input_mask,
            _input_sn_mask,
            _input_spec_mask,
            _input_wl_mask,
        ) = SpectraPlotter.prep(
            data,
            config,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
        )

        input_mask = np.logical_not(input_mask)

        x = np.ma.masked_array(wl, input_mask).mean(axis=(0, 1))
        y = np.ma.masked_array(amplitude, input_mask)
        yerr = np.ma.masked_array(sigma, input_mask)

        # Mean
        y_mean = y.mean(axis=(0, 1))
        y_std = y.std(axis=(0, 1))
        yerr_mean = yerr.mean(axis=(0, 1))

        order = np.argsort(x)
        x = x[order]
        y_mean = y_mean[order]
        y_std = y_std[order]
        yerr_mean = yerr_mean[order]
        fig, ax, ebar = Plotter.errorbar(
            x,
            y_mean,
            *args,
            fig=fig,
            ax=ax,
            yerr=yerr_mean,
            linestyle="-",
            **kwargs,
        )
        c = ebar.lines[0].get_color()
        fig, ax, _fill = Plotter.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            *args,
            fig=fig,
            ax=ax,
            color=c,
            alpha=0.2,
            **kwargs,
        )
        label = (config.plot_kwargs or {}).get("label")
        if label is not None:
            fig, ax, _lines = Plotter.lines(
                [],
                [],
                *args,
                fig=fig,
                ax=ax,
                label=label,
                linestyle="-",
                c=c,
                **kwargs,
            )

        y_min, y_max = ax.get_ylim()
        y_min = max(y_mean.min() * 0.9, y_min)
        y_max = min(y_mean.max() * 1.1, y_max)
        y_diff = max(abs(y_mean.min() - y_min), abs(y_mean.max() - y_max))
        y_min = y_mean.min() - y_diff
        y_max = y_mean.max() + y_diff
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Mean Amplitude")
        ax.legend()
        ax.set_title((config.plot_kwargs or {}).get("title", config.name.capitalize()))
        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None
        return fig, ax

    @staticmethod
    def plot_residual(
        data: "DataStepResult",
        config: "ResidualPlot",
        *args: "Any",
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | tuple[None, None]:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None, None
        (
            wl,
            amplitude,
            sigma,
            input_mask,
            _input_sn_mask,
            _input_spec_mask,
            _input_wl_mask,
        ) = SpectraPlotter.prep(
            data,
            config,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
        )

        input_mask = np.logical_not(input_mask)
        base_mask = config.base_mask if config.base_mask is not None else input_mask

        x = np.ma.masked_array(wl, input_mask).mean(axis=(0, 1))
        y = np.ma.masked_array(amplitude, input_mask)
        yerr = np.ma.masked_array(sigma, input_mask)
        x_prime = np.ma.masked_array(config.base_wl, base_mask).mean(axis=(0, 1))
        y_prime = np.ma.masked_array(config.base_amp, base_mask)
        yerr_prime = np.ma.masked_array(config.base_sigma, base_mask)

        if fig is None:
            fig = Plotter.figure(fig)
        if ax is None:
            spectra_ax = Plotter.axis(fig, 211)
            residual_ax = Plotter.axis(fig, 212, sharex=spectra_ax)
            spectra_ax.tick_params("x", labelbottom=False)
            fig.subplots_adjust(wspace=0, hspace=0)
            ax = [spectra_ax, residual_ax]
        else:
            spectra_ax, residual_ax = ax

        order = np.argsort(x)
        x = x[order]
        y_mean = y.mean(axis=(0, 1))[order]
        y_std = y.std(axis=(0, 1))[order]
        yerr_mean = yerr.mean(axis=(0, 1))[order]

        order_prime = np.argsort(x_prime)
        x_prime = x_prime[order_prime]
        y_prime_mean = y_prime.mean(axis=(0, 1))[order_prime]
        y_prime_std = y_prime.std(axis=(0, 1))[order_prime]
        yerr_prime_mean = yerr_prime.mean(axis=(0, 1))[order_prime]

        if config.plot_base:
            fig, spectra_ax, _ebar = Plotter.errorbar(
                x_prime,
                y_prime_mean,
                *args,
                fig=fig,
                ax=spectra_ax,
                yerr=yerr_prime_mean,
                linestyle="-",
                color="black",
                **kwargs,
            )
            fig, spectra_ax, _fill = Plotter.fill_between(
                x_prime,
                y_prime_mean - y_prime_std,
                y_prime_mean + y_prime_std,
                *args,
                fig=fig,
                ax=spectra_ax,
                color="black",
                alpha=0.2,
                **kwargs,
            )
            base_label = (config.plot_kwargs or {}).get("base_label", "Base")
            if base_label is not None:
                fig, spectra_ax, _lines = Plotter.lines(
                    [],
                    [],
                    *args,
                    fig=fig,
                    ax=spectra_ax,
                    label=base_label,
                    linestyle="-",
                    color="black",
                    **kwargs,
                )

        fig, spectra_ax, ebar = Plotter.errorbar(
            x,
            y_mean,
            *args,
            fig=fig,
            ax=spectra_ax,
            yerr=yerr_mean,
            linestyle="-",
            **kwargs,
        )
        c = ebar.lines[0].get_color()
        fig, spectra_ax, _fill = Plotter.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            *args,
            fig=fig,
            ax=spectra_ax,
            color=c,
            alpha=0.2,
            **kwargs,
        )
        label = (config.plot_kwargs or {}).get("label")
        if label is not None:
            fig, spectra_ax, _lines = Plotter.lines(
                [],
                [],
                *args,
                fig=fig,
                ax=spectra_ax,
                label=label,
                linestyle="-",
                c=c,
                **kwargs,
            )

        y_min, y_max = spectra_ax.get_ylim()
        y_min = max(y_mean.min() * 0.9, y_min)
        y_max = min(y_mean.max() * 1.1, y_max)
        y_diff = max(abs(y_mean.min() - y_min), abs(y_mean.max() - y_max))
        y_min = y_mean.min() - y_diff
        y_max = y_mean.max() + y_diff
        spectra_ax.set_ylim(y_min, y_max)
        spectra_ax.set_ylabel("Mean Amplitude")

        fig, residual_ax, _hline = Plotter.axhline(
            0, color="black", fig=fig, ax=residual_ax
        )

        y_residual = y - y_prime
        y_residual_mean = y_residual.mean(axis=(0, 1))[order_prime]

        fig, residual_ax, ebar = Plotter.errorbar(
            x_prime,
            y_residual_mean,
            *args,
            fig=fig,
            ax=residual_ax,
            # yerr=yerr_residual_mean,
            linestyle="-",
            **kwargs,
        )

        y_min, y_max = residual_ax.get_ylim()
        y_min = min(y_residual_mean.min() * 0.9, y_min)
        y_max = max(y_residual_mean.max() * 1.1, y_max)
        y_diff = max(
            abs(y_residual_mean.min() - y_min), abs(y_residual_mean.max() - y_max)
        )
        y_min = y_residual_mean.min() - y_diff
        y_max = y_residual_mean.max() + y_diff
        residual_ax.set_ylim(y_min, y_max)
        residual_ax.set_xlabel("Wavelength [Å]")
        residual_ax.set_ylabel("Mean Residual")

        spectra_ax.set_title(
            (config.plot_kwargs or {}).get("title", config.name.capitalize())
        )
        spectra_ax.legend()

        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None
        return fig, ax
