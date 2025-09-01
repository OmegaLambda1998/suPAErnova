from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np
from numpy import typing as npt  # noqa: TC002

from .analysis import Plotter, AbstractPlot

if TYPE_CHECKING:
    from typing import Any

    from supaernova.configs.steps.data import LazySNPAEData

    from .analysis import Axis, Figure


class SpectraPlot(AbstractPlot):
    filter: (
        dict[
            str,
            dict["Literal['min', 'max', 'equals', 'contains']", str | float],
        ]
        | None
    ) = None


class ComparisonPlot(SpectraPlot):
    base: str = ""
    base_wl: "npt.NDArray[int] | None" = None
    base_amp: "npt.NDArray[int] | None" = None
    base_sigma: "npt.NDArray[int] | None" = None
    base_mask: "npt.NDArray[int] | None" = None
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
        data: "LazySNPAEData",
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
        "npt.NDArray[str]",
        "npt.NDArray[float]",
    ]:
        wl = data.wavelength.copy()
        amplitude = data.amplitude.copy()
        sigma = data.sigma.copy()
        sn_name = data.sn_name.copy()
        time = data.time.copy()

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

        data.clear()

        return (
            wl,
            amplitude,
            sigma,
            sn_name,
            time,
            input_mask,
            input_sn_mask,
            input_spec_mask,
            input_wl_mask,
        )

    @staticmethod
    def plot_spectra(
        data: "LazySNPAEData",
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
            sn_name,
            time,
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
                    label=sn_name[sn, 0, 0],
                    **kwargs,
                )
                for spec in range(n_spec):
                    if input_spec_mask[sn, spec, 0]:
                        c = colours(time[sn, spec, 0])
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
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        ax.set_title((config.plot_kwargs or {}).get("title", config.name.capitalize()))
        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None
        return fig, ax

    @staticmethod
    def plot_summary(
        data: "LazySNPAEData",
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
            _sn_name,
            _time,
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
        yerr_mean = np.sqrt(
            (yerr * yerr).mean(axis=(0, 1)) / np.ones_like(yerr).sum(axis=(0, 1))
        )

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

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Amplitude")
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        ax.set_title((config.plot_kwargs or {}).get("title", config.name.capitalize()))
        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None
        return fig, ax

    @staticmethod
    def plot_comparison(
        data: "LazySNPAEData",
        config: "ComparisonPlot",
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
            _sn_name,
            _time,
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
            spectra_ax = Plotter.axis(fig, 311)
            residual_ax = Plotter.axis(fig, 312, sharex=spectra_ax)
            pull_ax = Plotter.axis(fig, 313, sharex=residual_ax)
            spectra_ax.tick_params("x", labelbottom=False)
            residual_ax.tick_params("x", labelbottom=False)
            fig.subplots_adjust(wspace=0, hspace=0)
            ax = [spectra_ax, residual_ax, pull_ax]
        else:
            spectra_ax, residual_ax, pull_ax = ax

        order = np.argsort(x)
        x = x[order]
        y = y[..., order]
        yerr = yerr[..., order]
        y_mean = y.mean(axis=(0, 1))
        y_std = y.std(axis=(0, 1))
        yerr_mean = np.sqrt(
            (yerr * yerr).mean(axis=(0, 1)) / np.ones_like(yerr).sum(axis=(0, 1))
        )

        order_prime = np.argsort(x_prime)
        x_prime = x_prime[order_prime]
        y_prime = y_prime[..., order_prime]
        yerr_prime = yerr_prime[..., order_prime]
        y_prime_mean = y_prime.mean(axis=(0, 1))
        y_prime_std = y_prime.std(axis=(0, 1))
        yerr_prime_mean = np.sqrt(
            (yerr_prime * yerr_prime).mean(axis=(0, 1))
            / np.ones_like(yerr_prime).sum(axis=(0, 1))
        )

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

        spectra_ax.set_ylabel("Amplitude")

        fig, residual_ax, _hline = Plotter.axhline(
            0, color="black", fig=fig, ax=residual_ax
        )

        # Restrict to overlap in x
        x_min = max(x.min(), x_prime.min())
        x_max = min(x.max(), x_prime.max())

        mask_overlap = (x >= x_min - 1) & (x <= x_max + 1)
        mask_overlap_prime = (x_prime >= x_min - 1) & (x_prime <= x_max + 1)

        # Extract overlapping regions
        x_common = x[mask_overlap]
        y_common = y[..., mask_overlap]
        yerr_common = yerr[..., mask_overlap]
        y_common_mean = y_common.mean(axis=(0, 1))
        yerr_common_mean = np.sqrt(
            (yerr_common * yerr_common).mean(axis=(0, 1))
            / np.ones_like(yerr_common).sum(axis=(0, 1))
        )

        x_prime_common = x_prime[mask_overlap_prime]
        y_prime_common = y_prime[..., mask_overlap_prime]
        yerr_prime_common = yerr_prime[..., mask_overlap_prime]
        y_prime_common_mean = y_prime_common.mean(axis=(0, 1))
        yerr_prime_common_mean = yerr_prime_common.mean(axis=(0, 1))
        yerr_prime_common_mean = np.sqrt(
            (yerr_prime_common * yerr_prime_common).mean(axis=(0, 1))
            / np.ones_like(yerr_prime_common).sum(axis=(0, 1))
        )

        # Residual with masks respected
        # y_residual = y_common - y_prime_common
        # y_residual_mean = y_residual.mean(axis=(0, 1))
        y_residual_mean = y_common_mean - y_prime_common_mean
        yerr_residual_mean = np.sqrt(
            yerr_common_mean * yerr_common_mean
            + yerr_prime_common_mean * yerr_prime_common_mean
        )

        fig, residual_ax, ebar = Plotter.errorbar(
            x_common,
            y_residual_mean,
            yerr=yerr_residual_mean,
            *args,
            fig=fig,
            ax=residual_ax,
            linestyle="-",
            **kwargs,
        )

        residual_ax.set_ylabel("Residual")

        # fig, pull_ax, _hline = Plotter.axhline(0, color="black", fig=fig, ax=pull_ax)

        y_pull_mean = np.abs(y_residual_mean) / yerr_residual_mean

        fig, pull_ax, ebar = Plotter.errorbar(
            x_common,
            y_pull_mean,
            *args,
            fig=fig,
            ax=pull_ax,
            linestyle="-",
            **kwargs,
        )

        pull_ax.set_xlabel("Wavelength [Å]")
        pull_ax.set_ylabel("Abs Pull")
        if y_pull_mean.min() != y_pull_mean.max():
            pull_ax.set_yscale("symlog", linthresh=1e-3)

        spectra_ax.set_title(
            (config.plot_kwargs or {}).get("title", config.name.capitalize())
        )
        spectra_ax.legend(bbox_to_anchor=(1.0, 1.0))

        fig.align_ylabels(ax)

        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None
        return fig, ax
