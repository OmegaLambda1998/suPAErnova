from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np
from numpy import typing as npt  # noqa: TC002
from pydantic import PositiveInt  # noqa: TC002
from matplotlib.colors import to_rgba
from matplotlib.transforms import blended_transform_factory

from supaernova.utils import pp

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
    stack: bool = False


class ComparisonPlot(SpectraPlot):
    base: str = ""
    base_wl: "npt.NDArray[int] | None" = None
    base_amp: "npt.NDArray[int] | None" = None
    base_sigma: "npt.NDArray[int] | None" = None
    base_mask: "npt.NDArray[int] | None" = None
    reduce: Literal["mean", "median", "max_central"] = "max_central"
    plot_base: bool = True


class ComparisonArrayPlot(ComparisonPlot):
    plot_best: bool = True
    plot_worst: bool = True
    plot_mean: bool = True
    plot_max_delta_m: bool = True
    plot_min_delta_m: bool = True
    plot_median: bool = True
    plot_max_delta_p: bool = True
    plot_min_delta_p: bool = True
    plot_names: list[str] | None = None
    plot_random: PositiveInt | Literal["auto", "none"] = "auto"


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
        phase: bool = False,
    ) -> tuple[
        "npt.NDArray[float]",
        "npt.NDArray[float]",
        "npt.NDArray[float]",
        "npt.NDArray[str]",
        "npt.NDArray[float]",
        "npt.NDArray[bool]",
        "npt.NDArray[bool]",
        "npt.NDArray[bool]",
        "npt.NDArray[bool]",
    ]:
        wl = data.wavelength.copy()
        amplitude = data.amplitude.copy()
        sigma = data.sigma.copy()
        sn_name = data.sn_name.copy()
        time = data.phase.copy() if phase else data.time.copy()
        input_mask = (
            np.ones_like(data.mask, dtype=np.bool) if mask is None else mask.copy()
        )

        # Wavelength Range Mask
        input_wl_mask = np.ones_like(input_mask) if wl_mask is None else wl_mask.copy()

        if config.filter is not None:
            for key, constraints in config.filter.items():
                value = getattr(data, key)
                for comparison, constraint in constraints.items():
                    compare = CONSTRAINTS[comparison]
                    input_wl_mask &= compare(value, constraint)

        data.clear()

        # Phase Range Mask
        input_spec_mask = (
            input_wl_mask.any(axis=-1, keepdims=True)
            if spec_mask is None
            else spec_mask.copy()
        )
        # Redshift Range Mask
        input_sn_mask = (
            input_spec_mask.any(axis=-2, keepdims=True)
            if sn_mask is None
            else sn_mask.copy()
        )

        input_spec_mask &= input_wl_mask.any(axis=-1, keepdims=True)
        input_sn_mask &= input_spec_mask.any(axis=-2, keepdims=True)
        input_mask &= input_sn_mask & input_spec_mask & input_wl_mask

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
        mask: "npt.NDArray[bool] | None" = None,
        sn_mask: "npt.NDArray[bool] | None" = None,
        spec_mask: "npt.NDArray[bool] | None" = None,
        wl_mask: "npt.NDArray[bool] | None" = None,
        decorate: bool = True,
        offset: int = 0,
        phase: bool = True,
        stack: bool = False,
        shift: float = 0.0,
        shift_min: float | None = None,
        shift_max: float | None = None,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | tuple[None, None]:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            Plotter.close(fig, ax)
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
            phase=True,
        )

        if stack:
            stacked_x = None
            stacked_y = None
            stacked_yerr = None

        if shift_max is None:
            shift_max = time[np.isfinite(time)].max()
        if shift_min is None:
            shift_min = time[np.isfinite(time)].min()

        def phase_shift(phase):
            return shift * (1 - ((phase - shift_min) / (shift_max - shift_min)))

        n_sn, n_spec, _n_wl = input_mask.shape

        y_max = -np.inf
        y_min = np.inf

        i = 0 + offset
        for sn in range(n_sn):
            t_last = -np.inf
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
                    label=sn_name[sn, 0, 0]
                    if decorate
                    else None,  # + config.plot_kwargs.get("label", ""),
                    **kwargs,
                )
                for spec in range(n_spec):
                    if input_spec_mask[sn, spec, 0]:
                        c = colours(0.5)
                        ma = input_mask[sn, spec, :].astype(bool)
                        if np.count_nonzero(ma) != np.count_nonzero(
                            input_mask[sn, 0, :]
                        ):
                            continue
                        x = wl[sn, spec, :][ma]
                        y = amplitude[sn, spec, :][ma] + phase_shift(time[sn, spec, 0])
                        yerr = sigma[sn, spec, :][ma]
                        order = np.argsort(x)
                        x = x[order]
                        y = y[order]
                        yerr = yerr[order]
                        if y.size > 0:
                            y_max = max(
                                y_max, np.abs(y + yerr).max(), np.abs(y - yerr).max()
                            )
                            y_min = min(
                                y_min, np.abs(y + yerr).min(), np.abs(y - yerr).min()
                            )
                            if stack:
                                if stacked_x is None:
                                    stacked_x = x
                                if stacked_y is None:
                                    stacked_y = y - phase_shift(time[sn, spec, 0])
                                else:
                                    stacked_y += y - phase_shift(time[sn, spec, 0])
                                if stacked_yerr is None:
                                    stacked_yerr = yerr
                                else:
                                    stacked_yerr = np.sqrt(
                                        stacked_yerr * stacked_yerr + yerr * yerr
                                    )
                                continue

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
                                yerr=None if save else yerr,
                                linestyle="-",
                                marker="",
                                c=c,
                            )
                            # Last point
                            x_last, y_last = x[-1], y[-1]

                            # Blend: x in axes coords, y in data coords
                            trans = blended_transform_factory(
                                ax.transAxes, ax.transData
                            )

                            t = np.round(time[sn, spec, 0], decimals=1)

                            if phase and np.abs(t - t_last) > 0.5:
                                ax.text(
                                    1.02,
                                    y_last,
                                    f"{time[sn, spec, 0]:+.2f} Days",  # 1.02 puts it just outside right spine
                                    transform=trans,
                                    va="center",
                                )
                            t_last = t
                if stack:
                    (
                        fig,
                        ax,
                        _ebar,
                    ) = Plotter.errorbar(
                        stacked_x,
                        stacked_y,
                        *args,
                        fig=fig,
                        ax=ax,
                        yerr=None if save else stacked_yerr,
                        linestyle="-",
                        marker="",
                        c=c,
                    )
                    y_min = stacked_y.min()
                    y_max = stacked_y.max()

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Amplitude + offset")
        ax.set_title((config.plot_kwargs or {}).get("title", config.name.capitalize()))
        symlog_max = 10 + shift
        symlog_scale = 2
        if y_max > symlog_max:
            ax.set_yscale("symlog", linthresh=symlog_max, linscale=symlog_scale)
        if save:
            ax.set_ylim(0.95 * y_min, 1.05 * y_max)
        leg = ax.legend(bbox_to_anchor=(1.0, 1.0))
        leg.set_in_layout(False)
        # Trigger a draw so that constrained layout is executed once
        # before we turn it off when printing....
        fig.canvas.draw()
        # We want the legend included in the bbox_inches='tight' calcs.
        leg.set_in_layout(True)
        # We don't want the layout to change at this point.
        fig.set_layout_engine("none")
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
            Plotter.close(fig, ax)
            return None, None
        (
            wl,
            amplitude,
            sigma,
            _sn_name,
            _time,
            input_mask,
            input_sn_mask,
            input_spec_mask,
            input_wl_mask,
        ) = SpectraPlotter.prep(
            data,
            config,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
        )

        # ~(~input_mask & input_wl_mask)
        # Extracts unmasked wavelengths from the valid wavelength range provided by wl_mask
        valid_wl_mask = np.logical_not(
            np.logical_and(np.logical_not(input_mask), input_wl_mask)
        )

        # Determine which spectra to keep
        # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
        mask_spec = np.logical_and(
            np.any(valid_wl_mask, axis=-1, keepdims=True), input_spec_mask
        )

        # Determine which SNe to keep
        # Will mask out any SN with *no* unmasked spectra
        mask_sn = np.logical_and(
            np.any(mask_spec, axis=-2, keepdims=True), input_sn_mask
        )

        summary_mask = np.logical_not(input_mask & mask_spec & mask_sn)
        x = np.ma.masked_array(wl, summary_mask).mean(axis=(0, 1))
        y = np.ma.masked_array(amplitude, summary_mask)
        yerr = np.ma.masked_array(sigma, summary_mask)

        # Mean
        scale = (~y.mask).sum(axis=(0, 1))
        y_mean = y.sum(axis=(0, 1)) / scale
        yerr_mean = np.ma.sqrt((yerr * yerr).sum(axis=(0, 1))) / scale
        y_var = ((y - y_mean) ** 2).sum(axis=(0, 1)) / scale
        y_std = np.sqrt(y_var)
        y_sem = np.sqrt(y_var / scale)

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
            zorder=10,
            **kwargs,
        )
        c = ebar.lines[0].get_color()
        fig, ax, _fill = Plotter.fill_between(
            x,
            y_mean - y_sem,
            y_mean + y_sem,
            *args,
            fig=fig,
            ax=ax,
            facecolor=to_rgba(c, alpha=0.5),
            edgecolor="none",
            zorder=5,
            **kwargs,
        )
        fig, ax, _fill = Plotter.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            *args,
            fig=fig,
            ax=ax,
            facecolor=to_rgba(c, alpha=0.25),
            edgecolor=to_rgba(c, alpha=0.75),
            zorder=1,
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
        ax.set_title((config.plot_kwargs or {}).get("title", config.name.capitalize()))
        symlog_max = 10
        symlog_scale = 2
        if (
            np.abs(y_mean + y_std).max() > symlog_max
            or np.abs(y_mean - y_std).max() > symlog_max
        ):
            ax.set_yscale("symlog", linthresh=symlog_max, linscale=symlog_scale)
        leg = ax.legend(bbox_to_anchor=(1.0, 1.0))
        leg.set_in_layout(False)
        # Trigger a draw so that constrained layout is executed once
        # before we turn it off when printing....
        fig.canvas.draw()
        # We want the legend included in the bbox_inches='tight' calcs.
        leg.set_in_layout(True)
        # We don't want the layout to change at this point.
        fig.set_layout_engine("none")
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
        decorate: bool = True,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
        **kwargs: "Any",
    ) -> tuple["Figure", "Axis"] | tuple[None, None]:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            Plotter.close(fig, ax)
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
            spectra_ax = Plotter.axis(
                fig,
                311,
            )
            residual_ax = Plotter.axis(
                fig,
                312,
                sharex=spectra_ax,
            )
            pull_ax = Plotter.axis(
                fig,
                313,
                sharex=residual_ax,
            )
            spectra_ax.tick_params("x", labelbottom=False, bottom=False)
            residual_ax.tick_params("x", labelbottom=False, bottom=False)
            fig.get_layout_engine().set(wspace=0, hspace=0, w_pad=0 / 72, h_pad=0 / 72)
            ax = [spectra_ax, residual_ax, pull_ax]
        else:
            spectra_ax, residual_ax, pull_ax = ax

        order = np.argsort(x)
        x = x[order]
        y = y[..., order]
        yerr = yerr[..., order]

        # Mean
        scale = (~y.mask).sum(axis=(0, 1))
        y_mean = y.sum(axis=(0, 1)) / scale
        yerr_mean = np.ma.sqrt((yerr * yerr).sum(axis=(0, 1))) / scale
        y_var = ((y - y_mean) ** 2).sum(axis=(0, 1)) / scale
        y_std = np.sqrt(y_var)
        y_sem = np.sqrt(y_var / scale)

        order_prime = np.argsort(x_prime)
        x_prime = x_prime[order_prime]
        y_prime = y_prime[..., order_prime]
        yerr_prime = yerr_prime[..., order_prime]

        # Weighted Mean
        scale_prime = (~y_prime.mask).sum(axis=(0, 1))
        y_prime_mean = y_prime.sum(axis=(0, 1)) / scale_prime
        yerr_prime_mean = (
            np.ma.sqrt((yerr_prime * yerr_prime).sum(axis=(0, 1))) / scale_prime
        )
        y_prime_var = ((y_prime - y_prime_mean) ** 2).sum(axis=(0, 1)) / scale_prime
        y_prime_std = np.sqrt(y_prime_var)
        y_prime_sem = np.sqrt(y_prime_var / scale_prime)

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
                zorder=10,
                **kwargs,
            )
            fig, spectra_ax, _fill = Plotter.fill_between(
                x_prime,
                y_prime_mean - y_prime_sem,
                y_prime_mean + y_prime_sem,
                *args,
                fig=fig,
                ax=spectra_ax,
                facecolor=to_rgba("black", alpha=0.5),
                edgecolor="none",
                zorder=5,
                **kwargs,
            )
            fig, spectra_ax, _fill = Plotter.fill_between(
                x_prime,
                y_prime_mean - y_prime_std,
                y_prime_mean + y_prime_std,
                *args,
                fig=fig,
                ax=spectra_ax,
                facecolor=to_rgba("black", alpha=0.25),
                edgecolor=to_rgba("black", alpha=0.75),
                zorder=1,
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
            zorder=10,
            **kwargs,
        )
        c = ebar.lines[0].get_color()
        fig, spectra_ax, _fill = Plotter.fill_between(
            x,
            y_mean - y_sem,
            y_mean + y_sem,
            *args,
            fig=fig,
            ax=spectra_ax,
            facecolor=to_rgba(c, alpha=0.5),
            edgecolor="none",
            zorder=5,
            **kwargs,
        )
        fig, spectra_ax, _fill = Plotter.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            *args,
            fig=fig,
            ax=spectra_ax,
            facecolor=to_rgba(c, alpha=0.25),
            edgecolor=to_rgba(c, alpha=0.75),
            zorder=1,
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
        scale_common = (~y_common.mask).sum(axis=(0, 1))
        y_common_mean = y_common.sum(axis=(0, 1)) / scale_common
        yerr_common_mean = (
            np.ma.sqrt((yerr_common * yerr_common).sum(axis=(0, 1))) / scale_common
        )
        y_common_var = ((y_common - y_common_mean) ** 2).sum(axis=(0, 1)) / scale_common
        y_common_std = np.sqrt(y_common_var)
        y_common_sem = np.sqrt(y_common_var / scale_common)

        x_prime_common = x_prime[mask_overlap_prime]
        y_prime_common = y_prime[..., mask_overlap_prime]
        yerr_prime_common = yerr_prime[..., mask_overlap_prime]

        scale_prime_common = (~y_prime_common.mask).sum(axis=(0, 1))
        y_prime_common_mean = y_prime_common.sum(axis=(0, 1)) / scale_prime_common
        yerr_prime_common_mean = (
            np.ma.sqrt((yerr_prime_common * yerr_prime_common).sum(axis=(0, 1)))
            / scale_prime_common
        )
        y_prime_common_var = ((y_prime_common - y_prime_common_mean) ** 2).sum(
            axis=(0, 1)
        ) / scale_prime_common
        y_prime_common_std = np.sqrt(y_prime_common_var)
        y_prime_common_sem = np.sqrt(y_prime_common_var / scale_prime_common)

        # Residual with masks respected
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

        symlog_max = 10
        symlog_scale = 2
        if (
            np.abs(y_residual_mean + yerr_residual_mean).max() > symlog_max
            or np.abs(y_residual_mean - yerr_residual_mean).max() > symlog_max
        ):
            residual_ax.set_yscale(
                "symlog", linthresh=symlog_max, linscale=symlog_scale
            )

        fig, pull_ax, _hline = Plotter.axhline(1, color="black", fig=fig, ax=pull_ax)

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

        symlog_max = 10
        symlog_scale = 2
        if y_pull_mean.max() > symlog_max:
            pull_ax.set_yscale("symlog", linthresh=symlog_max, linscale=symlog_scale)

        symlog_max = 10
        symlog_scale = 2
        if (
            np.abs(y_mean + yerr_mean).max() > symlog_max
            or np.abs(y_mean - yerr_mean).max() > symlog_max
            or np.abs(y_mean + y_std).max() > symlog_max
            or np.abs(y_mean - y_std).max() > symlog_max
            or np.abs(y_prime_mean + yerr_prime_mean).max() > symlog_max
            or np.abs(y_prime_mean - yerr_prime_mean).max() > symlog_max
            or np.abs(y_prime_mean + y_prime_std).max() > symlog_max
            or np.abs(y_prime_mean - y_prime_std).max() > symlog_max
        ):
            spectra_ax.set_yscale("symlog", linthresh=symlog_max, linscale=symlog_scale)

        fig.align_ylabels(ax)

        if decorate:
            spectra_ax.set_ylabel("Amplitude")
            residual_ax.set_ylabel("Residual")
            pull_ax.set_xlabel("Wavelength [Å]")
            pull_ax.set_ylabel("Abs Pull")
            spectra_ax.set_title(
                (config.plot_kwargs or {}).get("title", config.name.capitalize())
            )
            leg = spectra_ax.legend(bbox_to_anchor=(1.0, 1.0))
            leg.set_in_layout(False)
            # Trigger a draw so that constrained layout is executed once
            # before we turn it off when printing....
            fig.canvas.draw()
            # We want the legend included in the bbox_inches='tight' calcs.
            leg.set_in_layout(True)
            # We don't want the layout to change at this point.
            fig.set_layout_engine("none")

        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None
        return fig, ax
