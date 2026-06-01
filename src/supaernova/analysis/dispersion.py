from typing import TYPE_CHECKING, Any, Literal
from pathlib import Path

import numpy as np

from supaernova.utils import pp, max_central, jackknife_resample

from .spectra import SpectraPlot, SpectraPlotter
from .analysis import Plotter, scale_lightness

if TYPE_CHECKING:
    from numpy import typing as npt
    import pandas as pd

    from supaernova.configs.steps.data import LazySNPAEData
    from supaernova.configs.steps.posterior import PosteriorStepResult

    from .analysis import Axis, Figure


class DispersionPlot(SpectraPlot):
    subset: Literal["train", "test"]
    legacy: tuple[Path, ...] | None = None
    twins: str | None = None
    reduce: Literal["mean", "median", "max_central"] = "max_central"


class DispersionPlotter(Plotter):
    @staticmethod
    def plot_dispersion(
        data: "LazySNPAEData",
        hmcs: "list[PosteriorStepResult]",
        config: "DispersionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "tuple[tuple[Axis | None, Axis | None, Axis | None, Axis | None], tuple[Axis | None, Axis | None, Axis | None, Axis | None], tuple[Axis, Axis, Axis, Axis]] | None" = None,
        twins: "pd.DataFrame | None" = None,
        legacy: "dict[str, npt.NDArray[Any]] | None" = None,
        force: bool = False,
        mask: "npt.NDArray[float] | None" = None,
        sn_mask: "npt.NDArray[float] | None" = None,
        spec_mask: "npt.NDArray[float] | None" = None,
        wl_mask: "npt.NDArray[float] | None" = None,
    ) -> (
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float] | None,
            tuple[float, float, float, float] | None,
            tuple[float, float, float, float] | None,
            tuple[float, float, float, float],
        ]
        | None
    ):
        do_twins = twins is not None
        do_legacy = legacy is not None

        n_rows = 2
        n_cols = 2 * (1 + (1 if do_twins else 0) + (1 if do_legacy else 0))

        handles = []

        if fig is None:
            fig = Plotter.figure()
        if ax is None:
            ind = 1

            if do_twins:
                twins_spectra_ax = Plotter.axis(fig, n_rows, n_cols, ind)
                twins_spectra_ax.tick_params("x", labelbottom=False, bottom=False)
                ind += 1

                twins_spectra_hist_ax = Plotter.axis(
                    fig,
                    n_rows,
                    n_cols,
                    ind,
                    sharey=twins_spectra_ax,
                )
                twins_spectra_hist_ax.tick_params("x", labelbottom=False, bottom=False)
                twins_spectra_hist_ax.tick_params("y", labelleft=False, left=False)
                ind += 1
            else:
                twins_spectra_ax = None
                twins_spectra_hist_ax = None

            if do_legacy:
                legacy_spectra_ax = Plotter.axis(
                    fig,
                    n_rows,
                    n_cols,
                    ind,
                    sharey=twins_spectra_hist_ax if do_twins else None,
                )
                legacy_spectra_ax.tick_params(
                    "x",
                    labelbottom=False,
                    bottom=False,
                )
                legacy_spectra_ax.tick_params(
                    "y",
                    labelleft=not do_twins,
                    left=not do_twins,
                )
                ind += 1

                legacy_spectra_hist_ax = Plotter.axis(
                    fig,
                    n_rows,
                    n_cols,
                    ind,
                    sharey=legacy_spectra_ax,
                )
                legacy_spectra_hist_ax.tick_params("x", labelbottom=False, bottom=False)
                legacy_spectra_hist_ax.tick_params("y", labelleft=False, left=False)
                ind += 1
            else:
                legacy_spectra_ax = None
                legacy_spectra_hist_ax = None

            spectra_ax = Plotter.axis(
                fig,
                n_rows,
                n_cols,
                ind,
                sharey=legacy_spectra_hist_ax
                if do_legacy
                else twins_spectra_hist_ax
                if do_twins
                else None,
            )
            spectra_ax.tick_params(
                "x",
                labelbottom=False,
                bottom=False,
            )
            spectra_ax.tick_params(
                "y",
                labelleft=not (do_twins or do_legacy),
                left=not (do_twins or do_legacy),
            )
            ind += 1

            spectra_hist_ax = Plotter.axis(
                fig,
                n_rows,
                n_cols,
                ind,
                sharey=spectra_ax,
            )
            spectra_hist_ax.tick_params("x", labelbottom=False, bottom=False)
            spectra_hist_ax.tick_params("y", labelleft=False, left=False)
            ind += 1

            if do_twins:
                twins_pull_ax = Plotter.axis(
                    fig,
                    n_rows,
                    n_cols,
                    ind,
                    sharex=twins_spectra_ax,
                )
                ind += 1

                twins_pull_hist_ax = Plotter.axis(
                    fig,
                    n_rows,
                    n_cols,
                    ind,
                    sharey=twins_pull_ax,
                    sharex=spectra_hist_ax,
                )
                twins_pull_hist_ax.tick_params("y", labelleft=False, left=False)
                ind += 1
            else:
                twins_pull_ax = None
                twins_pull_hist_ax = None

            if do_legacy:
                legacy_pull_ax = Plotter.axis(
                    fig,
                    n_rows,
                    n_cols,
                    ind,
                    sharex=legacy_spectra_ax,
                    sharey=twins_pull_hist_ax if do_twins else None,
                )
                legacy_pull_ax.tick_params(
                    "y", labelleft=not do_twins, left=not do_twins
                )
                ind += 1

                legacy_pull_hist_ax = Plotter.axis(
                    fig,
                    n_rows,
                    n_cols,
                    ind,
                    sharey=legacy_pull_ax,
                    sharex=spectra_hist_ax,
                )
                legacy_pull_hist_ax.tick_params("y", labelleft=False, left=False)
                ind += 1
            else:
                legacy_pull_ax = None
                legacy_pull_hist_ax = None

            pull_ax = Plotter.axis(
                fig,
                n_rows,
                n_cols,
                ind,
                sharex=spectra_ax,
                sharey=legacy_pull_hist_ax
                if do_legacy
                else twins_pull_hist_ax
                if do_twins
                else None,
            )
            pull_ax.tick_params(
                "y",
                labelleft=not (do_twins or do_legacy),
                left=not (do_twins or do_legacy),
            )
            ind += 1

            pull_hist_ax = Plotter.axis(
                fig,
                n_rows,
                n_cols,
                ind,
                sharey=pull_ax,
                sharex=spectra_hist_ax,
            )
            pull_hist_ax.tick_params("y", labelleft=False, left=False)
            ind += 1

            fig.get_layout_engine().set(wspace=0, hspace=0, w_pad=0 / 72, h_pad=0 / 72)

            ax = (
                (
                    twins_spectra_ax,
                    twins_spectra_hist_ax,
                    twins_pull_ax,
                    twins_pull_hist_ax,
                ),
                (
                    legacy_spectra_ax,
                    legacy_spectra_hist_ax,
                    legacy_pull_ax,
                    legacy_pull_hist_ax,
                ),
                (spectra_ax, spectra_hist_ax, pull_ax, pull_hist_ax),
            )
        else:
            (
                (
                    twins_spectra_ax,
                    twins_spectra_hist_ax,
                    twins_pull_ax,
                    twins_pull_hist_ax,
                ),
                (
                    legacy_spectra_ax,
                    legacy_spectra_hist_ax,
                    legacy_pull_ax,
                    legacy_pull_hist_ax,
                ),
                (spectra_ax, spectra_hist_ax, pull_ax, pull_hist_ax),
            ) = ax

        twins_ax = ax[0]
        legacy_ax = ax[1]
        pae_ax = ax[2]

        if do_twins:
            fig, twins_spectra_ax, _hline = Plotter.axhline(
                0, fig=fig, ax=twins_spectra_ax, color="black"
            )
            fig, twins_spectra_hist_ax, _hline = Plotter.axhline(
                0, fig=fig, ax=twins_spectra_hist_ax, color="black"
            )

        if do_legacy:
            fig, legacy_spectra_ax, _hline = Plotter.axhline(
                0, fig=fig, ax=legacy_spectra_ax, color="black"
            )
            fig, legacy_spectra_hist_ax, _hline = Plotter.axhline(
                0, fig=fig, ax=legacy_spectra_hist_ax, color="black"
            )

        fig, spectra_ax, _hline = Plotter.axhline(
            0, fig=fig, ax=spectra_ax, color="black"
        )
        fig, spectra_hist_ax, _hline = Plotter.axhline(
            0, fig=fig, ax=spectra_hist_ax, color="black"
        )

        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            return None

        pae_redshift = data.redshift[:, 0, 0]

        (
            _wl,
            _amplitude,
            _sigma,
            sn_name,
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

        # Determine which spectra to keep
        # Will mask out any spectrum without at least one masked wavelength within the valid wavelength range
        mask_spec = np.any(input_mask, axis=-1)

        # Determine which SNe to keep
        # Will mask out any SN with *no* unmasked spectra
        mask_sn = np.any(mask_spec, axis=-1)

        pae_order = np.argsort(pae_redshift)
        pae_redshift = pae_redshift[pae_order]
        pae_redshift_error = (pae_redshift * 3e5 + 300.0) / 3e5
        pae_magshift_error = abs(-5 * np.log10(pae_redshift / pae_redshift_error))

        pae_mask = mask_sn[pae_order]
        pae_names = sn_name[:, 0, 0][pae_order]

        pae_phases = []
        pae_phase_errs_lower = []
        pae_phase_stds = []
        pae_phase_errs_upper = []

        pae_amplitudes = []
        pae_amplitude_errs_lower = []
        pae_amplitude_stds = []
        pae_amplitude_errs_upper = []
        for i, hmc in enumerate(hmcs):
            phases = []
            phase_errs_lower = []
            phase_stds = []
            phase_errs_upper = []

            amplitudes = []
            amplitude_errs_lower = []
            amplitude_stds = []
            amplitude_errs_upper = []
            for sn in range(hmc.hmc.samples.shape[-2]):
                name = f"{i}_{sn}"
                delta_m = hmc.hmc.samples[..., sn, 0]
                delta_p = hmc.hmc.samples[..., sn, 1]
                log_prob = hmc.hmc.log_prob[:, sn]
                lower, center, upper = max_central(delta_p, weight=log_prob)
                phases.append(center)
                phase_errs_lower.append(lower)
                phase_stds.append(0.5 * (upper - lower))
                phase_errs_upper.append(upper)

                lower, center, upper = max_central(delta_m, weight=log_prob)
                amplitudes.append(center)
                amplitude_errs_lower.append(lower)
                amplitude_stds.append(0.5 * (upper - lower))
                amplitude_errs_upper.append(upper)
            pae_phases.append(np.array(phases))
            pae_phase_errs_lower.append(np.array(phase_errs_lower))
            pae_phase_stds.append(np.array(phase_stds))
            pae_phase_errs_upper.append(np.array(phase_errs_upper))

            pae_amplitudes.append(np.array(amplitudes))
            pae_amplitude_errs_lower.append(np.array(amplitude_errs_lower))
            pae_amplitude_stds.append(np.array(amplitude_stds))
            pae_amplitude_errs_upper.append(np.array(amplitude_errs_upper))

        pae_phases = np.vstack(pae_phases)[..., pae_order]
        pae_phase_errs_lower = np.vstack(pae_phase_errs_lower)[..., pae_order]
        pae_phase_stds = np.vstack(pae_phase_stds)[..., pae_order]
        pae_phase_errs_upper = np.vstack(pae_phase_errs_upper)[..., pae_order]

        pae_amplitudes = np.vstack(pae_amplitudes)[..., pae_order]
        pae_amplitude_errs_lower = np.vstack(pae_amplitude_errs_lower)[..., pae_order]
        pae_amplitude_stds = np.vstack(pae_amplitude_stds)[..., pae_order]
        pae_amplitude_errs_upper = np.vstack(pae_amplitude_errs_upper)[..., pae_order]

        if config.reduce == "mean":
            pae_phases = np.vstack(
                [
                    np.mean(hmc.hmc.samples[..., 1], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]

            pae_phase_stds = np.vstack(
                [
                    np.std(hmc.hmc.samples[..., 1], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]

            pae_amplitudes = np.vstack(
                [
                    np.mean(hmc.hmc.samples[..., 0], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]

            pae_amplitude_stds = np.vstack(
                [
                    np.std(hmc.hmc.samples[..., 0], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]
        elif config.reduce == "median":
            pae_phases = np.vstack(
                [
                    np.median(hmc.hmc.samples[..., 1], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]

            pae_phase_stds = np.vstack(
                [
                    np.std(hmc.hmc.samples[..., 1], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]

            pae_amplitudes = np.vstack(
                [
                    np.median(hmc.hmc.samples[..., 0], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]

            pae_amplitude_stds = np.vstack(
                [
                    np.std(hmc.hmc.samples[..., 0], axis=0, keepdims=True)
                    for hmc in hmcs
                ],
            )[..., pae_order]

        pae_weights = 1 / np.clip(pae_phase_stds * pae_phase_stds, 1e-7, np.inf)
        pae_weighted_sum = pae_weights.sum(axis=0)
        pae_weighted_phases = (pae_weights * pae_phases).sum(axis=0) / pae_weighted_sum
        pae_weighted_phase_errs_lower = (pae_weights * pae_phase_errs_lower).sum(
            axis=0
        ) / pae_weighted_sum
        pae_weighted_phase_errs_lower = np.sqrt(
            pae_weighted_phase_errs_lower * pae_weighted_phase_errs_lower
            + pae_magshift_error * pae_magshift_error
        )
        pae_weighted_phase_errs_upper = (pae_weights * pae_phase_errs_upper).sum(
            axis=0
        ) / pae_weighted_sum
        pae_weighted_phase_errs_upper = np.sqrt(
            pae_weighted_phase_errs_upper * pae_weighted_phase_errs_upper
            + pae_magshift_error * pae_magshift_error
        )
        # TODO: Don't hardcode
        pae_mask &= np.abs(pae_weighted_phases) < 0.02

        pae_weights = 1 / np.clip(pae_amplitude_stds * pae_amplitude_stds, 1e-7, np.inf)
        pae_weighted_sum = pae_weights.sum(axis=0)
        pae_weighted_amplitudes = (pae_weights * pae_amplitudes).sum(
            axis=0
        ) / pae_weighted_sum
        pae_weighted_amplitude_errs_lower = (
            pae_weights * pae_amplitude_errs_lower
        ).sum(axis=0) / pae_weighted_sum
        pae_weighted_amplitude_errs_lower = np.sqrt(
            pae_weighted_amplitude_errs_lower * pae_weighted_amplitude_errs_lower
            + pae_magshift_error * pae_magshift_error
        )
        pae_weighted_amplitude_errs_upper = (
            pae_weights * pae_amplitude_errs_upper
        ).sum(axis=0) / pae_weighted_sum
        pae_weighted_amplitude_errs_upper = np.sqrt(
            pae_weighted_amplitude_errs_upper * pae_weighted_amplitude_errs_upper
            + pae_magshift_error * pae_magshift_error
        )

        pae_n_iter = len(hmcs)
        pae_n_eff = 1 if pae_n_iter == 1 else pae_n_iter / (pae_n_iter - 1)

        pae_weighted_variance = (
            (pae_weights * pae_amplitudes * pae_amplitudes).sum(axis=0)
            / pae_weighted_sum
        ) - (pae_weighted_amplitudes * pae_weighted_amplitudes)

        pae_weighted_deviations = np.sqrt(pae_n_eff * np.abs(pae_weighted_variance))

        pae_weighted_stds = np.sqrt(
            pae_weighted_deviations * pae_weighted_deviations
            + pae_magshift_error * pae_magshift_error
        )

        pae_twins_mask = np.ones_like(pae_mask, dtype=bool)
        pae_salt_mask = np.ones_like(pae_mask, dtype=bool)
        if twins is not None:
            pae_twins_mask = np.zeros_like(pae_mask, dtype=bool)
            pae_salt_mask = np.zeros_like(pae_mask, dtype=bool)
            twins_names = twins.name
            pae_intersection = set(pae_names) & set(twins_names)
            for name in pae_intersection:
                ind = np.argwhere(pae_names == name)[0]
                df = twins[twins.name == name]
                pae_twins_mask[ind] = df.mask_twins
                pae_salt_mask[ind] = df.mask_salt

        def _plot(
            x: "npt.NDArray[Any]",
            y: "npt.NDArray[Any]",
            yerr: "npt.NDArray[Any]",
            mask: "npt.NDArray[Any] | None",
            fig: "Figure",
            ax: "tuple[Axis, Axis, Axis, Axis]",
            color: str,
            alpha: float,
            title: str,
            residual_bins: "npt.NDArray[float]",
            pull_bins: "npt.NDArray[float]",
            yerr_lower: "npt.NDArray[Any] | None" = None,
            yerr_upper: "npt.NDArray[Any] | None" = None,
            names: "npt.NDArray[Any] | None" = None,
            tmp: bool = False,
        ) -> tuple[
            "Figure",
            tuple[
                "Axis",
                "Axis",
                "Axis",
                "Axis",
            ],
            tuple[float, float, float, float],
        ]:
            (
                s_ax,
                s_h_ax,
                p_ax,
                p_h_ax,
            ) = ax

            if mask is not None:
                mask = mask.astype(bool)
                x = x[mask]
                y = y[mask]
                yerr = yerr[mask]
                if yerr_lower is not None and yerr_upper is not None:
                    yerr_lower = yerr_lower[mask]
                    yerr_upper = yerr_upper[mask]

            pull_y = np.abs(y)
            pull_yerr = yerr
            if yerr_lower is not None and yerr_upper is not None:
                pull_yerr = np.abs(np.where(y > 0, yerr_lower, yerr_upper))
                yerr = (yerr_lower, yerr_upper)

            if names is None:
                names = pae_names
            names = names[mask]

            w_rms_jackknife = jackknife_resample(y, np.std)
            if tmp:
                rms_sort = np.argsort(w_rms_jackknife)
                rms_ = {}
                for i, rms in enumerate(w_rms_jackknife[rms_sort]):
                    if rms not in rms_:
                        rms_[rms] = []
                    rms_[rms].append(names[rms_sort][i])
                print("rms")
                pp(rms_)
            w_rms = np.mean(w_rms_jackknife)
            w_rms_std = np.std(w_rms_jackknife)

            n = len(w_rms_jackknife)
            w_rms = np.mean(w_rms_jackknife)
            w_rms_std = np.sqrt(np.sum((y - w_rms) ** 2 * pull_yerr**2)) / (
                (n - 1) * w_rms
            )

            k = 1.4826
            w_nmad_jackknife = k * jackknife_resample(
                y, lambda a: np.median(np.abs(a - np.median(a, axis=0)))
            )
            if tmp:
                nmad_sort = np.argsort(w_nmad_jackknife)
                nmad_ = {}
                for i, nmad in enumerate(w_nmad_jackknife[nmad_sort]):
                    if nmad not in nmad_:
                        nmad_[nmad] = []
                    nmad_[nmad].append(names[nmad_sort][i])
                print("nmad")
                pp(nmad_)
            w_nmad = np.mean(w_nmad_jackknife)
            w_nmad_std = np.std(w_nmad_jackknife)

            n = len(w_nmad_jackknife)
            w_nmad = np.mean(w_nmad_jackknife)
            w_nmad_std = 1.253 * w_nmad / np.sqrt(n)

            fig, s_ax, ebar = Plotter.errorbar(
                x,
                y,
                yerr=yerr,
                fig=fig,
                ax=s_ax,
                color=color,
                alpha=alpha,
                label=f"{title}\n{np.size(y)} SN\nRMS: {w_rms:.3f}±{w_rms_std:.3f}\nNMAD: {w_nmad:.3f}±{w_nmad_std:.3f}",
            )
            handles.append(ebar)
            fig, s_h_ax, _hist = Plotter.hist(
                y,
                bins=residual_bins,
                norm=True,
                orientation="horizontal",
                fig=fig,
                ax=s_h_ax,
                color=color,
                alpha=alpha,
            )

            fig, p_ax, _hline = Plotter.axhline(1, color="black", fig=fig, ax=p_ax)
            fig, p_h_ax, _hline = Plotter.axhline(1, color="black", fig=fig, ax=p_h_ax)
            fig, p_ax, _ebar = Plotter.errorbar(
                x,
                pull_y / pull_yerr,
                fig=fig,
                ax=p_ax,
                color=color,
                alpha=alpha,
            )
            fig, p_h_ax, _hist = Plotter.hist(
                pull_y / pull_yerr,
                bins=pull_bins,
                norm=True,
                orientation="horizontal",
                fig=fig,
                ax=p_h_ax,
                color=color,
                alpha=alpha,
            )

            return (
                fig,
                (
                    s_ax,
                    s_h_ax,
                    p_ax,
                    p_h_ax,
                ),
                (w_rms, w_rms_std, w_nmad, w_nmad_std),
            )

        pae_x = pae_redshift
        pae_y = pae_weighted_amplitudes
        pae_yerr = pae_weighted_stds
        pae_yerr_lower = pae_weighted_amplitude_errs_lower
        pae_yerr_upper = pae_weighted_amplitude_errs_upper
        no_plot_mask = None
        sn_plot_mask = pae_mask
        twins_plot_mask = pae_mask & pae_twins_mask
        salt_plot_mask = pae_mask & pae_salt_mask
        combined_plot_mask = pae_mask & pae_twins_mask & pae_salt_mask

        residual_max = np.log10(np.max(np.abs(pae_y[combined_plot_mask])))
        residual_scale_min = np.floor(residual_max)
        residual_scale_max = np.ceil(residual_max)
        residual_scale = 10 ** (
            residual_scale_min
            if np.abs(10**residual_max - 10**residual_scale_min)
            < np.abs(10**residual_max - 10**residual_scale_max)
            else residual_scale_max
        )
        residual_step = (
            10
            ** (
                residual_scale_min
                if np.abs(10**residual_max - 10**residual_scale_min)
                < 2 * np.abs(10**residual_max - 10**residual_scale_max)
                else residual_scale_max - np.log10(2)
            )
            / 4
        )

        if residual_step == 0:
            return None

        residual_bins = np.arange(
            -5 * residual_scale - 0.5 * residual_step,
            5 * residual_scale + 1.5 * residual_step,
            residual_step,
        )

        pull_y = np.abs(pae_y)
        sort_y = np.argsort(pull_y[combined_plot_mask])
        print("delta_m")
        pp(
            dict(
                zip(
                    [float(delta_m) for delta_m in pull_y[combined_plot_mask][sort_y]],
                    pae_names[combined_plot_mask][sort_y],
                    strict=True,
                )
            ),
        )

        yerr = np.where(pae_y > 0, pae_yerr_lower, pae_yerr_upper)
        pull_yerr = np.abs(yerr)
        sort_pull = np.argsort((pull_y / pull_yerr)[combined_plot_mask])
        print("pull")
        pp(
            dict(
                zip(
                    [
                        float(pull)
                        for pull in (pull_y / pull_yerr)[combined_plot_mask][sort_pull]
                    ],
                    pae_names[combined_plot_mask][sort_pull],
                    strict=True,
                )
            ),
        )

        max_pull = np.max((pull_y / pull_yerr)[combined_plot_mask])
        pull_max = np.log10(max_pull)
        pull_scale_min = np.floor(pull_max)
        pull_scale_max = np.ceil(pull_max)
        pull_scale = 10 ** (
            pull_scale_min
            if np.abs(10**pull_max - 10**pull_scale_min)
            < np.abs(10**pull_max - 10**pull_scale_max)
            else pull_scale_max
        )
        pull_step = (10**pull_scale_min) / 4

        pull_bins = np.arange(
            0 - 0.5 * pull_step, 5 * pull_scale + 1.5 * pull_step, pull_step
        )

        if twins is not None:
            sort = np.argsort(pae_names)

            twins_names = twins.name
            twins_intersection = set(pae_names) & set(twins_names)
            twins_mask = np.zeros_like(twins_names, dtype=np.int32)
            twins_pae_mask = np.zeros_like(pae_names, dtype=np.int32)
            for name in twins_intersection:
                ind = np.argwhere(twins_names == name)[0]
                twins_mask[ind] = 1
                ind = np.argwhere(pae_names == name)[0]
                twins_pae_mask[ind] = 1
            twins_mask = twins_mask.astype(bool)
            twins_pae_mask = twins_pae_mask.astype(bool)[sort]

            pae_twins = sort[twins_pae_mask]

            twins_redshift = pae_redshift[pae_twins]
            twins_order = np.argsort(twins_redshift)
            twins_redshift = twins_redshift[twins_order]
            twins_redshift_error = (twins_redshift * 3e5 + 300.0) / 3e5
            twins_magshift_error = abs(
                -5 * np.log10(twins_redshift / twins_redshift_error)
            )

            pae_twins = pae_twins[twins_order]

            twins_names = twins.name.to_numpy()[twins_mask][twins_order]

            twins_amplitudes = twins.dm_residuals_twins.to_numpy()[twins_mask][
                ..., twins_order
            ][None, ...]
            twins_amplitude_stds = twins.rbtl_dm_err.to_numpy()[twins_mask][
                ..., twins_order
            ][None, ...]

            twins_weights = 1 / (twins_amplitude_stds * twins_amplitude_stds)
            twins_weighted_sum = twins_weights.sum(axis=0)
            twins_weighted_amplitudes = (twins_weights * twins_amplitudes).sum(
                axis=0
            ) / twins_weighted_sum

            twins_n_eff = 1

            twins_weighted_variance = (
                (twins_weights * twins_amplitudes * twins_amplitudes).sum(axis=0)
                / twins_weighted_sum
            ) - (twins_weighted_amplitudes * twins_weighted_amplitudes)

            twins_weighted_deviations = np.sqrt(
                twins_n_eff * np.abs(twins_weighted_variance)
            )

            twins_weighted_stds = np.sqrt(
                twins_weighted_deviations * twins_weighted_deviations
                + twins_magshift_error * twins_magshift_error
            )

            # === No Mask ===
            twins_x = twins_redshift
            twins_y = twins_weighted_amplitudes
            twins_yerr = twins_weighted_stds
            fig, twins_ax, _ = _plot(
                twins_x,
                twins_y,
                twins_yerr,
                no_plot_mask,
                fig,
                twins_ax,
                "black",
                0.25,
                "Twins No Mask",
                residual_bins=residual_bins,
                pull_bins=pull_bins,
                names=twins_names,
            )

            if twins is not None:
                # === SN Mask ===
                fig, twins_ax, _ = _plot(
                    twins_x,
                    twins_y,
                    twins_yerr,
                    sn_plot_mask[pae_twins],
                    fig,
                    twins_ax,
                    "brown",
                    0.25,
                    "Twins SN Mask",
                    residual_bins=residual_bins,
                    pull_bins=pull_bins,
                    names=twins_names,
                )

                # === Twins Mask ===
                fig, twins_ax, _ = _plot(
                    twins_x,
                    twins_y,
                    twins_yerr,
                    twins_plot_mask[pae_twins],
                    fig,
                    twins_ax,
                    "blue",
                    0.25,
                    "Twins Twins Mask",
                    residual_bins=residual_bins,
                    pull_bins=pull_bins,
                    names=twins_names,
                )

                # === Salt Mask ===
                fig, twins_ax, _ = _plot(
                    twins_x,
                    twins_y,
                    twins_yerr,
                    salt_plot_mask[pae_twins],
                    fig,
                    twins_ax,
                    "purple",
                    0.25,
                    "Twins Salt Mask",
                    residual_bins=residual_bins,
                    pull_bins=pull_bins,
                    names=twins_names,
                )

            # === Combined Mask ===
            fig, twins_ax, _ = _plot(
                twins_x,
                twins_y,
                twins_yerr,
                combined_plot_mask[pae_twins],
                fig,
                twins_ax,
                "green",
                1,
                "Twins All Masks",
                residual_bins=residual_bins,
                pull_bins=pull_bins,
                names=twins_names,
            )
            twins_pull_ax.set_xlabel("z")
            twins_pull_hist_ax.set_xlabel("PDF")

        if legacy is not None:
            legacy_names = legacy["names"]

            legacy_intersection = set(pae_names) & set(legacy_names)
            legacy_mask = np.zeros_like(legacy_names, dtype=np.int32)
            for name in legacy_intersection:
                ind = np.argwhere(legacy_names == name)[0]
                legacy_mask[ind] = 1
            legacy_mask = legacy_mask.astype(bool)

            legacy_redshift = legacy["redshift"][legacy_mask]
            legacy_order = np.argsort(legacy_redshift)
            legacy_redshift = legacy_redshift[legacy_order]
            legacy_redshift_error = (legacy_redshift * 3e5 + 300.0) / 3e5
            legacy_magshift_error = abs(
                -5 * np.log10(legacy_redshift / legacy_redshift_error)
            )

            legacy_names = legacy["names"][legacy_mask][legacy_order]
            legacy_amplitudes = legacy["amplitude_mcmc"][legacy_mask][None, ...][
                ..., legacy_order
            ]
            legacy_amplitude_stds = legacy["amplitude_mcmc_err"][legacy_mask][
                None, ...
            ][..., legacy_order]

            legacy_weights = 1 / (legacy_amplitude_stds * legacy_amplitude_stds)
            legacy_weighted_sum = legacy_weights.sum(axis=0)
            legacy_weighted_amplitudes = (legacy_weights * legacy_amplitudes).sum(
                axis=0
            ) / legacy_weighted_sum

            legacy_n_eff = 1

            legacy_weighted_variance = (
                (legacy_weights * legacy_amplitudes * legacy_amplitudes).sum(axis=0)
                / legacy_weighted_sum
            ) - (legacy_weighted_amplitudes * legacy_weighted_amplitudes)

            legacy_weighted_deviations = np.sqrt(
                legacy_n_eff * np.abs(legacy_weighted_variance)
            )

            legacy_weighted_stds = np.sqrt(
                legacy_weighted_deviations * legacy_weighted_deviations
                + legacy_magshift_error * legacy_magshift_error
            )

            # === No Mask ===
            legacy_x = legacy_redshift
            legacy_y = legacy_weighted_amplitudes
            legacy_yerr = legacy_weighted_stds
            fig, legacy_ax, _ = _plot(
                legacy_x,
                legacy_y,
                legacy_yerr,
                no_plot_mask,
                fig,
                legacy_ax,
                "black",
                0.25,
                "Legacy No Mask",
                residual_bins=residual_bins,
                pull_bins=pull_bins,
            )

            if twins is not None:
                # === SN Mask ===
                fig, legacy_ax, _ = _plot(
                    legacy_x,
                    legacy_y,
                    legacy_yerr,
                    sn_plot_mask,
                    fig,
                    legacy_ax,
                    "brown",
                    0.25,
                    "Legacy SN Mask",
                    residual_bins=residual_bins,
                    pull_bins=pull_bins,
                )

                # === Twins Mask ===
                fig, legacy_ax, _ = _plot(
                    legacy_x,
                    legacy_y,
                    legacy_yerr,
                    twins_plot_mask,
                    fig,
                    legacy_ax,
                    "blue",
                    0.25,
                    "Legacy Twins Mask",
                    residual_bins=residual_bins,
                    pull_bins=pull_bins,
                )

                # === Salt Mask ===
                fig, legacy_ax, _ = _plot(
                    legacy_x,
                    legacy_y,
                    legacy_yerr,
                    salt_plot_mask,
                    fig,
                    legacy_ax,
                    "purple",
                    0.25,
                    "Legacy Salt Mask",
                    residual_bins=residual_bins,
                    pull_bins=pull_bins,
                )

            # === Combined Mask ===
            fig, legacy_ax, _ = _plot(
                legacy_x,
                legacy_y,
                legacy_yerr,
                combined_plot_mask,
                fig,
                legacy_ax,
                "green",
                1,
                "Legacy All Masks",
                residual_bins=residual_bins,
                pull_bins=pull_bins,
            )
            legacy_pull_ax.set_xlabel("z")
            legacy_pull_hist_ax.set_xlabel("PDF")

        # === No Mask ===
        fig, pae_ax, no_mask_stats = _plot(
            pae_x,
            pae_y,
            pae_yerr,
            no_plot_mask,
            fig,
            pae_ax,
            "black",
            0.25,
            "No Mask",
            residual_bins=residual_bins,
            pull_bins=pull_bins,
            yerr_lower=pae_yerr_lower,
            yerr_upper=pae_yerr_upper,
        )

        sn_mask_stats = None
        twins_mask_stats = None
        salt_mask_stats = None
        if twins is not None:
            # === SN Mask ===
            fig, pae_ax, sn_mask_stats = _plot(
                pae_x,
                pae_y,
                pae_yerr,
                sn_plot_mask,
                fig,
                pae_ax,
                "brown",
                0.25,
                "SN Mask",
                residual_bins=residual_bins,
                pull_bins=pull_bins,
                yerr_lower=pae_yerr_lower,
                yerr_upper=pae_yerr_upper,
            )

            # === Twins Mask ===
            fig, pae_ax, twins_mask_stats = _plot(
                pae_x,
                pae_y,
                pae_yerr,
                twins_plot_mask,
                fig,
                pae_ax,
                "blue",
                0.25,
                "Twins Mask",
                residual_bins=residual_bins,
                pull_bins=pull_bins,
                yerr_lower=pae_yerr_lower,
                yerr_upper=pae_yerr_upper,
            )

            # === Salt Mask ===
            fig, pae_ax, salt_mask_stats = _plot(
                pae_x,
                pae_y,
                pae_yerr,
                salt_plot_mask,
                fig,
                pae_ax,
                "purple",
                0.25,
                "Salt Mask",
                residual_bins=residual_bins,
                pull_bins=pull_bins,
                yerr_lower=pae_yerr_lower,
                yerr_upper=pae_yerr_upper,
            )

        # === Combined Mask ===
        fig, pae_ax, combined_mask_stats = _plot(
            pae_x,
            pae_y,
            pae_yerr,
            combined_plot_mask,
            fig,
            pae_ax,
            "green",
            1,
            "All Masks",
            residual_bins=residual_bins,
            pull_bins=pull_bins,
            yerr_lower=pae_yerr_lower,
            yerr_upper=pae_yerr_upper,
            tmp=True,
        )

        fig.suptitle((config.plot_kwargs or {}).get("title", config.name.capitalize()))
        if do_twins:
            twins_spectra_ax.set_ylim(
                -1.1 * np.abs((pae_y - yerr)[combined_plot_mask].min()),
                1.1 * np.abs(pae_y + yerr)[combined_plot_mask].max(),
            )
            twins_spectra_ax.set_ylabel("ΔM")
            twins_pull_ax.set_ylim(
                0,
                1.1 * np.abs(pae_y / yerr)[combined_plot_mask].max(),
            )
            twins_pull_ax.set_ylabel("Pull")
        elif do_legacy:
            legacy_spectra_ax.set_ylim(
                -1.1 * np.abs((pae_y - yerr)[combined_plot_mask].min()),
                1.1 * np.abs(pae_y + yerr)[combined_plot_mask].max(),
            )
            legacy_spectra_ax.set_ylabel("ΔM")
            legacy_pull_ax.set_ylim(
                0,
                1.1 * np.abs(pae_y / yerr)[combined_plot_mask].max(),
            )
            legacy_pull_ax.set_ylabel("Pull")
        else:
            spectra_ax.set_ylim(
                -1.1 * np.abs((pae_y - yerr)[combined_plot_mask].min()),
                1.1 * np.abs(pae_y + yerr)[combined_plot_mask].max(),
            )
            spectra_ax.set_ylabel("ΔM")
            pull_ax.set_ylim(
                0,
                1.1 * np.abs(pae_y / yerr)[combined_plot_mask].max(),
            )
            pull_ax.set_ylabel("Pull")

        pull_ax.set_xlabel("z")
        pull_hist_ax.set_xlabel("PDF")

        leg = spectra_hist_ax.legend(
            handles=handles, bbox_to_anchor=(1.0, 1.0), ncols=n_cols / 2
        )
        leg.set_in_layout(False)
        # Trigger a draw so that constrained layout is executed once
        # before we turn it off when printing....
        fig.canvas.draw()
        # We want the legend included in the bbox_inches='tight' calcs.
        leg.set_in_layout(True)
        # We don't want the layout to change at this point.
        fig.set_layout_engine("none")

        fig = Plotter.save(fig, savepath)
        Plotter.close(fig, [*ax[0], *ax[1], *ax[2]])

        return (
            no_mask_stats,
            sn_mask_stats,
            twins_mask_stats,
            salt_mask_stats,
            combined_mask_stats,
        )
