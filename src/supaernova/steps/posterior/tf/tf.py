# Copyright 2025 Patrick Armstrong
from supaernova.steps.pae.tf.photometry import (
    photometry,
    photometry_amplitude_setup,
    photometry_sigma_setup,
)
import os
import types
import contextlib
from typing import TYPE_CHECKING, override

from tqdm import tqdm
import numpy as np

from supaernova._tf import (
    HUGE,
    NPROC,
    ks,
    tf,
    tfb,
    tfd,
    tfp,
    clear_session,
    JIT_COMPILE,
)
from supaernova.utils.tf import db, pp

from .hmc import PosteriorHMCValue
from .map import PosteriorMap

if TYPE_CHECKING:
    from typing import Any, Literal
    from logging import Logger
    from pathlib import Path
    from collections.abc import Iterator, Sequence

    from numpy import typing as npt
    from tensorflow_probability.python.optimizer.lbfgs import LBfgsOptimizerResults
    from tensorflow_probability.python.mcmc.dual_averaging_step_size_adaptation import (
        DualAveragingStepSizeAdaptationResults,
    )

    from supaernova.steps.pae.tf import TFPAEModel
    from supaernova.steps.nflow.tf import TFNFlowModel
    from supaernova.steps.posterior import Posterior
    from supaernova.configs.steps.data import LazySNPAEData
    from supaernova.typing.backends.tf import Loss, TensorLike
    from supaernova.configs.steps.posterior import PosteriorMAPStage
    from supaernova.configs.steps.posterior.tf import TFPosteriorConfig

POSTERIORMODELSTEP: "Posterior"


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPosteriorModel(ks.Model):
    def __init__(
        self,
        config: "Posterior",
        subset: "Literal['train', 'test']",
        seed: int,
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=config.name.split()[-1], **kwargs)
        # --- Config ---
        global POSTERIORMODELSTEP
        POSTERIORMODELSTEP = config
        self.options: TFPosteriorConfig = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = seed
        self.subset: Literal["train", "test"] = subset
        self.step_size = config.step_sizes[self.subset]
        self.u_latent_bounds = config.u_latent_bounds[self.subset]
        self.generalised_u_latents: float = config.generalised_u_latents

        self.min_phase = config.min_phase
        self.max_phase = config.max_phase

        self.debug: bool = config.config.debug or self.options.debug
        self.profile: bool = self.options.profile

        self.data: LazySNPAEData = getattr(config, f"{self.subset}_data")
        self.data_time: npt.NDArray[float] = self.data.time
        self.data_amplitude: npt.NDArray[float] = self.data.amplitude
        self.data_sigma: npt.NDArray[float] = self.data.sigma
        self.data_wavelength: npt.NDArray[float] = self.data.wavelength
        self.data_throughput: npt.NDArray[float] = self.data.throughput
        self.data_effective_wavelength: npt.NDArray[float] = (
            self.data.effective_wavelength
        )
        self.data_spectra_mask: npt.NDArray[float] = self.data.spectra_mask
        self.data_phot_mask: npt.NDArray[float] = self.data.phot_mask
        self.data.clear()

        self.cached_amp: tuple[tf.Tensor, tf.Tensor] = photometry_amplitude_setup(
            tf.repeat(
                self.data_wavelength[..., None], self.data_throughput.shape[-1], axis=-1
            ),
            self.data_throughput,
        )
        self.cached_sigma: tuple[tf.Tensor, tf.Tensor] = photometry_sigma_setup(
            tf.repeat(
                self.data_wavelength[..., None], self.data_throughput.shape[-1], axis=-1
            ),
            self.data_throughput,
            self.cached_amp,
        )

        self.data_mask: npt.NDArray[bool] = getattr(config, f"{self.subset}_mask")
        self.sn_mask: npt.NDArray[bool] = getattr(config, f"{self.subset}_sn_mask")
        self.spec_mask: npt.NDArray[bool] = getattr(config, f"{self.subset}_spec_mask")
        self.wl_mask: npt.NDArray[bool] = getattr(config, f"{self.subset}_wl_mask")

        self.data_time[~self.spec_mask] = -HUGE
        self.data_amplitude[~self.data_mask] = -HUGE
        self.data_sigma[~self.data_mask] = -HUGE

        self.legacy_path = config.legacy_path
        self.data_dir = config.data_dir

        # Equivalent to `self.pae = ...` but avoids tf / ks from tracking self.pae
        self.pae: TFPAEModel
        vars(self)["pae"] = config.pae
        self.pae.trainable = False
        self.pae.encoder.trainable = False
        self.pae.decoder.trainable = False

        # Equivalent to `self.nflow = ...` but avoids tf / ks from tracking self.nflow
        self.nflow: TFNFlowModel
        vars(self)["nflow"] = config.nflow
        self.nflow.trainable = False
        self.nflow.flow.trainable = False

        self.sn_dim, self.spec_dim, self.wl_dim = self.data_mask.shape

        n_walkers: int | float = self.options.n_walkers
        if isinstance(n_walkers, float):
            n_walkers = int(NPROC * n_walkers)
        self.n_walkers = n_walkers
        self.n_chains = 1
        self.log_likelihood_scale = self.options.log_likelihood_scale
        self.log_likelihood_spec_sum = self.options.log_likelihood_spec_sum
        self.log_likelihood_sum = self.options.log_likelihood_sum
        self.step_size_scale = self.options.step_size_scale
        self.fractional_error = self.options.fractional_error
        self.weighted_error = self.options.weighted_error

        # MAP Variables
        self.map: PosteriorMap
        self.norm_prob: float | None = None
        self.tolerance: float = self.options.tolerance
        self.x_tolerance: float = self.options.x_tolerance
        self.f_relative_tolerance: float = self.options.f_relative_tolerance
        self.f_absolute_tolerance: float = self.options.f_absolute_tolerance
        self.max_iterations = self.options.max_iterations
        self.max_line_search_iterations = (
            self.options.max_line_search_iterations or int(np.sqrt(self.max_iterations))
        )
        self.num_correction_pairs = self.options.num_correction_pairs or max(
            1, int(0.1 * self.max_line_search_iterations)
        )

        # --- Training ---
        self.save_best: bool = self.options.save_best
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )
        self.log_path: str = f"{'best' if self.save_best else 'latest'}_logs/"

        self.recon_error = config.recon_error[self.subset]
        self.recon_error_centers = config.recon_error_centers[self.subset]
        self.sigma_recon = tf.transpose(
            tfp.math.interp_regular_1d_grid(
                x=tf.transpose(self.data_time[..., 0]),
                x_ref_min=self.recon_error_centers[0],
                x_ref_max=self.recon_error_centers[-1],
                y_ref=self.recon_error,
            )
        )

        loss: Loss = self.options.loss_cls()
        loss.model = self
        self._loss: Loss = loss

        # HMC Variables

        self.n_leapfrog: int = self.options.n_leapfrog
        max_leapfrog = (2**self.n_leapfrog) - 1

        self.n_run_steps: int = self.options.n_run_steps

        self.n_burnin_steps: int = self.options.n_burnin_steps
        if isinstance(self.n_burnin_steps, float):
            self.n_burnin_steps = int(self.n_run_steps * self.n_burnin_steps)

        self.n_adaption_steps: int = self.options.n_adaption_steps
        if isinstance(self.n_adaption_steps, float):
            self.n_adaption_steps = int(
                (
                    self.n_burnin_steps
                    + (self.n_run_steps if self.n_burnin_steps == 0 else 0)
                )
                * self.n_adaption_steps
            )

        self.max_samples: int = int(
            max_leapfrog * (self.n_burnin_steps + self.n_run_steps)
        )

        self.max_tree_depth: int = (2**self.n_leapfrog) - 1

        self.max_steps: int = self.n_run_steps * self.n_walkers

        self.n_thinning: int = self.options.n_thinning
        self.target_acceptance_rate: float = self.options.target_acceptance_rate

        self.hmc: PosteriorHMCValue

        self.r_hat: tf.Tensor

        self.set_seed()

    @override
    def call(
        self,
        input_position: tf.Tensor,
        *,
        training: bool | None = None,
        input_phase: tf.Tensor | None = None,
        input_amp: tf.Tensor | None = None,
        input_sigma: tf.Tensor | None = None,
        input_wavelength: tf.Tensor | None = None,
        input_throughput: tf.Tensor | None = None,
        input_effective_wavelength: tf.Tensor | None = None,
        input_spectra_mask: tf.Tensor | None = None,
        input_phot_mask: tf.Tensor | None = None,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        testing: bool | None = None,
        additional_outputs: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        training = False if training is None else training
        testing = False if testing is None else testing
        eps = ks.backend.epsilon()

        # === Inputs ===
        if input_phase is None:
            input_phase = tf.cast(
                tf.convert_to_tensor(self.data_time), dtype=tf.float32
            )
        if input_amp is None:
            input_amp = tf.cast(
                tf.convert_to_tensor(self.data_amplitude), dtype=tf.float32
            )
        if input_sigma is None:
            input_sigma = tf.cast(
                tf.convert_to_tensor(self.data_sigma), dtype=tf.float32
            )
        if input_wavelength is None:
            input_wavelength = tf.cast(
                tf.convert_to_tensor(self.data_wavelength), dtype=tf.float32
            )
        if input_throughput is None:
            input_throughput = tf.cast(
                tf.convert_to_tensor(self.data_throughput), dtype=tf.float32
            )
        if input_effective_wavelength is None:
            input_effective_wavelength = tf.cast(
                tf.convert_to_tensor(self.data_effective_wavelength), dtype=tf.float32
            )
        if input_spectra_mask is None:
            input_spectra_mask = tf.cast(
                tf.convert_to_tensor(self.data_spectra_mask), dtype=tf.bool
            )
        if input_phot_mask is None:
            input_phot_mask = tf.cast(
                tf.convert_to_tensor(self.data_phot_mask), dtype=tf.bool
            )

        # --- Masks ---
        # Data Mask
        input_mask = tf.ones_like(input_amp, dtype=tf.bool) if mask is None else mask

        # Wavelength Range Mask
        input_wl_mask = tf.ones_like(input_mask) if wl_mask is None else wl_mask

        # Phase Range Mask
        input_spec_mask = (
            tf.math.reduce_any(input_wl_mask, axis=-1, keepdims=True)
            if spec_mask is None
            else spec_mask
        )

        # Redshift Range Mask
        input_sn_mask = (
            tf.math.reduce_any(input_spec_mask, axis=-2, keepdims=True)
            if sn_mask is None
            else sn_mask
        )

        posterior_mask = input_mask & input_sn_mask & input_spec_mask & input_wl_mask

        mask_spec = tf.math.reduce_any(posterior_mask, axis=-1)

        # Determine which sn to keep
        mask_sn = tf.math.reduce_any(mask_spec, axis=-1)

        # Unconstrained -> Constrained
        input_position = self.map.constrain(input_position, full=True)

        log_prior = self.map.prior(input_position)

        # Ignore prior of fully masked SN
        # Important to avoid them affecting accept ratio / step size calculations
        log_prior = tf.where(mask_sn, log_prior, -np.inf * tf.ones_like(log_prior))

        log_prior = tf.where(
            tf.math.is_finite(log_prior), log_prior, -np.inf * tf.ones_like(log_prior)
        )

        delta_m = input_position[..., 0:1]
        delta_p = input_position[..., 1:2]
        bias = input_position[..., 2:3]
        u_delta_av = input_position[..., 3:4]
        u_latents = input_position[..., 4:]

        # Transform from u-latent to z-latent space
        if self.nflow.physical_latents:
            us = tf.concat([u_delta_av, u_latents], axis=-1)
        else:
            us = u_latents

        zs = self.nflow.u_to_z(us, permute=True)
        if self.nflow.physical_latents:
            delta_av = zs[..., :1]
            zs = zs[..., 1:]
        if self.pae.physical_latents:
            zs = tf.concat(
                [
                    delta_av,
                    zs,
                    delta_m,
                    delta_p,
                ],
                axis=-1,
            )

        zs = tf.repeat(tf.expand_dims(zs, axis=-2), repeats=[self.spec_dim], axis=-2)
        phase = tf.repeat(
            tf.expand_dims(input_phase, axis=0), repeats=zs.shape[0], axis=0
        )

        # Create synthetic spectra from z-latents
        decoder_inputs = tf.concat((phase, zs), axis=-1)
        synth_amp = self.pae.decoder(
            decoder_inputs,
            mask=input_mask,
            sn_mask=input_sn_mask,
            spec_mask=input_spec_mask,
            wl_mask=input_wl_mask,
            training=False,
        )

        if self.map.train_delta_m and not self.pae.physical_latents:
            delta_m = tf.expand_dims(delta_m, axis=-2)
            synth_amp *= delta_m

        if self.map.train_bias:
            bias = tf.expand_dims(bias, axis=-2)
            synth_amp += bias

        if self.map.train_delta_p and not self.pae.physical_latents:
            delta_p = tf.expand_dims(delta_p, axis=-2)
            phase += delta_p

            # Measured average AE reconstruction error at current times
            sigma_recon = tf.transpose(
                tfp.math.interp_regular_1d_grid(
                    x=tf.transpose(phase[..., 0]),
                    x_ref_min=self.recon_error_centers[0],
                    x_ref_max=self.recon_error_centers[-1],
                    y_ref=self.recon_error,
                )
            )
        else:
            sigma_recon = self.sigma_recon

        synth_scale = (
            tf.math.maximum(
                tf.sqrt(synth_amp * synth_amp + input_sigma * input_sigma), eps
            )
            if self.fractional_error
            else 1
        )
        synth_sigma = tf.sqrt(
            tf.square(synth_scale * sigma_recon) + (input_sigma * input_sigma)
        )

        (
            synth_amp,
            synth_sigma,
        ) = photometry(
            tf.repeat(input_wavelength[None, ...], synth_amp.shape[0], axis=0),
            synth_amp,
            synth_sigma,
            tf.repeat(input_throughput[None, ...], synth_amp.shape[0], axis=0),
            tf.repeat(
                input_effective_wavelength[None, ...], synth_amp.shape[0], axis=0
            ),
            tf.repeat(input_spectra_mask[None, ...], synth_amp.shape[0], axis=0),
            tf.repeat(input_phot_mask[None, ...], synth_amp.shape[0], axis=0),
            self.cached_amp,
            self.cached_sigma,
        )

        # Set missing values to 0 for all times
        synth_amp = tf.where(posterior_mask, synth_amp, tf.zeros_like(synth_amp))

        # Set missing values to 1 for all times
        synth_sigma = tf.where(posterior_mask, synth_sigma, tf.ones_like(synth_sigma))
        sigma_mask = synth_sigma > 0
        synth_sigma = tf.where(sigma_mask, synth_sigma, tf.ones_like(synth_sigma))

        likelihood = tfd.Normal(loc=synth_amp, scale=synth_sigma)

        # Set missing values to 0 for all times
        synth_sigma = tf.where(sigma_mask, synth_sigma, tf.zeros_like(synth_sigma))
        amp = tf.where(posterior_mask, input_amp, tf.zeros_like(input_amp))
        log_likelihood_wl = likelihood.log_prob(amp)
        log_likelihood_wl = tf.where(sigma_mask, log_likelihood_wl, 0)

        log_likelihood_spec_num = tf.reduce_sum(
            tf.where(
                posterior_mask,
                log_likelihood_wl,
                tf.zeros_like(log_likelihood_wl),
            ),
            axis=-1,
        )
        log_likelihood_spec_sum = tf.math.maximum(
            tf.math.count_nonzero(
                posterior_mask,
                axis=-1,
                dtype=log_likelihood_wl.dtype,
            ),
            1,
        )

        log_likelihood_spec = log_likelihood_spec_num  # / log_likelihood_spec_sum
        if self.log_likelihood_spec_sum:
            log_likelihood_spec /= log_likelihood_spec_sum

        log_likelihood_num = tf.reduce_sum(
            tf.where(
                mask_spec,
                log_likelihood_spec,
                tf.zeros_like(log_likelihood_spec),
            ),
            axis=-1,
        )
        log_likelihood_sum = tf.math.maximum(
            tf.math.count_nonzero(
                mask_spec,
                axis=-1,
                dtype=log_likelihood_spec.dtype,
            ),
            1,
        )

        log_likelihood = self.log_likelihood_scale * log_likelihood_num
        if self.log_likelihood_sum:
            log_likelihood /= log_likelihood_sum

        # Ignore likelihood of fully masked SN
        log_likelihood = tf.where(
            mask_sn, log_likelihood, -np.inf * tf.ones_like(log_likelihood)
        )

        log_likelihood = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            -np.inf * tf.ones_like(log_likelihood),
        )

        log_probability = log_likelihood + log_prior

        # Ignore probability of fully masked SN
        log_probability = tf.where(
            mask_sn, log_probability, -np.inf * tf.ones_like(log_probability)
        )

        log_probability = tf.where(
            tf.math.is_finite(log_probability),
            log_probability,
            -np.inf * tf.ones_like(log_probability),
        )

        if additional_outputs:
            return log_probability, log_likelihood, log_prior, synth_amp, synth_sigma
        return log_probability

    @override
    def __call__(
        self,
        inputs: "TensorLike",
        *,
        training: bool | None = None,
        input_phase: "TensorLike | None" = None,
        input_amp: "TensorLike | None" = None,
        input_sigma: "TensorLike | None" = None,
        mask: "TensorLike | None" = None,
        sn_mask: "TensorLike | None" = None,
        spec_mask: "TensorLike | None" = None,
        wl_mask: "TensorLike | None" = None,
        testing: bool | None = None,
        additional_outputs: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        training = False if training is None else training
        testing = False if testing is None else testing
        inputs = tf.cast(tf.convert_to_tensor(inputs), dtype=tf.float32)
        if input_phase is not None:
            input_phase = tf.cast(tf.convert_to_tensor(input_phase), dtype=tf.float32)
        if input_amp is not None:
            input_amp = tf.cast(tf.convert_to_tensor(input_amp), dtype=tf.float32)
        if input_sigma is not None:
            input_sigma = tf.cast(tf.convert_to_tensor(input_sigma), dtype=tf.float32)
        if mask is not None:
            mask = tf.cast(tf.convert_to_tensor(mask), dtype=tf.bool)
        if sn_mask is not None:
            sn_mask = tf.cast(tf.convert_to_tensor(sn_mask), dtype=tf.bool)
        if spec_mask is not None:
            spec_mask = tf.cast(tf.convert_to_tensor(spec_mask), dtype=tf.bool)
        if wl_mask is not None:
            wl_mask = tf.cast(tf.convert_to_tensor(wl_mask), dtype=tf.bool)
        return super().__call__(
            inputs,
            training=training,
            input_phase=input_phase,
            input_amp=input_amp,
            input_sigma=input_sigma,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            testing=testing,
            additional_outputs=additional_outputs,
        )

    def setup_map(self) -> None:
        if not hasattr(self, "map"):
            vars(self)["map"] = PosteriorMap(self)

    def get_mask_sn(self) -> tf.Tensor:
        """Compute the per-SN mask.

        Returns:
            Boolean tensor of shape `(sn_dim,)`, True where a SN has at
            least one unmasked spec/wl point.
        """
        mask = self.data_mask & self.sn_mask & self.spec_mask & self.wl_mask
        mask_spec = tf.math.reduce_any(mask, axis=-1)
        return tf.math.reduce_any(mask_spec, axis=-1)

    @contextlib.contextmanager
    def _restricted_to_valid_sn(self) -> "Iterator[tf.Tensor | None]":
        """Temporarily shrink every sn_dim-indexed attribute to unmasked SNe only.

        NUTS runs one shared `tf.while_loop` across the whole (n_walkers,
        sn_dim) batch, so fully-masked SNe (whose log-prob is always -inf)
        otherwise ride along for the full trajectory length of whichever
        real SN's chain is slowest to U-turn, wasting the max_tree_depth
        budget. Restricting the batch to real SNe for the scope of
        sampling fixes that; callers must restore/scatter back afterwards.

        Yields:
            The indices of the valid (unmasked) SNe, or `None` if no SN is
            masked (in which case nothing was changed).
        """
        valid_sn = self.get_mask_sn()
        if bool(tf.reduce_all(valid_sn)):
            yield None
        else:
            valid_indices = tf.where(valid_sn)[:, 0]

            sn_attrs = (
                "data_time",
                "data_amplitude",
                "data_sigma",
                "data_wavelength",
                "data_throughput",
                "data_effective_wavelength",
                "data_spectra_mask",
                "data_phot_mask",
                "data_mask",
                "sn_mask",
                "spec_mask",
                "wl_mask",
                "sigma_recon",
            )
            originals = {attr: getattr(self, attr) for attr in sn_attrs}
            original_cached_amp = self.cached_amp
            original_cached_sigma = self.cached_sigma
            original_sn_dim = self.sn_dim

            map_value_attrs = ("u_delta_av", "delta_m", "delta_p", "bias")
            map_originals = {attr: getattr(self.map, attr) for attr in map_value_attrs}
            original_map_sn_dim = self.map.sn_dim

            try:
                for attr in sn_attrs:
                    setattr(
                        self, attr, tf.gather(originals[attr], valid_indices, axis=0)
                    )
                self.cached_amp = tuple(
                    tf.gather(t, valid_indices, axis=0) for t in original_cached_amp
                )
                self.cached_sigma = tuple(
                    tf.gather(t, valid_indices, axis=0) for t in original_cached_sigma
                )
                self.sn_dim = int(valid_indices.shape[0])

                for attr in map_value_attrs:
                    value = map_originals[attr]
                    setattr(
                        self.map,
                        attr,
                        types.SimpleNamespace(
                            current=tf.gather(value.current, valid_indices, axis=1),
                            best=tf.gather(value.best, valid_indices, axis=1),
                        ),
                    )
                self.map.sn_dim = self.sn_dim

                yield valid_indices
            finally:
                for attr in sn_attrs:
                    setattr(self, attr, originals[attr])
                self.cached_amp = original_cached_amp
                self.cached_sigma = original_cached_sigma
                self.sn_dim = original_sn_dim

                for attr in map_value_attrs:
                    setattr(self.map, attr, map_originals[attr])
                self.map.sn_dim = original_map_sn_dim

    def _scatter_sn(
        self,
        value: tf.Tensor,
        *,
        axis: int,
        indices: tf.Tensor,
        fill: float,
    ) -> tf.Tensor:
        """Scatter a tensor computed over valid SNe only back to full sn_dim.

        `value` has `len(indices)` entries along `axis`; the returned
        tensor has `self.sn_dim` entries along `axis`, with masked SNe
        filled with `fill`.

        Returns:
            `value` scattered back to `self.sn_dim` entries along `axis`.
        """
        rank = len(value.shape)
        axis %= rank
        perm = [axis, *(a for a in range(rank) if a != axis)]
        moved = tf.transpose(value, perm)
        full = tf.fill(
            [self.sn_dim, *moved.shape[1:]], tf.constant(fill, dtype=value.dtype)
        )
        updated = tf.tensor_scatter_nd_update(full, indices[:, None], moved)
        inv_perm = list(np.argsort(perm))
        return tf.transpose(updated, inv_perm)

    def setup_hmc(self) -> None:
        self.setup_map()
        if not hasattr(self, "hmc"):
            vars(self)["hmc"] = PosteriorHMCValue(
                tf.Variable(  # Samples
                    tf.cast(
                        tf.convert_to_tensor(
                            [
                                [[[0] * self.map.n_pae_latents] * self.sn_dim]
                                * self.n_walkers
                            ]
                            * self.n_run_steps
                        ),
                        dtype=tf.float32,
                    ),
                    shape=(
                        self.n_run_steps,
                        self.n_walkers,
                        self.sn_dim,
                        self.map.n_pae_latents,
                    ),
                ),
                tf.Variable(  # Log Prior
                    tf.cast(
                        tf.convert_to_tensor([[0] * self.sn_dim] * self.max_steps),
                        dtype=tf.float32,
                    ),
                    shape=(
                        self.max_steps,
                        self.sn_dim,
                    ),
                ),
                tf.Variable(  # Log Like
                    tf.cast(
                        tf.convert_to_tensor([[0] * self.sn_dim] * self.max_steps),
                        dtype=tf.float32,
                    ),
                    shape=(
                        self.max_steps,
                        self.sn_dim,
                    ),
                ),
                tf.Variable(  # Log Prob
                    tf.cast(
                        tf.convert_to_tensor([[0] * self.sn_dim] * self.max_steps),
                        dtype=tf.float32,
                    ),
                    shape=(
                        self.max_steps,
                        self.sn_dim,
                    ),
                ),
                tf.Variable(  # ZLatents
                    tf.cast(
                        tf.convert_to_tensor(
                            [[[0] * self.map.n_flow_latents] * self.sn_dim]
                            * self.max_steps
                        ),
                        dtype=tf.float32,
                    ),
                    shape=(self.max_steps, self.sn_dim, self.map.n_flow_latents),
                ),
            )

    def save_checkpoint(
        self,
        savepath: "Path",
        *,
        save_map: bool = False,
        save_hmc: bool = False,
    ) -> None:
        (savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)

        if save_map and save_hmc:
            ckpt = tf.train.Checkpoint(self, map=self.map, hmc=self.hmc)
            del self.hmc
            del self.map
        elif save_map:
            ckpt = tf.train.Checkpoint(self, map=self.map)
        elif save_hmc:
            ckpt = tf.train.Checkpoint(self, hmc=self.hmc)
        else:
            ckpt = tf.train.Checkpoint(self)

        ckpt.save(f"{savepath / self.ckpt_path}/")

        clear_session()

    def load_checkpoint(
        self,
        loadpath: "Path",
        *,
        load_map: bool = False,
        load_hmc: bool = False,
    ) -> None:
        if load_map:
            self.setup_map()
        if load_hmc:
            self.setup_hmc()

        self.n_chains = 1
        if load_map and load_hmc:
            ckpt = tf.train.Checkpoint(self, map=self.map, hmc=self.hmc)
        elif load_map:
            ckpt = tf.train.Checkpoint(self, map=self.map)
        elif load_hmc:
            ckpt = tf.train.Checkpoint(self, hmc=self.hmc)
        else:
            ckpt = tf.train.Checkpoint(self)

        ckpt.restore(
            tf.train.latest_checkpoint(f"{loadpath / self.ckpt_path}/")
        ).expect_partial()

        if load_hmc:
            r_hat = tfp.mcmc.potential_scale_reduction(
                self.hmc.samples, independent_chain_ndims=1, split_chains=True
            )
            self.r_hat = r_hat
            mask_sn = self.get_mask_sn()
            samples = tf.boolean_mask(self.hmc.samples, mask_sn, axis=-2)
            r_hat = tfp.mcmc.potential_scale_reduction(
                samples, independent_chain_ndims=1, split_chains=True
            )
            r_hat = tfp.stats.percentile(
                tf.where(tf.math.is_finite(r_hat), r_hat, 0), 50.0, axis=0
            )
            self.log.info(f"R-Hat: {r_hat}")
        clear_session()

    @override
    def get_config(self) -> dict[str, "Any"]:
        return {**super().get_config()}

    @override
    @classmethod
    def from_config(cls, config: dict[str, "Any"]):
        global POSTERIORMODELSTEP
        return cls(POSTERIORMODELSTEP)

    @override
    def set_seed(self, seed: int = 0) -> None:
        seed = self.seed + seed
        tf.random.set_seed(seed)

    def train_model(
        self,
        stages: "Sequence[PosteriorMAPStage]",
        *,
        savepath: "Path | None" = None,
    ) -> None:
        if savepath is not None and (savepath / self.ckpt_path).exists():
            self.log.debug(f"Loading Posterior from {savepath}")
            self.load_checkpoint(savepath, load_map=True, load_hmc=True)

            return

        n_total = sum(stage.n_chains for stage in stages) - 1
        chain = 0
        progress = tqdm(total=n_total, leave=False, dynamic_ncols=True, position=1)

        summary_writer = None
        if self.profile and savepath is not None:
            log_dir = savepath.parent / self.log_path / savepath.stem / "map"
            summary_writer = tf.summary.create_file_writer(str(log_dir))

        for stage in stages:
            for c in range(stage.n_chains):
                self.set_seed(chain)

                self.train_map(stage, c, chain, savepath=savepath)

                log_prior = -self.map.negative_log_prior
                log_like = -self.map.negative_log_like
                log_prob = -self.map.negative_log_prob

                min_log_prior = float(
                    tf.reduce_min(
                        tf.where(
                            tf.math.is_finite(log_prior),
                            log_prior,
                            np.inf * tf.ones_like(log_prior),
                        )
                    ).numpy()
                )
                mean_log_prior = float(
                    tf.reduce_sum(
                        tf.where(
                            tf.math.is_finite(log_prior),
                            log_prior,
                            tf.zeros_like(log_prior),
                        )
                    ).numpy()
                    / max(
                        tf.reduce_sum(
                            tf.where(
                                tf.math.is_finite(log_prior),
                                tf.ones_like(log_prior),
                                tf.zeros_like(log_prior),
                            )
                        ).numpy(),
                        1,
                    )
                )
                max_log_prior = float(
                    tf.reduce_max(
                        tf.where(
                            tf.math.is_finite(log_prior),
                            log_prior,
                            -np.inf * tf.ones_like(log_prior),
                        )
                    ).numpy()
                )
                min_log_like = float(
                    tf.reduce_min(
                        tf.where(
                            tf.math.is_finite(log_like),
                            log_like,
                            np.inf * tf.ones_like(log_like),
                        )
                    ).numpy()
                )
                mean_log_like = float(
                    tf.reduce_sum(
                        tf.where(
                            tf.math.is_finite(log_like),
                            log_like,
                            tf.zeros_like(log_like),
                        )
                    ).numpy()
                    / max(
                        tf.reduce_sum(
                            tf.where(
                                tf.math.is_finite(log_like),
                                tf.ones_like(log_like),
                                tf.zeros_like(log_like),
                            )
                        ).numpy(),
                        1,
                    )
                )
                max_log_like = float(
                    tf.reduce_max(
                        tf.where(
                            tf.math.is_finite(log_like),
                            log_like,
                            -np.inf * tf.ones_like(log_like),
                        )
                    ).numpy()
                )
                min_log_prob = float(
                    tf.reduce_min(
                        tf.where(
                            tf.math.is_finite(log_prob),
                            log_prob,
                            np.inf * tf.ones_like(log_prob),
                        )
                    ).numpy()
                )
                mean_log_prob = float(
                    tf.reduce_sum(
                        tf.where(
                            tf.math.is_finite(log_prob),
                            log_prob,
                            tf.zeros_like(log_prob),
                        )
                    ).numpy()
                    / max(
                        tf.reduce_sum(
                            tf.where(
                                tf.math.is_finite(log_prob),
                                tf.ones_like(log_prob),
                                tf.zeros_like(log_prob),
                            )
                        ).numpy(),
                        1,
                    )
                )
                max_log_prob = float(
                    tf.reduce_max(
                        tf.where(
                            tf.math.is_finite(log_prob),
                            log_prob,
                            -np.inf * tf.ones_like(log_prob),
                        )
                    ).numpy()
                )

                progress.set_postfix({
                    "evals_tot": self.map.num_evaluations.value().numpy(),
                    "evals_prev": self.map.num_chain_evaluations.value().numpy(),
                    "improved": tf.reduce_sum(
                        tf.ones_like(self.map.improved, dtype=tf.int32)
                        * tf.cast(self.map.improved, tf.int32)
                    ).numpy(),
                    "log_prob": (
                        f"{min_log_prob:.3E}",
                        f"{mean_log_prob:.3E}",
                        f"{max_log_prob:.3E}",
                    ),
                })
                progress.update()

                if summary_writer is not None:
                    with summary_writer.as_default():
                        converged = self.map.converged
                        tf.summary.histogram(
                            "chain_min",
                            tf.boolean_mask(self.map.chain_min, converged),
                            step=chain,
                        )
                        tf.summary.scalar(
                            "improved",
                            tf.reduce_sum(
                                tf.ones_like(self.map.improved, dtype=tf.int32)
                                * tf.cast(self.map.improved, tf.int32)
                                / self.map.converged.shape[1],
                            ),
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/min_log_prior",
                            min_log_prior,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/mean_log_prior",
                            mean_log_prior,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/max_log_prior",
                            max_log_prior,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/min_log_like",
                            min_log_like,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/mean_log_like",
                            mean_log_like,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/max_log_like",
                            max_log_like,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/min_log_prob",
                            min_log_prob,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/mean_log_prob",
                            mean_log_prob,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "map/max_log_prob",
                            max_log_prob,
                            step=chain,
                        )
                        tf.summary.histogram(
                            "u_delta_av",
                            tf.boolean_mask(self.map.u_delta_av.best, converged),
                            step=chain,
                        )
                        tf.summary.histogram(
                            "delta_av",
                            tf.boolean_mask(self.map.delta_av.best, converged),
                            step=chain,
                        )
                        tf.summary.histogram(
                            "delta_m",
                            tf.boolean_mask(self.map.delta_m.best, converged),
                            step=chain,
                        )
                        tf.summary.histogram(
                            "delta_p",
                            tf.boolean_mask(self.map.delta_p.best, converged),
                            step=chain,
                        )
                        tf.summary.histogram(
                            "bias",
                            tf.boolean_mask(self.map.bias.best, converged),
                            step=chain,
                        )
                        for i in range(self.map.n_u_latents):
                            tf.summary.histogram(
                                f"us/u_{i + 1}",
                                tf.boolean_mask(
                                    self.map.u_latents.best[..., i], converged
                                ),
                                step=chain,
                            )
                        for i in range(self.map.n_z_latents):
                            tf.summary.histogram(
                                f"zs/z_{i + 1}",
                                tf.boolean_mask(
                                    self.map.z_latents.best[..., i], converged
                                ),
                                step=chain,
                            )

                        unconstrained = self.map.unconstrain(self.map.position.best)
                        valid = tf.math.reduce_all(
                            tf.math.is_finite(unconstrained), axis=-1
                        )
                        keep = tf.math.logical_and(converged, valid)

                        j = 0
                        if not isinstance(self.map.delta_m_transform, tfb.Identity):
                            tf.summary.histogram(
                                "unconstrained/delta_m",
                                tf.boolean_mask(unconstrained[..., j : j + 1], keep),
                                step=chain,
                            )
                            j += 1
                        if not isinstance(self.map.delta_p_transform, tfb.Identity):
                            tf.summary.histogram(
                                "unconstrained/delta_p",
                                tf.boolean_mask(unconstrained[..., j : j + 1], keep),
                                step=chain,
                            )
                            j += 1
                        if not isinstance(self.map.u_delta_av_transform, tfb.Identity):
                            tf.summary.histogram(
                                "unconstrained/u_delta_av",
                                tf.boolean_mask(unconstrained[..., j : j + 1], keep),
                                step=chain,
                            )
                            j += 1
                        if not isinstance(self.map.u_latents_transform, tfb.Identity):
                            for i in range(self.map.n_u_latents):
                                tf.summary.histogram(
                                    f"unconstrained/u_{i + 1}",
                                    tf.boolean_mask(
                                        unconstrained[..., j + i : j + i + 1], keep
                                    ),
                                    step=chain,
                                )
                chain += 1
        self.log.info(f"Minimum found at chains:\n{self.map.chain_min}")
        if summary_writer is not None:
            summary_writer.close()
        progress.close()
        self.set_seed()
        self.train_hmc(savepath=savepath)

        if savepath is not None:
            self.save_checkpoint(savepath, save_map=True, save_hmc=True)
        clear_session()

    @tf.function(jit_compile=JIT_COMPILE)
    def vals_and_grads(self, position: tf.Tensor) -> tf.Tensor:
        input_position = self.map.get_position(position)
        log_prob = self(
            input_position,
            training=False,
            input_phase=self.data_time,
            input_amp=self.data_amplitude,
            input_sigma=self.data_sigma,
            mask=self.data_mask,
            sn_mask=self.sn_mask,
            spec_mask=self.spec_mask,
            wl_mask=self.wl_mask,
        )

        loss = self._loss(self.norm_prob, log_prob)
        return loss

    def lbfgs(
        self,
        position: tf.Tensor,
    ) -> "LBfgsOptimizerResults":
        return tfp.optimizer.lbfgs_minimize(
            lambda x: tfp.math.value_and_gradient(
                self.vals_and_grads,
                x,
                auto_unpack_single_arg=False,
                use_gradient_tape=True,
            ),
            initial_position=position,
            tolerance=self.tolerance,
            x_tolerance=self.x_tolerance,
            f_relative_tolerance=self.f_relative_tolerance,
            f_absolute_tolerance=self.f_absolute_tolerance,
            max_iterations=self.max_iterations,
            max_line_search_iterations=self.max_line_search_iterations,
            parallel_iterations=NPROC,
            num_correction_pairs=self.num_correction_pairs,
        )

    def train_map(
        self,
        stage: "PosteriorMAPStage",
        chain: int,
        chain_total: int,
        *,
        savepath: "Path | None" = None,
    ) -> None:
        self.setup_map()
        self.n_chains = 1
        if self.norm_prob is None:
            self.map.setup(stage, chain)

            initial_position = self.map.position.current
            initial_position = self.map.unconstrain(initial_position)
            log_prob = self(
                self.map.get_position(initial_position),
                training=False,
                input_phase=self.data_time,
                input_amp=self.data_amplitude,
                input_sigma=self.data_sigma,
                mask=self.data_mask,
                sn_mask=self.sn_mask,
                spec_mask=self.spec_mask,
                wl_mask=self.wl_mask,
            )
            num_log_prob = tf.where(
                tf.math.is_finite(log_prob),
                tf.ones_like(log_prob),
                tf.zeros_like(log_prob),
            )
            log_prob = tf.where(
                tf.math.is_finite(log_prob),
                tf.math.abs(log_prob),
                tf.zeros_like(log_prob),
            )
            mean_log_prob = tf.reduce_sum(log_prob) / tf.reduce_sum(num_log_prob)
            scale_log_prob = tf.math.log(mean_log_prob) / tf.math.log(
                tf.constant(10, dtype=mean_log_prob.dtype)
            )
            scale_log_prob_min = tf.math.floor(scale_log_prob)
            scale_log_prob_max = tf.math.ceil(scale_log_prob)
            log_prob_scale = tf.where(
                tf.math.abs(
                    tf.math.pow(10, scale_log_prob)
                    - tf.math.pow(10, scale_log_prob_min)
                )
                < tf.math.abs(
                    tf.math.pow(10, scale_log_prob)
                    - tf.math.pow(10, scale_log_prob_max)
                ),
                scale_log_prob_min,
                scale_log_prob_max,
            )
            norm_prob = tf.math.pow(10, -log_prob_scale)
            self.norm_prob = norm_prob

        if savepath is not None:
            stage_savepath = savepath / "map" / f"{stage.fname}_{chain}"
            stage_savepath.mkdir(parents=True, exist_ok=True)
            if (stage_savepath / self.ckpt_path).exists():
                self.load_checkpoint(stage_savepath, load_map=True)

                return

        self.map.setup(stage, chain)

        initial_position = self.map.position.current
        initial_position = self.map.unconstrain(initial_position)

        if stage.setup:
            objective_value = self.vals_and_grads(initial_position) / self.norm_prob
            converged = tf.math.is_finite(objective_value)
            position = self.map.constrain(initial_position)
            failed = tf.math.logical_not(converged)
            num_objective_evaluations = 1
            improved = converged
        else:
            results = self.lbfgs(initial_position)
            objective_value = results.objective_value / self.norm_prob
            converged = results.converged
            position = self.map.constrain(results.position)
            failed = results.failed
            num_objective_evaluations = results.num_objective_evaluations
            improved = tf.math.logical_and(
                (objective_value < self.map.negative_log_prob), converged
            )

        _val, _grad = tfp.math.value_and_gradient(
            self.vals_and_grads,
            initial_position,
            auto_unpack_single_arg=False,
        )

        final_position = self.map.get_position(self.map.unconstrain(position))
        log_prob, log_like, log_prior, _, _ = self(
            final_position,
            training=False,
            input_phase=self.data_time,
            input_amp=self.data_amplitude,
            input_sigma=self.data_sigma,
            mask=self.data_mask,
            sn_mask=self.sn_mask,
            spec_mask=self.spec_mask,
            wl_mask=self.wl_mask,
            additional_outputs=True,
        )

        self.map.improved.assign(improved)

        self.map.chain_min.assign(
            tf.where(
                improved,
                chain_total * tf.ones(self.sn_dim, dtype=tf.int32),
                self.map.chain_min,
            )
        )

        self.map.converged.assign(
            tf.where(
                improved,
                converged,
                self.map.converged,
            )
        )
        self.map.failed.assign(
            tf.where(
                improved,
                failed,
                self.map.failed,
            )
        )
        self.map.num_evaluations.assign_add(num_objective_evaluations)
        self.map.num_chain_evaluations.assign(num_objective_evaluations)
        self.map.negative_log_prior.assign(
            tf.where(
                improved,
                -log_prior,
                self.map.negative_log_prior,
            )
        )
        self.map.negative_log_like.assign(
            tf.where(
                improved,
                -log_like,
                self.map.negative_log_like,
            )
        )
        self.map.negative_log_prob.assign(
            tf.where(
                improved,
                -log_prob,
                self.map.negative_log_prob,
            )
        )

        ind = 0
        initial_position = []
        current_position = []
        if self.map.train_delta_m:
            initial_delta_m = self.map.position.current[..., ind : ind + 1]
            delta_m = position[..., ind : ind + 1]
            ind += 1
            initial_position.append(initial_delta_m)
            current_position.append(delta_m)
        else:
            initial_delta_m = self.map.delta_m.original
            delta_m = self.map.delta_m.current
        self.map.delta_m.initial = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            initial_delta_m,
            self.map.delta_m.initial,
        )
        self.map.delta_m.best = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            delta_m,
            self.map.delta_m.best,
        )

        if self.map.train_delta_p:
            initial_delta_p = self.map.position.current[..., ind : ind + 1]
            delta_p = position[..., ind : ind + 1]
            ind += 1
            initial_position.append(initial_delta_p)
            current_position.append(delta_p)
        else:
            initial_delta_p = self.map.delta_p.original
            delta_p = self.map.delta_p.current
        self.map.delta_p.initial = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            initial_delta_p,
            self.map.delta_p.initial,
        )
        self.map.delta_p.best = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            delta_p,
            self.map.delta_p.best,
        )

        if self.map.train_bias:
            initial_bias = self.map.position.current[..., ind : ind + 1]
            bias = position[..., ind : ind + 1]
            ind += 1
            initial_position.append(initial_bias)
            current_position.append(bias)
        else:
            initial_bias = self.map.bias.original
            bias = self.map.bias.current
        self.map.bias.initial = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            bias,
            self.map.bias.initial,
        )
        self.map.bias.best = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            bias,
            self.map.bias.best,
        )

        if self.nflow.physical_latents:
            initial_u_delta_av = self.map.position.current[..., ind : ind + 1]
            u_delta_av = position[..., ind : ind + 1]
            ind += 1
            initial_position.append(initial_u_delta_av)
            current_position.append(u_delta_av)
        else:
            initial_u_delta_av = self.map.u_delta_av.original
            u_delta_av = self.map.u_delta_av.current
        self.map.u_delta_av.initial = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            initial_u_delta_av,
            self.map.u_delta_av.initial,
        )
        self.map.u_delta_av.best = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            u_delta_av,
            self.map.u_delta_av.best,
        )

        initial_u_latents = self.map.position.current[..., ind:]
        u_latents = position[..., ind:]
        initial_position.append(initial_u_latents)
        current_position.append(u_latents)
        self.map.u_latents.initial = tf.where(
            tf.repeat(
                tf.expand_dims(improved, axis=-1),
                repeats=self.map.n_u_latents,
                axis=-1,
            ),
            initial_u_latents,
            self.map.u_latents.initial,
        )
        self.map.u_latents.best = tf.where(
            tf.repeat(
                tf.expand_dims(improved, axis=-1),
                repeats=self.map.n_u_latents,
                axis=-1,
            ),
            u_latents,
            self.map.u_latents.best,
        )

        if self.nflow.physical_latents:
            initial_us = tf.concat([initial_u_delta_av, initial_u_latents], axis=-1)
            us = tf.concat([u_delta_av, u_latents], axis=-1)
        else:
            initial_us = initial_u_latents
            us = u_latents

        initial_z_latents = self.nflow.u_to_z(initial_us, permute=True)
        z_latents = self.nflow.u_to_z(us, permute=True)

        if self.nflow.physical_latents:
            initial_delta_av = initial_z_latents[..., 0:1]
            initial_z_latents = initial_z_latents[..., 1:]
            delta_av = z_latents[..., 0:1]
            z_latents = z_latents[..., 1:]
        else:
            initial_delta_av = self.map.delta_av.current
            delta_av = self.map.delta_av.current

        self.map.z_latents.initial = tf.where(
            tf.repeat(
                tf.expand_dims(improved, axis=-1),
                repeats=self.map.n_z_latents,
                axis=-1,
            ),
            initial_z_latents,
            self.map.z_latents.initial,
        )
        self.map.z_latents.best = tf.where(
            tf.repeat(
                tf.expand_dims(improved, axis=-1),
                repeats=self.map.n_z_latents,
                axis=-1,
            ),
            z_latents,
            self.map.z_latents.best,
        )

        self.map.delta_av.initial = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            initial_delta_av,
            self.map.delta_av.initial,
        )
        self.map.delta_av.best = tf.where(
            tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
            delta_av,
            self.map.delta_av.best,
        )

        initial_position = tf.concat(initial_position, axis=-1)
        current_position = tf.concat(current_position, axis=-1)

        self.map.position.initial = tf.where(
            tf.repeat(
                tf.expand_dims(improved, axis=-1),
                repeats=initial_position.shape[-1],
                axis=-1,
            ),
            initial_position,
            self.map.position.initial,
        )
        self.map.position.best = tf.where(
            tf.repeat(
                tf.expand_dims(improved, axis=-1),
                repeats=current_position.shape[-1],
                axis=-1,
            ),
            current_position,
            self.map.position.best,
        )

        if savepath is not None:
            (stage_savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(stage_savepath, save_map=True)
        clear_session()

    # === HMC Functions ===
    def update_sample_progress(
        self,
        log_prior: tf.Tensor,
        log_like: tf.Tensor,
        log_prob: tf.Tensor,
    ) -> None:
        if self.sample_progress.n % self.summary_interval != 0:
            with tf.init_scope():
                self.sample_progress.n += 1
            return

        min_log_prior = tf.reduce_min(
            tf.where(
                tf.math.is_finite(log_prior),
                log_prior,
                np.inf * tf.ones_like(log_prior),
            )
        )
        mean_log_prior = tf.reduce_sum(
            tf.where(
                tf.math.is_finite(log_prior),
                log_prior,
                tf.zeros_like(log_prior),
            )
        ) / tf.math.maximum(
            tf.reduce_sum(
                tf.where(
                    tf.math.is_finite(log_prior),
                    tf.ones_like(log_prior),
                    tf.zeros_like(log_prior),
                )
            ),
            1,
        )
        max_log_prior = tf.reduce_max(
            tf.where(
                tf.math.is_finite(log_prior),
                log_prior,
                -np.inf * tf.ones_like(log_prior),
            )
        )
        min_log_like = tf.reduce_min(
            tf.where(
                tf.math.is_finite(log_like),
                log_like,
                np.inf * tf.ones_like(log_like),
            )
        )
        mean_log_like = tf.reduce_sum(
            tf.where(
                tf.math.is_finite(log_like),
                log_like,
                tf.zeros_like(log_like),
            )
        ) / tf.math.maximum(
            tf.reduce_sum(
                tf.where(
                    tf.math.is_finite(log_like),
                    tf.ones_like(log_like),
                    tf.zeros_like(log_like),
                )
            ),
            1,
        )
        max_log_like = tf.reduce_max(
            tf.where(
                tf.math.is_finite(log_like),
                log_like,
                -np.inf * tf.ones_like(log_like),
            )
        )
        min_log_prob = tf.reduce_min(
            tf.where(
                tf.math.is_finite(log_prob),
                log_prob,
                np.inf * tf.ones_like(log_prob),
            )
        )
        mean_log_prob = tf.reduce_sum(
            tf.where(
                tf.math.is_finite(log_prob),
                log_prob,
                tf.zeros_like(log_prob),
            )
        ) / tf.math.maximum(
            tf.reduce_sum(
                tf.where(
                    tf.math.is_finite(log_prob),
                    tf.ones_like(log_prob),
                    tf.zeros_like(log_prob),
                )
            ),
            1,
        )
        max_log_prob = tf.reduce_max(
            tf.where(
                tf.math.is_finite(log_prob),
                log_prob,
                -np.inf * tf.ones_like(log_prob),
            )
        )

        @tf.py_function(Tout=[])
        def _update(
            min_log_prior: tf.Tensor,
            mean_log_prior: tf.Tensor,
            max_log_prior: tf.Tensor,
            min_log_like: tf.Tensor,
            mean_log_like: tf.Tensor,
            max_log_like: tf.Tensor,
            min_log_prob: tf.Tensor,
            mean_log_prob: tf.Tensor,
            max_log_prob: tf.Tensor,
        ) -> None:
            self.sample_progress.set_postfix({
                "log_prob": (
                    f"{min_log_prob:.3E}",
                    f"{mean_log_prob:.3E}",
                    f"{max_log_prob:.3E}",
                ),
            })
            self.sample_progress.update()

            if self.summary_writer is not None:
                with self.summary_writer.as_default():
                    tf.summary.scalar(
                        "samples/samples/min_log_prior",
                        min_log_prior,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/mean_log_prior",
                        mean_log_prior,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/max_log_prior",
                        max_log_prior,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/min_log_like",
                        min_log_like,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/mean_log_like",
                        mean_log_like,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/max_log_like",
                        max_log_like,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/min_log_prob",
                        min_log_prob,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/mean_log_prob",
                        mean_log_prob,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "samples/samples/max_log_prob",
                        max_log_prob,
                        step=self.sample_progress.n,
                    )

        _update(
            min_log_prior,
            mean_log_prior,
            max_log_prior,
            min_log_like,
            mean_log_like,
            max_log_like,
            min_log_prob,
            mean_log_prob,
            max_log_prob,
        )

    def update_run_progress(
        self,
        log_prior: tf.Tensor,
        log_like: tf.Tensor,
        log_prob: tf.Tensor,
        pkr: "DualAveragingStepSizeAdaptationResults",
    ) -> None:
        min_log_prior = tf.reduce_min(
            tf.where(
                tf.math.is_finite(log_prior),
                log_prior,
                np.inf * tf.ones_like(log_prior),
            )
        )
        mean_log_prior = tf.reduce_sum(
            tf.where(
                tf.math.is_finite(log_prior),
                log_prior,
                tf.zeros_like(log_prior),
            )
        ) / tf.math.maximum(
            tf.reduce_sum(
                tf.where(
                    tf.math.is_finite(log_prior),
                    tf.ones_like(log_prior),
                    tf.zeros_like(log_prior),
                )
            ),
            1,
        )
        max_log_prior = tf.reduce_max(
            tf.where(
                tf.math.is_finite(log_prior),
                log_prior,
                -np.inf * tf.ones_like(log_prior),
            )
        )
        min_log_like = tf.reduce_min(
            tf.where(
                tf.math.is_finite(log_like),
                log_like,
                np.inf * tf.ones_like(log_like),
            )
        )
        mean_log_like = tf.reduce_sum(
            tf.where(
                tf.math.is_finite(log_like),
                log_like,
                tf.zeros_like(log_like),
            )
        ) / tf.math.maximum(
            tf.reduce_sum(
                tf.where(
                    tf.math.is_finite(log_like),
                    tf.ones_like(log_like),
                    tf.zeros_like(log_like),
                )
            ),
            1,
        )
        max_log_like = tf.reduce_max(
            tf.where(
                tf.math.is_finite(log_like),
                log_like,
                -np.inf * tf.ones_like(log_like),
            )
        )
        min_log_prob = tf.reduce_min(
            tf.where(
                tf.math.is_finite(log_prob),
                log_prob,
                np.inf * tf.ones_like(log_prob),
            )
        )
        mean_log_prob = tf.reduce_sum(
            tf.where(
                tf.math.is_finite(log_prob),
                log_prob,
                tf.zeros_like(log_prob),
            )
        ) / tf.math.maximum(
            tf.reduce_sum(
                tf.where(
                    tf.math.is_finite(log_prob),
                    tf.ones_like(log_prob),
                    tf.zeros_like(log_prob),
                )
            ),
            1,
        )
        max_log_prob = tf.reduce_max(
            tf.where(
                tf.math.is_finite(log_prob),
                log_prob,
                -np.inf * tf.ones_like(log_prob),
            )
        )

        log_accept_ratio = pkr.inner_results.log_accept_ratio

        accept_ratio = (
            tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.0)))
            * tf.cast(self.map.sn_dim, tf.float32)
            / tf.math.count_nonzero(self.map.converged, dtype=tf.float32)
        )

        step_size = pkr.inner_results.step_size

        @tf.py_function(Tout=[])
        def _update(
            min_log_prior: tf.Tensor,
            mean_log_prior: tf.Tensor,
            max_log_prior: tf.Tensor,
            min_log_like: tf.Tensor,
            mean_log_like: tf.Tensor,
            max_log_like: tf.Tensor,
            min_log_prob: tf.Tensor,
            mean_log_prob: tf.Tensor,
            max_log_prob: tf.Tensor,
            accept_ratio: tf.Tensor,
            step_size: tf.Tensor,
        ) -> None:
            self.run_progress.set_postfix({
                "log_prob": (
                    f"{min_log_prob:.3E}",
                    f"{mean_log_prob:.3E}",
                    f"{max_log_prob:.3E}",
                ),
                "accept_ratio": f"{accept_ratio:.2%}",
            })
            self.run_progress.update()
            if self.summary_writer is not None and self.sample_progress.n > 1:
                with self.summary_writer.as_default():
                    tf.summary.scalar(
                        "run/min_log_prior",
                        min_log_prior,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/mean_log_prior",
                        mean_log_prior,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/max_log_prior",
                        max_log_prior,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/min_log_like",
                        min_log_like,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/mean_log_like",
                        mean_log_like,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/max_log_like",
                        max_log_like,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/min_log_prob",
                        min_log_prob,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/mean_log_prob",
                        mean_log_prob,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "run/max_log_prob",
                        max_log_prob,
                        step=self.run_progress.n,
                    )
                    tf.summary.scalar(
                        "accept_ratio", accept_ratio, step=self.run_progress.n
                    )
                    self.step_samples = self.sample_progress.n

        _update(
            min_log_prior,
            mean_log_prior,
            max_log_prior,
            min_log_like,
            mean_log_like,
            max_log_like,
            min_log_prob,
            mean_log_prob,
            max_log_prob,
            accept_ratio,
            step_size,
        )

    def unnormalized_posterior_log_prob(
        self,
        *pos: tf.Tensor,
        additional_outputs: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        input_position = self.map.get_position(tf.convert_to_tensor(pos)[0, ...])
        log_prob, log_like, log_prior, _, _ = self(
            input_position,
            training=False,
            input_phase=self.data_time,
            input_amp=self.data_amplitude,
            input_sigma=self.data_sigma,
            mask=self.data_mask,
            sn_mask=self.sn_mask,
            spec_mask=self.spec_mask,
            wl_mask=self.wl_mask,
            additional_outputs=True,
        )

        if self.summary_writer is not None:
            self.update_sample_progress(log_prior, log_like, log_prob)

        if additional_outputs:
            return log_prior, log_like, log_prob
        return log_prob

    def trace_fn(
        self, _state: tf.Tensor, pkr: "DualAveragingStepSizeAdaptationResults"
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # new_step_size: lets us check whether step-size adaptation has
        #   plateaued by the end of burn-in (diagnoses n_burnin_steps).
        # reach_max_depth: whether this step's NUTS trajectory was cut short
        #   by max_tree_depth rather than terminating on a U-turn/divergence
        #   (diagnoses n_leapfrog).
        # is_accepted / has_divergence: cheap standard health checks that
        #   help interpret the other two (e.g. a saturating chain with a
        #   low accept rate points at step size/geometry, not just a small
        #   n_leapfrog).
        new_step_size = pkr.new_step_size
        reach_max_depth = pkr.inner_results.reach_max_depth
        is_accepted = pkr.inner_results.is_accepted
        has_divergence = pkr.inner_results.has_divergence

        if self.summary_writer is not None:
            self.write_hmc_step_summary(
                new_step_size, reach_max_depth, is_accepted, has_divergence
            )

        return (new_step_size, reach_max_depth, is_accepted, has_divergence)

    def write_hmc_step_summary(
        self,
        step_size: tf.Tensor,
        reach_max_depth: tf.Tensor,
        is_accepted: tf.Tensor,
        has_divergence: tf.Tensor,
    ) -> None:
        """Stream one NUTS step's diagnostics to TensorBoard as they happen.

        Called from `trace_fn` during both burn-in and the run phase, on a
        single counter (`self.hmc_step`) shared across both, so the
        `hmc/*` charts in TensorBoard show one continuous timeline -- you
        can watch `hmc/step_size` plateau (or not) live during burn-in
        without waiting for sampling to finish.
        """
        mean_step_size = tf.reduce_mean(step_size)
        saturation_rate = tf.reduce_mean(tf.cast(reach_max_depth, tf.float32))
        accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
        divergence_rate = tf.reduce_mean(tf.cast(has_divergence, tf.float32))

        @tf.py_function(Tout=[])
        def _write(
            mean_step_size: tf.Tensor,
            saturation_rate: tf.Tensor,
            accept_rate: tf.Tensor,
            divergence_rate: tf.Tensor,
        ) -> None:
            with self.summary_writer.as_default():
                tf.summary.scalar("hmc/step_size", mean_step_size, step=self.hmc_step)
                tf.summary.scalar(
                    "hmc/tree_depth_saturation_rate",
                    saturation_rate,
                    step=self.hmc_step,
                )
                tf.summary.scalar("hmc/accept_rate", accept_rate, step=self.hmc_step)
                tf.summary.scalar(
                    "hmc/divergence_rate", divergence_rate, step=self.hmc_step
                )
            self.hmc_step += 1

        _write(mean_step_size, saturation_rate, accept_rate, divergence_rate)

    @tf.function(jit_compile=JIT_COMPILE)
    def sample_chain_burnin(
        self,
        position: tf.Tensor,
        kernel: tfp.mcmc.TransitionKernel,
    ) -> tuple[tf.Tensor, tf.Tensor, "DualAveragingStepSizeAdaptationResults"]:
        # Run as its own sample_chain call, and as its own eager Python
        # call (see train_hmc), for two reasons: (1) tfp.mcmc.sample_chain
        # never traces its num_burnin_steps steps, so this is the only way
        # to see the step-size adaptation trajectory; (2) returning to
        # Python before the run phase starts lets train_hmc print burn-in
        # diagnostics -- and lets you Ctrl+C -- before committing to the
        # (usually much longer) run phase.
        burnin = tfp.mcmc.sample_chain(
            num_results=self.n_burnin_steps,
            current_state=position,
            kernel=kernel,
            trace_fn=self.trace_fn,
            return_final_kernel_results=True,
            name="burnin",
            parallel_iterations=NPROC,
        )
        burnin_step_size, *_ = burnin.trace

        # burnin.all_states is empty when n_burnin_steps == 0 (nothing to
        # index) -- fall back to the original position, matching what
        # num_burnin_steps=0 does in a single-call sample_chain.
        final_state = position if self.n_burnin_steps == 0 else burnin.all_states[-1]

        return final_state, burnin_step_size, burnin.final_kernel_results

    @tf.function(jit_compile=JIT_COMPILE)
    def sample_chain_run(
        self,
        position: tf.Tensor,
        kernel: tfp.mcmc.TransitionKernel,
        previous_kernel_results: "DualAveragingStepSizeAdaptationResults",
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        run = tfp.mcmc.sample_chain(
            num_results=self.n_run_steps,
            current_state=position,
            previous_kernel_results=previous_kernel_results,
            kernel=kernel,
            num_steps_between_results=self.n_thinning,
            trace_fn=self.trace_fn,
            name="run",
            parallel_iterations=NPROC,
        )
        _run_step_size, reach_max_depth, is_accepted, has_divergence = run.trace

        samples = self.map.constrain(run.all_states)

        return samples, reach_max_depth, is_accepted, has_divergence

    def report_burnin_diagnostics(self, burnin_step_size: tf.Tensor) -> None:
        """Log the step-size adaptation health, to be called right after burn-in.

        Called between `sample_chain_burnin` and `sample_chain_run` in
        `train_hmc` -- deliberately *before* the run phase starts, so a
        too-short `n_burnin_steps` shows up here and the run (often the
        most expensive part) can be cancelled instead of run to completion
        first.
        """
        step_size_drift_warn_threshold = 0.05

        # Compare the step-size trace's last two burn-in windows: if it's
        # still moving this late in burn-in, adaptation hasn't converged
        # and n_burnin_steps/n_adaption_steps should grow. Too short a
        # burn-in to form two windows means there isn't enough signal to
        # judge convergence either way.
        min_steps_for_plateau_check = 20
        if self.n_burnin_steps < min_steps_for_plateau_check:
            self.log.debug(
                "Skipping burn-in step-size plateau check: "
                f"n_burnin_steps={self.n_burnin_steps} is too small to assess."
            )
            return

        window = max(1, self.n_burnin_steps // 10)
        penultimate = tf.reduce_mean(burnin_step_size[-2 * window : -window], axis=0)
        final = tf.reduce_mean(burnin_step_size[-window:], axis=0)
        relative_change = float(
            tf.reduce_mean(
                tf.abs(final - penultimate) / tf.maximum(tf.abs(penultimate), 1e-12)
            )
        )
        self.log.info(
            f"HMC step-size adaptation: {relative_change:.2%} relative "
            f"change between the last two {window}-step windows of the "
            f"{self.n_burnin_steps}-step burn-in "
            f"(n_adaption_steps={self.n_adaption_steps})."
        )
        if relative_change > step_size_drift_warn_threshold:
            self.log.warning(
                f"Step size is still changing by {relative_change:.2%} "
                "near the end of burn-in -- adaptation may not have "
                f"converged within n_burnin_steps={self.n_burnin_steps}. "
                "Consider increasing n_burnin_steps/n_adaption_steps "
                "(Ctrl+C now to cancel before the run phase)."
            )

    def report_run_diagnostics(
        self,
        reach_max_depth: tf.Tensor,
        is_accepted: tf.Tensor,
        has_divergence: tf.Tensor,
    ) -> None:
        """Log NUTS health diagnostics for tuning `n_leapfrog`."""
        saturation_warn_threshold = 0.01

        saturation_rate = float(tf.reduce_mean(tf.cast(reach_max_depth, tf.float32)))
        accept_rate = float(tf.reduce_mean(tf.cast(is_accepted, tf.float32)))
        divergence_rate = float(tf.reduce_mean(tf.cast(has_divergence, tf.float32)))
        self.log.info(
            f"HMC run-phase diagnostics: {saturation_rate:.2%} of steps hit "
            f"max_tree_depth={self.max_tree_depth} (n_leapfrog={self.n_leapfrog}), "
            f"{accept_rate:.2%} accept rate, {divergence_rate:.2%} divergence rate."
        )
        if saturation_rate > saturation_warn_threshold:
            self.log.warning(
                f"{saturation_rate:.2%} of run steps hit max_tree_depth="
                f"{self.max_tree_depth} before reaching a U-turn -- trajectories "
                "are being truncated by the cap rather than terminating "
                "naturally. Consider increasing n_leapfrog."
            )

    def train_hmc(
        self,
        *,
        savepath: "Path | None" = None,
    ) -> None:
        self.setup_hmc()
        self.n_chains = self.n_walkers
        if savepath is not None:
            hmc_savepath = savepath / "hmc"
            hmc_savepath.mkdir(parents=True, exist_ok=True)
            if (hmc_savepath / self.ckpt_path).exists():
                self.log.debug(f"Loading HMC from {hmc_savepath}")
                self.load_checkpoint(hmc_savepath, load_hmc=True)

                samples = self.hmc.samples

                vars(self)["hmc"] = PosteriorHMCValue(
                    tf.Variable(samples),
                    tf.Variable(self.hmc.log_prior),
                    tf.Variable(self.hmc.log_like),
                    tf.Variable(self.hmc.log_prob),
                    tf.Variable(self.hmc.zs),
                )
                return
        self.log.debug("Running HMC")

        initial_position = self.map.position.best

        if self.step_size_scale == "shift":
            original_position = self.map.position.original
            step_size = tf.where(
                self.map.converged[..., None],
                tf.abs(original_position - initial_position),
                tf.zeros_like(initial_position),
            )[0, ...]
            self.log.debug(f"Step Size: {tf.reduce_mean(step_size, axis=0)}")
        else:
            step_size_init = self.step_size
            step_size_std = tf.math.reduce_std(
                tf.boolean_mask(initial_position, self.map.converged), axis=0
            )
            step_size_std = tf.where(
                tf.math.is_finite(step_size_std), step_size_std, step_size_init
            )
            step_size_init = tf.where(
                tf.math.is_finite(step_size_init), step_size_init, step_size_std
            )

            if self.step_size_scale == "min":
                step_size_inner = tf.minimum(step_size_init, step_size_std)
            elif self.step_size_scale == "max":
                step_size_inner = tf.maximum(step_size_init, step_size_std)
            step_size_inner = self.map.unconstrain(step_size_inner)

            self.log.debug(f"Step Size: {step_size_inner}")

            # step_size_inner currently has shape (n_params)
            # We have n_walkers * n_sn chains
            # We want each sn to have their own step size, across their walkers
            # So we need step_size to have shape (n_sn, n_params)

            # shape = (1, n_params)
            step_size = tf.expand_dims(step_size_inner, axis=0)

            # shape = (n_sn, n_params)
            step_size = tf.repeat(
                step_size,
                repeats=initial_position.shape[-2],
                axis=0,
            )

        initial_position = self.map.unconstrain(initial_position)
        initial_position = tf.repeat(initial_position, repeats=self.n_walkers, axis=0)

        # Give each walker its own copy of the (per-SN) step size, so
        # DualAveragingStepSizeAdaptation adapts a fully independent step
        # size per (walker, SN) pair instead of pooling every walker's
        # accept ratio into one shared per-SN step size, where a single
        # stuck/divergent walker could otherwise drag the step size down
        # for every walker sampling that SN.
        step_size = tf.repeat(
            tf.expand_dims(step_size, axis=0), repeats=self.n_walkers, axis=0
        )

        if self.n_walkers > 1:
            # Start each walker from an independently-jittered point
            # around the MAP estimate (scaled by the per-SN step size)
            # rather than every walker replicating the exact same
            # starting position. Identical starts mean extra walkers are
            # pseudo-replicates of a single trajectory rather than
            # independent explorations of the posterior.
            initial_position += tf.random.normal(
                tf.shape(initial_position), stddev=step_size
            )

        self.log.debug(
            f"With {self.n_burnin_steps} [{self.max_tree_depth * self.n_burnin_steps}] burn-in steps [samples] ({self.n_adaption_steps} [{self.max_tree_depth * self.n_adaption_steps}] of which will be used for step-size adaption) and {self.n_run_steps} [{self.max_tree_depth * self.n_run_steps}] run steps [samples], a maximum of {(self.n_burnin_steps + self.n_run_steps)} [{self.max_samples}] steps [samples] will be drawn per-walker for a max leapfrog depth of {self.max_tree_depth}. Across all {self.n_walkers} walkers a maximum of {self.n_walkers * (self.n_burnin_steps + self.n_run_steps)} [{self.n_walkers * self.max_samples}] steps [samples] will be drawn."
        )

        self.summary_writer = None
        self.summary_interval = 1
        if self.profile and savepath is not None:
            log_dir = savepath.parent / self.log_path / savepath.stem / "hmc"
            self.summary_writer = tf.summary.create_file_writer(
                str(log_dir),
            )
        self.sample_progress = tqdm(
            total=self.max_samples,
            leave=False,
            dynamic_ncols=True,
            position=0,
        )
        # self.run_progress = tqdm(
        #     total=self.n_run_steps + 1,
        #     leave=False,
        #     dynamic_ncols=True,
        #     position=1,
        # )
        # self.run_progress.set_description("run")
        self.step_samples = 0
        self.hmc_step = 0

        # Masked SNe have -inf log-prob everywhere, but TFP's NUTS
        # tree-doubling loop is a single tf.while_loop shared across the
        # whole (n_walkers, sn_dim) batch -- it keeps running (re-evaluating
        # the full decoder/photometry pass for every SN) until the *last*
        # chain in the batch stops. Left in the batch, masked SNe would ride
        # along for the full trajectory length of whichever real SN is
        # slowest to U-turn, wasting the max_tree_depth budget. Restricting
        # to valid SNe for the scope of sampling avoids that; results are
        # scattered back to full sn_dim afterwards.
        with self._restricted_to_valid_sn() as valid_indices:
            position = (
                initial_position
                if valid_indices is None
                else tf.gather(initial_position, valid_indices, axis=-2)
            )
            step = (
                step_size
                if valid_indices is None
                else tf.gather(step_size, valid_indices, axis=-2)
            )

            sampler = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.unnormalized_posterior_log_prob,
                step_size=step,
                max_tree_depth=self.n_leapfrog,
                parallel_iterations=NPROC,
            )

            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=sampler,
                num_adaptation_steps=self.n_adaption_steps,
                target_accept_prob=self.target_acceptance_rate,
                reduce_fn=tfp.math.reduce_log_harmonic_mean_exp,
            )

            burnin_result = self.sample_chain_burnin(position, kernel)
            # Reported (and, if self.summary_writer is set, already streamed to
            # TensorBoard live throughout burn-in above) before the run phase
            # starts, so a too-short n_burnin_steps is visible here -- cancel
            # now rather than waiting through the whole run phase to find out.
            self.report_burnin_diagnostics(burnin_result[1])

            run_result = self.sample_chain_run(
                burnin_result[0], kernel, burnin_result[2]
            )
            samples = run_result[0]
            self.report_run_diagnostics(*run_result[1:])

            del kernel
            del sampler

            self.step_samples = None
            self.sample_progress.close()
            self.sample_progress = None
            if self.summary_writer is not None:
                self.summary_writer.close()
            self.summary_writer = None

            clear_session()

            self.log.debug(
                "Calculating prior, likelihood, and probability across all samples"
            )

            @tf.function
            def _fn(state):
                return self.unnormalized_posterior_log_prob(
                    state, additional_outputs=True
                )

            log_prior, log_like, log_prob = tf.map_fn(
                _fn,
                tf.convert_to_tensor(samples),
                swap_memory=True,
                fn_output_signature=(tf.float32, tf.float32, tf.float32),
            )

            self.log.debug("Calculating z-latents")

            @tf.function
            def _fn(u):
                return self.map.nflow.u_to_z(u, permute=True)

            us = samples[..., -self.map.nflow.n_flow_latents :]
            zs = tf.map_fn(_fn, us, swap_memory=True)

        if valid_indices is not None:
            samples = self._scatter_sn(
                samples, axis=-2, indices=valid_indices, fill=float("nan")
            )
            log_prior = self._scatter_sn(
                log_prior, axis=-1, indices=valid_indices, fill=float("-inf")
            )
            log_like = self._scatter_sn(
                log_like, axis=-1, indices=valid_indices, fill=float("-inf")
            )
            log_prob = self._scatter_sn(
                log_prob, axis=-1, indices=valid_indices, fill=float("-inf")
            )
            zs = self._scatter_sn(zs, axis=-2, indices=valid_indices, fill=float("nan"))

        log_prior = log_prior.numpy().reshape((
            log_prior.shape[0] * log_prior.shape[1],
            *log_prior.shape[2:],
        ))
        log_like = log_like.numpy().reshape((
            log_like.shape[0] * log_like.shape[1],
            *log_like.shape[2:],
        ))
        log_prob = log_prob.numpy().reshape((
            log_prob.shape[0] * log_prob.shape[1],
            *log_prob.shape[2:],
        ))
        zs = zs.numpy().reshape((
            zs.shape[0] * zs.shape[1],
            *zs.shape[2:],
        ))

        vars(self)["hmc"] = PosteriorHMCValue(
            tf.Variable(samples),
            tf.Variable(log_prior),
            tf.Variable(log_like),
            tf.Variable(log_prob),
            tf.Variable(zs),
        )

        if savepath is not None:
            self.save_checkpoint(hmc_savepath, save_hmc=True)
        clear_session()
