# Copyright 2025 Patrick Armstrong
from supaernova.steps.pae.tf.photometry import (
    photometry,
    photometry_amplitude_setup,
    photometry_sigma_setup,
)
import os
import json
import types
import shutil
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


def _finite_reduce(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    finite = tf.math.is_finite(x)
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    count = tf.math.maximum(tf.reduce_sum(tf.where(finite, ones, zeros)), 1)
    mean = tf.reduce_sum(tf.where(finite, x, zeros)) / count
    minimum = tf.reduce_min(tf.where(finite, x, np.inf * ones))
    maximum = tf.reduce_max(tf.where(finite, x, -np.inf * ones))
    return minimum, mean, maximum


def _reduce_trailing(x: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.cast(x, tf.float32), axis=list(range(1, len(x.shape))))


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

        self.n_leapfrog_adaption: int = self.options.n_leapfrog_adaption
        self.n_leapfrog_burnin: int = self.options.n_leapfrog_burnin
        self.n_leapfrog_run: int = self.options.n_leapfrog_run
        max_leapfrog_adaption = (2**self.n_leapfrog_adaption) - 1
        max_leapfrog_burnin = (2**self.n_leapfrog_burnin) - 1
        max_leapfrog_run = (2**self.n_leapfrog_run) - 1

        self.n_run_steps: int = self.options.n_run_steps

        self.n_burnin_steps: int = self.options.n_burnin_steps
        if isinstance(self.n_burnin_steps, float):
            self.n_burnin_steps = int(self.n_run_steps * self.n_burnin_steps)

        # `n_adaption_steps` is now its own phase, run before burn-in, so a
        # float is interpreted the same way as `n_burnin_steps` -- a fraction
        # of `n_run_steps` -- rather than a fraction *of* burn-in.
        self.n_adaption_steps: int = self.options.n_adaption_steps
        if isinstance(self.n_adaption_steps, float):
            self.n_adaption_steps = int(self.n_run_steps * self.n_adaption_steps)

        self.max_samples: int = int(
            max_leapfrog_adaption * self.n_adaption_steps
            + max_leapfrog_burnin * self.n_burnin_steps
            + max_leapfrog_run * self.n_run_steps
        )

        self.max_tree_depth_adaption: int = (2**self.n_leapfrog_adaption) - 1
        self.max_tree_depth_burnin: int = (2**self.n_leapfrog_burnin) - 1
        self.max_tree_depth_run: int = (2**self.n_leapfrog_run) - 1

        self.max_steps: int = self.n_run_steps * self.n_walkers

        self.n_thinning: int = self.options.n_thinning
        self.target_acceptance_rate: float = self.options.target_acceptance_rate

        # Chunk sizes for the Python-level sample_chain loop. `None`/`0` ->
        # one chunk per phase (behaviour matches a single sample_chain call);
        # a float is a fraction of that phase's step count.
        self.n_run_chunk_steps: int = self._resolve_chunk_steps(
            self.options.n_chunk_steps, self.n_run_steps
        )
        self.n_burnin_chunk_steps: int = self._resolve_chunk_steps(
            self.options.n_burnin_chunk_steps
            if self.options.n_burnin_chunk_steps is not None
            else self.options.n_chunk_steps,
            self.n_burnin_steps,
        )
        self.n_adaption_chunk_steps: int = self._resolve_chunk_steps(
            self.options.n_adaption_chunk_steps
            if self.options.n_adaption_chunk_steps is not None
            else self.options.n_chunk_steps,
            self.n_adaption_steps,
        )
        self._checkpoint_hmc: bool = self.options.checkpoint_hmc

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

        posterior_mask = (
            input_mask
            & input_sn_mask
            & input_spec_mask
            & input_wl_mask
            & (input_spectra_mask | input_phot_mask)
        )

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

    def get_mask_spec(self) -> tf.Tensor:
        """Compute the per-(sn, spec) mask.

        Returns:
            Boolean tensor of shape `(sn_dim, spec_dim)`, True where a
            spec/phot row has at least one unmasked wl point.
        """
        mask = (self.data_mask & self.sn_mask & self.spec_mask & self.wl_mask) & (
            self.data_spectra_mask | self.data_phot_mask
        )
        return tf.math.reduce_any(mask, axis=-1)

    def get_mask_sn(self) -> tf.Tensor:
        """Compute the per-SN mask.

        Returns:
            Boolean tensor of shape `(sn_dim,)`, True where a SN has at
            least one unmasked spec/wl point.
        """
        return tf.math.reduce_any(self.get_mask_spec(), axis=-1)

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

    @contextlib.contextmanager
    def _repacked_to_valid_spec(self) -> "Iterator[None]":
        """Temporarily shrink every spec_dim-indexed attribute to its valid rows.

        The decoder/photometry pass runs densely over the full (spec_dim,
        wl_dim) grid for every SN, but spec_dim is padded to the max
        number of observations across all SNe -- most SNe use only a
        fraction of it. Every layer in the decoder is a Dense op that
        broadcasts unmodified over spec_dim (axis 1), so repacking valid
        rows to the front and trimming to the largest per-SN valid count
        needed by the (already sn_dim-restricted) batch reduces decoder
        and photometry FLOPs proportionally, with no effect on the
        result: the likelihood reduction collapses spec_dim away before
        producing HMC's per-SN log-prob, so nothing needs to be scattered
        back afterwards. Callers must be nested inside a scope where
        `self.spec_dim` matches the current attributes' shape.

        Yields:
            None; spec_dim-indexed attributes are mutated in place for the
            scope of the context and restored on exit.
        """
        new_spec_dim = self.spec_dim
        if self.sn_dim:
            n_valid = tf.math.count_nonzero(self.get_mask_spec(), axis=-1)
            new_spec_dim = max(int(tf.reduce_max(n_valid)), 1)

        if new_spec_dim >= self.spec_dim:
            yield
        else:
            # Valid rows first (stable keeps relative order); trim to new_spec_dim.
            perm = tf.argsort(
                tf.cast(self.get_mask_spec(), tf.int32),
                axis=-1,
                direction="DESCENDING",
                stable=True,
            )[:, :new_spec_dim]

            spec_attrs = (
                "data_time",
                "data_amplitude",
                "data_sigma",
                "data_wavelength",
                "data_throughput",
                "data_effective_wavelength",
                "data_spectra_mask",
                "data_phot_mask",
                "data_mask",
                "spec_mask",
                "wl_mask",
                "sigma_recon",
            )
            originals = {attr: getattr(self, attr) for attr in spec_attrs}
            original_cached_amp = self.cached_amp
            original_cached_sigma = self.cached_sigma
            original_spec_dim = self.spec_dim

            try:
                for attr in spec_attrs:
                    setattr(
                        self,
                        attr,
                        tf.gather(originals[attr], perm, axis=1, batch_dims=1),
                    )
                self.cached_amp = tuple(
                    tf.gather(t, perm, axis=1, batch_dims=1)
                    for t in original_cached_amp
                )
                self.cached_sigma = tuple(
                    tf.gather(t, perm, axis=1, batch_dims=1)
                    for t in original_cached_sigma
                )
                self.spec_dim = new_spec_dim

                yield
            finally:
                for attr in spec_attrs:
                    setattr(self, attr, originals[attr])
                self.cached_amp = original_cached_amp
                self.cached_sigma = original_cached_sigma
                self.spec_dim = original_spec_dim

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
        clear: bool = True,
    ) -> None:
        # ``clear`` drops every traced ConcreteFunction (+ gc). Skip it when
        # checkpointing inside a hot loop (e.g. per MAP chain) so the shared
        # ``vals_and_grads`` / sampler graphs survive to the next iteration;
        # the caller clears once when the whole phase is done.
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

        if clear:
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
            log_prob = self._forward(self.map.get_position(initial_position))
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

        final_position = self.map.get_position(self.map.unconstrain(position))
        log_prob, log_like, log_prior, _, _ = self._forward(
            final_position, additional_outputs=True
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
            # Keep the traced graphs alive across MAP chains -- train_model
            # clears once when every stage/chain is done.
            self.save_checkpoint(stage_savepath, save_map=True, clear=False)

    # === HMC Functions ===

    @staticmethod
    def _resolve_chunk_steps(raw: float, total: int) -> int:
        if isinstance(raw, float):
            return max(1, min(total, int(total * raw)))
        return max(1, min(total, int(raw)))

    def _forward(
        self,
        position: tf.Tensor,
        *,
        additional_outputs: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run ``call`` with the model's own (subset-restricted) data tensors.

        A single place to thread ``self.data_*`` / masks into a forward pass;
        ``ks.Model.__call__`` already graph-compiles ``call`` keyed on input
        specs, so this is a DRY helper rather than an extra ``tf.function``.

        Args:
            position: Parameter vector(s) to evaluate, ``[..., n_params]``.
            additional_outputs: Forwarded to ``call``.

        Returns:
            ``call``'s output: ``log_probability`` alone, or the
            ``(log_probability, log_likelihood, log_prior, synth_amp,
            synth_sigma)`` tuple when ``additional_outputs`` is set.
        """
        return self(
            position,
            training=False,
            input_phase=self.data_time,
            input_amp=self.data_amplitude,
            input_sigma=self.data_sigma,
            mask=self.data_mask,
            sn_mask=self.sn_mask,
            spec_mask=self.spec_mask,
            wl_mask=self.wl_mask,
            additional_outputs=additional_outputs,
        )

    def unnormalized_posterior_log_prob(
        self,
        *pos: tf.Tensor,
        additional_outputs: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        input_position = self.map.get_position(tf.convert_to_tensor(pos)[0, ...])
        log_prob, log_like, log_prior, _, _ = self._forward(
            input_position, additional_outputs=True
        )

        if additional_outputs:
            return log_prior, log_like, log_prob
        return log_prob

    def _recompute_sample_diagnostics(
        self, samples: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Log-prior/like/prob and z-latents for every recorded HMC sample.

        ``samples`` is ``[n_run_steps, n_walkers, sn, n_params]``. ``call`` and
        ``get_position`` broadcast over a single leading axis, so the
        (steps, walkers) axes are folded into one batch axis and the run is
        evaluated a group of steps at a time -- fewer, larger forward passes
        than the old sequential ``tf.map_fn`` over the ``n_run_steps`` axis,
        while each pass stays close to the per-step sampling footprint so
        peak decoder memory does not blow up.

        The group size starts at ``_recompute_step_batch`` recorded steps and
        halves on OOM, down to a single step (which matches what the sampler
        itself held).

        Args:
            samples: Constrained sample tensor from ``_sample_hmc``.

        Returns:
            ``(log_prior, log_like, log_prob, zs)``; the log terms are
            ``[n_run_steps, n_walkers, sn]`` and ``zs`` is
            ``[n_run_steps, n_walkers, sn, n_flow_latents]``.

        Raises:
            tf.errors.ResourceExhaustedError: If a single recorded step
                (``n_walkers`` rows) still does not fit in device memory.
        """
        self.log.debug(
            "Calculating prior, likelihood, and probability across all samples"
        )
        n_steps, n_walkers = int(samples.shape[0]), int(samples.shape[1])
        flat = tf.reshape(samples, (n_steps * n_walkers, *samples.shape[2:]))

        # One recorded step is `n_walkers` rows -- exactly the batch the
        # sampler's target evaluated per leapfrog step. Group a few steps per
        # pass; back off to a single step if the GPU can't hold the group.
        step_batch = int(getattr(self, "_recompute_step_batch", 8))
        while True:
            try:
                parts = self._recompute_pass(flat, max(1, step_batch) * n_walkers)
                break
            except tf.errors.ResourceExhaustedError:
                if step_batch <= 1:
                    raise
                step_batch = max(1, step_batch // 2)
                self.log.warning(
                    f"OOM in sample-diagnostics recompute; retrying with "
                    f"step_batch={step_batch}"
                )
                clear_session()

        return tuple(  # type: ignore[return-value]
            tf.reshape(p, (n_steps, n_walkers, *p.shape[1:])) for p in parts
        )

    def _recompute_pass(
        self, flat: tf.Tensor, rows: int
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """One attempt at the batched recompute: ``rows`` sample-rows per pass.

        Args:
            flat: Samples flattened to ``[n_steps * n_walkers, sn, n_params]``.
            rows: Number of sample-rows to evaluate per forward pass.

        Returns:
            ``(log_prior, log_like, log_prob, zs)`` concatenated over the
            leading (flattened) axis.
        """
        n_flow = self.map.nflow.n_flow_latents
        stores: list[list[tf.Tensor]] = [[], [], [], []]
        for start in range(0, int(flat.shape[0]), rows):
            batch = flat[start : start + rows]
            lp, ll, lprob = self.unnormalized_posterior_log_prob(
                batch, additional_outputs=True
            )
            zsb = self.map.nflow.u_to_z(batch[..., -n_flow:], permute=True)
            for store, value in zip(stores, (lp, ll, lprob, zsb), strict=True):
                store.append(value)
        return tuple(tf.concat(s, axis=0) for s in stores)  # type: ignore[return-value]

    def trace_fn(
        self, _state: tf.Tensor, pkr: "DualAveragingStepSizeAdaptationResults"
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
    ]:
        lp_min, lp_mean, lp_max = _finite_reduce(pkr.inner_results.target_log_prob)
        return (
            pkr.new_step_size,
            pkr.inner_results.reach_max_depth,
            pkr.inner_results.is_accepted,
            pkr.inner_results.has_divergence,
            lp_min,
            lp_mean,
            lp_max,
        )

    def _emit_sample_summaries(self, trace: "Sequence[tf.Tensor]") -> None:
        n = int(trace[0].shape[0])
        if n == 0:
            return

        # tag -> per-step [n] series. `hmc/*` are means over the chain batch;
        # `samples/samples/*` come straight from trace_fn's _finite_reduce.
        series = {
            "hmc/step_size": _reduce_trailing(trace[0]),
            "hmc/tree_depth_saturation_rate": _reduce_trailing(trace[1]),
            "hmc/accept_rate": _reduce_trailing(trace[2]),
            "hmc/divergence_rate": _reduce_trailing(trace[3]),
            "samples/samples/min_log_prob": trace[4],
            "samples/samples/mean_log_prob": trace[5],
            "samples/samples/max_log_prob": trace[6],
        }

        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                for i in range(n):
                    for tag, values in series.items():
                        base = (
                            self.hmc_step
                            if tag.startswith("hmc/")
                            else self.sample_step
                        )
                        tf.summary.scalar(tag, values[i], step=base + i)
            self.summary_writer.flush()

        self.hmc_step += n
        self.sample_step += n

        if self.sample_progress is not None:
            self.sample_progress.update(n)
            self.sample_progress.set_postfix({
                "log_prob": f"{float(series['samples/samples/mean_log_prob'][-1]):.3E}",
                "accept": f"{float(series['hmc/accept_rate'][-1]):.2%}",
            })

    @tf.function(jit_compile=JIT_COMPILE)
    def _sample_chain_chunk(
        self,
        state: tf.Tensor,
        kernel: tfp.mcmc.TransitionKernel,
        previous_kernel_results: "DualAveragingStepSizeAdaptationResults | None",
        num_results: int,
        num_burnin_steps: int,
        num_steps_between_results: int,
        seed: tf.Tensor,
    ) -> tuple[
        tf.Tensor, "Sequence[tf.Tensor]", "DualAveragingStepSizeAdaptationResults"
    ]:
        chunk = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=state,
            previous_kernel_results=previous_kernel_results,
            kernel=kernel,
            num_steps_between_results=num_steps_between_results,
            trace_fn=self.trace_fn,
            return_final_kernel_results=True,
            seed=seed,
            parallel_iterations=NPROC,
        )
        return chunk.all_states, chunk.trace, chunk.final_kernel_results

    def _run_sampling_phase(
        self,
        *,
        phase: "Literal['adaption', 'burnin', 'run']",
        kernel: tfp.mcmc.TransitionKernel,
        state: tf.Tensor,
        total_steps: int,
        chunk_steps: int,
        thinning: int,
        hmc_savepath: "Path | None",
        pkr: "DualAveragingStepSizeAdaptationResults | None" = None,
        steps_done: int = 0,
        run_states: "list[tf.Tensor] | None" = None,
    ) -> tuple[
        tf.Tensor,
        "DualAveragingStepSizeAdaptationResults | None",
        "tuple[tf.Tensor, ...] | None",
        "tf.Tensor | None",
    ]:
        traces: list[Sequence[tf.Tensor]] = []
        run_states = list(run_states) if run_states is not None else []
        keep_states = phase == "run"

        # The Python chunk loop only earns its keep when it has something to do
        # between chunks: write a resume checkpoint, or emit per-chunk
        # TensorBoard summaries. With neither, run the whole phase in one
        # sample_chain call -- no chunk-boundary retraces, no per-chunk
        # concat/emit overhead.
        if not self._checkpoint_hmc and getattr(self, "summary_writer", None) is None:
            chunk_steps = max(chunk_steps, total_steps)

        # Fresh run phase (not a resume that pre-seeded run_states): drop any
        # per-chunk state shards left behind by an earlier interrupted run.
        if keep_states and hmc_savepath is not None and not run_states:
            for stale in (hmc_savepath / "run").glob("states_*.npy"):
                stale.unlink()

        while steps_done < total_steps:
            n = min(chunk_steps, total_steps - steps_done)
            # No leading gap before the very first recorded step of a phase
            # (matches a single-call sample_chain); re-insert the thinning
            # gap at every later chunk boundary.
            lead_gap = 0 if steps_done == 0 else thinning
            self._hmc_seed, chunk_seed = tfp.random.split_seed(self._hmc_seed, n=2)

            all_states, trace, pkr = self._sample_chain_chunk(
                state, kernel, pkr, n, lead_gap, thinning, chunk_seed
            )
            state = all_states[-1]
            steps_done += n
            traces.append(trace)
            if keep_states:
                run_states.append(all_states)

            self._emit_sample_summaries(trace)

            if self._checkpoint_hmc and hmc_savepath is not None:
                self._save_hmc_run_state(
                    phase=phase,
                    state=state,
                    pkr=pkr,
                    steps_done=steps_done,
                    latest_states=all_states if keep_states else None,
                    hmc_savepath=hmc_savepath,
                )

        full_trace = None
        if traces:
            full_trace = tuple(
                tf.concat([t[k] for t in traces], axis=0) for k in range(len(traces[0]))
            )
        all_states = tf.concat(run_states, axis=0) if run_states else None
        return state, pkr, full_trace, all_states

    # === HMC checkpoint / restore ===
    #
    # An in-progress HMC run is checkpointed between chunks (when
    # ``checkpoint_hmc`` is set) as a small ``hmc/run/run_state.npz`` +
    # ``run_state.json`` pair, so an interrupted run resumes without repeating
    # completed burn-in / sampling. The pair is deleted once the final
    # ``PosteriorHMCValue`` checkpoint is written. tfp kernel results are a
    # nested namedtuple of tensors: ``tf.nest.flatten`` on save,
    # ``tf.nest.pack_sequence_as`` against a fresh ``bootstrap_results``
    # template on restore.

    def _save_hmc_run_state(
        self,
        *,
        phase: str,
        state: tf.Tensor,
        pkr: "DualAveragingStepSizeAdaptationResults",
        steps_done: int,
        latest_states: "tf.Tensor | None",
        hmc_savepath: "Path",
    ) -> None:
        """Atomically write the intermediate HMC run state after one chunk.

        The run phase's recorded states are written as append-only per-chunk
        shards (``states_<steps_done>.npy``) rather than re-serialising the
        whole accumulated ``run_states`` list every chunk (which was
        O(n_chunks^2) in host copies and disk writes). ``_read_hmc_run_state``
        globs and concatenates the shards on resume.
        """
        run_dir = hmc_savepath / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        leaves = tf.nest.flatten(pkr)
        arrays: dict[str, "npt.NDArray[Any]"] = {
            "state": np.asarray(state),
            "hmc_seed": np.asarray(self._hmc_seed),
        }
        for i, leaf in enumerate(leaves):
            arrays[f"pkr_{i}"] = np.asarray(leaf)

        if phase == "run" and latest_states is not None:
            shard = run_dir / f"states_{steps_done:09d}.npy"
            tmp_shard = shard.parent / f"{shard.name}.tmp"
            with tmp_shard.open("wb") as handle:
                np.save(handle, np.asarray(latest_states))
            tmp_shard.replace(shard)

        meta = {
            "phase": phase,
            "steps_done": int(steps_done),
            "hmc_step": int(self.hmc_step),
            "sample_step": int(self.sample_step),
            "n_pkr_leaves": len(leaves),
            "n_adaption_steps": int(self.n_adaption_steps),
            "n_burnin_steps": int(self.n_burnin_steps),
            "n_run_steps": int(self.n_run_steps),
            "n_walkers": int(self.n_walkers),
        }

        tmp_npz = run_dir / "run_state.npz.tmp"
        with tmp_npz.open("wb") as handle:
            np.savez(handle, **arrays)
        tmp_npz.replace(run_dir / "run_state.npz")
        (run_dir / "run_state.json").write_text(json.dumps(meta))

    def _read_hmc_run_state(
        self, hmc_savepath: "Path | None"
    ) -> "dict[str, Any] | None":
        """Load an intermediate HMC run-state checkpoint if a usable one exists.

        Args:
            hmc_savepath: ``<savepath>/hmc`` or None.

        Returns:
            ``{"meta": <json dict>, "data": <NpzFile>}``, or None (start
            fresh) when checkpointing is off, no savepath is given, no
            checkpoint is present, or the checkpoint was written for a
            different sampler configuration.
        """
        if not self._checkpoint_hmc or hmc_savepath is None:
            return None
        run_dir = hmc_savepath / "run"
        npz_path = run_dir / "run_state.npz"
        meta_path = run_dir / "run_state.json"
        if not (npz_path.exists() and meta_path.exists()):
            return None

        meta = json.loads(meta_path.read_text())
        if (
            meta.get("n_adaption_steps") != int(self.n_adaption_steps)
            or meta.get("n_burnin_steps") != int(self.n_burnin_steps)
            or meta.get("n_run_steps") != int(self.n_run_steps)
            or meta.get("n_walkers") != int(self.n_walkers)
        ):
            self.log.warning(
                f"Ignoring HMC run-state checkpoint at {run_dir}: it was "
                "written for a different sampler configuration."
            )
            return None

        totals = {
            "adaption": self.n_adaption_steps,
            "burnin": self.n_burnin_steps,
            "run": self.n_run_steps,
        }
        total = totals[meta["phase"]]
        self.log.info(
            f"Resuming HMC from {run_dir} "
            f"(phase={meta['phase']}, {meta['steps_done']}/{total} steps done)."
        )

        data = np.load(npz_path)
        run_states = None
        shards = sorted(run_dir.glob("states_*.npy"))
        if shards:
            run_states = np.concatenate([np.load(shard) for shard in shards], axis=0)
        elif "run_states" in data.files:
            # Legacy checkpoint: recorded states lived inside run_state.npz
            # rather than as append-only shards.
            run_states = data["run_states"]

        return {"meta": meta, "data": data, "run_states": run_states}

    def _hmc_resume_plan(
        self,
        hmc_savepath: "Path | None",
        position: tf.Tensor,
        kernel_adaption: tfp.mcmc.TransitionKernel,
        kernel_burnin: tfp.mcmc.TransitionKernel,
        kernel_run: tfp.mcmc.TransitionKernel,
    ) -> "dict[str, Any]":
        """Resolve where to (re)start HMC, folding in the resume checkpoint.

        Advances `self.hmc_step` / `self.sample_step` / `self._hmc_seed` and
        the progress bar to match a restored checkpoint, then reports the
        per-phase starting parameters.

        Args:
            hmc_savepath: ``<savepath>/hmc`` or None.
            position: the fresh (unconstrained, restricted-SN) start position.
            kernel_adaption: the adaption kernel (for the resume template).
            kernel_burnin: the burn-in kernel (for the resume template).
            kernel_run: the run kernel (for the resume template).

        Returns:
            ``skip_adaption`` / ``skip_burnin`` (bool), ``adaption_state`` /
            ``adaption_pkr`` / ``adaption_steps_done`` (start of the adaption
            phase), ``burnin_state`` / ``burnin_pkr`` / ``burnin_steps_done``
            (start of the burn-in phase, or -- when ``skip_burnin`` -- its
            hand-off to the run phase), and ``run_steps_done`` /
            ``run_states_seed`` (resume state for the run phase).
        """
        plan: dict[str, Any] = {
            "skip_adaption": False,
            "skip_burnin": False,
            "adaption_state": position,
            "adaption_pkr": None,
            "adaption_steps_done": 0,
            "burnin_state": position,
            "burnin_pkr": None,
            "burnin_steps_done": 0,
            "run_steps_done": 0,
            "run_states_seed": None,
        }

        resume = self._read_hmc_run_state(hmc_savepath)
        if resume is None:
            return plan

        meta, data = resume["meta"], resume["data"]
        self.hmc_step = int(meta["hmc_step"])
        self.sample_step = int(meta["sample_step"])
        self._hmc_seed = tf.constant(data["hmc_seed"], dtype=tf.int32)
        state = tf.constant(data["state"])

        phase_order = ("adaption", "burnin", "run")
        totals = {
            "adaption": self.n_adaption_steps,
            "burnin": self.n_burnin_steps,
            "run": self.n_run_steps,
        }
        done_phase_offset = sum(
            totals[p] for p in phase_order[: phase_order.index(meta["phase"])]
        )
        self.sample_progress.update(meta["steps_done"] + done_phase_offset)

        if meta["phase"] == "run":
            plan["skip_adaption"] = True
            plan["skip_burnin"] = True
            plan["burnin_state"] = state
            plan["burnin_pkr"] = self._restore_pkr(
                kernel_run, state, data, meta["n_pkr_leaves"]
            )
            plan["run_steps_done"] = int(meta["steps_done"])
            if resume["run_states"] is not None:
                plan["run_states_seed"] = [tf.constant(resume["run_states"])]
        elif meta["phase"] == "burnin":
            plan["skip_adaption"] = True
            plan["burnin_state"] = state
            plan["burnin_pkr"] = self._restore_pkr(
                kernel_burnin, state, data, meta["n_pkr_leaves"]
            )
            plan["burnin_steps_done"] = int(meta["steps_done"])
        else:  # "adaption"
            plan["adaption_state"] = state
            plan["adaption_pkr"] = self._restore_pkr(
                kernel_adaption, state, data, meta["n_pkr_leaves"]
            )
            plan["adaption_steps_done"] = int(meta["steps_done"])
        return plan

    @staticmethod
    def _restore_pkr(
        kernel: tfp.mcmc.TransitionKernel,
        state: tf.Tensor,
        data: "Any",
        n_leaves: int,
    ) -> "DualAveragingStepSizeAdaptationResults":
        """Repack a flattened kernel-results checkpoint onto a fresh template.

        Args:
            kernel: the kernel whose ``bootstrap_results`` supplies the
                nested structure to pack into.
            state: current chain state (unconstrained).
            data: the loaded ``NpzFile`` holding ``pkr_<i>`` leaves.
            n_leaves: leaf count recorded when the checkpoint was written.

        Returns:
            The repacked kernel results, ready to pass as
            ``previous_kernel_results``.

        Raises:
            RuntimeError: if the checkpoint's leaf count does not match the
                current kernel (incompatible sampler configuration).
        """
        template = kernel.bootstrap_results(state)
        flat_template = tf.nest.flatten(template)
        if len(flat_template) != n_leaves:
            msg = (
                "HMC run-state checkpoint is incompatible with the current "
                f"kernel ({n_leaves} saved kernel-result leaves vs "
                f"{len(flat_template)} expected). Delete the 'hmc/run' "
                "directory to restart the HMC run from scratch."
            )
            raise RuntimeError(msg)
        leaves = [
            tf.constant(data[f"pkr_{i}"], dtype=leaf.dtype)
            for i, leaf in enumerate(flat_template)
        ]
        return tf.nest.pack_sequence_as(template, leaves)

    def report_adaption_diagnostics(self, adaption_step_size: tf.Tensor) -> None:
        """Log the step-size adaptation health, to be called right after adaption.

        Called from `_sample_hmc` between the adaption and burn-in phases --
        deliberately *before* burn-in/run start, so a too-short
        `n_adaption_steps` shows up here and the (often much more expensive)
        burn-in and run phases can be cancelled instead of run to completion
        first.
        """
        step_size_drift_warn_threshold = 0.05

        # Compare the step-size trace's last two adaption windows: if it's
        # still moving this late in adaption, it hasn't converged and
        # n_adaption_steps should grow. Too short an adaption phase to form
        # two windows means there isn't enough signal to judge convergence
        # either way.
        min_steps_for_plateau_check = 20
        if self.n_adaption_steps < min_steps_for_plateau_check:
            self.log.debug(
                "Skipping adaption step-size plateau check: "
                f"n_adaption_steps={self.n_adaption_steps} is too small to assess."
            )
            return

        window = max(1, self.n_adaption_steps // 10)
        penultimate = tf.reduce_mean(adaption_step_size[-2 * window : -window], axis=0)
        final = tf.reduce_mean(adaption_step_size[-window:], axis=0)
        relative_change = float(
            tf.reduce_mean(
                tf.abs(final - penultimate) / tf.maximum(tf.abs(penultimate), 1e-12)
            )
        )
        self.log.info(
            f"HMC step-size adaptation: {relative_change:.2%} relative "
            f"change between the last two {window}-step windows of the "
            f"{self.n_adaption_steps}-step adaption phase."
        )
        if relative_change > step_size_drift_warn_threshold:
            self.log.warning(
                f"Step size is still changing by {relative_change:.2%} "
                "near the end of adaption -- it may not have converged "
                f"within n_adaption_steps={self.n_adaption_steps}. Consider "
                "increasing n_adaption_steps (Ctrl+C now to cancel before "
                "the burn-in/run phases)."
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
            f"max_tree_depth={self.max_tree_depth_run} (n_leapfrog_run={self.n_leapfrog_run}), "
            f"{accept_rate:.2%} accept rate, {divergence_rate:.2%} divergence rate."
        )
        if saturation_rate > saturation_warn_threshold:
            self.log.warning(
                f"{saturation_rate:.2%} of run steps hit max_tree_depth="
                f"{self.max_tree_depth_run} before reaching a U-turn -- trajectories "
                "are being truncated by the cap rather than terminating "
                "naturally. Consider increasing n_leapfrog_run."
            )

    def _hmc_initial_position(self) -> tuple[tf.Tensor, tf.Tensor]:
        """Build the per-(walker, SN) start position and step size for HMC.

        Both come from the MAP result: the start position is the MAP best
        (unconstrained, replicated per walker, jittered by the step size when
        ``n_walkers > 1``); the step size is either the MAP shift
        (``step_size_scale == "shift"``) or a min/max blend of a configured
        scale and the per-parameter posterior spread.

        Returns:
            ``(initial_position, step_size)``, both shaped
            ``[n_walkers, sn_dim, n_params]``.
        """
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

            # step_size_inner has shape (n_params); we want (n_sn, n_params)
            # so each SN has its own step size across its walkers.
            step_size = tf.repeat(
                tf.expand_dims(step_size_inner, axis=0),
                repeats=initial_position.shape[-2],
                axis=0,
            )

        initial_position = self.map.unconstrain(initial_position)
        initial_position = tf.repeat(initial_position, repeats=self.n_walkers, axis=0)

        # Give each walker its own copy of the (per-SN) step size, so
        # DualAveragingStepSizeAdaptation adapts a fully independent step size
        # per (walker, SN) pair instead of pooling every walker's accept ratio
        # into one shared per-SN step size, where a single stuck/divergent
        # walker could otherwise drag the step size down for every walker
        # sampling that SN.
        step_size = tf.repeat(
            tf.expand_dims(step_size, axis=0), repeats=self.n_walkers, axis=0
        )

        # if self.n_walkers > 1:
        #     # Start each walker from an independently-jittered point around the
        #     # MAP estimate (scaled by the per-SN step size) rather than every
        #     # walker replicating the exact same starting position -- identical
        #     # starts make extra walkers pseudo-replicates of a single
        #     # trajectory rather than independent explorations of the posterior.
        #     initial_position += tf.random.normal(
        #         tf.shape(initial_position), stddev=(1 - 1 / self.n_walkers) * step_size
        #     )

        return initial_position, step_size

    def _sample_hmc(
        self,
        position: tf.Tensor,
        step: tf.Tensor,
        hmc_savepath: "Path | None",
    ) -> tf.Tensor:
        sampler_adaption = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.unnormalized_posterior_log_prob,
            step_size=step,
            max_tree_depth=self.n_leapfrog_adaption,
            parallel_iterations=NPROC,
        )
        kernel_adaption = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=sampler_adaption,
            num_adaptation_steps=self.n_adaption_steps,
            target_accept_prob=self.target_acceptance_rate,
        )
        # Burn-in and run both sample with the step size the adaption phase
        # converged to -- num_adaptation_steps=0 freezes it immediately, so
        # DualAveragingStepSizeAdaptation just forwards the inner NUTS
        # kernel's results without further adapting. Burn-in exists purely to
        # let the chain equilibrate under its own (typically larger than run,
        # smaller than adaption) max_tree_depth before the cheap run phase
        # starts. A separate kernel/sampler is used per phase --
        # max_tree_depth is baked into NoUTurnSampler at construction, but
        # previous_kernel_results (step size, adaptation step counter, etc.)
        # carries over unaffected, since none of that state depends on
        # max_tree_depth.
        sampler_run = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.unnormalized_posterior_log_prob,
            step_size=step,
            max_tree_depth=self.n_leapfrog_run,
            parallel_iterations=NPROC,
        )
        kernel_run = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=sampler_run,
            num_adaptation_steps=0,
            target_accept_prob=self.target_acceptance_rate,
        )
        # Reuse the run kernel for burn-in when the tree depth matches -- one
        # fewer distinct kernel object means one fewer full retrace of the
        # sample_chain graph. (The pkr state carried across phases doesn't
        # depend on max_tree_depth, so this is only a construction-time
        # dedup, not a behaviour change.)
        if self.n_leapfrog_burnin == self.n_leapfrog_run:
            kernel_burnin = kernel_run
        else:
            sampler_burnin = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.unnormalized_posterior_log_prob,
                step_size=step,
                max_tree_depth=self.n_leapfrog_burnin,
                parallel_iterations=NPROC,
            )
            kernel_burnin = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=sampler_burnin,
                num_adaptation_steps=0,
                target_accept_prob=self.target_acceptance_rate,
            )

        plan = self._hmc_resume_plan(
            hmc_savepath, position, kernel_adaption, kernel_burnin, kernel_run
        )

        if plan["skip_adaption"]:
            adaption_state = plan["adaption_state"]
            adaption_pkr = plan["adaption_pkr"]
        else:
            adaption_state, adaption_pkr, adaption_trace, _ = self._run_sampling_phase(
                phase="adaption",
                kernel=kernel_adaption,
                state=plan["adaption_state"],
                total_steps=self.n_adaption_steps,
                chunk_steps=self.n_adaption_chunk_steps,
                thinning=0,
                hmc_savepath=hmc_savepath,
                pkr=plan["adaption_pkr"],
                steps_done=plan["adaption_steps_done"],
            )
            # None when the adaption loop ran no chunks (already complete on
            # resume): fall back to the restored results.
            adaption_pkr = (
                adaption_pkr if adaption_pkr is not None else plan["adaption_pkr"]
            )
            # Reported before burn-in/run start, so a too-short
            # n_adaption_steps is visible here -- Ctrl+C now rather than
            # waiting through the whole (usually much longer) burn-in/run.
            # Skipped when resuming part-way through adaption (the trace is
            # then partial).
            full_adaption = adaption_trace is not None and (
                int(adaption_trace[0].shape[0]) >= self.n_adaption_steps
            )
            if full_adaption:
                self.report_adaption_diagnostics(adaption_trace[0])

        if plan["skip_burnin"]:
            burnin_state = plan["burnin_state"]
            burnin_pkr = plan["burnin_pkr"]
        else:
            burnin_start_state = (
                plan["burnin_state"] if plan["skip_adaption"] else adaption_state
            )
            burnin_start_pkr = (
                plan["burnin_pkr"] if plan["skip_adaption"] else adaption_pkr
            )
            burnin_state, burnin_pkr, _, _ = self._run_sampling_phase(
                phase="burnin",
                kernel=kernel_burnin,
                state=burnin_start_state,
                total_steps=self.n_burnin_steps,
                chunk_steps=self.n_burnin_chunk_steps,
                thinning=0,
                hmc_savepath=hmc_savepath,
                pkr=burnin_start_pkr,
                steps_done=plan["burnin_steps_done"],
            )
            # None when the burn-in loop ran no chunks (n_burnin_steps == 0,
            # or already complete on resume): fall back to whatever it was
            # handed.
            burnin_pkr = burnin_pkr if burnin_pkr is not None else burnin_start_pkr

        _, _, run_trace, all_states = self._run_sampling_phase(
            phase="run",
            kernel=kernel_run,
            state=burnin_state,
            total_steps=self.n_run_steps,
            chunk_steps=self.n_run_chunk_steps,
            thinning=self.n_thinning,
            hmc_savepath=hmc_savepath,
            pkr=burnin_pkr,
            steps_done=plan["run_steps_done"],
            run_states=plan["run_states_seed"],
        )
        # run_trace = (step_size, reach_max_depth, is_accepted,
        #              has_divergence, lp_min, lp_mean, lp_max)
        # run_trace is None when the run phase ran no chunks -- i.e. it was
        # already complete on resume (all recorded states came from the
        # checkpoint); there are then no fresh diagnostics to report.
        if run_trace is not None:
            self.report_run_diagnostics(run_trace[1], run_trace[2], run_trace[3])
        return self.map.constrain(all_states)

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

        initial_position, step_size = self._hmc_initial_position()

        n_total_steps = self.n_adaption_steps + self.n_burnin_steps + self.n_run_steps
        self.log.debug(
            f"With {self.n_adaption_steps} [{self.max_tree_depth_adaption * self.n_adaption_steps}] step-size adaption steps [samples] (max leapfrog depth {self.max_tree_depth_adaption}), {self.n_burnin_steps} [{self.max_tree_depth_burnin * self.n_burnin_steps}] burn-in steps [samples] (max leapfrog depth {self.max_tree_depth_burnin}) and {self.n_run_steps} [{self.max_tree_depth_run * self.n_run_steps}] run steps [samples] (max leapfrog depth {self.max_tree_depth_run}), a maximum of {n_total_steps} [{self.max_samples}] steps [samples] will be drawn per-walker. Across all {self.n_walkers} walkers a maximum of {self.n_walkers * n_total_steps} [{self.n_walkers * self.max_samples}] steps [samples] will be drawn."
        )

        self.summary_writer = None
        self.sample_progress = None
        if self.profile and savepath is not None:
            log_dir = savepath.parent / self.log_path / savepath.stem / "hmc"
            self.summary_writer = tf.summary.create_file_writer(
                str(log_dir),
            )
            self.sample_progress = tqdm(
                total=n_total_steps,
                leave=False,
                dynamic_ncols=True,
                position=0,
            )
        # hmc_step keys the `hmc/*` TensorBoard charts (shared across burn-in
        # and the run so they form one continuous timeline); sample_step keys
        # `samples/samples/*_log_prob`. Both advance in eager Python from
        # _emit_sample_summaries, once per recorded step.
        self.hmc_step = 0
        self.sample_step = 0
        # Every HMC run is seeded (stateless) so chunked runs are
        # deterministic and a resumed run reproduces an uninterrupted one.
        self._hmc_seed = tfp.random.sanitize_seed(self.seed)

        hmc_ckpt_path = hmc_savepath if savepath is not None else None

        # Masked SNe have -inf log-prob everywhere, but TFP's NUTS
        # tree-doubling loop is a single tf.while_loop shared across the
        # whole (n_walkers, sn_dim) batch -- it keeps running (re-evaluating
        # the full decoder/photometry pass for every SN) until the *last*
        # chain in the batch stops. Left in the batch, masked SNe would ride
        # along for the full trajectory length of whichever real SN is
        # slowest to U-turn, wasting the max_tree_depth budget. Restricting
        # to valid SNe for the scope of sampling avoids that; results are
        # scattered back to full sn_dim afterwards.
        # spec_dim is padded to the max observation count across all SNe;
        # repacking valid rows to the front and trimming to what the
        # (already sn_dim-restricted) batch actually needs cuts decoder
        # and photometry FLOPs proportionally on every HMC step. spec_dim
        # is fully reduced away before log-prob/samples/zs are produced,
        # so no scatter-back is needed for it.
        with (
            self._restricted_to_valid_sn() as valid_indices,
            self._repacked_to_valid_spec(),
        ):
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

            samples = self._sample_hmc(position, step, hmc_ckpt_path)

            if self.sample_progress is not None:
                self.sample_progress.close()
            self.sample_progress = None
            if self.summary_writer is not None:
                self.summary_writer.close()
            self.summary_writer = None

            # Free the (large) NUTS trajectory graph before the recompute
            # passes below build their own, smaller trace.
            clear_session()

            log_prior, log_like, log_prob, zs = self._recompute_sample_diagnostics(
                samples
            )

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
            # The full HMC result is now checkpointed -- drop the
            # intermediate per-chunk run state.
            shutil.rmtree(hmc_savepath / "run", ignore_errors=True)
        clear_session()
