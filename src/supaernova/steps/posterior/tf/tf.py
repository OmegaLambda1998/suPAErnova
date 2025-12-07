# Copyright 2025 Patrick Armstrong
import os
from typing import TYPE_CHECKING, override

from tqdm import tqdm
import numpy as np

from supaernova._tf import HUGE, ks, tf, tfd, tfp
from supaernova.utils.tf import db, pp

from .hmc import PosteriorHMCValue
from .map import PosteriorMap

if TYPE_CHECKING:
    from typing import Any, Literal
    from logging import Logger
    from pathlib import Path
    from collections.abc import Sequence

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

NPROC = os.cpu_count()

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

        self.debug: bool = config.config.debug or self.options.debug
        self.profile: bool = self.options.profile

        self.data: LazySNPAEData = getattr(config, f"{self.subset}_data")
        self.data_time: npt.NDArray[float] = self.data.time
        self.data_amplitude: npt.NDArray[float] = self.data.amplitude
        self.data_sigma: npt.NDArray[float] = self.data.sigma
        self.data.clear()

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

        # MAP Variables
        self.map: PosteriorMap
        vars(self)["map"] = PosteriorMap(self)
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
                (self.n_burnin_steps + self.n_run_steps) * self.n_adaption_steps
            )

        self.max_samples: int = int(
            max_leapfrog * (self.n_burnin_steps + self.n_run_steps)
        )

        self.n_thinning: int = self.options.n_thinning
        self.target_acceptance_rate: float = self.options.target_acceptance_rate

        self.hmc: PosteriorHMCValue
        vars(self)["hmc"] = PosteriorHMCValue(
            tf.Variable(  # Samples
                tf.convert_to_tensor(
                    [[[0] * self.map.n_pae_latents] * self.sn_dim] * self.n_run_steps,
                    dtype=tf.float32,
                ),
                shape=(self.n_run_steps, self.sn_dim, self.map.n_pae_latents),
            ),
            tf.Variable(  # Step Sizes Final
                tf.convert_to_tensor(
                    [[[0] * self.map.n_pae_latents] * self.sn_dim] * self.n_run_steps,
                    dtype=tf.float32,
                ),
                shape=(self.n_run_steps, self.sn_dim, self.map.n_pae_latents),
            ),
            tf.Variable(  # Is Accepted
                tf.convert_to_tensor(
                    [[False] * self.sn_dim] * self.n_run_steps, dtype=tf.bool
                ),
                shape=(
                    self.n_run_steps,
                    self.sn_dim,
                ),
            ),
            tf.Variable(  # Log Prior
                tf.convert_to_tensor(
                    [[0] * self.sn_dim] * self.n_run_steps, dtype=tf.float32
                ),
                shape=(
                    self.n_run_steps,
                    self.sn_dim,
                ),
            ),
            tf.Variable(  # Log Like
                tf.convert_to_tensor(
                    [[0] * self.sn_dim] * self.n_run_steps, dtype=tf.float32
                ),
                shape=(
                    self.n_run_steps,
                    self.sn_dim,
                ),
            ),
            tf.Variable(  # Log Prob
                tf.convert_to_tensor(
                    [[0] * self.sn_dim] * self.n_run_steps, dtype=tf.float32
                ),
                shape=(
                    self.n_run_steps,
                    self.sn_dim,
                ),
            ),
            tf.Variable(  # UDeltaAv
                tf.convert_to_tensor(
                    [[[0] * 1] * self.sn_dim] * self.n_run_steps, dtype=tf.float32
                ),
                shape=(self.n_run_steps, self.sn_dim, 1),
            ),
            tf.Variable(  # ULatents
                tf.convert_to_tensor(
                    [[[0] * self.map.n_u_latents] * self.sn_dim] * self.n_run_steps,
                    dtype=tf.float32,
                ),
                shape=(self.n_run_steps, self.sn_dim, self.map.n_u_latents),
            ),
            tf.Variable(  # DeltaAv
                tf.convert_to_tensor(
                    [[[0] * 1] * self.sn_dim] * self.n_run_steps, dtype=tf.float32
                ),
                shape=(self.n_run_steps, self.sn_dim, 1),
            ),
            tf.Variable(  # ZLatents
                tf.convert_to_tensor(
                    [[[0] * self.map.n_z_latents] * self.sn_dim] * self.n_run_steps,
                    dtype=tf.float32,
                ),
                shape=(self.n_run_steps, self.sn_dim, self.map.n_z_latents),
            ),
            tf.Variable(  # DeltaM
                tf.convert_to_tensor(
                    [[[0] * 1] * self.sn_dim] * self.n_run_steps, dtype=tf.float32
                ),
                shape=(self.n_run_steps, self.sn_dim, 1),
            ),
            tf.Variable(  # DeltaP
                tf.convert_to_tensor(
                    [[[0] * 1] * self.sn_dim] * self.n_run_steps, dtype=tf.float32
                ),
                shape=(self.n_run_steps, self.sn_dim, 1),
            ),
        )

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
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        testing: bool | None = None,
        additional_outputs: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        training = False if training is None else training
        testing = False if testing is None else testing

        # === Inputs ===
        if input_phase is None:
            input_phase = self.data_time
        if input_amp is None:
            input_amp = self.data_amplitude
        if input_sigma is None:
            input_sigma = self.data_sigma

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
            delta_av = zs[..., 0:1]
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

        # Create synthetic spectra from z-latents
        decoder_inputs = tf.concat((input_phase, zs), axis=-1)
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

        phase = input_phase
        if self.map.train_delta_p:  # and not self.pae.physical_latents:
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

        synth_sigma = tf.sqrt(
            ((synth_amp * sigma_recon) ** 2) + (input_sigma * input_sigma)
        )

        # Set missing values to 1 for all times
        synth_sigma = tf.where(posterior_mask, synth_sigma, tf.ones_like(synth_sigma))

        # Set missing values to 0 for all times
        synth_amp = tf.where(posterior_mask, synth_amp, tf.zeros_like(synth_amp))

        likelihood = tfd.Normal(loc=synth_amp, scale=synth_sigma)

        # Set missing values to 0 for all times
        amp = tf.where(posterior_mask, input_amp, tf.zeros_like(input_amp))

        log_likelihood_wl = likelihood.log_prob(amp)

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

        log_likelihood_spec = log_likelihood_spec_num

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

        log_likelihood = log_likelihood_num / log_likelihood_sum

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
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if input_phase is not None:
            input_phase = tf.convert_to_tensor(input_phase, dtype=tf.float32)
        if input_amp is not None:
            input_amp = tf.convert_to_tensor(input_amp, dtype=tf.float32)
        if input_sigma is not None:
            input_sigma = tf.convert_to_tensor(input_sigma, dtype=tf.float32)
        if mask is not None:
            mask = tf.convert_to_tensor(mask, dtype=tf.bool)
        if sn_mask is not None:
            sn_mask = tf.convert_to_tensor(sn_mask, dtype=tf.bool)
        if spec_mask is not None:
            spec_mask = tf.convert_to_tensor(spec_mask, dtype=tf.bool)
        if wl_mask is not None:
            wl_mask = tf.convert_to_tensor(wl_mask, dtype=tf.bool)
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
        elif save_map:
            ckpt = tf.train.Checkpoint(self, map=self.map)
        elif save_hmc:
            ckpt = tf.train.Checkpoint(self, hmc=self.hmc)
        else:
            ckpt = tf.train.Checkpoint(self)

        ckpt.save(f"{savepath / self.ckpt_path}/")

    def load_checkpoint(
        self,
        loadpath: "Path",
        *,
        load_map: bool = False,
        load_hmc: bool = False,
    ) -> None:
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
            ess = tfp.mcmc.effective_sample_size(
                self.hmc.samples, filter_beyond_positive_pairs=True
            )
            self.log.debug(f"Effective Sample Size: {ess}")
            valid_ess = tf.math.is_finite(ess)
            mean_ess = tf.reduce_sum(
                tf.where(valid_ess, ess, tf.zeros_like(ess)), axis=0
            ) / tf.math.count_nonzero(
                tf.math.reduce_all(valid_ess, axis=-1), dtype=ess.dtype
            )
            self.log.info(f"ESS: {mean_ess} ({mean_ess / self.n_run_steps})")
            self.log.info(
                f"R-Hat: {tfp.mcmc.potential_scale_reduction(self.hmc.samples, split_chains=True)}"
            )

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
        progress = tqdm(
            total=n_total, leave=False, dynamic_ncols=True, smoothing=1, position=1
        )

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
                            "converged",
                            tf.reduce_sum(
                                tf.ones_like(self.map.converged, dtype=tf.int32)
                                * tf.cast(self.map.converged, tf.int32)
                            )
                            / self.map.converged.shape[0],
                            step=chain,
                        )
                        tf.summary.scalar(
                            "failed",
                            tf.reduce_sum(
                                tf.ones_like(self.map.failed, dtype=tf.int32)
                                * tf.cast(self.map.failed, tf.int32)
                            )
                            / self.map.failed.shape[0],
                            step=chain,
                        )
                        tf.summary.scalar(
                            "improved",
                            tf.reduce_sum(
                                tf.ones_like(self.map.improved, dtype=tf.int32)
                                * tf.cast(self.map.improved, tf.int32)
                                / self.map.converged.shape[0],
                            ),
                            step=chain,
                        )
                        tf.summary.scalar(
                            "num_evaluations", self.map.num_evaluations, step=chain
                        )
                        tf.summary.scalar(
                            "num_chain_evaluations",
                            self.map.num_chain_evaluations,
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
                                f"u_{i + 1}",
                                tf.boolean_mask(
                                    self.map.u_latents.best[..., i], converged
                                ),
                                step=chain,
                            )
                        for i in range(self.map.n_z_latents):
                            tf.summary.histogram(
                                f"z_{i + 1}",
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
                        tf.summary.histogram(
                            "unconstrained/delta_m",
                            tf.boolean_mask(unconstrained[..., j : j + 1], keep),
                            step=chain,
                        )
                        j += 1
                        tf.summary.histogram(
                            "unconstrained/delta_p",
                            tf.boolean_mask(unconstrained[..., j : j + 1], keep),
                            step=chain,
                        )
                        j += 1
                        tf.summary.histogram(
                            "unconstrained/u_delta_av",
                            tf.boolean_mask(unconstrained[..., j : j + 1], keep),
                            step=chain,
                        )
                        j += 1
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
        self.hmc_train(savepath=savepath)

        if savepath is not None:
            self.save_checkpoint(savepath, save_map=True, save_hmc=True)

    def update_map_progress(self, log_prob: tf.Tensor) -> None:
        if self.map_progress.n % 10 != 0:
            self.map_progress.n += 1
            return

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
            min_log_prob: tf.Tensor,
            mean_log_prob: tf.Tensor,
            max_log_prob: tf.Tensor,
        ) -> None:
            self.map_progress.set_postfix({
                "log_prob": (
                    f"{min_log_prob:.3E}",
                    f"{mean_log_prob:.3E}",
                    f"{max_log_prob:.3E}",
                ),
            })
            self.map_progress.update()

        _update(min_log_prob, mean_log_prob, max_log_prob)

    @tf.function(jit_compile=False)
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

        self.update_map_progress(log_prob)

        return self._loss(self.norm_prob, log_prob)

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

        self.map_progress = tqdm(
            leave=False, dynamic_ncols=True, smoothing=1, position=0
        )

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
        self.map_progress.close()

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
            initial_delta_m = self.map.position.current[:, ind : ind + 1]
            delta_m = position[:, ind : ind + 1]
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
            initial_delta_p = self.map.position.current[:, ind : ind + 1]
            delta_p = position[:, ind : ind + 1]
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
            initial_bias = self.map.position.current[:, ind : ind + 1]
            bias = position[:, ind : ind + 1]
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
            initial_u_delta_av = self.map.position.current[:, ind : ind + 1]
            u_delta_av = position[:, ind : ind + 1]
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

        initial_u_latents = self.map.position.current[:, ind:]
        u_latents = position[:, ind:]
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
            initial_delta_av = initial_z_latents[:, 0:1]
            initial_z_latents = initial_z_latents[:, 1:]
            delta_av = z_latents[:, 0:1]
            z_latents = z_latents[:, 1:]
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
                        "sample/min_log_prior",
                        min_log_prior,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/mean_log_prior",
                        mean_log_prior,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/max_log_prior",
                        max_log_prior,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/min_log_like",
                        min_log_like,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/mean_log_like",
                        mean_log_like,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/max_log_like",
                        max_log_like,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/min_log_prob",
                        min_log_prob,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/mean_log_prob",
                        mean_log_prob,
                        step=self.sample_progress.n,
                    )
                    tf.summary.scalar(
                        "sample/max_log_prob",
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
                    tf.summary.scalar(
                        "step_samples",
                        self.sample_progress.n - self.step_samples,
                        step=self.run_progress.n,
                    )
                    self.step_samples = self.sample_progress.n
                    for i in range(step_size.shape[-1]):
                        tf.summary.histogram(
                            f"step_size/{i}",
                            step_size[..., i],
                            step=self.run_progress.n,
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
            accept_ratio,
            step_size,
        )

    def unnormalized_posterior_log_prob(
        self,
        *pos: tf.Tensor,
        sample: bool | None = None,
        pkr: "DualAveragingStepSizeAdaptationResults | None" = None,
        additional_outputs: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if sample is None:
            sample = pkr is None

        def _step(
            position: tf.Tensor,
        ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            input_position = self.map.get_position(position)
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

            return log_prob, log_like, log_prior

        log_prob, log_like, log_prior = tf.map_fn(
            _step,
            tf.convert_to_tensor(pos),
            parallel_iterations=NPROC,
            fn_output_signature=(
                tf.float32,
                tf.float32,
                tf.float32,
            ),
        )

        log_prob = tf.reduce_sum(log_prob, axis=0)
        log_like = tf.reduce_sum(log_like, axis=0)
        log_prior = tf.reduce_sum(log_prior, axis=0)

        if sample:
            self.update_sample_progress(log_prior, log_like, log_prob)
        if pkr is not None:
            self.update_run_progress(log_prior, log_like, log_prob, pkr)

        if additional_outputs:
            return log_prior, log_like, log_prob
        return log_prob

    def trace_fn(
        self, state: tf.Tensor, pkr: "DualAveragingStepSizeAdaptationResults"
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        log_prior, log_like, log_prob = self.unnormalized_posterior_log_prob(
            state, pkr=pkr, additional_outputs=True
        )
        step_size = pkr.inner_results.step_size
        is_accepted = pkr.inner_results.is_accepted
        log_accept_ratio = pkr.inner_results.log_accept_ratio
        return step_size, is_accepted, log_accept_ratio, log_prior, log_like, log_prob

    @tf.function(jit_compile=False)
    def sample_chain(
        self,
        position: tf.Tensor,
        kernel: tfp.mcmc.TransitionKernel,
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
    ]:
        (
            samples,
            (
                step_sizes_final,
                is_accepted,
                log_accept_ratio,
                log_prior,
                log_like,
                log_prob,
            ),
        ) = tfp.mcmc.sample_chain(
            num_results=self.n_run_steps,
            current_state=position,
            kernel=kernel,
            num_burnin_steps=self.n_burnin_steps,
            num_steps_between_results=self.n_thinning,
            trace_fn=self.trace_fn,
            name="run",
            parallel_iterations=NPROC,
        )

        samples = self.map.constrain(samples)

        return (
            samples,
            step_sizes_final,
            is_accepted,
            log_accept_ratio,
            log_prior,
            log_like,
            log_prob,
        )

    def hmc_train(
        self,
        *,
        savepath: "Path | None" = None,
    ) -> None:
        if savepath is not None:
            hmc_savepath = savepath / "hmc"
            hmc_savepath.mkdir(parents=True, exist_ok=True)
            if (hmc_savepath / self.ckpt_path).exists():
                self.log.debug(f"Loading HMC from {hmc_savepath}")
                self.load_checkpoint(hmc_savepath, load_hmc=True)

                samples = self.hmc.samples.numpy()

                ind = 0
                if self.map.train_delta_m:
                    delta_m = samples[..., ind : ind + 1]
                    ind += 1
                else:
                    delta_m = self.hmc.delta_m
                if self.map.train_delta_p:
                    delta_p = samples[..., ind : ind + 1]
                    ind += 1
                else:
                    delta_p = self.hmc.delta_p
                if self.nflow.physical_latents:
                    u_delta_av = samples[..., ind : ind + 1]
                    ind += 1
                else:
                    u_delta_av = self.hmc.u_delta_av
                u_latents = samples[..., ind:]
                if self.nflow.physical_latents:
                    us = np.concatenate([u_delta_av, u_latents], axis=-1)
                else:
                    us = u_latents
                us = us.reshape(-1, self.map.n_flow_latents)
                # Transform u_latents to z_latents
                z_latents = (
                    self.nflow.u_to_z(us, permute=True)
                    .numpy()
                    .reshape(*samples.shape[:-1], self.map.n_flow_latents)
                )
                if self.pae.physical_latents:
                    delta_av = z_latents[..., 0:1]
                    z_latents = z_latents[..., 1:]
                else:
                    delta_av = self.hmc.delta_av

                vars(self)["hmc"] = PosteriorHMCValue(
                    tf.Variable(samples),
                    tf.Variable(self.hmc.step_sizes_final),
                    tf.Variable(self.hmc.is_accepted),
                    tf.Variable(self.hmc.log_prior),
                    tf.Variable(self.hmc.log_like),
                    tf.Variable(self.hmc.log_prob),
                    tf.Variable(u_delta_av),
                    tf.Variable(u_latents),
                    tf.Variable(delta_av),
                    tf.Variable(z_latents),
                    tf.Variable(delta_m),
                    tf.Variable(delta_p),
                )
                return
        self.log.debug("Running HMC")
        step_size_init = self.step_size
        pp(step_size_init, name="step_size_init")

        initial_position = self.map.position.best
        step_size_std = tf.math.reduce_std(
            tf.boolean_mask(initial_position, self.map.converged), axis=0
        )
        pp(step_size_std, name="step_size_std")

        step_size_std = tf.where(
            tf.math.is_finite(step_size_std), step_size_std, step_size_init
        )
        step_size_init = tf.where(
            tf.math.is_finite(step_size_init), step_size_init, step_size_std
        )
        step_size_inner = tf.math.minimum(step_size_init, step_size_std)
        pp(step_size_inner, name="step_size_inner")
        step_size_inner = self.map.unconstrain(step_size_inner)
        pp(step_size_inner, name="unconstrained step_size_inner")

        step_size = tf.repeat(
            tf.expand_dims(step_size_inner, axis=0),
            repeats=initial_position.shape[0],
            axis=0,
        )

        initial_position = self.map.unconstrain(initial_position)

        self.log.debug(
            f"With {self.n_burnin_steps} burn-in steps and {self.n_run_steps} run steps ({self.n_adaption_steps} of which will be used for step-size adaption), a maximum of {self.max_samples} samples will be generated for a max leapfrog depth of {(2**self.n_leapfrog) - 1}"
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
            smoothing=1,
            position=0,
        )
        self.sample_progress.set_description("samples")
        self.run_progress = tqdm(
            total=self.n_run_steps + 1,
            leave=False,
            dynamic_ncols=True,
            smoothing=1,
            position=1,
        )
        self.run_progress.set_description("run")
        self.step_samples = 0

        sampler = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.unnormalized_posterior_log_prob,
            step_size=step_size,
            max_tree_depth=self.n_leapfrog,
            parallel_iterations=NPROC,
        )

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=sampler,
            num_adaptation_steps=self.n_adaption_steps,
            target_accept_prob=self.target_acceptance_rate,
            # reduce_fn = tfp.math.reduce_log_harmonic_mean_exp
        )

        (
            samples,
            step_sizes_final,
            is_accepted,
            log_accept_ratio,
            log_prior,
            log_like,
            log_prob,
        ) = self.sample_chain(initial_position, kernel)

        self.run_progress.close()
        self.run_progress = None
        self.step_samples = None
        self.sample_progress.close()
        self.sample_progress = None
        if self.summary_writer is not None:
            self.summary_writer.close()
        self.summary_writer = None

        (
            samples,
            step_sizes_final,
            is_accepted,
            log_accept_ratio,
            log_prior,
            log_like,
            log_prob,
        ) = (
            samples.numpy(),
            step_sizes_final.numpy(),
            is_accepted.numpy(),
            log_accept_ratio.numpy(),
            log_prior.numpy(),
            log_like.numpy(),
            log_prob.numpy(),
        )

        ind = 0
        if self.map.train_delta_m:
            delta_m = samples[..., ind : ind + 1]
            ind += 1
        else:
            delta_m = self.hmc.delta_m
        if self.map.train_delta_p:
            delta_p = samples[..., ind : ind + 1]
            ind += 1
        else:
            delta_p = self.hmc.delta_p
        if self.nflow.physical_latents:
            u_delta_av = samples[..., ind : ind + 1]
            ind += 1
        else:
            u_delta_av = self.hmc.u_delta_av
        u_latents = samples[..., ind:]
        if self.nflow.physical_latents:
            us = np.concatenate([u_delta_av, u_latents], axis=-1)
        else:
            us = u_latents
        us = us.reshape(-1, self.map.n_flow_latents)
        # Transform u_latents to z_latents
        z_latents = (
            self.nflow.u_to_z(us, permute=True)
            .numpy()
            .reshape(*samples.shape[:-1], self.map.n_flow_latents)
        )
        if self.pae.physical_latents:
            delta_av = z_latents[..., 0:1]
            z_latents = z_latents[..., 1:]
        else:
            delta_av = self.hmc.delta_av

        vars(self)["hmc"] = PosteriorHMCValue(
            tf.Variable(samples),
            tf.Variable(step_sizes_final),
            tf.Variable(is_accepted),
            tf.Variable(log_prior),
            tf.Variable(log_like),
            tf.Variable(log_prob),
            tf.Variable(u_delta_av),
            tf.Variable(u_latents),
            tf.Variable(delta_av),
            tf.Variable(z_latents),
            tf.Variable(delta_m),
            tf.Variable(delta_p),
        )

        if savepath is not None:
            self.save_checkpoint(hmc_savepath, save_hmc=True)
