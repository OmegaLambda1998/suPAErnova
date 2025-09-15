# Copyright 2025 Patrick Armstrong
import os
from typing import TYPE_CHECKING, Self, override
from collections.abc import Iterable

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from .hmc import PosteriorHMCValue
from .map import PosteriorMap

if TYPE_CHECKING:
    from typing import Any, Self, Literal
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
        self: "Self",
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

        self.debug: bool = self.options.debug
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
        self.tolerance = self.options.tolerance
        self.x_tolerance = self.options.x_tolerance
        self.f_relative_tolerance = self.options.f_relative_tolerance
        self.f_absolute_tolerance = self.options.f_absolute_tolerance
        self.max_iterations = self.options.max_iterations

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
        self.n_burnin: int = self.options.n_burnin
        self.n_samples: int = self.options.n_samples
        self.n_leapfrog: int = self.options.n_leapfrog
        self.n_thinning: int = self.options.n_thinning
        self.target_acceptance_rate: float = self.options.target_acceptance_rate

        self.hmc: PosteriorHMCValue
        vars(self)["hmc"] = PosteriorHMCValue(
            tf.Variable(  # Samples
                [[[0] * self.map.n_pae_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_pae_latents),
            ),
            tf.Variable(  # Step Sizes Final
                [[[0] * self.map.n_pae_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_pae_latents),
            ),
            tf.Variable(  # Is Accepted
                [[False] * self.sn_dim] * self.n_samples,
                dtype=tf.bool,
                shape=(
                    self.n_samples,
                    self.sn_dim,
                ),
            ),
            tf.Variable(  # UDeltaAv
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
            tf.Variable(  # ULatents
                [[[0] * self.map.n_u_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_u_latents),
            ),
            tf.Variable(  # DeltaAv
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
            tf.Variable(  # ZLatents
                [[[0] * self.map.n_z_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_z_latents),
            ),
            tf.Variable(  # DeltaM
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
            tf.Variable(  # DeltaP
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
        )

        self.set_seed()

    @override
    def call(
        self: "Self",
        inputs: tuple[tf.Tensor, ...],
        *,
        training: bool | None = None,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        testing: bool | None = None,
    ) -> tf.Tensor:
        training = False if training is None else training
        testing = False if testing is None else testing

        # === Unpack Inputs ===
        input_position = inputs[0]
        input_phase = inputs[1]
        input_amp = inputs[2]
        input_sigma = inputs[3]

        # --- Masks ---
        # Data Mask
        input_mask = tf.ones_like(input_amp, dtype=tf.int32) if mask is None else mask
        # Wavelength Range Mask
        input_wl_mask = tf.ones_like(input_mask) if wl_mask is None else wl_mask
        # Phase Range Mask
        input_spec_mask = (
            tf.reduce_max(input_wl_mask, axis=-1, keepdims=True)
            if spec_mask is None
            else spec_mask
        )
        # Redshift Range Mask
        input_sn_mask = (
            tf.reduce_max(input_spec_mask, axis=-2, keepdims=True)
            if sn_mask is None
            else sn_mask
        )

        posterior_mask = tf.cast(
            input_mask * input_sn_mask * input_spec_mask * input_wl_mask, tf.float32
        )

        # Determine the prior probability early to avoid exploring non-physical parameter spaces.
        log_prior = self.map.prior(input_position)

        delta_m = input_position[:, 0:1]
        delta_p = input_position[:, 1:2]
        bias = input_position[:, 2:3]
        u_delta_av = input_position[:, 3:4]
        u_latents = input_position[:, 4:]

        # Transform from u-latent to z-latent space
        if self.nflow.physical_latents:
            us = tf.concat([u_delta_av, u_latents], axis=-1)
        else:
            us = u_latents

        zs = self.nflow.u_to_z(us, permute=True)
        if self.nflow.physical_latents:
            delta_av = zs[:, 0:1]
            zs = zs[:, 1:]
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

        zs = ks.layers.RepeatVector(self.spec_dim)(zs)

        # Create synthetic spectra from z-latents
        decoder_inputs = tf.concat((input_phase, zs), axis=-1)
        synth_amp = self.pae.decoder(
            decoder_inputs,
            mask=input_mask,
            sn_mask=input_sn_mask,
            spec_mask=input_spec_mask,
            wl_mask=input_wl_mask,
        )

        if self.map.train_delta_m and not self.pae.physical_latents:
            delta_m = ks.layers.RepeatVector(1)(delta_m)
            synth_amp *= delta_m

        if self.map.train_bias:
            bias = ks.layers.RepeatVector(1)(bias)
            synth_amp += bias

        phase = input_phase
        if self.map.train_delta_p and not self.pae.physical_latents:
            delta_p = ks.layers.RepeatVector(1)(delta_p)
            phase += delta_p

        # Measured average AE reconstruction error at current times
        sigma_recon = tf.transpose(
            tfp.math.interp_regular_1d_grid(
                x=tf.transpose(phase[:, :, 0]),
                x_ref_min=self.recon_error_centers[0],
                x_ref_max=self.recon_error_centers[-1],
                y_ref=self.recon_error,
            )
        )

        synth_sigma = tf.sqrt(((synth_amp * sigma_recon) ** 2) + (input_sigma**2))

        # Set missing values to 1 for all times
        synth_sigma *= posterior_mask
        synth_sigma += 1 - posterior_mask

        # Create likelihood distribution
        likelihood = tfd.Independent(
            tfd.MultivariateNormalDiag(
                loc=synth_amp,
                scale_diag=synth_sigma,
            ),
            reinterpreted_batch_ndims=0,
        )

        # Determine which spectra to keep
        # Will mask out any spectrum with *no* unmasked wavelengths in the valid wavelength range
        mask_spec = tf.reduce_max(posterior_mask, axis=-1)

        # Determine which SNe to keep
        # Will mask out any SN with *no* unmasked spectra
        mask_sn = tf.reduce_max(mask_spec, axis=-1)

        mask_spec = tf.where(
            tf.cast(mask_spec, tf.bool), mask_spec, tf.zeros_like(mask_spec)
        )

        mask_sn = tf.where(
            tf.cast(mask_sn, tf.bool), mask_sn, -np.inf * tf.ones_like(mask_sn)
        )

        log_likelihood = likelihood.log_prob(input_amp) * mask_spec

        log_likelihood_num = tf.reduce_sum(log_likelihood, axis=-1)  # * mask_sn

        # The number of unmasked spectra
        log_likelihood_sum = tf.reduce_sum(mask_spec, axis=-1)

        # log_likelihood_sum = tf.reduce_sum(
        #     tf.cast(tf.math.greater(input_amp[:, :, 0], -1), tf.float32), axis=1
        # )

        log_likelihood = log_likelihood_num / log_likelihood_sum

        log_likelihood += log_prior

        inf_likelihood = -tf.ones_like(log_likelihood) * np.inf

        return tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            inf_likelihood,
        )

    @override
    def __call__(
        self: "Self",
        inputs: "TensorLike",
        *,
        training: bool | None = None,
        mask: "TensorLike | None" = None,
        sn_mask: "TensorLike | None" = None,
        spec_mask: "TensorLike | None" = None,
        wl_mask: "TensorLike | None" = None,
        testing: bool | None = None,
    ) -> tf.Tensor:
        training = False if training is None else training
        testing = False if testing is None else testing
        if isinstance(inputs, Iterable):
            inputs = tuple(tf.convert_to_tensor(i) for i in inputs)
        if mask is not None:
            mask = tf.convert_to_tensor(mask)
        if sn_mask is not None:
            sn_mask = tf.convert_to_tensor(sn_mask)
        if spec_mask is not None:
            spec_mask = tf.convert_to_tensor(spec_mask)
        if wl_mask is not None:
            wl_mask = tf.convert_to_tensor(wl_mask)
        return super().__call__(
            inputs,
            training=training,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            testing=testing,
        )

    def save_checkpoint(
        self: "Self",
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
        self: "Self",
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
        ).assert_existing_objects_matched()

    @override
    def get_config(self: "Self") -> dict[str, "Any"]:
        return {**super().get_config()}

    @override
    @classmethod
    def from_config(cls: type["Self"], config: dict[str, "Any"]) -> "Self":
        global POSTERIORMODELSTEP
        return cls(POSTERIORMODELSTEP)

    @override
    def set_seed(self: "Self", seed: int = 0) -> None:
        seed = self.seed + seed
        tf.random.set_seed(seed)

    def train_model(
        self: "Self",
        stages: "Sequence[PosteriorMAPStage]",
        *,
        savepath: "Path | None" = None,
    ) -> None:
        if savepath is not None and (savepath / self.ckpt_path).exists():
            self.log.debug(f"Loading Posterior from {savepath}")
            self.load_checkpoint(savepath, load_map=True, load_hmc=True)

            return

        n_total = sum(stage.n_chains for stage in stages)
        chain = 0
        progress = tqdm(total=n_total)

        summary_writer = None
        if self.profile and savepath is not None:
            log_dir = savepath.parent / self.log_path / savepath.stem / "map"
            summary_writer = tf.summary.create_file_writer(str(log_dir))

        for stage in stages:
            for c in range(stage.n_chains):
                self.set_seed(chain)

                self.train_map(stage, c, chain, savepath=savepath)

                log_prob = -self.map.negative_log_prob
                num_total = tf.reduce_sum(tf.ones_like(log_prob)).numpy()
                num_finite = tf.math.count_nonzero(tf.math.is_finite(log_prob)).numpy()
                finite_ratio = num_finite / num_total
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
                    / tf.reduce_sum(
                        tf.where(
                            tf.math.is_finite(log_prob),
                            tf.ones_like(log_prob),
                            tf.zeros_like(log_prob),
                        )
                    ).numpy()
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

                progress.set_description(f"Stage: {chain}/{n_total}")
                progress.set_postfix({
                    "evals_tot": self.map.num_evaluations.value().numpy(),
                    "evals_prev": self.map.num_chain_evaluations.value().numpy(),
                    "improved": tf.reduce_sum(
                        tf.ones_like(self.map.improved, dtype=tf.int32)
                        * tf.cast(self.map.improved, tf.int32)
                    ).numpy(),
                    "log_prob": (
                        f"{min_log_prob:.2E} ({(min_log_prob / max_log_prob) * finite_ratio:.2%})",
                        f"{mean_log_prob:.2E} ({(mean_log_prob / max_log_prob) * finite_ratio:.2%})",
                        f"{max_log_prob:.2E} ({finite_ratio:.2%})",
                    ),
                })
                progress.update()

                # TODO: Remove non-finite log_prob parameters
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
                            "min_log_prob",
                            min_log_prob,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "norm_min_log_prob",
                            (min_log_prob / max_log_prob) * finite_ratio,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "mean_log_prob",
                            mean_log_prob,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "norm_mean_log_prob",
                            (mean_log_prob / max_log_prob) * finite_ratio,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "max_log_prob",
                            max_log_prob,
                            step=chain,
                        )
                        tf.summary.scalar(
                            "norm_max_log_prob",
                            finite_ratio,
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
                                f"u_{i}",
                                tf.boolean_mask(
                                    self.map.u_latents.best[..., i], converged
                                ),
                                step=chain,
                            )
                        for i in range(self.map.n_z_latents):
                            tf.summary.histogram(
                                f"z_{i}",
                                tf.boolean_mask(
                                    self.map.z_latents.best[..., i], converged
                                ),
                                step=chain,
                            )
                chain += 1
        self.log.info(f"Minimum found at chains:\n{self.map.chain_min}")
        self.set_seed()
        self.hmc_train(savepath=savepath)

        if savepath is not None:
            self.save_checkpoint(savepath, save_map=True, save_hmc=True)

    def train_map(
        self: "Self",
        stage: "PosteriorMAPStage",
        chain: int,
        chain_total: int,
        *,
        savepath: "Path | None" = None,
    ) -> None:
        if savepath is not None:
            stage_savepath = savepath / "map" / f"{stage.fname}_{chain}"
            stage_savepath.mkdir(parents=True, exist_ok=True)
            if (stage_savepath / self.ckpt_path).exists():
                self.log.debug(
                    f"Loading MAP stage: {stage.name}_{chain} from {stage_savepath}"
                )
                self.load_checkpoint(stage_savepath, load_map=True)

                return
        self.log.debug(f"Running MAP stage: {stage.name}_{chain}")

        self.map.setup(stage, chain)

        progress = tqdm()

        @tf.py_function(Tout=[])
        def update_progress(log_prob) -> None:
            num_total = tf.math.reduce_sum(tf.ones_like(log_prob)).numpy()
            num_finite = tf.math.count_nonzero(tf.math.is_finite(log_prob)).numpy()
            finite_ratio = num_finite / num_total

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
                / tf.reduce_sum(
                    tf.where(
                        tf.math.is_finite(log_prob),
                        tf.ones_like(log_prob),
                        tf.zeros_like(log_prob),
                    )
                ).numpy()
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
                "log_prob": (
                    f"{min_log_prob:.2E} ({(min_log_prob / max_log_prob) * finite_ratio:.2%})",
                    f"{mean_log_prob:.2E} ({(mean_log_prob / max_log_prob) * finite_ratio:.2%})",
                    f"{max_log_prob:.2E} ({finite_ratio:.2%})",
                ),
            })
            progress.update()

        @tf.function
        def vals_and_grads(position: tf.Tensor) -> tf.Tensor:
            input_position = self.map.get_position(position)
            log_prob = self(
                (input_position, self.data_time, self.data_amplitude, self.data_sigma),
                training=False,
                mask=self.data_mask,
                sn_mask=self.sn_mask,
                spec_mask=self.spec_mask,
                wl_mask=self.wl_mask,
            )
            update_progress(log_prob)
            return self._loss(tf.zeros_like(log_prob), log_prob)

        def lbfgs() -> "LBfgsOptimizerResults":
            return tfp.optimizer.lbfgs_minimize(
                lambda x: tfp.math.value_and_gradient(
                    vals_and_grads, x, auto_unpack_single_arg=False
                ),
                initial_position=self.map.position.current,
                tolerance=self.tolerance,
                x_tolerance=self.x_tolerance,
                f_relative_tolerance=self.f_relative_tolerance,
                f_absolute_tolerance=self.f_absolute_tolerance,
                max_iterations=self.max_iterations,
                parallel_iterations=NPROC,
            )

        results = lbfgs()

        improved = (
            results.objective_value < self.map.negative_log_prob
        ) * results.converged
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
                results.converged,
                self.map.converged,
            )
        )
        self.map.failed.assign(
            tf.where(
                improved,
                results.failed,
                self.map.failed,
            )
        )
        self.map.num_evaluations.assign_add(results.num_objective_evaluations)
        self.map.num_chain_evaluations.assign(results.num_objective_evaluations)
        self.map.negative_log_prob.assign(
            tf.where(
                improved,
                results.objective_value,
                self.map.negative_log_prob,
            )
        )

        ind = 0
        initial_position = []
        current_position = []
        if self.map.train_delta_m:
            initial_delta_m = self.map.position.current[:, ind : ind + 1]
            delta_m = results.position[:, ind : ind + 1]
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
            delta_p = results.position[:, ind : ind + 1]
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
            bias = results.position[:, ind : ind + 1]
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
            u_delta_av = results.position[:, ind : ind + 1]
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
        u_latents = results.position[:, ind:]
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
            self.log.debug(
                f"Saving MAP stage: {stage.name}_{chain} from {stage_savepath}"
            )
            (stage_savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(stage_savepath, save_map=True)

    # === HMC Functions ===

    def hmc_train(
        self: "Self",
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
                    tf.Variable(u_delta_av),
                    tf.Variable(u_latents),
                    tf.Variable(delta_av),
                    tf.Variable(z_latents),
                    tf.Variable(delta_m),
                    tf.Variable(delta_p),
                )
                return
        self.log.debug("Running HMC")
        initial_position = self.map.position.best
        summary_writer = None
        if self.profile and savepath is not None:
            log_dir = savepath.parent / self.log_path / savepath.stem / "hmc"
            summary_writer = tf.summary.create_file_writer(str(log_dir))

        step_size_std = tf.math.reduce_std(initial_position, axis=0)
        step_size_inner = tf.math.sqrt(self.step_size * step_size_std)
        step_size = tf.repeat(
            tf.expand_dims(step_size_inner, axis=0),
            repeats=initial_position.shape[0],
            axis=0,
        )
        tf.print(self.step_size, step_size_std, step_size_inner)

        progress = tqdm(
            total=self.n_samples + 1,
        )

        tf.summary.experimental.set_step(tf.Variable(tf.constant(0, dtype=tf.int64)))

        def unnormalized_posterior_log_prob(
            *pos: tf.Tensor, results=False
        ) -> tf.Tensor:
            input_position = self.map.get_position(*pos)
            log_prob = self(
                (input_position, self.data_time, self.data_amplitude, self.data_sigma),
                training=False,
                mask=self.data_mask,
                sn_mask=self.sn_mask,
                spec_mask=self.spec_mask,
                wl_mask=self.wl_mask,
            )

            if summary_writer is not None:
                with summary_writer.as_default():
                    tf.summary.experimental.set_step(
                        tf.summary.experimental.get_step().assign_add(
                            tf.constant(1, dtype=tf.int64)
                        )
                    )

                    num_total = tf.math.reduce_sum(tf.ones_like(log_prob))
                    num_finite = tf.math.count_nonzero(
                        tf.math.is_finite(log_prob), dtype=tf.float32
                    )
                    finite_ratio = num_finite / num_total

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
                    ) / tf.reduce_sum(
                        tf.where(
                            tf.math.is_finite(log_prob),
                            tf.ones_like(log_prob),
                            tf.zeros_like(log_prob),
                        )
                    )

                    max_log_prob = tf.reduce_max(
                        tf.where(
                            tf.math.is_finite(log_prob),
                            log_prob,
                            -np.inf * tf.ones_like(log_prob),
                        )
                    )

                    tf.summary.scalar(
                        "min_log_prob",
                        min_log_prob,
                        step=tf.summary.experimental.get_step(),
                    )
                    tf.summary.scalar(
                        "norm_min_log_prob",
                        (min_log_prob / max_log_prob) * finite_ratio,
                        step=tf.summary.experimental.get_step(),
                    )
                    tf.summary.scalar(
                        "mean_log_prob",
                        mean_log_prob,
                        step=tf.summary.experimental.get_step(),
                    )
                    tf.summary.scalar(
                        "norm_mean_log_prob",
                        (mean_log_prob / max_log_prob) * finite_ratio,
                        step=tf.summary.experimental.get_step(),
                    )
                    tf.summary.scalar("max_log_prob", max_log_prob)
                    tf.summary.scalar(
                        "norm_max_log_prob",
                        finite_ratio,
                        step=tf.summary.experimental.get_step(),
                    )
                    if results:
                        tf.summary.scalar(
                            "result_min_log_prob",
                            min_log_prob,
                            step=tf.summary.experimental.get_step(),
                        )
                        tf.summary.scalar(
                            "result_norm_min_log_prob",
                            (min_log_prob / max_log_prob) * finite_ratio,
                            step=tf.summary.experimental.get_step(),
                        )
                        tf.summary.scalar(
                            "result_mean_log_prob",
                            mean_log_prob,
                            step=tf.summary.experimental.get_step(),
                        )
                        tf.summary.scalar(
                            "result_norm_mean_log_prob",
                            (mean_log_prob / max_log_prob) * finite_ratio,
                            step=tf.summary.experimental.get_step(),
                        )
                        tf.summary.scalar(
                            "result_max_log_prob",
                            max_log_prob,
                            step=tf.summary.experimental.get_step(),
                        )
                        tf.summary.scalar(
                            "result_norm_max_log_prob",
                            finite_ratio,
                            step=tf.summary.experimental.get_step(),
                        )

            return log_prob

        @tf.py_function(Tout=[])
        def update_progress() -> None:
            progress.update()

        def trace_fn(
            state: tf.Tensor, pkr: "DualAveragingStepSizeAdaptationResults"
        ) -> tuple[tf.Tensor, tf.Tensor]:
            unnormalized_posterior_log_prob(state, results=True)
            update_progress()
            step_size = pkr.inner_results.step_size
            is_accepted = pkr.inner_results.is_accepted
            return step_size, is_accepted

        @tf.function
        def sample_chain() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            sampler = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                step_size=step_size,
                max_tree_depth=self.n_leapfrog,
                parallel_iterations=NPROC,
            )

            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=sampler,
                num_adaptation_steps=int(self.n_burnin * 0.8),
                target_accept_prob=self.target_acceptance_rate,
                # reduce_fn=tfp.math.reduce_log_harmonic_mean_exp,
            )

            samples, [step_sizes_final, is_accepted] = tfp.mcmc.sample_chain(
                num_results=self.n_samples,
                current_state=initial_position,
                kernel=kernel,
                num_burnin_steps=self.n_burnin,
                num_steps_between_results=self.n_thinning,
                trace_fn=trace_fn,
                name="run",
                parallel_iterations=NPROC,
            )

            return samples, step_sizes_final, is_accepted

        samples, step_sizes_final, is_accepted = sample_chain()
        samples, step_sizes_final, is_accepted = (
            samples.numpy(),
            step_sizes_final.numpy(),
            is_accepted.numpy(),
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
            tf.Variable(u_delta_av),
            tf.Variable(u_latents),
            tf.Variable(delta_av),
            tf.Variable(z_latents),
            tf.Variable(delta_m),
            tf.Variable(delta_p),
        )

        if savepath is not None:
            self.save_checkpoint(hmc_savepath, save_hmc=True)
