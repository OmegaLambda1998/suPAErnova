# Copyright 2025 Patrick Armstrong
import os
from typing import TYPE_CHECKING, override

from tqdm import tqdm
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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

    from suPAErnova.steps.pae.tf import S, TFPAEModel, TensorCompatible
    from suPAErnova.steps.nflow.tf import TFNFlowModel
    from suPAErnova.configs.steps.data import DataStepResult
    from suPAErnova.steps.posterior.model import PosteriorModelStep
    from suPAErnova.configs.steps.posterior.tf import TFPosteriorModelConfig
    from suPAErnova.configs.steps.posterior.posterior import PosteriorMapStage


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPosteriorModel(ks.Model):
    def __init__(
        self: "TFPosteriorModel",
        config: "PosteriorModelStep[TFPosteriorModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        ks.backend.clear_session()

        super().__init__(
            *args, name=f"{config.name.split()[-1]}PosteriorModel", **kwargs
        )
        # --- Config ---
        global POSTERIORMODELSTEP
        POSTERIORMODELSTEP = config
        options = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed
        self.subset: Literal["train", "test"] = config.options.subset

        self.debug: bool = options.debug
        self.profile: bool = options.profile
        # Equivalent to `self.name = ...` but avoids tf / ks from tracking self.name
        vars(self)["nflow"]: TFNFlowModel = config.nflow.model
        vars(self)["pae"]: TFPAEModel = self.nflow.pae
        self.nflow.trainable = False
        self.nflow.flow.trainable = False

        self.data: DataStepResult = (
            self.pae.stage.train_data
            if self.subset == "train"
            else self.pae.stage.test_data
        )
        self.sn_mask: npt.NDArray[np.int32] = (
            np.array(self.pae.stage.train_sn_mask).astype(np.int32)
            if self.subset == "train"
            else np.array(self.pae.stage.test_sn_mask).astype(np.int32)
        )
        self.spec_mask: npt.NDArray[np.int32] = (
            np.array(self.pae.stage.train_spec_mask).astype(np.int32)
            if self.subset == "train"
            else np.array(self.pae.stage.test_spec_mask).astype(np.int32)
        )
        self.data.mask *= self.sn_mask * self.spec_mask

        self.sn_dim = self.data.time.shape[0]
        self.spec_dim = self.data.time.shape[1]

        # MAP Variables
        vars(self)["map"]: PosteriorMap = PosteriorMap(
            options, self.nflow, self.pae, self.data
        )
        self.tolerance = options.tolerance
        self.max_iterations = options.max_iterations

        # --- Training ---
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )
        self.log_path: str = f"{'best' if self.save_best else 'latest'}_logs/"

        inds = np.squeeze(self.sn_mask.astype(np.bool_), axis=(1, 2))
        self.recon_error, _recon_error_edges, self.recon_error_centers = (
            self.pae.recon_error((
                tf.convert_to_tensor(self.data.time[inds], dtype=tf.float32),
                tf.convert_to_tensor(self.data.amplitude[inds], dtype=tf.float32),
                tf.convert_to_tensor(self.data.sigma[inds], dtype=tf.float32),
                tf.convert_to_tensor(self.data.mask[inds], dtype=tf.int32),
            ))
        )

        loss: ks.losses.Loss = options.loss_cls()
        loss.model = self
        self._loss: ks.losses.Loss = loss

        # HMC Variables
        self.n_burnin: int = options.n_burnin
        self.n_samples: int = options.n_samples
        self.n_leapfrog: int = options.n_leapfrog
        self.target_acceptance_rate: float = options.target_acceptance_rate

        vars(self)["hmc"]: PosteriorHMCValue = PosteriorHMCValue(
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
    @tf.function
    def call(
        self,
        inputs: "PosteriorInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "PosteriorOutputs":
        training = False if training is None else training

        input_position = inputs[0]
        input_phase = inputs[1]
        input_amp = inputs[2]
        input_sigma = inputs[3]
        input_mask = (
            mask if mask is not None else tf.ones_like(input_amp, dtype=tf.int32)
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
        synth_amp = self.pae.decoder(decoder_inputs, mask=input_mask)

        if self.map.train_delta_m and not self.pae.physical_latents:
            delta_m = ks.layers.RepeatVector(1)(delta_m)
            synth_amp *= delta_m

        if self.map.train_bias:
            bias = ks.layers.RepeatVector(1)(bias)
            synth_amp += bias

        phase = input_phase
        # XXX: Test whether this fixes things
        if self.map.train_delta_p:  # and not self.pae.physical_latents:
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
        synth_sigma *= tf.cast(input_mask, tf.float32)
        synth_sigma += 1 - tf.cast(input_mask, tf.float32)

        # Create likelihood distribution
        likelihood = tfd.Independent(
            tfd.MultivariateNormalDiag(
                loc=synth_amp * tf.cast(input_mask, tf.float32), scale_diag=synth_sigma
            ),
            reinterpreted_batch_ndims=0,
        )

        # Determine the log likelihood
        is_kept = tf.cast(tf.reduce_min(input_mask, axis=-1), tf.float32)
        log_likelihood_num = tf.reduce_sum(
            likelihood.log_prob(input_amp * tf.cast(input_mask, tf.float32)) * is_kept,
            axis=1,
        )
        log_likelihood_sum = tf.reduce_sum(
            tf.cast(tf.math.greater(input_amp[:, :, 0], -1), tf.float32), axis=1
        )
        log_likelihood = log_likelihood_num / log_likelihood_sum

        inf_likelihood = -tf.ones_like(log_likelihood) * np.inf
        log_likelihood = tf.where(
            tf.math.is_nan(log_likelihood),
            inf_likelihood,
            log_likelihood,
        )

        return log_prior + log_likelihood

    def save_checkpoint(
        self, savepath: "Path", *, save_map: bool = False, save_hmc: bool = False
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
        self, loadpath: "Path", *, load_map: bool = False, load_hmc: bool = False
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
    def get_config(self) -> dict[str, "Any"]:
        return {**super().get_config()}

    @override
    @classmethod
    def from_config(cls, config: dict[str, "Any"]) -> "Self":
        global POSTERIORMODELSTEP
        return cls(POSTERIORMODELSTEP)

    @override
    def set_seed(self, seed: int = 0) -> None:
        seed = self.seed + seed
        tf.random.set_seed(seed)

    def train_model(
        self,
        stages: "Sequence[PosteriorMapStage]",
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
                progress.set_description(f"Stage: {chain}/{n_total}")
                progress.set_postfix({
                    "evals_tot": self.map.num_evaluations.value().numpy(),
                    "evals_prev": self.map.num_chain_evaluations.value().numpy(),
                    "improved": tf.reduce_sum(
                        tf.ones_like(self.map.improved, dtype=tf.int32)
                        * tf.cast(self.map.improved, tf.int32)
                    ).numpy(),
                    "neg_log_prob": (
                        f"{float(tf.reduce_min(self.map.negative_log_prob).numpy()):.2E}",
                        f"{float(tf.reduce_mean(self.map.negative_log_prob).numpy()):.2E}",
                        f"{float(tf.reduce_max(self.map.negative_log_prob).numpy()):.2E}",
                    ),
                })
                self.train_map(stage, c, chain, savepath=savepath)
                progress.update()

                if summary_writer is not None:
                    with summary_writer.as_default():
                        tf.summary.histogram(
                            "chain_min", self.map.chain_min, step=chain
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
                            tf.reduce_min(-self.map.negative_log_prob),
                            step=chain,
                        )
                        tf.summary.scalar(
                            "mean_log_prob",
                            tf.reduce_mean(-self.map.negative_log_prob),
                            step=chain,
                        )
                        tf.summary.scalar(
                            "max_log_prob",
                            tf.reduce_max(-self.map.negative_log_prob),
                            step=chain,
                        )
                        tf.summary.histogram(
                            "u_delta_av", self.map.u_delta_av.best, step=chain
                        )
                        tf.summary.histogram(
                            "delta_av", self.map.delta_av.best, step=chain
                        )
                        tf.summary.histogram(
                            "delta_m", self.map.delta_m.best, step=chain
                        )
                        tf.summary.histogram(
                            "delta_p", self.map.delta_p.best, step=chain
                        )
                        tf.summary.histogram("bias", self.map.bias.best, step=chain)
                        for i in range(self.map.n_u_latents):
                            tf.summary.histogram(
                                f"u_{i}", self.map.u_latents.best[..., i], step=chain
                            )
                        for i in range(self.map.n_z_latents):
                            tf.summary.histogram(
                                f"z_{i}", self.map.z_latents.best[..., i], step=chain
                            )

                chain += 1
        self.log.info(f"Minimum found at chains:\n{self.map.chain_min}")
        self.set_seed()
        self.hmc_train(savepath=savepath)

        if savepath is not None:
            self.save_checkpoint(savepath, save_map=True, save_hmc=True)

    def vals_and_grads(self, position):
        input_position = self.map.get_position(position)
        log_prob = self(
            (input_position, self.data.time, self.data.amplitude, self.data.sigma),
            training=False,
            mask=self.data.mask,
        )
        return self._loss(tf.zeros_like(log_prob), log_prob)

    @tf.function
    def lbfgs(self, x):
        return tfp.math.value_and_gradient(
            self.vals_and_grads, x, auto_unpack_single_arg=False
        )

    def train_map(
        self,
        stage: "PosteriorMapStage",
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

        results = tfp.optimizer.lbfgs_minimize(
            self.lbfgs,
            initial_position=tf.identity(self.map.position.current),
            tolerance=self.tolerance,
            x_tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            num_correction_pairs=1,
        )

        improved = results.objective_value < self.map.negative_log_prob
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
        self.map.delta_m.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_delta_m,
                self.map.delta_m.initial,
            )
        )
        self.map.delta_m.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                delta_m,
                self.map.delta_m.best,
            )
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
        self.map.delta_p.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_delta_p,
                self.map.delta_p.initial,
            )
        )
        self.map.delta_p.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                delta_p,
                self.map.delta_p.best,
            )
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
        self.map.bias.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                bias,
                self.map.bias.initial,
            )
        )
        self.map.bias.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                bias,
                self.map.bias.best,
            )
        )

        if self.map.nflow.physical_latents:
            initial_u_delta_av = self.map.position.current[:, ind : ind + 1]
            u_delta_av = results.position[:, ind : ind + 1]
            ind += 1
            initial_position.append(initial_u_delta_av)
            current_position.append(u_delta_av)
        else:
            initial_u_delta_av = self.map.u_delta_av.original
            u_delta_av = self.map.u_delta_av.current
        self.map.u_delta_av.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_u_delta_av,
                self.map.u_delta_av.initial,
            )
        )
        self.map.u_delta_av.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                u_delta_av,
                self.map.u_delta_av.best,
            )
        )

        initial_u_latents = self.map.position.current[:, ind:]
        u_latents = results.position[:, ind:]
        initial_position.append(initial_u_latents)
        current_position.append(u_latents)
        self.map.u_latents.initial.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_u_latents,
                    axis=-1,
                ),
                initial_u_latents,
                self.map.u_latents.initial,
            )
        )
        self.map.u_latents.best.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_u_latents,
                    axis=-1,
                ),
                u_latents,
                self.map.u_latents.best,
            )
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

        self.map.z_latents.initial.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_z_latents,
                    axis=-1,
                ),
                initial_z_latents,
                self.map.z_latents.initial,
            )
        )
        self.map.z_latents.best.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_z_latents,
                    axis=-1,
                ),
                z_latents,
                self.map.z_latents.best,
            )
        )

        self.map.delta_av.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_delta_av,
                self.map.delta_av.initial,
            )
        )
        self.map.delta_av.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                delta_av,
                self.map.delta_av.best,
            )
        )

        initial_position = tf.concat(initial_position, axis=-1)
        current_position = tf.concat(current_position, axis=-1)

        self.map.position.initial.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=initial_position.shape[-1],
                    axis=-1,
                ),
                initial_position,
                self.map.position.initial,
            )
        )
        self.map.position.best.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=current_position.shape[-1],
                    axis=-1,
                ),
                current_position,
                self.map.position.best,
            )
        )

        if savepath is not None:
            self.log.debug(
                f"Saving MAP stage: {stage.name}_{chain} from {stage_savepath}"
            )
            (stage_savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(stage_savepath, save_map=True)

    # === HMC Functions ===

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
                if self.map.nflow.physical_latents:
                    u_delta_av = samples[..., ind : ind + 1]
                    ind += 1
                else:
                    u_delta_av = self.hmc.u_delta_av
                u_latents = samples[..., ind:]
                if self.map.nflow.physical_latents:
                    us = np.concat([u_delta_av, u_latents], axis=-1)
                else:
                    us = u_latents
                us = us.reshape(-1, self.map.n_flow_latents)
                # Transform u_latents to z_latents
                z_latents = (
                    self.nflow.u_to_z(us, permute=True)
                    .numpy()
                    .reshape(*samples.shape[:-1], self.map.n_flow_latents)
                )
                if self.map.pae.physical_latents:
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
        if self.profile and savepath is not None:
            log_dir = savepath.parent / self.log_path / savepath.stem / "hmc"
            summary_writer = tf.summary.create_file_writer(str(log_dir))
            tf.summary.experimental.set_step(0)

        stds = []
        if self.map.train_delta_m:
            stds.append(np.std(self.map.delta_m.best, axis=0))
        if self.map.train_delta_p:
            stds.append(np.std(self.map.delta_p.best, axis=0))
        if self.map.nflow.physical_latents:
            stds.append(np.std(self.map.u_delta_av.best, axis=0))
        stds.append(np.std(self.map.u_latents.best, axis=0))
        stds = tf.concat(stds, axis=-1)
        step_size = tf.repeat(
            tf.expand_dims(stds, axis=0), repeats=initial_position.shape[0], axis=0
        )

        progress = tqdm(
            total=(2 * self.n_leapfrog) * (self.n_burnin + self.n_samples),
            leave=True,
        )

        @tf.py_function(Tout=[])
        def update_progress(pos, log_prob) -> None:
            progress.set_description(
                "Burnin"
                if tf.summary.experimental.get_step()
                < self.n_burnin * (2 * self.n_leapfrog)
                else "Run"
            )
            progress.update()

            if summary_writer is not None:
                with summary_writer.as_default():
                    tf.summary.experimental.set_step(
                        tf.summary.experimental.get_step() + 1
                    )

                    lp = log_prob.numpy()
                    lp = lp[lp != -np.inf]

                    tf.summary.scalar("min_log_prob", np.nanmin(lp))
                    tf.summary.scalar("mean_log_prob", np.nanmean(lp))
                    tf.summary.scalar("max_log_prob", np.nanmax(lp))

        def unnormalized_posterior_log_prob(*pos):
            input_position = self.map.get_position(*pos)
            # tf.print("input_position", input_position.shape, input_position)
            log_prob = self(
                (input_position, self.data.time, self.data.amplitude, self.data.sigma),
                training=False,
                mask=self.data.mask,
            )
            update_progress(input_position, log_prob)
            # tf.print("log_prob", log_prob.shape, log_prob)
            return log_prob

        def trace_fn(_, pkr):
            step_size = pkr.inner_results.accepted_results.step_size
            is_accepted = pkr.inner_results.is_accepted
            return [step_size, is_accepted]

        def sample_chain():
            # run hmc
            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                num_leapfrog_steps=self.n_leapfrog,
                step_size=step_size,
            )

            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=hmc,
                num_adaptation_steps=int(self.n_burnin * 0.8),
                target_accept_prob=self.target_acceptance_rate,
                reduce_fn=tfp.math.reduce_log_harmonic_mean_exp,
            )

            samples, [step_sizes_final, is_accepted] = tfp.mcmc.sample_chain(
                num_results=self.n_samples,
                current_state=initial_position,
                kernel=kernel,
                num_burnin_steps=self.n_burnin,
                trace_fn=trace_fn,
                name="run",
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
        if self.map.nflow.physical_latents:
            u_delta_av = samples[..., ind : ind + 1]
            ind += 1
        else:
            u_delta_av = self.hmc.u_delta_av
        u_latents = samples[..., ind:]
        if self.map.nflow.physical_latents:
            us = np.concat([u_delta_av, u_latents], axis=-1)
        else:
            us = u_latents
        us = us.reshape(-1, self.map.n_flow_latents)
        # Transform u_latents to z_latents
        z_latents = (
            self.nflow.u_to_z(us, permute=True)
            .numpy()
            .reshape(*samples.shape[:-1], self.map.n_flow_latents)
        )
        if self.map.pae.physical_latents:
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
