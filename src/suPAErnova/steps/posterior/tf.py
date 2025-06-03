# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    NamedTuple,
    cast,
    override,
)

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

if TYPE_CHECKING:
    from typing import Any
    from logging import Logger
    from pathlib import Path

    from suPAErnova.steps.pae.tf import (
        S,
        FTensor,
        TensorCompatible,
    )
    from suPAErnova.steps.nflow.tf import TFNFlowModel
    from suPAErnova.steps.posterior.model import PosteriorModel
    from suPAErnova.configs.steps.posterior.tf import TFPosteriorModelConfig

    from .model import Stage

    # === Custom Types ===


class MinimiserResults(NamedTuple):
    converged: tf.Tensor
    failed: tf.Tensor
    num_objective_evaluations: int
    position: tf.Tensor
    objective_value: tf.Tensor
    objective_gradient: tf.Tensor
    position_deltas: tf.Tensor
    gradient_deltas: tf.Tensor


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPosteriorLoss(ks.losses.Loss):
    @override
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return -y_pred


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPosteriorModel(ks.Model):
    def __init__(
        self,
        config: "PosteriorModel[TFPosteriorModel, TFPosteriorModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(
            *args, name=f"{config.name.split()[-1]}PosteriorModel", **kwargs
        )
        # --- Config ---
        options = cast("TFPosteriorModelConfig", config.options)
        self.log: "Logger" = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force

        # --- Previous Steps ---
        self.nflow: "TFNFlowModel" = cast("TFNFlowModel", config.nflow.model)

        # --- Data ---
        self.recon_error: tf.Tensor
        self.recon_error_centers: tf.Tensor
        self.batch_dim: int
        self.nspec_dim: int
        self.n_latents: int = self.nflow.pae.n_zs + (
            1 if self.nflow.physical_latents else 0
        )

        # --- Training ---
        self.trained: bool = False
        self.built: bool = False
        self._chain: int = 0
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

        self.random_initial_positions: bool = options.random_initial_positions
        self.tolerance: float = options.tolerance
        self.max_iterations: int = options.max_iterations

        self._loss: ks.losses.Loss = (
            options.loss_cls() if options.loss_cls is not None else TFPosteriorLoss()
        )
        self.stage: Stage

        # --- Variables ---
        # Initial values
        self.init_zs: "FTensor[S['batch_dim n_zs']] | None" = None
        self.init_latents: "FTensor[S['batch_dim n_latents']] | None" = None
        self.init_delta_av: "FTensor[S['batch_dim 1']] | None" = None
        self.init_delta_m: "FTensor[S['batch_dim 1']] | None" = None
        self.init_delta_p: "FTensor[S['batch_dim 1']] | None" = None
        self.init_bias: "FTensor[S['batch_dim 1']] | None" = None

        # Current values
        self.curr_zs: "FTensor[S['batch_dim n_zs']]"
        self.curr_latents: "FTensor[S['batch_dim n_latents']]"
        self.curr_delta_av: "FTensor[S['batch_dim 1']]"
        self.curr_delta_m: "FTensor[S['batch_dim 1']]"
        self.curr_delta_p: "FTensor[S['batch_dim 1']]"
        self.curr_bias: "FTensor[S['batch_dim 1']]"

        # Best values
        self.best_zs: tf.Variable
        self.best_latents: tf.Variable
        self.best_delta_av: tf.Variable
        self.best_delta_m: tf.Variable
        self.best_delta_p: tf.Variable
        self.best_bias: tf.Variable

        # --- Layers ---
        self.latents_mean: float = options.latents_mean
        self.latents_std: float = options.latents_std
        self.latents_prior: tfd.MultivariateNormalDiag = tfd.MultivariateNormalDiag(
            loc=self.latents_mean * tf.ones(self.n_latents),
            scale_diag=self.latents_std * tf.ones(self.n_latents),
        )

        self.train_delta_av: bool = options.train_delta_av
        self.delta_av_min: float = options.delta_av_min
        self.delta_av_max: float = options.delta_av_max
        self.delta_av_mean: float = options.delta_av_mean
        self.delta_av_std: float = options.delta_av_std
        self.delta_av_prior: tdf.Normal = tfd.Normal(
            loc=self.delta_av_mean, scale=self.delta_av_std
        )

        self.train_delta_m: bool = options.train_delta_m
        self.delta_m_min: float = options.delta_m_min
        self.delta_m_max: float = options.delta_m_max
        self.delta_m_mean: float = options.delta_m_mean
        self.delta_m_std: float = options.delta_m_std
        self.delta_m_prior: tdf.Normal = tfd.Normal(
            loc=self.delta_m_mean, scale=self.delta_m_std
        )

        self.train_delta_p: bool = options.train_delta_p
        self.delta_p_min: float = options.delta_p_min
        self.delta_p_max: float = options.delta_p_max
        self.delta_p_mean: float = options.delta_p_mean
        self.delta_p_std: float = options.delta_p_std
        self.delta_p_prior: tfd.Normal = tfd.Normal(
            loc=self.delta_p_mean, scale=self.delta_p_std
        )

        self.train_bias: bool = options.train_bias
        self.bias_min: float = options.bias_min
        self.bias_max: float = options.bias_max
        self.bias_mean: float = options.bias_mean
        self.bias_std: float = options.bias_std
        self.bias_prior: tfd.Normal = tfd.Normal(
            loc=self.bias_mean, scale=self.bias_std
        )

        self.best_chain: tf.Variable
        self.best_converged: tf.Variable
        self.best_objective_value: tf.Variable
        self.num_evals: tf.Variable

        self.u_likelihood: tfd.Independent

        # --- Metrics ---
        self.results: MinimiserResults

        # --- Losses ---
        self._loss_terms: dict[str, "FTensor[S['']]"]
        self.loss_tracker: ks.metrics.Metric = ks.metrics.Mean("loss")

    @property
    @override
    def metrics(self) -> list[ks.metrics.Metric]:
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each chain
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]

    @override
    def call(
        self,
        inputs: "PosteriorInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "PosteriorOutputs":
        training = False if training is None else training
        (data, position) = inputs
        (input_phase, input_amplitude, input_d_amplitude) = data
        input_mask = tf.cast(
            (
                mask
                if mask is not None
                else tf.ones_like(input_amplitude, dtype=tf.int32)
            ),
            tf.float32,
        )

        latents = position[:, : self.n_latents]
        self.curr_latents = latents

        if self.train_delta_av or self.nflow.physical_latents:
            delta_av = tf.expand_dims(latents[:, 0], axis=-1)
            self.curr_delta_av = delta_av
        else:
            delta_av = self.curr_delta_av

        idx = 0
        if self.train_delta_m:
            delta_m = tf.expand_dims(position[:, self.n_latents + idx], axis=-1)
            self.curr_delta_m = delta_m
            idx += 1
        else:
            delta_m = self.curr_delta_m

        if self.train_delta_p:
            delta_p = tf.expand_dims(position[:, self.n_latents + idx], axis=-1)
            self.curr_delta_p = delta_p
            idx += 1
        else:
            delta_p = self.curr_delta_p

        if self.train_bias:
            bias = tf.expand_dims(position[:, self.n_latents + idx], axis=-1)
            self.curr_bias = bias
        else:
            bias = self.curr_bias

        pae_latents = tf.concat(
            (
                self.nflow.flow.bijector.forward(latents),
                delta_m,
                delta_p,
            ),
            axis=-1,
        )
        pae_latents = ks.layers.RepeatVector(self.nspec_dim)(pae_latents)

        amplitude = self.nflow.pae.decoder(
            (pae_latents, input_phase), training=False, mask=input_mask
        )

        if self.train_delta_m:
            amplitude *= tf.expand_dims(delta_m, axis=-1)
        if self.train_bias:
            amplitude += tf.expand_dims(bias, axis=-1)
        phase = input_phase
        if self.train_delta_p:
            phase += tf.expand_dims(delta_p, axis=-1)
        phase = phase[:, 0, :]

        # Measured average AE reconstruction error at current times
        mean_pae_recon_error = tf.transpose(
            tfp.math.interp_regular_1d_grid(
                x=tf.transpose(phase),
                x_ref_min=self.recon_error_centers[0],
                x_ref_max=self.recon_error_centers[-1],
                y_ref=self.recon_error,
            )
        )

        std_amplitude = tf.sqrt(
            (amplitude * mean_pae_recon_error) ** 2 + input_d_amplitude**2
        )
        std_amplitude *= input_mask
        std_amplitude += 1 - input_mask

        self.u_likelihood = tfd.Independent(
            tfd.MultivariateNormalDiag(loc=amplitude, scale_diag=std_amplitude),
            reinterpreted_batch_ndims=0,
        )

        log_prior = self.latents_prior.log_prob(latents)

        log_likelihood = self.u_likelihood.log_prob(input_amplitude * input_mask)

        spec_mask = tf.reduce_min(input_mask, axis=-1)
        n_unmasked_specs = tf.reduce_sum(
            tf.cast(input_amplitude[:, :, 0] > 0, tf.float32), axis=1
        )
        mean_log_likelihood = (
            tf.reduce_sum(log_likelihood * spec_mask, axis=-1) / n_unmasked_specs
        )

        return log_prior + mean_log_likelihood

    @override
    def __call__(
        self,
        inputs: "PosteriorInputs",
        *,
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "FTensor[S['batch_dim n_latents']]":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)

    @override
    def train_step(
        self,
        data: "TensorCompatible",
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:
        training = False

        # === Per Chain Setup ===
        # Fixed RNG
        tf.random.set_seed(self._chain)
        self._chain += 1
        (phase, amplitude, d_amplitude), mask, init_position = self.prep_data()

        def vals_and_grads(position: "PosteriorVars") -> "FTensor[S['']]":
            inputs = ((phase, amplitude, d_amplitude), position)
            log_prob = self(inputs, training=training, mask=mask)
            return self._loss(tf.zeros_like(log_prob), log_prob)

        def lbfgs(x: "PosteriorVars") -> "tuple(FTensor[S['']], FTensor[S['']])":
            return tfp.math.value_and_gradient(
                vals_and_grads, x, auto_unpack_single_arg=False
            )

        self.results = tfp.optimizer.lbfgs_minimize(
            lbfgs,
            initial_position=init_position,
            tolerance=self.tolerance,
            x_tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            num_correction_pairs=1,
        )

        position = self.results.position

        latents = position[:, : self.n_latents]
        zs = self.nflow.flow.bijector.forward(latents)
        if self.train_delta_av or self.nflow.physical_latents:
            delta_av = tf.expand_dims(latents[:, 0], axis=-1)
        else:
            delta_av = self.curr_delta_av

        idx = 0
        if self.train_delta_m:
            delta_m = tf.expand_dims(position[:, self.n_latents + idx], axis=-1)
            idx += 1
        else:
            delta_m = self.curr_delta_m

        if self.train_delta_p:
            delta_p = tf.expand_dims(position[:, self.n_latents + idx], axis=-1)
            idx += 1
        else:
            delta_p = self.curr_delta_p

        if self.train_bias:
            bias = tf.expand_dims(position[:, self.n_latents + idx], axis=-1)
        else:
            bias = self.curr_bias

        improved = (
            tf.expand_dims(self.results.objective_value, axis=-1)
            < self.best_objective_value
        )
        self.best_objective_value.assign(
            tf.where(
                improved,
                tf.expand_dims(self.results.objective_value, axis=-1),
                self.best_objective_value,
            )
        )
        self.best_chain.assign(
            tf.where(
                improved, self._chain * tf.ones_like(self.best_chain), self.best_chain
            )
        )
        self.best_converged.assign(
            tf.where(
                improved,
                tf.expand_dims(self.results.converged, axis=-1),
                self.best_converged,
            )
        )
        self.num_evals.assign(self.num_evals + self.results.num_objective_evaluations)

        self.best_zs.assign(tf.where(improved, zs, self.best_zs))
        self.best_latents.assign(tf.where(improved, latents, self.best_latents))
        self.best_delta_av.assign(tf.where(improved, delta_av, self.best_delta_av))
        self.best_delta_m.assign(tf.where(improved, delta_m, self.best_delta_m))
        self.best_delta_p.assign(tf.where(improved, delta_p, self.best_delta_p))
        self.best_bias.assign(tf.where(improved, bias, self.best_bias))

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train_model(
        self,
        stage: "Stage",
    ) -> ks.callbacks.History:
        self.stage = stage

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        if self.stage.savepath is not None:
            # --- Backup & Restore ---
            # Backup checkpoints each chain and restore if training got cancelled midway through
            if not self.force:
                backup_dir = self.stage.savepath / "backups"
                backup_callback = ks.callbacks.BackupAndRestore(str(backup_dir))
                callbacks.append(backup_callback)

            # --- Model Checkpoint ---
            checkpoint_callback = ks.callbacks.ModelCheckpoint(
                str(self.stage.savepath / self.weights_path),
                save_best_only=self.save_best,
                save_weights_only=True,
                # verbose=0,
            )
            callbacks.append(checkpoint_callback)

        # --- Terminate on NaN ---
        # Terminate training when a NaN loss is encountered
        callbacks.append(ks.callbacks.TerminateOnNaN())

        if stage.loadpath is not None:
            self.load_checkpoint(stage.loadpath)
        else:
            self.build_model()

        # === Train ===
        self._chain = 0
        return self.fit(
            self.prep_data(),
            initial_epoch=self._chain,
            epochs=self.stage.n_chains,
            batch_size=self.batch_dim,
            callbacks=callbacks,
            # verbose=0,
        )

    def build_model(
        self,
    ) -> None:
        if not self.built:
            # === Prep Data ===
            train_phase = self.stage.train_data.phase
            train_d_phase = self.stage.train_data.dphase
            train_amplitude = self.stage.train_data.amplitude
            train_d_amplitude = self.stage.train_data.sigma
            train_mask = self.stage.train_data.mask
            self.batch_dim = train_phase.shape[0]
            self.nspec_dim = train_phase.shape[1]

            self.recon_error, _recon_error_edges, self.recon_error_centers = (
                self.nflow.pae.recon_error((
                    tf.convert_to_tensor(train_phase, dtype=tf.float32),
                    tf.convert_to_tensor(train_amplitude, dtype=tf.float32),
                    tf.convert_to_tensor(train_d_amplitude, dtype=tf.float32),
                    tf.convert_to_tensor(train_mask, dtype=tf.int32),
                ))
            )

            data, mask, position = self.prep_data()
            loss = self._loss
            self.compile(
                loss=loss,
                run_eagerly=self.stage.debug,
            )
            self(
                (data, position),
                training=False,
                mask=mask,
            )

            self.log.debug("Trainable variables:")
            for var in self.trainable_variables:
                self.log.debug(f"{var.name}: {var.shape}")
            self.summary(print_fn=self.log.debug)  # Will show number of parameters

            self.built = True
        if not self.trained:
            self.best_zs = tf.Variable(
                self.init_zs,
                trainable=False,
            )
            self.best_latents = tf.Variable(
                self.init_latents,
                trainable=False,
            )
            self.best_delta_av = tf.Variable(self.init_delta_av, trainable=False)
            self.best_delta_m = tf.Variable(self.init_delta_m, trainable=False)
            self.best_delta_p = tf.Variable(self.init_delta_p, trainable=False)
            self.best_bias = tf.Variable(self.init_bias, trainable=False)
            self.best_objective_value = tf.Variable(
                np.inf * tf.ones((self.batch_dim, 1)), trainable=False
            )
            self.best_chain = tf.Variable(
                tf.zeros((self.batch_dim, 1)), trainable=False
            )
            self.best_converged = tf.Variable(
                tf.cast(tf.zeros((self.batch_dim, 1)), tf.bool), trainable=False
            )
            self.num_evals = tf.Variable(0, trainable=False)
            self.trained = True

    def save_checkpoint(self, savepath: "Path") -> None:
        self.save_weights(savepath / self.weights_path)
        self.save(savepath / self.model_path)

    def load_checkpoint(self, loadpath: "Path") -> None:
        self.trained = True
        self.build_model()
        self.load_weights(loadpath / self.weights_path)

    def prep_data(self) -> "tuple[PosteriorData, PosteriorMask, PosteriorPosition]":
        train_phase = self.stage.train_data.phase
        train_amplitude = self.stage.train_data.amplitude
        train_d_amplitude = self.stage.train_data.sigma
        train_mask = self.stage.train_data.mask

        data = (train_phase, train_amplitude, train_d_amplitude)
        mask = train_mask

        # --- Latents / Zs ---
        if self.stage.init_latents == "init":
            if self.random_initial_positions:
                self.stage.init_latents = "random"
            else:
                self.stage.init_latents = "data"
        if self.stage.init_latents == "random":
            init_latents = self.latents_prior.sample(self.batch_dim)
            init_zs = self.nflow.flow.bijector.forward(init_latents)
        elif self.stage.init_latents == "data":
            init_zs = self.nflow.pae.encoder(
                (
                    tf.convert_to_tensor(train_phase),
                    tf.convert_to_tensor(train_amplitude),
                ),
                training=False,
                mask=train_mask,
            )
            init_latents = self.nflow.flow.bijector.inverse(init_zs)
        else:
            init_latents = tf.zeros((self.batch_dim, self.n_latents))
            init_zs = self.nflow.flow.bijector.forward(init_latents)

        self.curr_zs = init_zs
        if self.init_zs is None:
            self.init_zs = self.curr_zs
        self.curr_latents = init_latents
        if self.init_latents is None:
            self.init_latents = self.curr_latents

        # --- ΔAᵥ ---
        if self.stage.init_delta_av == "init":
            if self.random_initial_positions:
                self.stage.init_delta_av = "random"
            else:
                self.stage.init_delta_av = "data"
        if self.stage.init_delta_av == "random":
            init_delta_av = tf.expand_dims(
                self.delta_av_prior.sample(self.batch_dim), axis=-1
            )
        elif self.stage.init_delta_av == "data":
            init_delta_av = self.init_zs[:, 0]
        elif self.stage.init_delta_av == "scale":
            delta_av_slope = (
                self.delta_av_max - self.delta_av_min
            ) / self.stage.n_chains
            delta_av_scale = (
                self.delta_av_min + (self.stage.n_chains - self._chain) * delta_av_slope
            )
            init_delta_av = tf.zeros((self.batch_dim, 1)) + delta_av_scale
        else:
            init_delta_av = tf.zeros((self.batch_dim, 1))
        self.curr_delta_av = init_delta_av
        if self.init_delta_av is None:
            self.init_delta_av = self.curr_delta_av

        # --- Δℳ  ---
        if self.stage.init_delta_m == "init":
            if self.random_initial_positions:
                self.stage.init_delta_m = "random"
            else:
                self.stage.init_delta_m = "data"
        if self.stage.init_delta_m == "random":
            init_delta_m = tf.expand_dims(
                self.delta_m_prior.sample(self.batch_dim), axis=-1
            )
        elif self.stage.init_delta_m == "data":
            init_delta_m = self.init_zs[:, self.nflow.pae.n_zs + 1]
        elif self.stage.init_delta_m == "scale":
            delta_m_slope = (self.delta_m_max - self.delta_m_min) / self.stage.n_chains
            delta_m_scale = (
                self.delta_m_min + (self.stage.n_chains - self._chain) * delta_m_slope
            )
            init_delta_m = tf.zeros((self.batch_dim, 1)) + delta_m_scale
        else:
            init_delta_m = tf.ones((self.batch_dim, 1))
        self.curr_delta_m = init_delta_m
        if self.init_delta_m is None:
            self.init_delta_m = self.curr_delta_m

        # --- Δ𝓅 ---
        if self.stage.init_delta_p == "init":
            if self.random_initial_positions:
                self.stage.init_delta_p = "random"
            else:
                self.stage.init_delta_p = "data"
        if self.stage.init_delta_p == "random":
            init_delta_p = tf.expand_dims(
                self.delta_p_prior.sample(self.batch_dim), axis=-1
            )
        elif self.stage.init_delta_p == "data":
            init_delta_p = self.init_zs[:, self.nflow.pae.n_zs + 2]
        elif self.stage.init_delta_p == "scale":
            delta_p_slope = (self.delta_p_pax - self.delta_p_min) / self.stage.n_chains
            delta_p_scale = (
                self.delta_p_min + (self.stage.n_chains - self._chain) * delta_p_slope
            )
            init_delta_p = tf.zeros((self.batch_dim, 1)) + delta_p_scale
        else:
            init_delta_p = tf.zeros((self.batch_dim, 1))
        self.curr_delta_p = init_delta_p
        if self.init_delta_p is None:
            self.init_delta_p = self.curr_delta_p

        # --- Bias ---
        if self.stage.init_bias == "init":
            if self.random_initial_positions:
                self.stage.init_bias = "random"
            else:
                self.stage.init_bias = "data"
        if self.stage.init_bias == "random":
            init_bias = tf.expand_dims(self.bias_prior.sample(self.batch_dim), axis=-1)
        else:
            init_bias = tf.zeros((self.batch_dim, 1))
        self.curr_bias = init_bias
        if self.init_bias is None:
            self.init_bias = self.curr_bias

        init_pos = []
        if self.train_delta_av:
            init_pos.append(self.curr_delta_av)
        init_pos.append(self.curr_latents)
        if self.train_delta_m:
            init_pos.append(self.curr_delta_m)
        if self.train_delta_p:
            init_pos.append(self.curr_delta_p)
        if self.train_bias:
            init_pos.append(self.curr_bias)
        init_position = tf.concat(init_pos, axis=-1)
        return data, mask, init_position
