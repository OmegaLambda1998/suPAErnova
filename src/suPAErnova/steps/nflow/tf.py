# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    override,
)

import keras
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import (
    bijectors as tfb,
    distributions as tfd,
)

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.pae.tf import S, FTensor, TFPAEModel, TensorCompatible
    from suPAErnova.configs.steps.nflow.tf import TFNFlowModelConfig

    from .model import NFlowModelStep

    # === Custom Types ===
    type NFlowInputs = FTensor[S["batch_dim n_flow_latents"]]
    type NFlowOutputs = FTensor[S["batch_dim"]]


@keras.saving.register_keras_serializable("SuPAErnova")
class TFNFlowLoss(ks.losses.Loss):
    @override
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return -y_pred


@keras.saving.register_keras_serializable("SuPAErnova")
class TFNFlowModel(ks.Model):
    def __init__(
        self,
        config: "NFlowModelStep[TFNFlowModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=f"{config.name.split()[-1]}NFlowModel", **kwargs)
        # --- Config ---
        options = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.debug: bool = options.debug
        self.pae: TFPAEModel = cast("TFPAEModel", config.pae.model)

        # --- Training ---
        self.built: bool = False
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

        self.activation: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn
        self._optimiser: type[ks.optimizers.Optimizer] = options.optimiser_cls
        self._loss: ks.losses.Loss = (
            options.loss_cls() if options.loss_cls is not None else TFNFlowLoss()
        )

        self.n_hidden_units: int = options.n_hidden_units
        self.n_layers: int = options.n_layers
        self.batch_normalisation: bool = options.batch_normalisation
        self.physical_latents: bool = options.physical_latents

        self.learning_rate: float = options.learning_rate
        self.epochs: int = options.epochs
        self._epoch: int = 0

        self.data: "NFlowInputs"

        # --- Latent Dimensions ---
        self.n_flow_latents: int = self.pae.n_zs
        if self.physical_latents:
            self.n_flow_latents += 1

        # --- Layers ---
        self.gaussian: tfd.MultivariateNormalDiag = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.n_flow_latents),
            scale_diag=tf.ones(self.n_flow_latents),
        )

        permutations = [
            tf.roll(tf.range(self.n_flow_latents), shift=i, axis=0)
            for i in range(self.n_layers)
        ]

        bijectors = []

        for permutation in permutations:
            # First permute input dimensions
            bijectors.append(tfb.Permute(permutation=permutation))

            # Then (optionally) apply batch normalisation
            if self.batch_normalisation:
                bijectors.append(tfb.BatchNormalization(training=True))

            # Finally, pass to a Masked Autoregressive Flow
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                        params=2,
                        hidden_units=[self.n_hidden_units, self.n_hidden_units],
                        activation=self.activation,
                        use_bias=True,
                    ),
                )
            )

        # Optionally apply one last batch normalisation layer
        if self.batch_normalisation:
            bijectors.append(tfb.BatchNormalization(training=True))

        self.bijectors: tfb.Chain = tfb.Chain(bijectors)

        self.flow: tfd.TransformedDistribution = (
            tfp.distributions.TransformedDistribution(
                distribution=self.gaussian, bijector=self.bijectors
            )
        )

    @override
    def call(
        self,
        inputs: "NFlowInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "NFlowOutputs":
        training = False if training is None else training

        log_prob = self.flow.log_prob(inputs, training=training)
        if TYPE_CHECKING:
            log_prob = cast("FTensor[S['batch_dim']]", log_prob)

        return log_prob

    def train_model(
        self,
        *,
        savepath: "Path | None" = None,
    ) -> ks.callbacks.History:
        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        if savepath is not None:
            # --- Backup & Restore ---
            # Backup checkpoints each epoch and restore if training got cancelled midway through
            if not self.force:
                backup_dir = savepath / "backups"
                backup_callback = ks.callbacks.BackupAndRestore(str(backup_dir))
                callbacks.append(backup_callback)

            # --- Model Checkpoint ---
            checkpoint_callback = ks.callbacks.ModelCheckpoint(
                str(savepath / self.weights_path),
                save_best_only=self.save_best,
                save_weights_only=True,
                # verbose=0,
            )
            callbacks.append(checkpoint_callback)

        # --- Terminate on NaN ---
        # Terminate training when a NaN loss is encountered
        callbacks.append(ks.callbacks.TerminateOnNaN())

        self.build_model()

        # === Train ===
        self._epoch = 0
        return self.fit(
            x=self.data,
            y=tf.zeros_like(self.data),
            initial_epoch=self._epoch,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=callbacks,
            # verbose=0,
        )

    def get_data(
        self, latents: "FTensor[S['batch_dim nspec_dim pae_latents']]"
    ) -> "FTensor[S['batch_dim nspec_dim n_flow_latents']]":
        if self.physical_latents:
            data = latents[:, :, : self.pae.n_zs + 1]
        elif self.pae.physical_latents:
            data = latents[:, :, 1 : self.pae.n_zs + 1]
        else:
            data = latents
        return data

    def build_model(
        self,
    ) -> None:
        if not self.built:
            # === Prep Data ===
            train_phase = tf.convert_to_tensor(self.pae.stage.train_data.phase)
            train_amplitude = tf.convert_to_tensor(self.pae.stage.train_data.amplitude)
            train_mask = self.pae.stage.train_data.mask
            latents = self.pae.encoder(
                (train_phase, train_amplitude), training=False, mask=train_mask
            )
            self.data = self.get_data(latents)

            optimiser = self._optimiser(
                learning_rate=self.learning_rate,
            )
            loss = self._loss
            self.compile(
                optimizer=optimiser,
                loss=loss,
                run_eagerly=self.debug,
            )
            self((self.data), training=False)

            self.log.debug("Trainable variables:")
            for var in self.trainable_variables:
                self.log.debug(f"{var.name}: {var.shape}")
            self.summary(print_fn=self.log.debug)  # Will show number of parameters

            self.built = True

    def save_checkpoint(self, savepath: "Path") -> None:
        self.save_weights(savepath / self.weights_path)
        self.save(savepath / self.model_path)

    def load_checkpoint(self, loadpath: "Path") -> None:
        self.build_model()
        self.load_weights(loadpath / self.weights_path)
