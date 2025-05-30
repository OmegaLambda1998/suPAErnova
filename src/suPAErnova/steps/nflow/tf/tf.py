# Copyright 2025 Patrick Armstrong
import os
import random as rn
from typing import TYPE_CHECKING, cast, override

import numpy as np
import tf_keras as ks
import tensorflow as tf
from tqdm.keras import TqdmCallback
import tensorflow_probability as tfp
from tensorflow_probability import (
    bijectors as tfb,
    distributions as tfd,
)

if TYPE_CHECKING:
    from typing import Any, Self
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.pae.tf import S, FTensor, TFPAEModel, TensorCompatible
    from suPAErnova.configs.steps.nflow.tf import TFNFlowModelConfig

    from .model import NFlowModelStep

    # === Custom Types ===
    type NFlowInputs = FTensor[S["batch_dim n_flow_latents"]]
    type NFlowOutputs = FTensor[S["batch_dim"]]


@ks.utils.register_keras_serializable("SuPAErnova")
class TFNFlowModel(ks.Model):
    def __init__(
        self,
        config: "NFlowModelStep[TFNFlowModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=f"{config.name.split()[-1]}NFlowModel", **kwargs)
        # --- Config ---
        global NFLOWMODELSTEP
        NFLOWMODELSTEP = config
        options = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed
        self.set_seed()

        self.debug: bool = options.debug
        self.pae: TFPAEModel = cast("TFPAEModel", config.pae.model)
        self.pae.trainable = False
        self.pae.encoder.trainable = False
        self.pae.decoder.trainable = False

        # --- Training ---
        self.built: bool = False
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )

        self.activation: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn
        self._optimiser: type[ks.optimizers.Optimizer] = options.optimiser_cls
        loss: ks.losses.Loss = options.loss_cls()
        loss.model = self
        self._loss: ks.losses.Loss = loss

        self.n_hidden_units: int = options.n_hidden_units
        self.n_layers: int = options.n_layers
        self.batch_normalisation: bool = options.batch_normalisation
        # Only include physical latents (ΔAᵥ) if the PAE includes physical latents
        self.physical_latents: bool = (
            options.physical_latents and self.pae.physical_latents
        )

        # self.physical_latents doesn't match options.physical_latents
        if self.physical_latents ^ options.physical_latents:
            self.log.warning(
                "Can't include physical latents (ΔAᵥ) in NFlow model as it wasn't included in PAE model."
            )

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

        permute = tf.roll(tf.range(self.n_flow_latents), shift=1, axis=0)

        bijectors = []

        for _ in range(self.n_layers):
            # First permute input dimensions
            bijectors.append(tfb.Permute(permutation=permute))

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

        # The first element is a permutation, but we don't want to immediately permute the latents
        # So remove the first element
        self.bijectors: tfb.Chain = tfb.Chain(bijectors[1:])

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
        log_prob = self.flow.log_prob(inputs)
        if TYPE_CHECKING:
            log_prob = cast("FTensor[S['batch_dim']]", log_prob)

        return log_prob

    def train_model(
        self,
        *,
        savepath: "Path | None" = None,
    ) -> ks.callbacks.History:
        n_batches_per_epoch = (
            self.pae.stage.train_data.amplitude.shape[0] / self.batch_size
        )

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        # --- Backup & Restore ---
        # Backup checkpoints each epoch and restore if training got cancelled midway through
        if not self.force and savepath is not None:
            backup_dir = savepath / "backups"
            backup_callback = ks.callbacks.BackupAndRestore(
                str(backup_dir),
                save_freq=max(1, int(0.01 * self.epochs * n_batches_per_epoch)),
            )
            callbacks.append(backup_callback)

        # --- Terminate on NaN ---
        # Terminate training when a NaN loss is encountered
        callbacks.append(ks.callbacks.TerminateOnNaN())

        # --- TQDM Progress Bar ---
        callbacks.append(
            cast(
                "ks.callbacks.Callback",
                cast(
                    "object",
                    TqdmCallback(
                        epochs=self.epochs,
                        data_size=self.pae.stage.train_data.amplitude.shape[0],
                        batch_size=self.batch_size,
                        verbose=0,
                    ),
                ),
            )
        )

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
            verbose=0,
        )

    def get_data(
        self, latents: "FTensor[S['batch_dim nspec_dim pae_latents']]"
    ) -> "FTensor[S['batch_dim nspec_dim n_flow_latents']]":
        # Get the first n_zs + 1 latents
        # If there are no physical pae latents, this is all latents
        # If there are physical pae latents, this includes ΔAᵥ
        data = latents[:, :, : self.pae.n_zs + 1]
        # If there are physical pae latents, and we don't want to include them, remove the first element (ΔAᵥ)
        if self.pae.physical_latents and (not self.physical_latents):
            data = data[:, :, 1:]
        return data

    def build_model(self) -> None:
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

            self.built = True

    def save_checkpoint(self, savepath: "Path") -> None:
        self.save_weights(savepath / self.weights_path)
        self.save(savepath / self.model_path)
        (savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
        tf.train.Checkpoint(
            self,
        ).save(f"{savepath / self.ckpt_path}/")

    def load_checkpoint(self, loadpath: "Path") -> None:
        self.build_model()

        self.train_step((
            tf.zeros_like(self.data),
            tf.zeros_like(self.data),
        ))

        tf.train.Checkpoint(
            self,
        ).restore(
            tf.train.latest_checkpoint(f"{loadpath / self.ckpt_path}/")
        ).assert_existing_objects_matched()

    @override
    def get_config(self) -> dict[str, "Any"]:
        return {**super().get_config()}

    @override
    @classmethod
    def from_config(cls, config: dict[str, "Any"]) -> "Self":
        global NFLOWMODELSTEP
        return cls(NFLOWMODELSTEP)

    def build_from_config(self, _config: dict[str, "Any"]) -> None:
        self.build_model()

    def set_seed(self, seed: int = 0) -> None:
        seed = self.seed + seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        rn.seed(seed)
