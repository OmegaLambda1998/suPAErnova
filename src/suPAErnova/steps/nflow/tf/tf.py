# Copyright 2025 Patrick Armstrong
import os
import random as rn
from typing import TYPE_CHECKING, cast, override

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras as ks
from tqdm.keras import TqdmCallback
from tensorflow_probability import (
    bijectors as tfb,
    distributions as tfd,
)

if TYPE_CHECKING:
    from typing import Any, Self, Literal
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.pae.tf import S, FTensor, TFPAEModel, TensorCompatible
    from suPAErnova.steps.nflow.model import NFlowModelStep
    from suPAErnova.typing.backends.tf import GenericTensor
    from suPAErnova.configs.steps.nflow.tf import TFNFlowModelConfig
    from suPAErnova.typing.steps.nflow.nflow import (
        NULatents,
        NPhysicalLatents,
    )

    # === Custom Types ===
    type NFlowInputs = FTensor[S["batch_dim n_flow_latents"]]
    type NFlowOutputs = FTensor[S["batch_dim"]]


@ks.utils.register_keras_serializable("SuPAErnova")
class TFNFlowModel(ks.Model):
    def __init__(
        self,
        config: "NFlowModelStep[Literal['tf']]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        ks.backend.clear_session()
        super().__init__(*args, name=f"{config.name.split()[-1]}NFlowModel", **kwargs)
        # --- Config ---
        global NFLOWMODELSTEP
        NFLOWMODELSTEP = config
        self.options: "TFNFlowModelConfig" = cast("TFNFlowModelConfig", config.options)
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed

        self.debug: bool = self.options.debug
        # Equivalent to `self.pae = ...` but avoids tf / ks from tracking self.pae
        vars(self)["pae"]: TFPAEModel = cast("TFPAEModel", config.pae.model)
        self.pae.trainable = False
        self.pae.encoder.trainable = False
        self.pae.decoder.trainable = False

        # --- Training ---
        self.built: bool = False
        self.batch_size: int = self.options.batch_size
        self.save_best: bool = self.options.save_best
        self.patience: float = self.options.patience
        self.validation_frac: float = self.options.validation_frac
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )

        self.activation: Callable[[tf.Tensor], tf.Tensor] = self.options.activation_fn
        self._optimiser: type[ks.optimizers.Optimizer] = self.options.optimiser_cls
        loss: ks.losses.Loss = self.options.loss_cls()
        loss.model = self
        self._loss: ks.losses.Loss = loss
        self._loss_terms: dict[str, FTensor[tuple[None]]]
        self._val_loss_terms: dict[str, FTensor[tuple[None]]]

        self.n_hidden_units: int = self.options.n_hidden_units
        self.n_layers: int = self.options.n_layers
        self.batch_normalisation: bool = self.options.batch_normalisation
        # Only include physical latents (ΔAᵥ) if the PAE includes physical latents
        self.physical_latents: bool = (
            self.options.physical_latents and self.pae.physical_latents
        )

        # self.physical_latents doesn't match options.physical_latents
        if self.physical_latents ^ self.options.physical_latents:
            self.log.warning(
                "Can't include physical latents (ΔAᵥ) in NFlow model as it wasn't included in PAE model."
            )

        self.learning_rate: float = self.options.learning_rate
        self.epochs: int = self.options.epochs
        self._epoch: int = 0

        self.data: "NFlowInputs"

        # --- Latent Dimensions ---
        self.n_u_latents: NULatents = self.pae.n_z_latents
        self.n_physical_latents: NPhysicalLatents = 1 if self.physical_latents else 0
        self.n_flow_latents = self.n_u_latents + self.n_physical_latents

        self.shift: int = self.n_layers - 1
        self.u_to_z_permute: tf.Tensor = tf.constant(
            tf.roll(
                tf.range(self.n_flow_latents),
                shift=self.shift,
                axis=0,
            )
        )
        self.z_to_u_permute: tf.Tensor = tf.constant(
            tf.roll(
                tf.range(self.n_flow_latents),
                shift=-self.shift,
                axis=0,
            )
        )

        # --- Layers ---
        self.flow: tfd.TransformedDistribution

        self.set_seed()

    @override
    def build(self, input_shape) -> None:
        gaussian = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.n_flow_latents),
            scale_diag=tf.ones(self.n_flow_latents),
            name="NFlowGaussian",
        )

        permute = tf.constant(tf.roll(tf.range(self.n_flow_latents), shift=1, axis=0))
        bijectors = []
        for n in range(self.n_layers):
            # First permute input dimensions
            bijectors.append(
                tfb.Permute(
                    permutation=permute,
                    name=f"NFlowPermute_{n}",
                )
            )

            # Then (optionally) apply batch normalisation
            if self.batch_normalisation:
                bijectors.append(
                    tfb.BatchNormalization(
                        training=True,
                        name=f"NFlowBatchNorm_{n}",
                    )
                )

            # Finally, pass to a Masked Autoregressive Flow
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                        params=2,
                        hidden_units=[self.n_hidden_units, self.n_hidden_units],
                        activation=self.activation,
                        use_bias=True,
                        name=f"NFlowARNetwork_{n}",
                    ),
                    name=f"NFlowARFlow_{n}",
                )
            )

        # Optionally apply one last batch normalisation layer
        if self.batch_normalisation:
            bijectors.append(
                tfb.BatchNormalization(
                    training=True,
                    name="NFlowBatchNorm",
                )
            )

        # The first element is a permutation, but we don't want to immediately permute the latents
        # So remove the first element
        bijectors = tfb.Chain(
            bijectors[1:],
            name="NFlowChain",
        )

        self.flow = tfd.TransformedDistribution(
            distribution=gaussian,
            bijector=bijectors,
            name="NFlowFlow",
        )

    @override
    def call(
        self,
        inputs: "NFlowInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "NFlowOutputs":
        return self.flow.log_prob(inputs)

    def u_to_z(self, inputs: "NFlowInputs", *, permute: bool = False) -> "NFlowInputs":
        # If permute is True, then the incoming u_latents need to be permuted correctly
        if permute:
            inputs = tf.gather(inputs, self.u_to_z_permute, axis=-1)
        return self.flow.bijector.forward(inputs)

    def z_to_u(self, inputs: "NFlowInputs", *, permute: bool = False) -> "NFlowInputs":
        u_latents = self.flow.bijector.inverse(inputs)
        # If permute is True, then the outgoing u_latents need to be un-permuted correctly
        # Reverse, permute, reverse undoes the initial permutation
        if permute:
            u_latents = tf.reverse(
                tf.gather(
                    tf.reverse(u_latents, axis=(-1,)), self.z_to_u_permute, axis=-1
                ),
                axis=(-1,),
            )
        return u_latents

    def z_to_u_steps(
        self, inputs: "NFlowInputs", step: int, *, permute: bool = False
    ) -> tuple["NFlowInputs", bool]:
        if step <= 0:
            return tf.convert_to_tensor(inputs), False
        bijectors = self.flow.bijector.bijectors
        step = max(1, step)
        shift = 0
        u_latents = inputs
        for bijector in bijectors[:step]:
            u_latents = bijector.inverse(u_latents)
            if isinstance(bijector, tfb.Permute):
                shift -= 1
        # If permute is True, then the outgoing u_latents need to be un-permuted correctly
        # Reverse, permute, reverse undoes the initial permutation
        if permute:
            z_to_u_permute = tf.constant(
                tf.roll(
                    tf.range(self.n_flow_latents),
                    shift=shift,
                    axis=0,
                )
            )
            u_latents = tf.reverse(
                tf.gather(tf.reverse(u_latents, axis=(-1,)), z_to_u_permute, axis=-1),
                axis=(-1,),
            )
        return u_latents, isinstance(bijectors[:step][-1], tfb.Permute)

    def train_model(
        self,
        *,
        savepath: "Path | None" = None,
    ) -> ks.callbacks.History:
        self.build_model()

        n_batches_per_epoch = self.data.shape[0] / self.batch_size

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        # --- Backup & Restore ---
        # Backup checkpoints each epoch and restore if training got cancelled midway through
        if not self.force and savepath is not None:
            backup_dir = savepath / "backups"
            backup_callback = ks.callbacks.BackupAndRestore(
                str(backup_dir),
                save_freq=max(1, int(0.1 * self.epochs * n_batches_per_epoch)),
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
                        verbose=0,
                    ),
                ),
            )
        )

        # callbacks.append(
        #     ks.callbacks.EarlyStopping(
        #         monitor="val_loss",
        #         patience=self.patience,
        #     )
        # )

        # === Train ===
        self._epoch = 0
        self.set_seed()
        return self.fit(
            x=self.data,
            y=tf.zeros_like(self.data, dtype=tf.float32),
            validation_split=self.validation_frac,
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_epoch=self._epoch,
            # steps_per_epoch=self.data.shape[0] // self.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

    def get_data(
        self, latents: "FTensor[S['batch_dim nspec_dim pae_latents']]"
    ) -> "FTensor[S['batch_dim nspec_dim n_flow_latents']]":
        # Get the first n_z_latents + 1 latents
        # If there are no physical pae latents, this is all latents
        # If there are physical pae latents, this includes ΔAᵥ
        data = latents[:, 0, : self.pae.n_z_latents + 1]
        # If there are physical pae latents, and we don't want to include them, remove the first element (ΔAᵥ)
        if self.pae.physical_latents and (not self.physical_latents):
            data = data[:, 1:]
        return data

    def build_model(self) -> None:
        if not self.built:
            # === Prep Data ===
            train_phase = tf.convert_to_tensor(self.pae.stage.train_data.time)
            train_amplitude = tf.convert_to_tensor(self.pae.stage.train_data.amplitude)
            train_mask = tf.convert_to_tensor(self.pae.stage.train_data.mask)
            train_sn_mask = tf.convert_to_tensor(self.pae.stage.train_sn_mask)
            train_spec_mask = tf.convert_to_tensor(self.pae.stage.train_spec_mask)
            latents = self.pae.encoder(
                (train_phase, train_amplitude),
                training=False,
                mask=train_mask * train_spec_mask,
            )
            inds = tf.squeeze(tf.cast(train_sn_mask, tf.bool), axis=(1, 2))
            train_latents = tf.boolean_mask(latents, inds)
            self.data = self.get_data(train_latents)
            self.build(self.data.shape)

            optimiser = self._optimiser(
                learning_rate=self.learning_rate,
            )
            loss = self._loss
            self.compile(optimizer=optimiser, loss=loss, run_eagerly=self.debug)
            dummy = self(self.data, training=False)

            self.log.debug("Trainable variables:")
            for var in self.trainable_variables:
                self.log.debug(f"{var.name}: {var.shape}")
            self.summary(
                print_fn=self.log.debug, show_trainable=True
            )  # Will show number of parameters

            self.built = True

    def save_checkpoint(self, savepath: "Path") -> None:
        (savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
        tf.train.Checkpoint(
            self,
        ).save(f"{savepath / self.ckpt_path}/")

    def load_checkpoint(self, loadpath: "Path") -> None:
        self.build_model()

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

    @override
    def set_seed(self, seed: int = 0) -> None:
        seed = self.seed + seed
        tf.random.set_seed(seed)
