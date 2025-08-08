# Copyright 2025 Patrick Armstrong
import gc
import os
from typing import TYPE_CHECKING, cast, override

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
    from typing import Any, Self
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    import numpy as np

    from supaernova.steps.pae import PAE
    from supaernova.steps.data import Data
    from supaernova.steps.nflow import NFlow
    from supaernova.steps.pae.tf import TFPAEModel
    from supaernova.configs.steps.nflow.tf import TFNFlowConfig


@ks.utils.register_keras_serializable("SuPAErnova")
class TFNFlowModel(ks.Model):
    def __init__(
        self,
        config: "NFlow",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        ks.backend.clear_session()
        gc.collect()

        super().__init__(*args, name=config.name.split()[-1], **kwargs)
        # --- Config ---
        global NFLOWMODELSTEP
        NFLOWMODELSTEP = config
        self.options: TFNFlowConfig = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed
        self.set_seed()

        self.debug: bool = self.options.debug
        self.profile: bool = self.options.profile
        self.data_step: Data = config.data_step
        self.kfold = config.kfold
        # Equivalent to `self.pae = ...` but avoids tf / ks from tracking self.pae
        vars(self)["pae"]: PAE = config.pae_step.model
        self.pae.trainable = False
        self.pae.encoder.trainable = False
        self.pae.decoder.trainable = False

        # --- Training ---
        self.built: bool = False
        self.batch_size: int = config.batch_size
        self.save_best: bool = self.options.save_best
        self.patience: int = self.options.patience
        self.validation_frac: float = self.options.validation_frac
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )
        self.log_path: str = f"{'best' if self.save_best else 'latest'}_logs/"

        self.lr: float = self.options.lr
        self.lr_decay_steps: float = self.options.lr_decay_steps
        self.lr_decay_rate: float = self.options.lr_decay_rate
        self.lr_weight_decay_rate: float = self.options.lr_weight_decay_rate

        self.activation: Callable[[tf.Tensor], tf.Tensor] = self.options.activation_fn
        self._scheduler: type[ks.optimizers.schedules.LearningRateSchedule] = (
            self.options.scheduler_cls
        )
        self._optimiser: type[ks.optimizers.Optimizer] = self.options.optimiser_cls
        loss: ks.losses.Loss = self.options.loss_cls()
        loss.model = self
        self._loss: ks.losses.Loss = loss
        self._loss_terms: dict[str, tf.Tensor]
        self._val_loss_terms: dict[str, tf.Tensor]

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

        self.epochs: int = self.options.epochs
        self._epoch: int = 0

        self.train_data: tf.Tensor
        self.test_data: tf.Tensor

        # --- Latent Dimensions ---
        self.n_u_latents: int = self.pae.n_z_latents
        self.n_physical_latents = 1 if self.physical_latents else 0
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

    @override
    def build(self, input_shape: tuple[int, ...]) -> None:
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
    @tf.function
    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
        mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        return self.flow.log_prob(inputs)

    def u_to_z(self, inputs: tf.Tensor, *, permute: bool = False) -> tf.Tensor:
        # If permute is True, then the incoming u_latents need to be permuted correctly
        if permute:
            inputs = tf.gather(inputs, self.u_to_z_permute, axis=-1)
        return self.flow.bijector.forward(inputs)

    def z_to_u(self, inputs: tf.Tensor, *, permute: bool = False) -> tf.Tensor:
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
        self, inputs: tf.Tensor, step: int, *, permute: bool = False
    ) -> tuple[tf.Tensor, bool]:
        if step <= 0:
            return tf.convert_to_tensor(inputs, dtype=tf.float32), False
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
            z_to_u_permute = tf.roll(
                tf.range(self.n_flow_latents),
                shift=shift,
                axis=0,
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

        n_batches_per_epoch = self.train_data.shape[0] / self.batch_size

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

        patience = self.patience
        if isinstance(patience, float):
            patience = int(self.epochs * patience)
        callbacks.append(
            ks.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                start_from_epoch=patience,
            )
        )

        if self.profile and savepath is not None:
            callbacks.append(
                ks.callbacks.TensorBoard(
                    log_dir=savepath.parent / self.log_path / savepath.stem,
                    write_graph=False,
                    write_images=False,
                    write_steps_per_second=False,
                    update_freq="epoch",
                    # profile_batch=(
                    #     int(n_batches_per_epoch * self.epochs * 0.1),
                    #     int(n_batches_per_epoch * self.epochs * 0.1) + 10,
                    # ),
                    embeddings_freq=0,
                ),
            )

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

        # === Train ===
        self._epoch = 0
        return self.fit(
            x=self.train_data,
            y=tf.zeros_like(self.train_data, dtype=tf.float32),
            initial_epoch=self._epoch,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            validation_data=(
                self.test_data,
                tf.zeros_like(self.test_data, dtype=tf.float32),
            ),
            validation_freq=1,
        )

    def get_data(self, latents: tf.Tensor) -> tf.Tensor:
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
            train_data = self.data_step.train_data[self.kfold]
            train_phase = tf.convert_to_tensor(train_data.time, dtype=tf.float32)
            train_amplitude = tf.convert_to_tensor(
                train_data.amplitude, dtype=tf.float32
            )
            train_mask = tf.convert_to_tensor(train_data.mask, dtype=tf.int32)
            train_spec_mask = tf.convert_to_tensor(
                self.pae.stage.train_spec_mask, dtype=tf.int32
            )
            train_sn_mask = tf.convert_to_tensor(
                self.pae.stage.train_sn_mask, dtype=tf.int32
            )

            train_pae_input = tf.concat((train_phase, train_amplitude), axis=-1)
            train_latents = self.pae.encoder(
                train_pae_input,
                training=False,
                mask=train_mask * train_spec_mask * train_sn_mask,
            )
            # train_inds = tf.squeeze(tf.cast(train_sn_mask, tf.bool), axis=(1, 2))
            # train_latents = tf.boolean_mask(train_latents, train_inds)
            self.train_data = self.get_data(train_latents)

            test_data = self.data_step.test_data[self.kfold]
            test_phase = tf.convert_to_tensor(test_data.time, dtype=tf.float32)
            test_amplitude = tf.convert_to_tensor(test_data.amplitude, dtype=tf.float32)
            test_mask = tf.convert_to_tensor(test_data.mask, dtype=tf.int32)
            test_spec_mask = tf.convert_to_tensor(
                self.pae.stage.test_spec_mask, dtype=tf.int32
            )
            test_sn_mask = tf.convert_to_tensor(
                self.pae.stage.test_sn_mask, dtype=tf.int32
            )

            test_pae_input = tf.concat((test_phase, test_amplitude), axis=-1)
            test_latents = self.pae.encoder(
                test_pae_input,
                training=False,
                mask=test_mask * test_spec_mask * test_sn_mask,
            )
            # test_inds = tf.squeeze(tf.cast(test_sn_mask, tf.bool), axis=(1, 2))
            # test_latents = tf.boolean_mask(test_latents, test_inds)
            self.test_data = self.get_data(test_latents)

            self.build(self.train_data.shape)

            if self._scheduler is not None:
                schedule = self._scheduler(
                    initial_learning_rate=self.lr,
                    decay_steps=self.lr_decay_steps,
                    decay_rate=self.lr_decay_rate,
                )
            else:
                schedule = self.lr
            optimiser = self._optimiser(
                learning_rate=schedule,
                weight_decay=self.lr_weight_decay_rate,
            )

            loss = self._loss
            self.compile(optimizer=optimiser, loss=loss, run_eagerly=self.debug)
            self(self.train_data, training=False)

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
