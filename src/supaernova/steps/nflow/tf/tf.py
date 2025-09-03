# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, Self, cast, override

import numpy as np
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

    from numpy import typing as npt

    from supaernova.steps.pae import TFPAEModel
    from supaernova.steps.nflow import NFlow
    from supaernova.configs.steps.data import SNPAEData
    from supaernova.typing.backends.tf import Loss, LearningRateSchedule
    from supaernova.configs.steps.nflow.tf import TFNFlowConfig

NFLOWMODELSTEP: "NFlow"


@ks.utils.register_keras_serializable("SuPAErnova")
class TFNFlowModel(ks.Model):
    def __init__(
        self: "Self",
        config: "NFlow",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=config.name.split()[-1], **kwargs)
        # --- Config ---
        global NFLOWMODELSTEP
        NFLOWMODELSTEP = config
        self.options: TFNFlowConfig = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed
        self.debug: bool = self.options.debug
        self.profile: bool = self.options.profile
        self.set_seed()

        # Data Dimensions
        self.sn_dim = config.sn_dim
        self.spec_dim = config.spec_dim
        self.wl_dim = config.wl_dim

        self.latents: tf.Tensor
        self.data: SNPAEData = config.data
        self.data_mask: npt.NDArray[bool] = config.mask
        self.sn_mask: npt.NDArray[bool] = config.sn_mask
        self.spec_mask: npt.NDArray[bool] = config.spec_mask
        self.wl_mask: npt.NDArray[bool] = config.wl_mask
        input_mask = self.data_mask * self.sn_mask * self.spec_mask * self.wl_mask
        valid_wl_mask = tf.cast(
            tf.logical_not(
                tf.logical_and(
                    tf.logical_not(tf.cast(input_mask, tf.bool)),
                    tf.cast(self.wl_mask, tf.bool),
                )
            ),
            tf.int32,
        )
        mask_spec = tf.cast(
            tf.reduce_max(valid_wl_mask, axis=-1),
            tf.int32,
        )
        mask_sn = tf.reduce_max(mask_spec, axis=-1, keepdims=True)
        self.mask = tf.cast(mask_sn, tf.float32)

        self.train_latents: tf.Tensor
        self.train_data: SNPAEData = config.train_data
        self.train_data_mask: npt.NDArray[bool] = config.train_mask
        self.train_sn_mask: npt.NDArray[bool] = config.train_sn_mask
        self.train_spec_mask: npt.NDArray[bool] = config.train_spec_mask
        self.train_wl_mask: npt.NDArray[bool] = config.train_wl_mask
        input_train_mask = (
            self.train_data_mask
            * self.train_sn_mask
            * self.train_spec_mask
            * self.train_wl_mask
        )
        valid_wl_train_mask = tf.cast(
            tf.logical_not(
                tf.logical_and(
                    tf.logical_not(tf.cast(input_train_mask, tf.bool)),
                    tf.cast(self.train_wl_mask, tf.bool),
                )
            ),
            tf.int32,
        )
        train_mask_spec = tf.cast(
            tf.reduce_max(valid_wl_train_mask, axis=-1),
            tf.int32,
        )
        train_mask_sn = tf.reduce_max(train_mask_spec, axis=-1, keepdims=True)
        self.train_mask = tf.cast(train_mask_sn, tf.float32)

        self.test_latents: tf.Tensor
        self.test_data: SNPAEData = config.test_data
        self.test_data_mask: npt.NDArray[bool] = config.test_mask
        self.test_sn_mask: npt.NDArray[bool] = config.test_sn_mask
        self.test_spec_mask: npt.NDArray[bool] = config.test_spec_mask
        self.test_wl_mask: npt.NDArray[bool] = config.test_wl_mask
        input_test_mask = (
            self.test_data_mask
            * self.test_sn_mask
            * self.test_spec_mask
            * self.test_wl_mask
        )
        valid_wl_test_mask = tf.cast(
            tf.logical_not(
                tf.logical_and(
                    tf.logical_not(tf.cast(input_test_mask, tf.bool)),
                    tf.cast(self.test_wl_mask, tf.bool),
                )
            ),
            tf.int32,
        )
        test_mask_spec = tf.cast(
            tf.reduce_max(valid_wl_test_mask, axis=-1),
            tf.int32,
        )
        test_mask_sn = tf.reduce_max(test_mask_spec, axis=-1, keepdims=True)
        self.test_mask = tf.cast(test_mask_sn, tf.float32)

        self.val_latents: tf.Tensor
        self.val_data: SNPAEData = config.val_data
        self.val_data_mask: npt.NDArray[bool] = config.val_mask
        self.val_sn_mask: npt.NDArray[bool] = config.val_sn_mask
        self.val_spec_mask: npt.NDArray[bool] = config.val_spec_mask
        self.val_wl_mask: npt.NDArray[bool] = config.val_wl_mask
        input_val_mask = (
            self.val_data_mask
            * self.val_sn_mask
            * self.val_spec_mask
            * self.val_wl_mask
        )
        valid_wl_val_mask = tf.cast(
            tf.logical_not(
                tf.logical_and(
                    tf.logical_not(tf.cast(input_val_mask, tf.bool)),
                    tf.cast(self.val_wl_mask, tf.bool),
                )
            ),
            tf.int32,
        )
        val_mask_spec = tf.cast(
            tf.reduce_max(valid_wl_val_mask, axis=-1),
            tf.int32,
        )
        val_mask_sn = tf.reduce_max(val_mask_spec, axis=-1, keepdims=True)
        self.val_mask = tf.cast(val_mask_sn, tf.float32)

        # Equivalent to `self.pae = ...` but avoids tf / ks from tracking self.pae
        self.pae: TFPAEModel
        vars(self)["pae"] = config.pae
        self.pae.trainable = False
        self.pae.encoder.trainable = False
        self.pae.decoder.trainable = False

        # --- Training ---
        self.built: bool = False
        self._epoch: int = 0

        self.batch_size: int = config.batch_size
        self.save_best: bool = self.options.save_best
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )
        self.log_path: str = f"{'best' if self.save_best else 'latest'}_logs/"
        self.patience: int = self.options.patience

        self.lr: float = self.options.lr
        self.lr_decay_steps: int = self.options.lr_decay_steps
        self.lr_decay_rate: float = self.options.lr_decay_rate
        self.lr_weight_decay_rate: float = self.options.lr_weight_decay_rate

        self.activation: Callable[[tf.Tensor], tf.Tensor] = self.options.activation_fn
        self._scheduler: type[LearningRateSchedule] = self.options.scheduler_cls
        self._optimiser: type[ks.optimizers.Optimizer] = self.options.optimiser_cls
        loss: Loss = self.options.loss_cls()
        loss.model = self
        self._loss: Loss = loss
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
                "Can't include physical latents (ΔAᵥ) in NFlow model as it wasn't included in the PAE model."
            )

        self.epochs: int = self.options.epochs
        self._epoch: int = 0

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

        self._get_latents()

    @override
    def build(self: "Self", input_shape: tf.TensorShape) -> None:
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
        self: "Self",
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        training = False if training is None else training

        # === Unpack Inputs ===
        latents = inputs[..., :-1]
        mask = inputs[..., -1:][..., 0]

        # === Calculate Log Probability ===
        log_prob = self.flow.log_prob(latents)

        # Replace NaN and inf with infinitely low log prob
        inf_log_prob = -np.inf * tf.ones_like(log_prob)

        masked_log_prob = tf.where(tf.cast(mask, tf.bool), log_prob, inf_log_prob)

        return tf.where(
            tf.math.is_finite(masked_log_prob),
            masked_log_prob,
            inf_log_prob,
        )

    def u_to_z(self: "Self", inputs: tf.Tensor, *, permute: bool = False) -> tf.Tensor:
        # If permute is True, then the incoming u_latents need to be permuted correctly
        if permute:
            inputs = tf.gather(inputs, self.u_to_z_permute, axis=-1)
        return self.flow.bijector.forward(inputs)

    def z_to_u(self: "Self", inputs: tf.Tensor, *, permute: bool = False) -> tf.Tensor:
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
        self: "Self", inputs: tf.Tensor, step: int, *, permute: bool = False
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
        self: "Self",
        *,
        savepath: "Path | None" = None,
    ) -> ks.callbacks.History:
        self.build_model()

        n_batches_per_epoch = self.data_mask.shape[0] / self.batch_size

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

        # === Prep Data ===
        data = tf.concat(
            (self.latents, self.mask),
            axis=-1,
        )

        train_data = tf.concat(
            (self.train_latents, self.train_mask),
            axis=-1,
        )

        test_data = tf.concat(
            (self.test_latents, self.test_mask),
            axis=-1,
        )

        val_data = tf.concat(
            (self.val_latents, self.val_mask),
            axis=-1,
        )

        # === Train ===
        self._epoch = 0
        return self.fit(
            x=train_data,
            y=tf.zeros_like(self.train_latents),
            initial_epoch=self._epoch,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            validation_data=(val_data, tf.zeros_like(self.val_latents)),
            validation_freq=1,
        )

    def _get_latents(self: "Self") -> None:
        for dt in ["train_", "test_", "val_", ""]:
            data: SNPAEData = getattr(self, f"{dt}data")
            phase = tf.convert_to_tensor(data.time, dtype=tf.float32)
            amplitude = tf.convert_to_tensor(data.amplitude, dtype=tf.float32)
            data_mask = tf.convert_to_tensor(
                getattr(self, f"{dt}data_mask"), dtype=tf.int32
            )
            sn_mask = tf.convert_to_tensor(
                getattr(self, f"{dt}sn_mask"), dtype=tf.int32
            )
            spec_mask = tf.convert_to_tensor(
                getattr(self, f"{dt}spec_mask"), dtype=tf.int32
            )
            wl_mask = tf.convert_to_tensor(
                getattr(self, f"{dt}wl_mask"), dtype=tf.int32
            )
            pae_inputs = tf.concat((phase, amplitude), axis=-1)
            latents = self.pae.encoder(
                pae_inputs,
                training=False,
                mask=data_mask,
                sn_mask=sn_mask,
                spec_mask=spec_mask,
                wl_mask=wl_mask,
            )

            # Get the first n_z_latents + 1 latents
            # If there are no physical pae latents, this is all latents
            # If there are physical pae latents, this includes ΔAᵥ
            latents = latents[:, 0, : self.pae.n_z_latents + 1]
            # If there are physical pae latents, and we don't want to include them, remove the first element (ΔAᵥ)
            if self.pae.physical_latents and (not self.physical_latents):
                latents = latents[:, 1:]
            setattr(self, f"{dt}latents", latents)

    def build_model(self: "Self") -> None:
        if not self.built:
            self.build(self.train_latents.shape)

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

            train_data = tf.concat(
                (self.train_latents, self.train_mask),
                axis=-1,
            )
            self(train_data, training=False)

            if self.debug:
                self.log.debug("Trainable variables:")
                for var in self.trainable_variables:
                    self.log.debug(f"{var.name}: {var.shape}")
                self.summary(
                    print_fn=self.log.debug, show_trainable=True
                )  # Will show number of parameters

            self.built = True

    def save_checkpoint(self: "Self", savepath: "Path") -> None:
        (savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
        tf.train.Checkpoint(
            self,
        ).save(f"{savepath / self.ckpt_path}/")

    def load_checkpoint(self: "Self", loadpath: "Path") -> None:
        self.build_model()

        tf.train.Checkpoint(
            self,
        ).restore(
            tf.train.latest_checkpoint(f"{loadpath / self.ckpt_path}/")
        ).assert_existing_objects_matched()

    @override
    def get_config(self: "Self") -> dict[str, "Any"]:
        return {**super().get_config()}

    @override
    @classmethod
    def from_config(cls: type["Self"], config: dict[str, "Any"]) -> "Self":
        global NFLOWMODELSTEP
        return cls(NFLOWMODELSTEP)

    def build_from_config(self: "Self", _config: dict[str, "Any"]) -> None:
        self.build_model()

    @override
    def set_seed(self: "Self", seed: int = 0) -> None:
        seed = self.seed + seed
        tf.random.set_seed(seed)
