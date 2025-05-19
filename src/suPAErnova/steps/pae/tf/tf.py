# Copyright 2025 Patrick Armstrong
from typing import (
    TYPE_CHECKING,
    cast,
    override,
)

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from tqdm.keras import TqdmCallback
import tensorflow_probability as tfp

if TYPE_CHECKING:
    from typing import (
        Any,
        Literal,
        Annotated,
    )
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable, Sequence

    from numpy import typing as npt

    from suPAErnova.steps.pae.model import PAEModelStep
    from suPAErnova.configs.steps.pae import PAEStage
    from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig

    # === Custom Types ===
    S = Literal
    type FTensor[Shape: str] = Annotated[tf.Tensor, tf.float32, Shape]
    type ITensor[Shape: str] = Annotated[tf.Tensor, tf.int32, Shape]
    type FRTensor[Shape: str] = Annotated[tf.RaggedTensor, tf.float32, Shape]
    type IRTensor[Shape: str] = Annotated[tf.RaggedTensor, tf.int32, Shape]
    type Tensor[Shape: str] = FTensor[Shape] | ITensor[Shape]
    type RaggedTensor[Shape: str] = FRTensor[Shape] | IRTensor[Shape]
    type GenericTensor[Shape: str] = Tensor[Shape] | RaggedTensor[Shape]

    type TensorCompatible = (
        tf.Tensor
        | str
        | float
        | np.ndarray[Any, Any]
        | np.number[Any]
        | Sequence[TensorCompatible]
    )

    # --- Encoder Tensors ---
    EncoderInputsShape = tuple[tuple[int, int, int], tuple[int, int, int]]
    EncoderInputs = tuple[
        FTensor[S["batch_dim nspec_dim phase_dim"]],
        FTensor[S["batch_dim nspec_dim wl_dim"]],
    ]
    EncoderOutputs = FTensor[S["batch_dim nspec_dim n_pae_latents"]]

    # --- Decoder Tensors ---
    DecoderInputsShape = tuple[tuple[int, int, int], tuple[int, int, int]]
    DecoderInputs = tuple[
        FTensor[S["batch_dim nspec_dim n_pae_latents"]],
        FTensor[S["batch_dim nspec_dim phase_dim"]],
    ]
    DecoderOutputs = FTensor[S["batch_dim nspec_dim wl_dim"]]

    # --- Model Tensors ---
    RawData = tuple[
        npt.NDArray[np.float32],  # phase
        npt.NDArray[np.float32],  # dphase
        npt.NDArray[np.float32],  # amp
        npt.NDArray[np.float32],  # damp
        npt.NDArray[np.int32],  # mask
    ]
    PrepData = tuple[
        ITensor[S["n_sn nspec_dim"]],
        IRTensor[S["n_sn n_spec_to_shuffle(sn)"]],
        IRTensor[S["n_sn n_spec_to_shuffle(sn)"]],
    ]

    EpochInputs = tuple[
        tuple[
            FTensor[S["batch_dim nspec_dim phase_dim"]],  # phase
            FTensor[S["batch_dim nspec_dim phase_dim"]],  # dphase
            FTensor[S["batch_dim nspec_dim wl_dim"]],  # amp
            FTensor[S["batch_dim nspec_dim wl_dim"]],  # damp
            ITensor[S["batch_dim nspec_dim wl_dim"]],  # mask
        ],
        tuple[
            ITensor[S["batch_dim nspec_dim"]],
            IRTensor[S["batch_dim n_spec_to_shuffle(sn)"]],
            IRTensor[S["batch_dim n_spec_to_shuffle(sn)"]],
        ],
    ]

    ModelInputs = tuple[
        FTensor[S["batch_dim nspec_dim phase_dim"]],  # phase
        FTensor[S["batch_dim nspec_dim wl_dim"]],  # amp
        FTensor[S["batch_dim nspec_dim wl_dim"]],  # damp
        ITensor[S["batch_dim nspec_dim wl_dim"]],  # mask
    ]


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEEncoder(ks.layers.Layer):
    def __init__(
        self,
        options: "TFPAEModelConfig",
        name: str,
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=f"{name.split()[-1]}Encoder", **kwargs)

        # --- Config Params ---
        n_physical: "Literal[0, 3]" = 3 if options.physical_latents else 0
        n_zs: int = options.n_z_latents
        self.n_pae_latents: int = n_physical + n_zs
        self.latents_z_mask: "ITensor[S['n_pae_latents']]"
        self.latents_physical_mask: "ITensor[S['n_pae_latents']]"

        self.stage_num: int
        self.moving_means: "FTensor[S['n_pae_latents']]"

        self.encode_dims: list[int] = options.encode_dims

        self.activation: "Callable[[tf.Tensor], tf.Tensor]" = options.activation_fn
        self.regulariser: ks.regularizers.Regularizer | None = (
            options.kernel_regulariser_cls(options.kernel_regulariser_penalty)
            if options.kernel_regulariser_cls is not None
            else None
        )

        self.dropout: float = options.dropout
        self.batch_normalisation: bool = options.batch_normalisation

        # --- Layers ---
        self.encode_layers: list[ks.layers.Dense]
        self.dropout_layers: list[ks.layers.Dropout | ks.layers.Identity]
        self.batch_normalisation_layers: list[
            ks.layers.BatchNormalization | ks.layers.Identity
        ]
        self.encode_nspec_layer: ks.layers.Dense
        self.encode_output_layer: ks.layers.Dense
        self.repeat_latent_layer: ks.layers.RepeatVector

    @override
    def build(self, input_shape: "EncoderInputsShape") -> None:
        (_batch_dim, nspec_dim, _phase_dim), (_batch_dim, _nspec_dim, _wl_dim) = (
            input_shape
        )

        # Encode from input layer dimensions into intermediate dimensions
        self.encode_layers = [
            ks.layers.Dense(
                encode_dim,
                activation=self.activation,
                kernel_regularizer=self.regulariser,
            )
            for encode_dim in self.encode_dims
        ]
        # Dropout layer
        self.dropout_layers = [
            ks.layers.Dropout(self.dropout, noise_shape=[None, 1, None])
            if self.dropout > 0
            else ks.layers.Identity(trainable=False)
            for _ in self.encode_dims
        ]
        # Batch normalisation layer
        self.batch_normalisation_layers = [
            ks.layers.BatchNormalization()
            if self.batch_normalisation
            else ks.layers.Identity(trainable=False)
            for _ in self.encode_dims
        ]

        # Encode from intermediate dimensions into nspec_dim dimensions
        self.encode_nspec_layer = ks.layers.Dense(
            nspec_dim,
            activation=self.activation,
            kernel_regularizer=self.regulariser,
        )

        # Encode from nspec_dim dimensions into output (latent) dimensions
        self.encode_output_layer = ks.layers.Dense(
            self.n_pae_latents,
            kernel_regularizer=self.regulariser,
            use_bias=False,
        )

        # Repeat latent vector to match nspec_dim
        self.repeat_latent_layer = ks.layers.RepeatVector(nspec_dim)

    @override
    def call(
        self,
        inputs: "EncoderInputs",
        mask: "ITensor[S['batch_dim nspec_dim wl_dim']] | None" = None,
        training: bool | None = None,
    ) -> "EncoderOutputs":
        training = False if training is None else training

        input_phase = inputs[0]
        input_amp = inputs[1]
        input_mask = (
            mask if mask is not None else tf.ones_like(input_amp, dtype=tf.int32)
        )

        # Create initial input layer
        x: "FTensor[S['batch_dim nspec_dim wl_dim+1']]" = ks.layers.concatenate([
            input_amp,
            input_phase,
        ])

        # Encode from input layers to intermediate dimensions
        for i, encode_layer in enumerate(self.encode_layers):
            tf.debugging.check_numerics(x, "1")
            x = encode_layer(x)
            tf.debugging.check_numerics(x, "2")
            x = self.dropout_layers[i](x, training=training)
            tf.debugging.check_numerics(x, "3")
            x = self.batch_normalisation_layers[i](x)
            tf.debugging.check_numerics(x, "4")

        # Encode from intermediate dimensions to nspec_dim dimensions
        x = self.encode_nspec_layer(x)

        # Encode from nspec_dim dimensions to output (latent) dimensions
        x = self.encode_output_layer(x)

        # Determine which spectra to keep
        is_kept = tf.cast(tf.reduce_min(input_mask, axis=-1, keepdims=True), tf.float32)
        if TYPE_CHECKING:
            is_kept = cast("FTensor[S['batch_dim nspec_dim 1']]", is_kept)

        # Latent tensor is the average of the latent values over all unmasked spectra
        # First sum latents over all unmasked spectra
        batch_sum = tf.reduce_sum(x * is_kept, axis=-2)
        if TYPE_CHECKING:
            batch_sum = cast("FTensor[S['batch_dim n_pae_latents']]", batch_sum)

        # Then determine the number of unmasked spectra
        batch_num = tf.math.maximum(tf.reduce_sum(is_kept, axis=-2), y=1)
        if TYPE_CHECKING:
            batch_num = cast("FTensor[S['batch_dim 1']]", batch_num)

        # Finally, calculate the average
        latents = batch_sum / batch_num
        if TYPE_CHECKING:
            latents = cast("FTensor[S['batch_dim n_pae_latents']]", latents)

        # Mask latents which aren't being trained
        # The latents are ordered by training stage
        # ΔAᵥ -> zs -> Δℳ  -> Δ𝓅
        # Note that this differs from the order used in the legacy SuPAErnova code:
        # Δ𝓅 -> Δℳ  -> ΔAᵥ -> zs
        masked_latents = tf.zeros(self.n_pae_latents - self.stage_num)
        if TYPE_CHECKING:
            masked_latents = cast(
                "FTensor[S['n_pae_latents-stage_num']]", masked_latents
            )
        unmasked_latents = tf.ones(self.stage_num)
        if TYPE_CHECKING:
            unmasked_latents = cast("FTensor[S['stage_num']]", unmasked_latents)
        latent_mask = tf.concat((unmasked_latents, masked_latents), axis=0)
        if TYPE_CHECKING:
            latent_mask = cast("FTensor[S['n_pae_latents']]", latent_mask)
        latents *= latent_mask

        if training:
            # Normalise the physical latents within this batch such that they have a mean of 0
            latents_sum = tf.reduce_sum(latents, axis=0, keepdims=True)
            latents_num = tf.reduce_sum(tf.ones_like(latents), axis=0, keepdims=True)
            latents_mean = self.latents_physical_mask * latents_sum / latents_num
            latents -= latents_mean
            if TYPE_CHECKING:
                latents_sum = cast("FTensor[S['1 n_pae_latents']]", latents_sum)
                latents_num = cast("FTensor[S['1 n_pae_latents']]", latents_num)
                latents_mean = cast("FTensor[S['1 n_pae_latents']]", latents_mean)
        else:
            # Normalise the physical latents within this batch such that the entire unbatched sample has a mean of 0
            latents -= self.latents_physical_mask * self.moving_means
        if TYPE_CHECKING:
            latents = cast("FTensor[S['batch_dim n_pae_latents']]", latents)

        # Repeat latent layers across NSpecDim
        encoded = self.repeat_latent_layer(latents)
        if TYPE_CHECKING:
            encoded = cast("FTensor[S['batch_dim nspec_dim n_pae_latents']]", encoded)

        return encoded

    @override
    def __call__(
        self,
        inputs: "EncoderInputs",
        *,
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "FTensor[S['batch_dim nspec_dim n_pae_latents']]":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEDecoder(ks.layers.Layer):
    def __init__(
        self, options: "TFPAEModelConfig", name: str, *args: "Any", **kwargs: "Any"
    ) -> None:
        super().__init__(*args, name=f"{name.split()[-1]}Decoder", **kwargs)
        # --- Config Params ---
        self.physical_latents: bool = options.physical_latents
        self.n_physical: "Literal[0, 3]" = 3 if options.physical_latents else 0
        self.n_zs: int = options.n_z_latents

        self.wl_dim: int
        self.decode_dims: list[int] = options.decode_dims

        self.activation: "Callable[[tf.Tensor], tf.Tensor]" = options.activation_fn

        self.regulariser: ks.regularizers.Regularizer | None = (
            options.kernel_regulariser_cls(options.kernel_regulariser_penalty)
            if options.kernel_regulariser_cls is not None
            else None
        )
        self.batch_normalisation: bool = options.batch_normalisation

        colourlaw = options.colourlaw
        if colourlaw is not None:
            _, colourlaw = np.loadtxt(colourlaw, unpack=True)
        self.colourlaw: "npt.NDArray[np.float64] | None" = colourlaw

        # --- Layers ---
        self.decode_nspec_layer: ks.layers.Dense
        self.decode_layers: list[ks.layers.Dense]
        self.batch_normalisation_layers: list[
            ks.layers.BatchNormalization | ks.layers.Identity
        ]
        self.decode_output_layer: ks.layers.Dense
        self.colourlaw_layer: ks.layers.Dense | ks.layers.Identity

    @override
    def build(self, input_shape: "DecoderInputsShape") -> None:
        (
            (_batch_dim, nspec_dim, _n_pae_latents),
            (_batch_dim, _nspec_dim, _phase_dim),
        ) = input_shape
        # Project from input dimensions into nspec_dim dimensions
        self.decode_nspec_layer = ks.layers.Dense(
            nspec_dim,
            activation=self.activation,
            kernel_regularizer=self.regulariser,
        )

        # Decode from nspec_dim dimensions into intermediate dimensions
        self.decode_layers = [
            ks.layers.Dense(
                decode_dim,
                activation=self.activation,
                kernel_regularizer=self.regulariser,
            )
            for decode_dim in self.decode_dims
        ]
        self.batch_normalisation_layers = [
            ks.layers.BatchNormalization()
            if self.batch_normalisation
            else ks.layers.Identity(trainable=False)
            for decode_dim in self.decode_dims
        ]

        # Decode from intermediate dimensions to output dimensions
        self.decode_output_layer = ks.layers.Dense(
            self.wl_dim, kernel_regularizer=self.regulariser
        )

        # Colourlaw
        self.colourlaw_layer = (
            ks.layers.Dense(
                self.wl_dim,
                kernel_initializer=None
                if self.colourlaw is None
                else tf.constant_initializer(self.colourlaw),
                use_bias=False,
                trainable=True,
                kernel_constraint=None
                if self.colourlaw is None
                else ks.constraints.NonNeg(),
            )
            if self.physical_latents
            else ks.layers.Identity(trainable=False)
        )

    @override
    def call(
        self,
        inputs: "DecoderInputs",
        mask: "ITensor[S['batch_dim nspec_dim wl_dim']] | None" = None,
        training: bool | None = None,
    ) -> "DecoderOutputs":
        training = False if training is None else training

        input_latents = inputs[0]
        input_phase = inputs[1]
        input_mask = (
            mask
            if mask is not None
            else tf.ones((tf.shape(input_phase)[:-1], self.wl_dim), dtype=tf.int32)
        )

        # Extract physical parameters (if applicable)
        if self.physical_latents:
            delta_av_latent = input_latents[:, :, 0:1]
            zs_latent = input_latents[:, :, 1 : self.n_zs + 1]
            delta_m_latent = input_latents[:, :, self.n_zs + 1 : self.n_zs + 2]
            delta_p_latent = input_latents[:, :, self.n_zs + 2 : self.n_zs + 3]

            # Apply Δ𝓅 shift
            input_phase += delta_p_latent
        else:
            zs_latent = input_latents
            delta_av_latent = delta_m_latent = delta_p_latent = tf.zeros_like(
                zs_latent[:, :, 0]
            )  # Define here even though they never get used, as it allows for better XLA compilation
        if TYPE_CHECKING:
            delta_av_latent = cast(
                "FTensor[S['batch_dim nspec_dim 1']]", delta_av_latent
            )
            zs_latent = cast("FTensor[S['batch_dim nspec_dim n_zs']]", zs_latent)
            delta_m_latent = cast("FTensor[S['batch_dim nspec_dim 1']]", delta_m_latent)
            delta_p_latent = cast("FTensor[S['batch_dim nspec_dim 1']]", delta_p_latent)

        # Create initial input layer
        x: FTensor[S["batch_dim nspec_dim n_zs+phase_dim"]] = ks.layers.concatenate([
            zs_latent,
            input_phase,
        ])

        # Decode from input (latent) dimensions to nspec_dim dimensions
        x = self.decode_nspec_layer(x)

        # Decode from nspec_dim dimensions to intermediate dimensions
        for i, decode_layer in enumerate(self.decode_layers):
            x = decode_layer(x)
            x = self.batch_normalisation_layers[i](x)

        # Decode from intermediate dimensions to output dimension
        amplitude = self.decode_output_layer(x)

        # Apply ΔAᵥ / Δℳ  shift
        if self.physical_latents:
            # Calculate Colourlaw
            colourlaw = self.colourlaw_layer(delta_av_latent)

            amplitude *= tf.pow(10.0, -0.4 * (colourlaw + delta_m_latent))

        # Apply RELU activation function
        if not training:
            amplitude = tf.nn.relu(amplitude)

        # Zero out masked elements
        return amplitude * tf.cast(
            tf.reduce_max(input_mask, axis=-1, keepdims=True), tf.float32
        )

    @override
    def __call__(
        self,
        inputs: "DecoderInputs",
        *,
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "FTensor[S['batch_dim nspec_dim wl_dim']]":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)


@keras.saving.register_keras_serializable("SuPAErnova")
class TFPAEModel(ks.Model):
    def __init__(
        self,
        config: "PAEModelStep[Literal['tf']]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=f"{config.name.split()[-1]}PAEModel", **kwargs)
        # --- Config ---
        self.options: "TFPAEModelConfig" = cast("TFPAEModelConfig", config.options)
        self.log: "Logger" = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force

        # --- Latent Dimensions ---
        self.physical_latents: bool = self.options.physical_latents
        self.n_physical: "Literal[0, 3]" = 3 if self.options.physical_latents else 0
        self.n_zs: int = self.options.n_z_latents
        self.n_pae_latents: int = self.n_physical + self.n_zs

        # --- Layers ---
        self.encoder: TFPAEEncoder = TFPAEEncoder(self.options, config.name)
        self.decoder: TFPAEDecoder = TFPAEDecoder(self.options, config.name)
        self.decoder.wl_dim = config.wl_dim

        # --- Training ---
        self.built: bool = False
        self._epoch: int = 0
        self.batch_size: int = self.options.batch_size
        self.save_best: bool = self.options.save_best
        self.weights_path: str = f"{'best' if self.save_best else 'latest'}.weights.h5"
        self.model_path: str = f"{'best' if self.save_best else 'latest'}.model.keras"

        # Data Offsets
        self.phase_offset_scale: "Literal[0, -1] | float" = (
            self.options.phase_offset_scale
        )
        self.amplitude_offset_scale: float = self.options.amplitude_offset_scale
        self.mask_fraction: float = self.options.mask_fraction

        # Training functions
        self._scheduler: type[ks.optimizers.schedules.LearningRateSchedule] = (
            self.options.scheduler_cls
        )
        self._optimiser: type[ks.optimizers.Optimizer] = self.options.optimiser_cls

        self.stage: PAEStage
        self.latents_z_mask: "ITensor[S['n_pae_latents']]"
        self.latents_physical_mask: "ITensor[S['n_pae_latents']]"

        # --- Loss ---
        self._loss: ks.losses.Loss
        self._loss_terms: dict[str, "FTensor[S['']]"]
        self.loss_residual_penalty: float = self.options.loss_residual_penalty

        self.loss_delta_av_penalty: float = self.options.loss_delta_av_penalty
        self.loss_delta_m_penalty: float = self.options.loss_delta_m_penalty
        self.loss_delta_p_penalty: float = self.options.loss_delta_p_penalty
        self.loss_physical_penalty: float = sum((
            self.loss_delta_av_penalty,
            self.loss_delta_m_penalty,
            self.loss_delta_p_penalty,
        ))  # Only calculate physical latent penalties if at least one penalty scaling is > 0

        self.loss_covariance_penalty: float = self.options.loss_covariance_penalty
        self.loss_decorrelate_all: bool = self.options.loss_decorrelate_all
        self.loss_decorrelate_dust: bool = self.options.loss_decorrelate_dust

        self.loss_clip_delta: float = self.options.loss_clip_delta

        # --- Metrics ---
        self.loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss")
        self.pred_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss_pred")
        self.model_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss_model")
        self.resid_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss_resid")
        self.delta_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss_delta")
        self.cov_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="loss_cov")

    @property
    @override
    def metrics(self) -> list[ks.metrics.Metric]:
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [
            self.loss_tracker,
            self.pred_loss_tracker,
            self.model_loss_tracker,
        ]
        if self.loss_residual_penalty > 0:
            metrics.append(self.resid_loss_tracker)
        if self.physical_latents and self.loss_physical_penalty > 0:
            metrics.append(self.delta_loss_tracker)
        if self.loss_covariance_penalty > 0:
            metrics.append(self.cov_loss_tracker)
        return metrics

    @override
    def call(
        self,
        inputs: "EncoderInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "tuple[EncoderOutputs, DecoderOutputs]":
        training = False if training is None else training

        input_phase = inputs[0]
        encoded = self.encoder(inputs, training=training, mask=mask)
        if TYPE_CHECKING:
            encoded = cast("FTensor[S['batch_dim nspec_dim n_pae_latents']]", encoded)

        decoded = self.decoder((encoded, input_phase), training=training, mask=mask)
        if TYPE_CHECKING:
            decoded = cast("FTensor[S['batch_dim nspec_dim wl_dim']]", decoded)

        return encoded, decoded

    @override
    def __call__(
        self,
        inputs: "EncoderInputs",
        *,
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "tuple[FTensor[S['batch_dim nspec_dim n_pae_latents']], FTensor[S['batch_dim nspec_dim wl_dim']]]":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)

    @override
    def compute_loss(
        self,
        x: "TensorCompatible | None" = None,
        y: "TensorCompatible | None" = None,
        y_pred: "TensorCompatible | None" = None,
        sample_weight: "TensorCompatible | None" = None,
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "FTensor[S['']] | None":
        if x is None or y is None or y_pred is None or sample_weight is None:
            return None
        training = False if training is None else training

        latents = tf.convert_to_tensor(x, dtype=tf.float32)
        input_amp = tf.convert_to_tensor(y, dtype=tf.float32)
        output_amp = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        input_d_amp = tf.convert_to_tensor(sample_weight, dtype=tf.float32)
        input_mask = tf.cast(
            (
                tf.convert_to_tensor(mask, dtype=tf.int32)
                if mask is not None
                else tf.ones_like(input_amp, dtype=tf.int32)
            ),
            tf.float32,
        )
        latents_mask = tf.reduce_max(
            tf.reduce_min(input_mask, axis=-1, keepdims=True), axis=-2
        )

        if TYPE_CHECKING:
            latents = cast("FTensor[S['batch_dim nspec_dim n_pae_latents']]", latents)
            input_amp = cast("FTensor[S['batch_dim nspec_dim wl_dim']]", input_amp)
            output_amp = cast("FTensor[S['batch_dim nspec_dim wl_dim']]", output_amp)
            input_mask = cast("ITensor[S['batch_dim nspec_dim wl_dim']]", input_mask)
            latents_mask = cast("FTensor[S['batch_dim 1']]", latents_mask)

        loss = self.options.loss_cls()
        loss.latents = latents
        loss.input_amp = input_amp
        loss.input_d_amp = input_d_amp
        loss.output_amp = output_amp
        loss.input_mask = input_mask
        loss.latents_mask = latents_mask
        loss.model = self
        self._loss = loss

        pred_loss = loss(y_true=input_amp, y_pred=output_amp)

        model_loss = tf.reduce_sum(self.losses)
        if TYPE_CHECKING:
            pred_loss = cast("FTensor[S['']]", pred_loss)
            model_loss = cast("FTensor[S['']]", model_loss)
        loss_terms = {
            self.pred_loss_tracker.name: pred_loss,
            self.model_loss_tracker.name: model_loss,
        }

        # --- Penalties ---
        # Penalise larger residuals between the input amplitude and the output amplitude
        if self.loss_residual_penalty > 0:
            residual_penalty = self.loss_residual_penalty * tf.reduce_mean(
                tf.abs(
                    tf.reduce_sum(input_mask * (input_amp - output_amp), axis=(-2, -1))
                )
            )
            loss_terms[self.resid_loss_tracker.name] = residual_penalty

        # Penalise phyical latents which are far from unity (one for multiplicative, zero for additive)
        if self.physical_latents and self.loss_physical_penalty > 0:
            latents_penalty_scale = tf.concat(
                (
                    tf.constant((self.loss_delta_av_penalty,), dtype=tf.float32),
                    tf.zeros(self.n_zs, dtype=tf.float32),
                    tf.constant((self.loss_delta_m_penalty,), dtype=tf.float32),
                    tf.constant((self.loss_delta_p_penalty,), dtype=tf.float32),
                ),
                axis=0,
            )
            preferred_latent_values = tf.concat(
                (
                    tf.constant((1.0,)),  # ΔAᵥ = 1
                    tf.zeros(self.n_zs),  # zs = 0
                    tf.constant((0.0,)),  # Δℳ  = 0
                    tf.constant((0.0,)),  # Δ𝓅 = 0
                ),
                axis=0,
            )
            median_latent_values_per_sn = latents_mask * tfp.stats.percentile(
                latents,
                50,
                interpolation="midpoint",
                axis=[1],
            )
            median_latent_values = tfp.stats.percentile(
                median_latent_values_per_sn,
                50,
                interpolation="midpoint",
                axis=[0],
            )
            physical_latents_offset = (
                preferred_latent_values - median_latent_values
            ) ** 2
            physical_latents_penalty = tf.reduce_sum(
                self.latents_physical_mask
                * latents_penalty_scale
                * physical_latents_offset
            )
            loss_terms[self.delta_loss_tracker.name] = physical_latents_penalty

        if self.loss_covariance_penalty > 0:
            eps = tf.constant(1e-10)
            masked_latents = latents[:, 0, :] * latents_mask
            num_masked_latents = tf.reduce_sum(latents_mask) + eps
            latents_mean = (
                tf.reduce_sum(masked_latents, axis=0, keepdims=True)
                / num_masked_latents
            )
            latents_norm = masked_latents - latents_mean
            latents_cov = (
                tf.matmul(latents_norm, latents_norm, transpose_a=True)
                / num_masked_latents
            )
            latents_var = tf.reduce_sum(latents_norm**2, axis=0) / num_masked_latents
            std_outer = (
                tf.sqrt(
                    tf.matmul(
                        tf.expand_dims(latents_var, -1), tf.expand_dims(latents_var, 0)
                    )
                )
            ) + eps
            latents_cov_norm = latents_cov / std_outer

            latent_dim = tf.shape(latents_cov_norm)[0]
            cov_mask = 1.0 - tf.eye(latent_dim)

            # Decorrelate all -> Punish all latents for off-diagonal terms
            if not self.loss_decorrelate_all:
                decorrelate = tf.concat(
                    (
                        tf.constant((
                            1.0 if self.loss_decorrelate_dust else 0.0,
                        )),  # delta_av,
                        tf.zeros(self.n_zs),  # zs
                        tf.constant((1.0,)),  # delta_m,
                        tf.constant((0.0,)),  # delta_p,
                    ),
                    axis=0,
                )
                cov_mask *= tf.expand_dims(decorrelate, axis=0)
                cov_mask += tf.transpose(cov_mask)
                cov_mask = tf.clip_by_value(cov_mask, 0.0, 1.0)

            loss_cov = tf.reduce_sum(tf.square(latents_cov_norm * cov_mask)) / (
                tf.reduce_sum(cov_mask) + eps
            )

            loss_covariance_penalty = self.loss_covariance_penalty * loss_cov
            loss_terms[self.cov_loss_tracker.name] = loss_covariance_penalty

        total_loss = tf.add_n(loss_terms.values())
        loss_terms[self.loss_tracker.name] = total_loss

        self._loss_terms = loss_terms
        return total_loss

    @override
    def train_step(
        self,
        data: "TensorCompatible",
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:
        training = True

        # === Per Epoch Setup ===
        # Fixed RNG
        tf.random.set_seed(self._epoch)
        self._epoch += 1
        # tf.print(
        #     "epoch",
        #     self._epoch,
        #     [np.min(abs(w)) for w in self.encoder.weights if np.isclose(w, 0).any()],
        # )

        # --- Setup Data ---
        (phase, amplitude, d_amplitude, mask) = self.prep_data_per_epoch(
            cast("EpochInputs", data)
        )

        with tf.GradientTape() as tape:
            latents, pred_amplitude = self(
                (phase, amplitude), training=training, mask=mask
            )
            if TYPE_CHECKING:
                latents = cast(
                    "FTensor[S['batch_dim nspec_dim n_pae_latents']]", latents
                )
                pred_amplitude = cast(
                    "FTensor[S['batch_dim nspec_dim wl_dim']]", pred_amplitude
                )
            loss = self.compute_loss(
                x=latents,
                y=amplitude,
                y_pred=pred_amplitude,
                sample_weight=d_amplitude,
                training=training,
                mask=mask,
            )
        if loss is None:
            return {m.name: m.result() for m in self.metrics}

        gradients = tape.gradient(loss, self.trainable_variables)
        cast("ks.optimizers.Optimizer", self.optimizer).apply_gradients(
            zip(gradients, self.trainable_variables, strict=True)
        )

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            metric.update_state(self._loss_terms[metric.name])

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train_model(
        self,
        stage: "PAEStage",
    ) -> None:
        self.stage = stage

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        if self.stage.savepath is not None:
            # --- Backup & Restore ---
            # Backup checkpoints each epoch and restore if training got cancelled midway through
            if not self.force:
                backup_dir = self.stage.savepath / "backups"
                backup_callback = ks.callbacks.BackupAndRestore(str(backup_dir))
                callbacks.append(backup_callback)

            # --- Model Checkpoint ---
            checkpoint_callback = ks.callbacks.ModelCheckpoint(
                str(self.stage.savepath / self.weights_path),
                save_best_only=self.save_best,
                save_weights_only=True,
            )
            callbacks.append(checkpoint_callback)

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
                        epochs=self.stage.epochs,
                        data_size=self.stage.train_data.amplitude.shape[0],
                        batch_size=self.batch_size,
                        verbose=0,
                    ),
                ),
            )
        )

        if stage.loadpath is not None:
            self.load_checkpoint(stage.loadpath)
        else:
            self.build_model()

        # === Prep Data ===
        data = (
            self.stage.train_data.phase,
            self.stage.train_data.dphase,
            self.stage.train_data.amplitude,
            self.stage.train_data.sigma,
            self.stage.train_data.mask,
        )
        prep = self.prep_data(data)

        test_data = (
            self.stage.test_data.phase,
            self.stage.test_data.dphase,
            self.stage.test_data.amplitude,
            self.stage.test_data.sigma,
            self.stage.test_data.mask,
        )

        val_data = (
            self.stage.val_data.phase,
            self.stage.val_data.dphase,
            self.stage.val_data.amplitude,
            self.stage.val_data.sigma,
            self.stage.val_data.mask,
        )

        # TODO: Validation Epoch Callback
        # TODO: Test Callback

        # === Train ===
        self._epoch = 0
        self.fit(
            x=data,
            y=prep,
            initial_epoch=self._epoch,
            epochs=self.stage.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

    def build_model(self) -> None:
        if not self.built:
            # Mask tensors to select specific latents
            if self.physical_latents:
                self.latents_z_mask = tf.concat(
                    (tf.zeros(1), tf.ones(self.n_zs), tf.zeros(1), tf.zeros(1)), axis=0
                )
            else:
                self.latents_z_mask = tf.ones(self.n_zs)
            self.latents_physical_mask = tf.abs(
                self.latents_z_mask - 1
            )  # Swap 0s and 1s

            # === Setup Encoder ===
            self.encoder.stage_num = self.stage.stage
            self.encoder.moving_means = tf.convert_to_tensor(self.stage.moving_means)
            self.encoder.latents_z_mask = self.latents_z_mask
            self.encoder.latents_physical_mask = self.latents_physical_mask

            schedule = self._scheduler(
                initial_learning_rate=self.stage.learning_rate,
                decay_steps=self.stage.learning_rate_decay_steps,
                decay_rate=self.stage.learning_rate_decay_rate,
            )
            optimiser = self._optimiser(
                learning_rate=schedule,
                weight_decay=self.stage.learning_rate_weight_decay_rate,
            )
            self.compile(
                optimizer=optimiser,
                loss=self.options.loss_cls(),
                run_eagerly=self.stage.debug,  # or True,
            )
            self(
                (
                    tf.convert_to_tensor(self.stage.train_data.phase),
                    tf.convert_to_tensor(self.stage.train_data.amplitude),
                ),
                training=False,
                mask=self.stage.train_data.mask,
            )

            self.log.debug("Trainable variables:")
            for var in self.trainable_variables:
                self.log.debug(f"{var.name}: {var.shape}")
            self.summary(print_fn=self.log.debug)  # Will show number of parameters
            self.built = True

    def get_loss(self, loss: str):
        return self._loss_terms.get(loss, tf.constant(0, dtype=tf.float32))

    def save_checkpoint(self, savepath: "Path") -> None:
        # Normalise mean of physical latents to 0 across all batches
        if self.physical_latents:
            phase = tf.convert_to_tensor(self.stage.train_data.phase)
            amplitude = tf.convert_to_tensor(self.stage.train_data.amplitude)
            mask = tf.convert_to_tensor(self.stage.train_data.mask)
            encoded = self.encoder((phase, amplitude), training=False, mask=mask)
            if TYPE_CHECKING:
                encoded = cast("FTensor[S['n_sn nspec_dim n_pae_latents']]", encoded)

            latents_sum = tf.reduce_sum(encoded, axis=(0, 1))
            latents_num = tf.reduce_sum(tf.ones_like(encoded), axis=(0, 1))
            moving_means = latents_sum / latents_num
            if TYPE_CHECKING:
                latents_sum = cast("FTensor[S['n_pae_latents']]", latents_sum)
                latents_num = cast("FTensor[S['n_pae_latents']]", latents_num)
                moving_means = cast("FTensor[S['n_pae_latents']]", moving_means)
            self.encoder.moving_means = moving_means
        checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        checkpoint.save(savepath / "checkpoint/")
        # self.save_weights(savepath / self.weights_path)
        # self.save(savepath / self.model_path)

    def load_checkpoint(
        self, loadpath: "Path", *, reset_weights: bool | None = None
    ) -> None:
        # tf.print(
        #     1, [np.min(abs(w)) for w in self.encoder.weights if np.isclose(w, 0).any()]
        # )
        self.build_model()
        init_weights = self.encoder.encode_output_layer.get_weights()[0]
        # self.load_weights(loadpath / self.weights_path)
        checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        checkpoint.restore(tf.train.latest_checkpoint(loadpath / "checkpoint/"))
        # tf.print(
        #     2, [np.min(abs(w)) for w in self.encoder.weights if np.isclose(w, 0).any()]
        # )
        reset_weights = (
            (self.stage.stage < self.n_pae_latents)
            if reset_weights is None
            else reset_weights
        )
        if reset_weights:
            weights = self.encoder.encode_output_layer.get_weights()[0]
            # Set the weights of the newly introduced latent parameter to effectively 0
            #   Since the initial weights are random values which we then divide by 100
            weights[:, self.stage.stage - 1] = (
                init_weights[:, self.stage.stage - 1] / 100
            )
            self.encoder.encode_output_layer.set_weights([weights])
            # tf.print(
            #     3,
            #     [
            #         np.min(abs(w))
            #         for w in self.encoder.weights
            #         if np.isclose(w, 0).any()
            #     ],
            # )

    def prep_data(self, data: "RawData") -> "PrepData":
        (_phase, _d_phase, _amplitude, _d_amplitude, mask) = data
        n_sn, _nspec_dim, _wl_dim = tf.shape(mask)

        # === Randomised Spectral Masking ===
        # We want a subset of each SN's spectra to be randomly masked
        # We do this by first building a mask tensor, masking the first n spectra per SN
        # Where n is the number of spectra we want to mask for that SNe

        # We then shuffle this mask to randomise which spectra make up this subset.

        # --- Build Masking tensor ---

        # Which spectra are currently masked for each SN
        spec_mask = tf.reduce_max(mask, axis=-1)
        if TYPE_CHECKING:
            spec_mask = cast("ITensor[S['n_sn nspec_dim']]", spec_mask)

        # The number of unmasked spectra for each SN
        n_unmasked_spectra_per_sn = tf.reduce_sum(spec_mask, axis=-1)
        if TYPE_CHECKING:
            n_unmasked_spectra_per_sn = cast(
                "ITensor[S['n_sn']]", n_unmasked_spectra_per_sn
            )

        # Determine how many spectra to mask for each SN
        #   by multiplying the number of unmasked SN by the mask_fraction
        n_spectra_to_mask_per_sn = tf.cast(
            tf.round(
                self.mask_fraction * tf.cast(n_unmasked_spectra_per_sn, tf.float32)
            ),
            tf.int32,
        )
        if TYPE_CHECKING:
            n_spectra_to_mask_per_sn = cast(
                "ITensor[S['n_sn']]", n_spectra_to_mask_per_sn
            )

        # How many spectra we are going to mask in total

        # Get the indices (batch_ind, nspec_ind) of each unmasked spectrum
        unmasked_inds = tf.cast(tf.where(tf.reduce_max(mask, axis=-1)), tf.int32)
        if TYPE_CHECKING:
            unmasked_inds = cast("ITensor[S['n_unmasked 2']]", unmasked_inds)

        # For each unmasked spectra, get the index of its corresponding SN
        unmasked_sn_inds = unmasked_inds[:, 0]
        if TYPE_CHECKING:
            unmasked_sn_inds = cast("ITensor[S['n_unmasked']]", unmasked_sn_inds)

        # For each unmasked spectra, get its index
        unmasked_spec_inds = unmasked_inds[:, 1]
        if TYPE_CHECKING:
            unmasked_spec_inds = cast("ITensor[S['n_unmasked']]", unmasked_spec_inds)

        # For each unmasked spectra, get the number of spectra we want to mask in its corresponding SN
        n_spectra_to_mask_per_spectra = tf.cast(
            tf.gather(n_spectra_to_mask_per_sn, unmasked_sn_inds), tf.int32
        )
        if TYPE_CHECKING:
            n_spectra_to_mask_per_spectra = cast(
                "ITensor[S['n_unmasked']]", n_spectra_to_mask_per_spectra
            )

        # Determine which spectrum should be masked.
        #   If a spectrum's index is less than the number of spectra to mask for that SN, then mask that spectrum.
        #   This essentially means we are masking the first `n_spectra_to_mask` spectra of each SN.
        to_mask = unmasked_spec_inds <= n_spectra_to_mask_per_spectra
        if TYPE_CHECKING:
            to_mask = cast("ITensor[S['n_unmasked']]", to_mask)

        # Get the indices of each spectrum we are going to mask.
        inds_to_mask = tf.boolean_mask(unmasked_inds, to_mask)
        if TYPE_CHECKING:
            inds_to_mask = cast("ITensor[S['n_to_mask 2']]", inds_to_mask)

        # Since we want to mask these spectra, we will update the mask at these indices with 0
        #   So this tensor just contains the value (0) we want to update the mask with
        zeros_mask = tf.zeros(tf.shape(inds_to_mask)[0], dtype=tf.int32)
        if TYPE_CHECKING:
            zeros_mask = cast("ITensor[S['n_unmasked']]", zeros_mask)

        # The masking tensor we are going to apply these updates to
        ones_mask = tf.ones_like(spec_mask, dtype=tf.int32)
        if TYPE_CHECKING:
            ones_mask = cast("ITensor[S['n_sn nspec_dim']]", ones_mask)

        # Update the values in `ones_mask` at the indices in `inds_to_mask` with the corresponding values in `zeros_mask`.
        unshuffled_mask = tf.tensor_scatter_nd_update(
            ones_mask, inds_to_mask, zeros_mask
        )
        if TYPE_CHECKING:
            unshuffled_mask = cast("ITensor[S['n_sn nspec_dim']]", unshuffled_mask)

        # --- Shuffle Masking Tensor ---

        # For each unmasked spectra, get the number of spectra we want to shuffle (i.e. the original number of unmasked spectra) in its corresponding SN
        n_spectra_to_shuffle_per_spectra = tf.cast(
            tf.gather(n_unmasked_spectra_per_sn, unmasked_sn_inds), tf.int32
        )
        if TYPE_CHECKING:
            n_spectra_to_shuffle_per_spectra = cast(
                "ITensor[S['n_unmasked']]", n_spectra_to_shuffle_per_spectra
            )

        # Determine which spectrum should be shuffled.
        #   If a spectrum's index is less than the number of spectra to shuffle for that SN, then shuffle that spectrum.
        #   This essentially means we are shuffling the first `n_spectra_to_shuffle` spectra of each SN.
        to_shuffle = unmasked_spec_inds <= n_spectra_to_shuffle_per_spectra
        if TYPE_CHECKING:
            to_shuffle = cast("ITensor[S['n_unmasked']]", to_shuffle)

        # Get the indices of each spectrum we are going to shuffle.
        inds_to_shuffle = tf.boolean_mask(unmasked_inds, to_shuffle)
        if TYPE_CHECKING:
            inds_to_shuffle = cast("ITensor[S['n_to_shuffle 2']]", inds_to_shuffle)

        # For each spectrum we will shuffle, get the index of its corresponding SN
        sn_inds = inds_to_shuffle[:, 0]
        if TYPE_CHECKING:
            sn_inds = cast("ITensor[S['n_to_shuffle']]", sn_inds)

        # For each spectrum we will shuffle, get its index
        spec_inds_to_shuffle = inds_to_shuffle[:, 1]
        if TYPE_CHECKING:
            spec_inds_to_shuffle = cast(
                "ITensor[S['n_to_shuffle']]", spec_inds_to_shuffle
            )

        # For each SN, get the index of each spectra we want to shuffle.
        #   Since different SNe will have a different number of spectra to shuffle, this is a ragged tensor
        spec_inds_to_shuffle_per_sn = tf.RaggedTensor.from_value_rowids(
            spec_inds_to_shuffle, sn_inds, nrows=n_sn
        )
        if TYPE_CHECKING:
            spec_inds_to_shuffle_per_sn = cast(
                "IRTensor[S['n_sn n_spec_to_shuffle(sn)']]", spec_inds_to_shuffle_per_sn
            )

        # For each SN, construct a tensor of the (sn_ind, nspec_ind) indices of each spectrum we are going to shuffle
        # This is basically splitting spec_inds_to_shuffle into n_sn tensors, each of length n_spec_to_shuffle(sn)
        # We need to do this so that these indices can be batched properly
        shuffle_inds = tf.ragged.constant([
            p.numpy()
            for p in tf.dynamic_partition(
                spec_inds_to_shuffle, sn_inds, num_partitions=n_sn
            )
        ])
        if TYPE_CHECKING:
            shuffle_inds = cast(
                "IRTensor[S['n_sn n_spec_to_shuffle(sn)']]", shuffle_inds
            )

        return (
            unshuffled_mask,
            shuffle_inds,
            spec_inds_to_shuffle_per_sn,
        )

    @tf.function
    def prep_data_per_epoch(self, data: "EpochInputs") -> "ModelInputs":
        (
            (phase, d_phase, amplitude, d_amplitude, mask),
            (
                unshuffled_mask,
                shuffle_inds,
                spec_inds_to_shuffle_per_sn,
            ),
        ) = data

        # === Randomised Data Offsets ===
        # Every epoch we want to randomly shift some of the data as a countermeasure to overfitting

        # --- Phase Offset ---
        if self.phase_offset_scale != 0:
            d_phase_shape = tf.shape(d_phase)
            if self.phase_offset_scale < 0:
                phase_offset = (
                    abs(self.phase_offset_scale)
                    * d_phase
                    * tf.random.normal(d_phase_shape)
                )
            else:
                phase_offset = (
                    tf.ones_like(d_phase)
                    * self.phase_offset_scale
                    * tf.random.normal(d_phase_shape)
                )
            if TYPE_CHECKING:
                phase_offset = cast(
                    "FTensor[S['batch_dim nspec_dim phase_dim']]", phase_offset
                )

            phase += phase_offset
            if TYPE_CHECKING:
                phase = cast("FTensor[S['batch_dim nspec_dim phase_dim']]", phase)

        # --- Amplitude Offset ---
        if self.amplitude_offset_scale != 0:
            amplitude_offset = (
                d_amplitude
                * self.amplitude_offset_scale
                * tf.random.normal(tf.shape(d_amplitude))
            )
            if TYPE_CHECKING:
                amplitude_offset = cast(
                    "FTensor[S['batch_dim nspec_dim wl_dim']]", amplitude_offset
                )

            amplitude += amplitude_offset
            if TYPE_CHECKING:
                amplitude = cast("FTensor[S['batch_dim nspec_dim wl_dim']]", amplitude)

        # --- Spectral Masking ---
        if self.mask_fraction != 0:
            # For each SN, shuffle the indices of each spectra we want to shuffle
            shuffled_inds = tf.ragged.map_flat_values(
                tf.random.shuffle, spec_inds_to_shuffle_per_sn
            )
            if TYPE_CHECKING:
                shuffled_inds = cast(
                    "IRTensor[S['batch_dim n_spec_to_shuffle(sn)']]",
                    shuffled_inds,
                )

            # Now that we've shuffled the indices of each spectra for each SN,
            #   We need to shuffle the actual spectra of each SN.
            #   Since the spectra were only shuffled with others from the same SN,
            #   the SN index associated with each shuffled spectra is the same too
            #   So this just repeats the SN index once for each shuffled spectra associated with that SN.
            sn_inds_per_spec = tf.repeat(
                tf.range(shuffled_inds.nrows()), shuffled_inds.row_lengths()
            )
            if TYPE_CHECKING:
                sn_inds_per_spec = cast(
                    "IRTensor[S['n_sn n_spec_to_shuffle(sn)']]", sn_inds_per_spec
                )

            # Flatten out the ragged shuffled_indices tensor, and stack with the gather_sn_indices tensor.
            # This tensor encodes where each spectrum will be shuffled to.
            orig_inds_to_shuffled_inds = tf.stack(
                (sn_inds_per_spec, shuffled_inds.flat_values), axis=1
            )
            if TYPE_CHECKING:
                orig_inds_to_shuffled_inds = cast(
                    "ITensor[S['n_spec_to_shuffle 2']]", orig_inds_to_shuffled_inds
                )

            # For each shuffled index in `orig_inds_to_shuffled_inds`, get the corresponding element in `unshuffled_mask`
            # This allows us to randomise which spectra are masked, rather than always masking the first `n_spectra_to_mask` spectra
            spec_masks = tf.gather_nd(unshuffled_mask, orig_inds_to_shuffled_inds)
            if TYPE_CHECKING:
                spec_masks = cast("ITensor[S['n_spec_to_shuffle']]", spec_masks)

            # The original indices we are shuffling
            sn_inds = tf.cast(
                tf.repeat(tf.range(shuffle_inds.nrows()), shuffle_inds.row_lengths()),
                tf.int32,
            )
            spec_inds_to_shuffle = tf.stack(
                [sn_inds, shuffled_inds.flat_values], axis=1
            )
            if TYPE_CHECKING:
                spec_inds_to_shuffle = cast(
                    "ITensor[S['n_spec_to_shuffle 2']]", spec_inds_to_shuffle
                )
            # The final mask which randomly masks a fraction of spectra per SN
            shuffled_mask = tf.tensor_scatter_nd_update(
                unshuffled_mask, spec_inds_to_shuffle, spec_masks
            )
            if TYPE_CHECKING:
                shuffled_mask = cast("ITensor[S['batch_dim nspec_dim']]", shuffled_mask)

            # Mask every amplitude for any spectra which has been randomly masked
            shuffled_amp_mask = tf.tile(
                tf.expand_dims(shuffled_mask, axis=-1), [1, 1, tf.shape(mask)[-1]]
            )
            if TYPE_CHECKING:
                shuffled_amp_mask = cast(
                    "ITensor[S['batch_dim nspec_dim wl_dim']]", shuffled_amp_mask
                )
            mask *= shuffled_amp_mask

        return (phase, amplitude, d_amplitude, mask)

    def recon_error(
        self,
        data: "ModelInputs",
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        (phase, amp_true, d_amp, mask) = data
        _, amp_pred = self((phase, amp_true), training=False, mask=mask)
        wl_dim = tf.shape(mask)[-1]

        outlier_cut = 100

        time_bin_width = 0.1
        time_min = 0.0
        time_max = 1.0
        num_time_bins = tf.cast((time_max - time_min) // time_bin_width + 2, tf.int32)

        # Mask to keep only spectra with any valid data
        has_valid_data = tf.reduce_max(mask, axis=-1) == 1

        # Bin edges and centers
        time_bin_edges = tf.linspace(
            time_min - time_bin_width / 2,
            time_max + time_bin_width / 2,
            num_time_bins + 1,
        )
        time_bin_centers = 0.5 * (time_bin_edges[:-1] + time_bin_edges[1:])

        # Filter and reshape
        amp_true = tf.boolean_mask(amp_true, has_valid_data)
        amp_pred = tf.boolean_mask(amp_pred, has_valid_data)
        d_amp = tf.boolean_mask(d_amp, has_valid_data)
        mask = tf.boolean_mask(mask, has_valid_data)
        phase = tf.boolean_mask(phase, has_valid_data)
        min_phase = tf.reduce_min(phase, keepdims=True)
        max_phase = tf.reduce_min(phase, keepdims=True)
        time = (phase - min_phase) / (max_phase - min_phase)

        amp_true = tf.reshape(amp_true, [-1, wl_dim])
        amp_pred = tf.reshape(amp_pred, [-1, wl_dim])
        d_amp = tf.reshape(d_amp, [-1, wl_dim])
        mask = tf.reshape(mask, [-1, wl_dim])

        amp_true = tf.clip_by_value(amp_true, 1e-3, tf.float32.max)
        amp_pred = tf.clip_by_value(amp_pred, 1e-3, tf.float32.max)

        error = tf.abs((amp_true - amp_pred) / amp_pred)

        # Digitize: get bin indices (digitize returns 1-based, so subtract 1)
        bin_indices = tf.reshape(
            (
                tf.raw_ops.Bucketize(
                    input=time, boundaries=time_bin_edges.numpy().tolist()
                )
                - 1
            ),
            [-1],
        )  # no direct tf op, use raw with numpy for now
        unique_bins = tf.unique(bin_indices).y

        binned_error = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )

        for bin_id in tf.unstack(unique_bins):
            in_bin_idx = tf.where(bin_indices == bin_id)[:, 0]
            bin_error = tf.gather(error, in_bin_idx)
            bin_mask = tf.gather(mask, in_bin_idx)

            upper_clip = tfp.stats.percentile(bin_error, outlier_cut, axis=0)
            clip_mask = tf.cast(bin_error > upper_clip, tf.int32)
            bin_mask = tf.cast(bin_mask * (1 - clip_mask), tf.float32)

            # Compute std of masked values
            numerator = tf.reduce_sum((bin_error**2) * bin_mask, axis=0)
            denominator = tf.reduce_sum(bin_mask, axis=0) + 1e-8
            rms_error = tf.sqrt(numerator / denominator)
            binned_error = binned_error.write(bin_id, rms_error)

        binned_error = tf.transpose(binned_error.stack())
        return binned_error, time_bin_edges, time_bin_centers
