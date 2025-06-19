# Copyright 2025 Patrick Armstrong
import os
import random as rn
from typing import (
    TYPE_CHECKING,
    Self,
    cast,
    override,
)

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras as ks
from tqdm.keras import TqdmCallback
import tensorflow_probability as tfp

if TYPE_CHECKING:
    from typing import (
        Any,
        Literal,
    )
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt

    from suPAErnova.steps.pae.model import PAEModelStep
    from suPAErnova.configs.steps.pae import PAEStage
    from suPAErnova.typing.dimensions import SNDim, SpecDim
    from suPAErnova.typing.backends.tf import (
        FTensor,
        ITensor,
        GenericTensor,
    )
    from suPAErnova.typing.steps.pae.tf import (
        AmpTensor,
        MaskTensor,
        EpochInputs,
        ModelInputs,
        SigmaTensor,
        DecoderInputs,
        EncoderInputs,
        DecoderOutputs,
        EncoderOutputs,
        ZLatentsTensor,
        PAELatentsTensor,
        DecoderInputsShape,
        EncoderInputsShape,
        PhysicalLatentsTensor,
    )
    from suPAErnova.configs.steps.pae.tf import TFPAEModelConfig
    from suPAErnova.typing.steps.pae.pae import NZLatents, NPAELatents, NPhysicalLatents

    type StageNum = int


@ks.utils.register_keras_serializable("SuPAErnova")
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
        n_physical_latents: NPhysicalLatents = 3 if options.physical_latents else 0
        n_z_latents: NZLatents = options.n_z_latents
        self.n_pae_latents: NPAELatents = n_physical_latents + n_z_latents
        self.latents_z_mask: ITensor[tuple[NPAELatents]]
        self.latents_physical_mask: ITensor[tuple[NPAELatents]]

        self.stage_num: int
        self.moving_means: FTensor[tuple[NPAELatents]] = tf.zeros(self.n_pae_latents)

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
        self.encode_spec_layer: ks.layers.Dense
        self.encode_output_layer: ks.layers.Dense
        self.repeat_latent_layer: ks.layers.RepeatVector

    @override
    def build(self, input_shape: "EncoderInputsShape") -> None:
        (_batch_dim, spec_dim, _phase_dim), (_batch_dim, _spec_dim, _wl_dim) = (
            input_shape
        )

        # Encode from input layer dimensions into intermediate dimensions
        self.encode_layers = [
            ks.layers.Dense(
                encode_dim,
                activation=self.activation,
                kernel_regularizer=self.regulariser,
                name=f"PAEEDense{encode_dim}",
            )
            for encode_dim in self.encode_dims
        ]
        # Dropout layer
        self.dropout_layers = [
            ks.layers.Dropout(
                self.dropout,
                noise_shape=[None, 1, None],
                name=f"PAEEDropout_{encode_dim}",
            )
            if self.dropout > 0
            else ks.layers.Identity(trainable=False)
            for encode_dim in self.encode_dims
        ]
        # Batch normalisation layer
        self.batch_normalisation_layers = [
            ks.layers.BatchNormalization(name=f"PAEEBatchNorm_{encode_dim}")
            if self.batch_normalisation
            else ks.layers.Identity(trainable=False)
            for encode_dim in self.encode_dims
        ]

        # Encode from intermediate dimensions into spec_dim dimensions
        self.encode_spec_layer = ks.layers.Dense(
            spec_dim,
            activation=self.activation,
            kernel_regularizer=self.regulariser,
            name="PAEEDenseSpec",
        )

        # Encode from spec_dim dimensions into output (latent) dimensions
        self.encode_output_layer = ks.layers.Dense(
            self.n_pae_latents,
            kernel_regularizer=self.regulariser,
            use_bias=False,
            name="PAEEDenseLatents",
        )

        # Repeat latent vector to match spec_dim
        self.repeat_latent_layer = ks.layers.RepeatVector(spec_dim, name="PAEERepeat")

    @override
    def call(
        self,
        inputs: "EncoderInputs",
        mask: "MaskTensor | None" = None,
        training: bool | None = None,
    ) -> "EncoderOutputs":
        training = False if training is None else training

        input_phase = inputs[0]
        input_amp = inputs[1]
        input_mask = (
            mask if mask is not None else tf.ones_like(input_amp, dtype=tf.int32)
        )

        # Create initial input layer
        x: FTensor[tuple[SNDim, SpecDim, Literal["WLDim + PhaseDim"]]] = (
            ks.layers.concatenate([
                input_amp,
                input_phase,
            ])
        )

        # Encode from input layers to intermediate dimensions
        for i, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x, training=training)
            x = self.dropout_layers[i](x, training=training)
            x = self.batch_normalisation_layers[i](x, training=training)
        # Encode from intermediate dimensions to spec_dim dimensions
        x: FTensor[tuple[SNDim, SpecDim, SpecDim]] = self.encode_spec_layer(
            x, training=training
        )

        # Encode from spec_dim dimensions to output (latent) dimensions
        x: FTensor[tuple[SNDim, SpecDim, NPAELatents]] = self.encode_output_layer(
            x, training=training
        )

        # Determine which spectra to keep
        # Will mask out any spectrum which has at least one masked wavelength
        is_kept: FTensor[tuple[SNDim, SpecDim, Literal[1]]] = tf.cast(
            tf.reduce_min(input_mask, axis=-1, keepdims=True), tf.float32
        )

        # Latent tensor is the average of the latent values over all unmasked spectra
        # First sum latents over all unmasked spectra
        batch_sum: FTensor[tuple[SNDim, NPAELatents]] = tf.reduce_sum(
            x * is_kept, axis=-2
        )

        # Then determine the number of unmasked spectra
        batch_num: FTensor[tuple[SNDim, Literal[1]]] = tf.math.maximum(
            tf.reduce_sum(is_kept, axis=-2), y=1
        )

        # Finally, calculate the average
        latents: FTensor[tuple[SNDim, NPAELatents]] = batch_sum / batch_num

        # Mask latents which aren't being trained
        # The latents are ordered by training stage
        # ΔAᵥ -> zs -> Δℳ  -> Δ𝓅
        # Note that this differs from the order used in the legacy SuPAErnova code:
        # Δ𝓅 -> Δℳ  -> ΔAᵥ -> zs
        masked_latents: FTensor[tuple[Literal["NPAELatents - StageNum"]]] = tf.zeros(
            self.n_pae_latents - self.stage_num
        )
        unmasked_latents: FTensor[tuple[StageNum]] = tf.ones(self.stage_num)
        latent_mask: FTensor[tuple[NPAELatents]] = tf.concat(
            (unmasked_latents, masked_latents), axis=0
        )
        latents *= latent_mask

        if training:
            # Normalise the physical latents within this batch such that they have a mean of 0
            # Don't account for latents of SNe which have *no* unmasked spectra
            sn_mask: FTensor[tuple[SNDim, Literal[1]]] = tf.reduce_max(
                is_kept[..., 0], axis=-1, keepdims=True
            )
            latents_sum: FTensor[tuple[NPAELatents]] = tf.reduce_sum(
                latents * sn_mask, axis=0
            )
            latents_num: FTensor[tuple[None]] = tf.reduce_sum(sn_mask)
            latents_mean: FTensor[tuple[NPAELatents]] = latents_sum / latents_num
            latents -= self.latents_physical_mask * latents_mean
        else:
            # Normalise the physical latents within this batch such that the entire unbatched sample has a mean of 0
            latents -= self.latents_physical_mask * self.moving_means

        # Repeat latent layers across specDim
        encoded: EncoderOutputs = self.repeat_latent_layer(latents)
        return encoded

    @override
    def __call__(
        self,
        inputs: "EncoderInputs",
        *,
        training: bool | None = None,
        mask: "GenericTensor | None" = None,
    ) -> "EncoderOutputs":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPAEDecoder(ks.layers.Layer):
    def __init__(
        self, options: "TFPAEModelConfig", name: str, *args: "Any", **kwargs: "Any"
    ) -> None:
        super().__init__(*args, name=f"{name.split()[-1]}Decoder", **kwargs)
        # --- Config Params ---
        self.physical_latents: bool = options.physical_latents
        self.n_physical_latents: NPhysicalLatents = 3 if options.physical_latents else 0
        self.n_z_latents: NZLatents = options.n_z_latents
        self.n_z_latents: NZLatents = options.n_z_latents

        self.wl_dim: int
        self.decode_dims: list[int] = options.decode_dims

        self.activation: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn

        self.regulariser: ks.regularizers.Regularizer | None = (
            options.kernel_regulariser_cls(options.kernel_regulariser_penalty)
            if options.kernel_regulariser_cls is not None
            else None
        )
        self.batch_normalisation: bool = options.batch_normalisation

        colourlaw = options.colourlaw
        if colourlaw is not None:
            _, colourlaw = np.loadtxt(colourlaw, unpack=True)
        self.colourlaw: npt.NDArray[np.float64] | None = colourlaw

        # --- Layers ---
        self.decode_spec_layer: ks.layers.Dense
        self.decode_layers: list[ks.layers.Dense]
        self.batch_normalisation_layers: list[
            ks.layers.BatchNormalization | ks.layers.Identity
        ]
        self.decode_output_layer: ks.layers.Dense
        self.colourlaw_layer: ks.layers.Dense | ks.layers.Identity

    @override
    def build(self, input_shape: "DecoderInputsShape") -> None:
        (
            (_batch_dim, spec_dim, _n_pae_latents),
            (_batch_dim, _spec_dim, _phase_dim),
        ) = input_shape
        # Project from input dimensions into spec_dim dimensions
        self.decode_spec_layer = ks.layers.Dense(
            spec_dim,
            activation=self.activation,
            kernel_regularizer=self.regulariser,
            name="PAEDDenseSpec",
        )

        # Decode from spec_dim dimensions into intermediate dimensions
        self.decode_layers = [
            ks.layers.Dense(
                decode_dim,
                activation=self.activation,
                kernel_regularizer=self.regulariser,
                name=f"PAEDDense{decode_dim}",
            )
            for decode_dim in self.decode_dims
        ]
        self.batch_normalisation_layers = [
            ks.layers.BatchNormalization(
                name=f"PAEDBatchNorm{decode_dim}",
            )
            if self.batch_normalisation
            else ks.layers.Identity(trainable=False)
            for decode_dim in self.decode_dims
        ]

        # Decode from intermediate dimensions to output dimensions
        self.decode_output_layer = ks.layers.Dense(
            self.wl_dim,
            kernel_regularizer=self.regulariser,
            name="PAEDDenseAmp",
        )

        # Colourlaw
        self.colourlaw_layer = (
            ks.layers.Dense(
                self.wl_dim,
                kernel_initializer=None
                if self.colourlaw is None
                else tf.constant_initializer(self.colourlaw),
                use_bias=False,
                trainable=self.colourlaw is None,
                kernel_constraint=ks.constraints.NonNeg()
                if self.colourlaw is None
                else None,
                name="PAEDDenseCL",
            )
            if self.physical_latents
            else ks.layers.Identity(trainable=False)
        )

    @override
    def call(
        self,
        inputs: "DecoderInputs",
        mask: "MaskTensor | None" = None,
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
        physical_latents = (
            input_latents
            if self.physical_latents
            else tf.zeros_like((*input_latents.shape[:-1], self.n_z_latents + 3))
        )
        delta_av_latent: PhysicalLatentsTensor = physical_latents[:, :, 0:1]
        zs_latent: ZLatentsTensor = (
            input_latents[:, :, 1 : self.n_z_latents + 1]
            if self.physical_latents
            else input_latents
        )
        delta_m_latent: PhysicalLatentsTensor = physical_latents[
            :, :, self.n_z_latents + 1 : self.n_z_latents + 2
        ]
        delta_p_latent: PhysicalLatentsTensor = physical_latents[
            :, :, self.n_z_latents + 2 : self.n_z_latents + 3
        ]

        # Apply Δ𝓅 shift
        input_phase += delta_p_latent

        # Create initial input layer
        x: FTensor[tuple[SNDim, SpecDim, Literal["NZLatents + PhaseDim"]]] = (
            ks.layers.concatenate([
                zs_latent,
                input_phase,
            ])
        )

        # Decode from input (latent) dimensions to spec_dim dimensions
        x = self.decode_spec_layer(x, training=training)

        # Decode from spec_dim dimensions to intermediate dimensions
        for i, decode_layer in enumerate(self.decode_layers):
            x = decode_layer(x, training=training)
            x = self.batch_normalisation_layers[i](x, training=training)

        # Decode from intermediate dimensions to output dimension
        amplitude = self.decode_output_layer(x, training=training)

        # Apply ΔAᵥ / Δℳ  shift
        if self.physical_latents:
            # Calculate Colourlaw
            colourlaw = self.colourlaw_layer(delta_av_latent, training=training)

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
        mask: "GenericTensor | None" = None,
    ) -> "DecoderOutputs":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPAEModel(ks.Model):
    def __init__(
        self,
        config: "PAEModelStep[Literal['tf']]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        ks.backend.clear_session()

        super().__init__(*args, name=f"{config.name.split()[-1]}PAEModel", **kwargs)
        # --- Config ---
        global PAEMODELSTEP
        PAEMODELSTEP = config
        self.options: "TFPAEModelConfig" = cast("TFPAEModelConfig", config.options)
        self.log: "Logger" = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed

        # --- Latent Dimensions ---
        self.physical_latents: bool = self.options.physical_latents
        self.n_physical_latents: NPhysicalLatents = (
            3 if self.options.physical_latents else 0
        )
        self.n_z_latents: NZLatents = self.options.n_z_latents
        self.n_pae_latents: NPAELatents = self.n_physical_latents + self.n_z_latents

        # --- Layers ---
        self.encoder: TFPAEEncoder = TFPAEEncoder(self.options, config.name)
        self.decoder: TFPAEDecoder = TFPAEDecoder(self.options, config.name)
        self.decoder.wl_dim = config.wl_dim

        # --- Training ---
        self.built: bool = False
        self._epoch: tf.Variable = tf.Variable(0, name="epoch", trainable=False)

        self.batch_size: int = self.options.batch_size
        self.save_best: bool = self.options.save_best
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )

        # Data Offsets
        self.phase_offset_scale: Literal[0, -1] | float = (
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
        self.latents_z_mask: ITensor[tuple[NPAELatents]]
        self.latents_physical_mask: ITensor[tuple[NPAELatents]]

        # --- Loss ---
        self._loss: ks.losses.Loss
        self._loss_terms: dict[str, FTensor[tuple[None]]]
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

        self.val_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(name="val_loss")
        self.val_pred_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(
            name="val_loss_pred"
        )
        self.val_model_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(
            name="val_loss_model"
        )
        self.val_resid_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(
            name="val_loss_resid"
        )
        self.val_delta_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(
            name="val_loss_delta"
        )
        self.val_cov_loss_tracker: ks.metrics.Metric = ks.metrics.Mean(
            name="val_loss_cov"
        )

        self.set_seed()

    @override
    def get_config(self) -> dict[str, "Any"]:
        return {**super().get_config(), "stage": self.stage.name}

    @override
    @classmethod
    def from_config(cls, config: dict[str, "Any"]) -> Self:
        global PAEMODELSTEP
        self = cls(PAEMODELSTEP)
        self.stage = next(
            stage for stage in PAEMODELSTEP.run_stages if stage.name == config["stage"]
        )
        return self

    def build_from_config(self, _config: dict[str, "Any"]) -> None:
        self.build_model()

    @override
    def set_seed(self, seed: int = 0) -> None:
        seed = self.seed + seed
        tf.random.set_seed(seed)

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

    @property
    def val_metrics(self) -> list[ks.metrics.Metric]:
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [
            self.val_loss_tracker,
            self.val_pred_loss_tracker,
            self.val_model_loss_tracker,
        ]
        if self.loss_residual_penalty > 0:
            metrics.append(self.val_resid_loss_tracker)
        if self.physical_latents and self.loss_physical_penalty > 0:
            metrics.append(self.val_delta_loss_tracker)
        if self.loss_covariance_penalty > 0:
            metrics.append(self.val_cov_loss_tracker)
        return metrics

    @override
    def call(
        self,
        inputs: "EncoderInputs",
        training: bool | None = None,
        mask: "GenericTensor | None" = None,
    ) -> "tuple[EncoderOutputs, DecoderOutputs]":
        training = False if training is None else training
        input_phase = inputs[0]

        encoded: EncoderOutputs = self.encoder(inputs, training=training, mask=mask)
        decoded: DecoderOutputs = self.decoder(
            (encoded, input_phase), training=training, mask=mask
        )
        return encoded, decoded

    @override
    def __call__(
        self,
        inputs: "EncoderInputs",
        *,
        training: bool | None = None,
        mask: "GenericTensor | None" = None,
    ) -> "tuple[EncoderOutputs, DecoderOutputs]":
        training = False if training is None else training
        return super().__call__(inputs, training=training, mask=mask)

    @override
    def compute_loss(
        self,
        x: "GenericTensor | None" = None,
        y: "GenericTensor | None" = None,
        y_pred: "GenericTensor | None" = None,
        sample_weight: "GenericTensor | None" = None,
        training: bool | None = None,
        mask: "GenericTensor | None" = None,
    ) -> "FTensor[tuple[None]] | None":
        if x is None or y is None or y_pred is None or sample_weight is None:
            return None
        training = False if training is None else training

        latents: PAELatentsTensor = tf.convert_to_tensor(x, dtype=tf.float32)
        input_amp: AmpTensor = tf.convert_to_tensor(y, dtype=tf.float32)
        output_amp: AmpTensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        input_d_amp: SigmaTensor = tf.convert_to_tensor(sample_weight, dtype=tf.float32)
        input_mask: MaskTensor = tf.cast(
            (
                tf.convert_to_tensor(mask, dtype=tf.int32)
                if mask is not None
                else tf.ones_like(input_amp, dtype=tf.int32)
            ),
            tf.float32,
        )

        # If any wavelengths of a spectra are masked, mask the entire spectra
        # Then remove all masked spectra
        latents_mask: FTensor[tuple[SNDim, Literal[1]]] = tf.reduce_max(
            tf.reduce_min(input_mask, axis=-1, keepdims=True), axis=-2
        )

        loss = self.options.loss_cls()
        loss.latents = latents
        loss.input_amp = input_amp
        loss.input_d_amp = input_d_amp
        loss.output_amp = output_amp
        loss.input_mask = input_mask
        loss.latents_mask = latents_mask
        loss.model = self
        self._loss = loss

        pred_loss: FTensor[tuple[None]] = loss(y_true=input_amp, y_pred=output_amp)
        model_loss: FTensor[tuple[None]] = tf.reduce_sum(self.losses)

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
                    tf.zeros(self.n_z_latents, dtype=tf.float32),
                    tf.constant((self.loss_delta_m_penalty,), dtype=tf.float32),
                    tf.constant((self.loss_delta_p_penalty,), dtype=tf.float32),
                ),
                axis=0,
            )
            preferred_latent_values = tf.concat(
                (
                    tf.constant((1.0,)),  # ΔAᵥ = 1
                    tf.zeros(self.n_z_latents),  # zs = 0
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
            eps = tf.constant(1e-3)
            z_cov_latents = latents[:, 0, :]
            masked_latents = z_cov_latents * latents_mask
            num_masked_latents = tf.reduce_sum(latents_mask)  # + eps
            latents_mean = (
                tf.reduce_sum(masked_latents, axis=0, keepdims=True)
                / num_masked_latents
            )
            latents_norm = latents_mask * (z_cov_latents - latents_mean)
            latents_cov = (
                tf.matmul(latents_norm, latents_norm, transpose_a=True)
                / num_masked_latents
            )

            latents_std = tf.sqrt(
                tf.reduce_sum(latents_norm**2, axis=0) / num_masked_latents
            )
            # Avoid nans
            # TODO: eps * tf.ones???
            latents_std = tf.where(
                latents_std < eps, tf.ones_like(latents_std), latents_std
            )

            std_outer = tf.matmul(
                tf.expand_dims(latents_std, axis=-1),
                tf.expand_dims(latents_std, axis=0),
            )
            # + eps

            latents_cov_norm = latents_cov / std_outer

            cov_dim = tf.shape(latents_cov_norm)[0]
            cov_mask = 1.0 - tf.eye(cov_dim)

            # Decorrelate all -> Punish all latents for off-diagonal terms
            if not self.loss_decorrelate_all:
                decorrelate_delta_av = (
                    1.0 if self.loss_decorrelate_dust else 0.0
                ) * tf.ones((1, cov_dim))
                decorrelate_z_latents = tf.zeros((self.n_z_latents, cov_dim))
                decorrelate_delta_m = tf.ones((1, cov_dim))
                decorrelate_delta_p = tf.ones((1, cov_dim))

                decorrelate_latents = []
                if self.physical_latents:
                    decorrelate_latents.append(decorrelate_delta_av)
                decorrelate_latents.append(decorrelate_z_latents)
                if self.physical_latents:
                    decorrelate_latents.extend((
                        decorrelate_delta_m,
                        decorrelate_delta_p,
                    ))

                decorrelate: ITensor[tuple[NPAELatents, NPAELatents]] = tf.concat(
                    decorrelate_latents, axis=0
                )
                decorrelate *= tf.transpose(decorrelate)
                cov_mask *= decorrelate

            loss_cov = tf.reduce_sum(tf.square(latents_cov_norm * cov_mask)) / (
                tf.reduce_sum(cov_mask)  # + eps
            )

            loss_covariance_penalty = self.loss_covariance_penalty * loss_cov
            loss_terms[self.cov_loss_tracker.name] = loss_covariance_penalty

        total_loss = tf.add_n(loss_terms.values())
        loss_terms[self.loss_tracker.name] = total_loss

        self._loss_terms = loss_terms
        return total_loss

    @override
    def train_step(
        self, data: "GenericTensor", *, dummy: bool = False
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:
        training = not dummy

        # === Per Epoch Setup ===
        self._epoch.assign_add(1)

        # --- Setup Data ---
        if dummy:
            (phase, amplitude, d_amplitude, mask) = cast("ModelInputs", data)
        else:
            (phase, amplitude, d_amplitude, mask) = self.prep_data_per_epoch(
                cast("EpochInputs", data)
            )

        with tf.GradientTape() as tape:
            latents, pred_amplitude = self(
                (phase, amplitude), training=training, mask=mask
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

    @override
    def test_step(
        self, data: "GenericTensor"
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:
        training = False

        # --- Setup Data ---
        (phase, _d_phase, amplitude, d_amplitude, mask, sn_mask, spec_mask) = cast(
            "EpochInputs", data
        )[0]
        mask = mask * sn_mask * spec_mask

        latents, pred_amplitude = self((phase, amplitude), training=training, mask=mask)

        loss = self.compute_loss(
            x=latents,
            y=amplitude,
            y_pred=pred_amplitude,
            sample_weight=d_amplitude,
            training=training,
            mask=mask,
        )
        if loss is None:
            return {m.name: m.result() for m in self.val_metrics}

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            metric.update_state(self._loss_terms[metric.name])
        for i, metric in enumerate(self.val_metrics):
            metric.update_state(self._loss_terms[self.metrics[i].name])

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train_model(self, stage: "PAEStage") -> None:
        self.stage = stage

        self.stage.train_sn_mask = tf.convert_to_tensor(self.stage.train_sn_mask)
        self.stage.train_spec_mask = tf.convert_to_tensor(self.stage.train_spec_mask)
        self.stage.test_sn_mask = tf.convert_to_tensor(self.stage.test_sn_mask)
        self.stage.test_spec_mask = tf.convert_to_tensor(self.stage.test_spec_mask)
        self.stage.val_sn_mask = tf.convert_to_tensor(self.stage.val_sn_mask)
        self.stage.val_spec_mask = tf.convert_to_tensor(self.stage.val_spec_mask)

        n_batches_per_epoch = self.stage.train_data.amplitude.shape[0] / self.batch_size

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        # --- Backup & Restore ---
        # Backup checkpoints each epoch and restore if training got cancelled midway through
        if not self.force and self.stage.savepath is not None:
            backup_dir = self.stage.savepath / "backups"
            backup_callback = ks.callbacks.BackupAndRestore(
                str(backup_dir),
                save_freq=max(1, int(0.1 * self.stage.epochs * n_batches_per_epoch)),
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
                    TqdmCallback(epochs=self.stage.epochs, verbose=0),
                ),
            )
        )

        if stage.loadpath is not None:
            self.load_checkpoint(stage.loadpath)
        self.build_model()

        # === Prep Data ===
        train_data = (
            self.stage.train_data.time,
            self.stage.train_data.dphase,
            self.stage.train_data.amplitude,
            self.stage.train_data.sigma,
            self.stage.train_data.mask,
            self.stage.train_sn_mask,
            self.stage.train_spec_mask,
        )

        test_data = (
            self.stage.test_data.time,
            self.stage.test_data.dphase,
            self.stage.test_data.amplitude,
            self.stage.test_data.sigma,
            self.stage.test_data.mask,
            self.stage.test_sn_mask,
            self.stage.test_spec_mask,
        )

        val_data = (
            self.stage.val_data.time,
            self.stage.val_data.dphase,
            self.stage.val_data.amplitude,
            self.stage.val_data.sigma,
            self.stage.val_data.mask,
            self.stage.val_sn_mask,
            self.stage.val_spec_mask,
        )

        all_data = (
            self.stage.all_data.time,
            self.stage.all_data.dphase,
            self.stage.all_data.amplitude,
            self.stage.all_data.sigma,
            self.stage.all_data.mask,
            self.stage.all_sn_mask,
            self.stage.all_spec_mask,
        )

        # === Train ===
        self._epoch.assign(0)
        self.set_seed(self.stage.stage)
        self.fit(
            x=train_data,
            initial_epoch=self._epoch.numpy(),
            epochs=self.stage.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            validation_data=(val_data,),
            validation_freq=max(
                1, int(0.1 * self.stage.epochs)
            ),  # TODO: Get val_freq > 1 working with tqdm
        )

    def build_model(self, *, update: bool = False) -> None:
        if not self.built or update:
            self.set_seed(self.stage.stage)
            # Mask tensors to select specific latents
            if self.physical_latents:
                self.latents_z_mask = tf.concat(
                    (tf.zeros(1), tf.ones(self.n_z_latents), tf.zeros(1), tf.zeros(1)),
                    axis=0,
                )
            else:
                self.latents_z_mask = tf.ones(self.n_z_latents)
            self.latents_physical_mask = 1 - self.latents_z_mask  # Swap 0s and 1s

            # === Setup Encoder ===
            self.encoder.stage_num = self.stage.stage
            self.encoder.latents_z_mask = self.latents_z_mask
            self.encoder.latents_physical_mask = self.latents_physical_mask

            schedule = self._scheduler(
                initial_learning_rate=self.stage.learning_rate,
                decay_steps=self.stage.learning_rate_decay_steps,
                decay_rate=self.stage.learning_rate_decay_rate,
            )
            if update:
                self.optimizer.learning_rate = schedule
                self.optimizer.weight_decay = self.stage.learning_rate_weight_decay_rate

            if not update:
                optimiser = self._optimiser(
                    learning_rate=schedule,
                    weight_decay=self.stage.learning_rate_weight_decay_rate,
                )
                loss = self.options.loss_cls()
                self.compile(
                    optimizer=optimiser,
                    loss=loss,
                    metrics=self.metrics,
                    run_eagerly=self.stage.debug,
                )
                self(
                    (
                        tf.convert_to_tensor(self.stage.train_data.time),
                        tf.convert_to_tensor(self.stage.train_data.amplitude),
                    ),
                    training=False,
                    mask=self.stage.train_data.mask,
                )

            self.log.debug("Trainable variables:")
            for var in self.trainable_variables:
                self.log.debug(f"{var.name}: {var.shape}")
            self.summary(
                print_fn=self.log.debug, show_trainable=True
            )  # Will show number of parameters
            self.built = True

    def get_loss(self, loss: str):
        return self._loss_terms.get(loss, tf.constant(0, dtype=tf.float32))

    def save_checkpoint(self, savepath: "Path") -> None:
        (savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
        tf.train.Checkpoint(
            self,
        ).save(f"{savepath / self.ckpt_path}/")

    def load_checkpoint(
        self,
        loadpath: "Path",
        *,
        reset_weights: bool | None = None,
    ) -> None:
        self.set_seed(self.stage.stage)
        stage_num = self.stage.stage
        if self.stage.prev_stage is not None:
            self.stage.stage = self.stage.prev_stage

        self.build_model()
        init_weights = self.encoder.encode_output_layer.get_weights()[0]

        self.train_step(
            (
                tf.zeros_like(self.stage.train_data.time, dtype=tf.float32),
                tf.zeros_like(self.stage.train_data.amplitude, dtype=tf.float32),
                tf.zeros_like(self.stage.train_data.sigma, dtype=tf.float32),
                tf.zeros_like(self.stage.train_data.mask, dtype=tf.int32),
            ),
            dummy=True,
        )

        tf.train.Checkpoint(
            self,
        ).restore(
            tf.train.latest_checkpoint(f"{loadpath / self.ckpt_path}/")
        ).assert_existing_objects_matched()

        self.stage.stage = stage_num

        self.build_model(update=True)

        reset_weights = (
            (self.stage.prev_stage is not None)
            and (self.stage.stage < self.n_pae_latents)
            if reset_weights is None
            else reset_weights
        )
        if reset_weights:
            weights = self.encoder.encode_output_layer.get_weights()[0]
            # Set the weights of the newly introduced latent parameter to effectively 0
            weights[:, self.stage.prev_stage : self.stage.stage] = (
                init_weights[:, self.stage.prev_stage : self.stage.stage] / 100
            )
            self.encoder.encode_output_layer.set_weights([weights])
        # Normalise mean of physical latents to 0 across all batches
        if self.physical_latents:
            phase = tf.convert_to_tensor(self.stage.train_data.time)
            amplitude = tf.convert_to_tensor(self.stage.train_data.amplitude)
            mask = (
                tf.convert_to_tensor(self.stage.train_data.mask)
                * self.stage.train_sn_mask
                * self.stage.train_spec_mask
            )
            encoded = self.encoder((phase, amplitude), training=False, mask=mask)

            latents_sum: FTensor[tuple[NPAELatents]] = tf.reduce_sum(
                (encoded * tf.cast(self.stage.train_sn_mask, tf.float32))[:, 0, :],
                axis=0,
            )
            latents_num: FTensor[tuple[NPAELatents]] = tf.cast(
                tf.reduce_sum(self.stage.train_sn_mask), tf.float32
            )
            moving_means: FTensor[tuple[NPAELatents]] = latents_sum / latents_num

            self.encoder.moving_means = moving_means

    @tf.function
    def prep_data_per_epoch(self, data: "EpochInputs") -> "ModelInputs":
        (phase, d_phase, amplitude, d_amplitude, mask, sn_mask, spec_mask) = data[0]

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

            phase += phase_offset

        # --- Amplitude Offset ---
        # TODO: Remove abs..?
        if self.amplitude_offset_scale != 0:
            amplitude_offset: AmpTensor = self.amplitude_offset_scale * tf.abs(
                d_amplitude * tf.random.normal(tf.shape(d_amplitude))
            )

            amplitude += amplitude_offset

        # --- Spectral Masking ---
        if self.mask_fraction != 0:
            # Identify any spectra which aren't completely masked out
            unmasked_spectra = tf.reduce_max(mask, axis=-1)

            # The number of unmasked spectra for each SN
            n_unmasked_spectra_per_sn = tf.reduce_sum(
                unmasked_spectra, axis=-1, keepdims=True
            )

            # Determine how many spectra to mask for each SN
            #   by multiplying the number of unmasked SN by the mask_fraction
            n_spectra_to_mask_per_sn = tf.cast(
                self.mask_fraction * tf.cast(n_unmasked_spectra_per_sn, tf.float32),
                tf.int32,
            )
            spec_inds = tf.repeat(
                tf.expand_dims(tf.range(unmasked_spectra.shape[-1]), axis=0),
                repeats=unmasked_spectra.shape[0],
                axis=0,
            )

            unshuffled_mask = spec_inds >= n_spectra_to_mask_per_sn

            def shuffle_and_pad(n_unmasked):
                n_total = tf.shape(unmasked_spectra)[-1]
                shuffled = tf.random.shuffle(tf.range(n_unmasked))
                padding = tf.range(n_unmasked, n_total)
                return tf.concat([shuffled, padding], axis=0)

            shuffled_inds = tf.map_fn(
                shuffle_and_pad,
                n_unmasked_spectra_per_sn[:, 0],
            )

            shuffled_mask = tf.cast(
                tf.expand_dims(
                    tf.gather(unshuffled_mask, shuffled_inds, axis=1, batch_dims=1),
                    axis=-1,
                ),
                dtype=tf.int32,
            )

            mask *= shuffled_mask

        mask *= sn_mask * spec_mask

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
        num_time_bins = tf.cast(((time_max - time_min) // time_bin_width) + 2, tf.int32)

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

        amp_true = tf.reshape(amp_true, [-1, wl_dim])
        amp_pred = tf.reshape(amp_pred, [-1, wl_dim])
        d_amp = tf.reshape(d_amp, [-1, wl_dim])
        mask = tf.reshape(mask, [-1, wl_dim])

        amp_true = tf.clip_by_value(amp_true, 1e-3, tf.float32.max)
        amp_pred = tf.clip_by_value(amp_pred, 1e-3, tf.float32.max)

        error = tf.abs((amp_true - amp_pred) / amp_pred)

        bin_indices = tf.reshape(
            (
                tf.raw_ops.Bucketize(
                    input=phase, boundaries=time_bin_edges.numpy().tolist()
                )
                - 1
            ),
            [-1],
        )
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
