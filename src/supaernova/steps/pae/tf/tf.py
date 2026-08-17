# Copyright 2025 Patrick Armstrong

from supaernova.steps.pae.tf.photometry import photometry
from typing import (
    TYPE_CHECKING,
    cast,
    override,
)

from tqdm.keras import TqdmCallback

from supaernova._tf import HUGE, JIT_COMPILE, ks, tf, tfp, clear_session
from supaernova.utils import max_central
from supaernova.utils.tf import db, pp

if TYPE_CHECKING:
    from typing import (
        Any,
        Literal,
    )
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.steps.pae import PAE
    from supaernova.configs.steps.pae import PAEStage
    from supaernova.typing.backends.tf import (
        Loss,
        TensorLike,
        LearningRateSchedule,
    )
    from supaernova.configs.steps.pae.tf import TFPAEConfig

    type StageNum = int


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPAEEncoder(ks.layers.Layer):
    def __init__(
        self,
        options: "TFPAEConfig",
        name: str,
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=f"{name.rsplit(maxsplit=1)[-1]}Encoder", **kwargs)

        # --- Config Params ---
        n_physical_latents = 3 if options.physical_latents else 0
        n_z_latents: int = options.n_z_latents
        self.n_pae_latents: int = n_physical_latents + n_z_latents

        self.latents_physical_mask: tf.Variable = tf.Variable(
            tf.zeros(self.n_pae_latents, dtype=tf.bool),
            trainable=False,
            name="PAEELatentsPhysicalMask",
        )
        # dtype is int64 as int32 Variables can't go on GPUs.
        self.stage_num: tf.Variable = tf.Variable(
            tf.constant(-1, dtype=tf.int64), trainable=False, name="PAEEStageNum"
        )
        self.moving_means: tf.Variable = tf.Variable(
            tf.zeros(self.n_pae_latents), trainable=False, name="PAEEMovingMeans"
        )

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
    def build(self, input_shape: tuple[int, int, int]) -> None:
        (_batch_dim, spec_dim, _encoder_dim) = input_shape

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
        inputs: tf.Tensor,
        *,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        training: bool | None = None,
        testing: bool | None = None,
    ) -> tf.Tensor:
        training = False if training is None else training
        testing = False if testing is None else testing

        # === Unpack Inputs ===
        # --- Data Phase ---
        input_phase = inputs[..., :1]

        # --- Data Amplitude ---
        input_amp = inputs[..., 1:]

        # --- Masks ---
        # Data Mask
        input_mask = tf.ones_like(input_amp, dtype=tf.bool) if mask is None else mask

        # Wavelength Range Mask
        input_wl_mask = tf.ones_like(input_mask) if wl_mask is None else wl_mask

        # Phase Range Mask
        input_spec_mask = (
            tf.math.reduce_any(input_wl_mask, axis=-1, keepdims=True)
            if spec_mask is None
            else spec_mask
        )

        # Redshift Range Mask
        input_sn_mask = (
            tf.math.reduce_any(input_spec_mask, axis=-2, keepdims=True)
            if sn_mask is None
            else sn_mask
        )

        # === Setup Masks ===
        # Apply sn and spec masks
        input_mask &= input_sn_mask & input_spec_mask & input_wl_mask

        # ~(~input_mask & input_wl_mask)
        # Extracts unmasked wavelengths from the valid wavelength range provided by wl_mask
        valid_wl_mask = tf.logical_not(
            tf.logical_and(tf.logical_not(input_mask), input_wl_mask)
        )

        # Determine which spectra to keep
        # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
        mask_spec = tf.logical_and(
            tf.math.reduce_all(valid_wl_mask, axis=-1, keepdims=True), input_spec_mask
        )

        # The number of unmasked spectra
        n_unmasked_spec = tf.math.count_nonzero(
            mask_spec[..., 0], axis=-1, keepdims=True, dtype=tf.float32
        )
        n_unmasked_spec = tf.math.maximum(n_unmasked_spec, 1)

        # Determine which SNe to keep
        # Will mask out any SN with *no* unmasked spectra
        mask_sn = tf.logical_and(
            tf.math.reduce_any(mask_spec, axis=-2, keepdims=True), input_sn_mask
        )[..., 0]

        # The number of unmasked SNe
        n_unmasked_sn = tf.math.count_nonzero(mask_sn[:, 0], dtype=tf.float32)

        # Determine which latents to keep
        # The latents are ordered by training stage
        # ΔAᵥ -> zs -> ΔM  -> Δp
        # Note that this differs from the order used in the legacy SuPAErnova code:
        # Δp -> ΔM  -> ΔAᵥ -> zs
        latent_mask = tf.sequence_mask(
            self.stage_num, self.n_pae_latents, dtype=tf.bool
        )

        # === Run Encoder ===
        # Create initial input layer
        x = ks.layers.concatenate([
            input_amp,
            input_phase,
        ])

        # Encode from input layers to intermediate dimensions
        for i, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x, training=training)
            x = self.dropout_layers[i](x, training=training)
            x = self.batch_normalisation_layers[i](x, training=training)

        # Encode from intermediate dimensions to spec_dim dimensions
        x = self.encode_spec_layer(x, training=training)

        # Encode from spec_dim dimensions to output (latent) dimensions
        x = self.encode_output_layer(x, training=training)

        # Latent tensor is the average of the latent values over all unmasked spectra
        latents = (
            tf.reduce_sum(tf.where(mask_spec, x, tf.zeros_like(x)), axis=-2)
            / n_unmasked_spec
        )

        # Zero out latents of masked SNe
        latents = tf.where(mask_sn, latents, tf.zeros_like(latents))

        if training:
            # latents_mean = tf.nn.weighted_moments(
            #     latents, [0], n_unmasked_spec * tf.cast(mask_sn, tf.float32)
            # )[0]
            # latents_mean = tfp.stats.percentile(latents, 50.0, axis=0)
            latents_mean = (
                tf.reduce_sum(
                    tf.where(mask_sn, latents, tf.zeros_like(latents)), axis=0
                )
                / n_unmasked_sn
            )
        else:
            latents_mean = self.moving_means

        latents -= tf.where(
            self.latents_physical_mask,
            latents_mean,
            tf.zeros_like(latents_mean),
        )

        # Zero out latents of masked SNe
        latents = tf.where(mask_sn, latents, tf.zeros_like(latents))

        # Mask latents which aren't being trained
        latents = tf.where(latent_mask, latents, tf.zeros_like(latents))

        # Repeat latent layers across spec_dim
        return self.repeat_latent_layer(latents)

    @override
    def __call__(
        self,
        inputs: tf.Tensor,
        *,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        training: bool | None = None,
        testing: bool | None = None,
    ) -> tf.Tensor:
        training = False if training is None else training
        testing = False if testing is None else testing
        return super().__call__(
            inputs,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            training=training,
            testing=testing,
        )


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPAEDecoder(ks.layers.Layer):
    def __init__(
        self, options: "TFPAEConfig", name: str, *args: "Any", **kwargs: "Any"
    ) -> None:
        super().__init__(*args, name=f"{name.rsplit(maxsplit=1)[-1]}Decoder", **kwargs)
        # --- Config Params ---
        self.physical_latents: bool = options.physical_latents
        self.n_physical_latents = 3 if options.physical_latents else 0
        self.n_z_latents: int = options.n_z_latents

        self.wl_dim: int
        self.decode_dims: list[int] = options.decode_dims

        self.activation: Callable[[tf.Tensor], tf.Tensor] = options.activation_fn

        self.regulariser: ks.regularizers.Regularizer | None = (
            options.kernel_regulariser_cls(options.kernel_regulariser_penalty)
            if options.kernel_regulariser_cls is not None
            else None
        )
        self.batch_normalisation: bool = options.batch_normalisation

        self.colourlaw: npt.NDArray[float] | None

        # --- Layers ---
        self.decode_spec_layer: ks.layers.Dense
        self.decode_layers: list[ks.layers.Dense]
        self.batch_normalisation_layers: list[
            ks.layers.BatchNormalization | ks.layers.Identity
        ]
        self.decode_output_layer: ks.layers.Dense
        self.colourlaw_layer: ks.layers.Dense | ks.layers.Identity

    @override
    def build(self, input_shape: tuple[int, int, int]) -> None:
        (_batch_dim, spec_dim, _decoder_dim) = input_shape
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
        inputs: tf.Tensor,
        *,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        training: bool | None = None,
        testing: bool | None = None,
    ) -> tf.Tensor:
        training = False if training is None else training
        testing = False if testing is None else testing

        # === Unpack Inputs ===
        # --- Data Phase ---
        input_phase = inputs[..., :1]
        phase_shape = tf.shape(input_phase)
        sn_dim = phase_shape[-3]
        spec_dim = phase_shape[-2]

        # --- Encoder Latents ---
        input_latents = inputs[..., 1:]

        # --- Physical Latents ---
        physical_latents = (
            input_latents
            if self.physical_latents
            else tf.zeros_like((sn_dim, spec_dim, self.n_z_latents + 3))
        )
        # ΔAᵥ
        delta_av_latent = physical_latents[..., :1]
        # zs
        zs_latent = (
            input_latents[..., 1 : self.n_z_latents + 1]
            if self.physical_latents
            else input_latents
        )
        # ΔM
        delta_m_latent = physical_latents[
            ..., self.n_z_latents + 1 : self.n_z_latents + 2
        ]
        # Δp
        delta_p_latent = physical_latents[
            ..., self.n_z_latents + 2 : self.n_z_latents + 3
        ]

        # --- Masks ---
        # Data Mask
        input_mask = (
            tf.ones((sn_dim, spec_dim, self.wl_dim), dtype=tf.bool)
            if mask is None
            else mask
        )
        # Wavelength Range Mask
        input_wl_mask = tf.ones_like(input_mask) if wl_mask is None else wl_mask
        # Phase Range Mask
        input_spec_mask = (
            tf.math.reduce_any(input_wl_mask, axis=-1, keepdims=True)
            if spec_mask is None
            else spec_mask
        )
        # Redshift Range Mask
        input_sn_mask = (
            tf.math.reduce_any(input_spec_mask, axis=-2, keepdims=True)
            if sn_mask is None
            else sn_mask
        )

        # === Setup Masks ===
        # Apply sn, spec, and wl masks
        input_mask &= input_sn_mask & input_spec_mask & input_wl_mask

        # === Run Decoder ===
        # Apply Δp shift
        input_phase += delta_p_latent

        # Create initial input layer
        x = ks.layers.concatenate([
            zs_latent,
            input_phase,
        ])

        # Decode from input (latent) dimensions to spec_dim dimensions
        x = self.decode_spec_layer(x, training=training)

        # Decode from spec_dim dimensions to intermediate dimensions
        for i, decode_layer in enumerate(self.decode_layers):
            x = decode_layer(x, training=training)
            x = self.batch_normalisation_layers[i](x, training=training)

        # Decode from intermediate dimensions to output dimension
        amplitude = self.decode_output_layer(x, training=training)

        # Apply ΔAᵥ / ΔM shift
        if self.physical_latents:
            # Calculate Colourlaw
            colourlaw = self.colourlaw_layer(delta_av_latent, training=training)
            amplitude *= tf.pow(10.0, -0.4 * (colourlaw + delta_m_latent))

        if not (training or testing):
            amplitude = tf.nn.relu(amplitude)

        return tf.where(input_mask, amplitude, tf.zeros_like(amplitude))

    @override
    def __call__(
        self,
        inputs: tf.Tensor,
        *,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        wavelength: tf.Tensor | None = None,
        sigma: tf.Tensor | None = None,
        throughput: tf.Tensor | None = None,
        effective_wavelength: tf.Tensor | None = None,
        spectra_mask: tf.Tensor | None = None,
        phot_mask: tf.Tensor | None = None,
        training: bool | None = None,
        testing: bool | None = None,
    ) -> tf.Tensor:
        training = False if training is None else training
        testing = False if testing is None else testing
        synth_amp = super().__call__(
            inputs,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            training=training,
            testing=testing,
        )

        if not any(
            v is None
            for v in (
                wavelength,
                sigma,
                throughput,
                effective_wavelength,
                spectra_mask,
                phot_mask,
            )
        ):
            (
                synth_amp,
                _,
            ) = photometry(
                wavelength,
                synth_amp,
                sigma,
                throughput,
                effective_wavelength,
                spectra_mask,
                phot_mask,
            )
        return synth_amp


PAEMODELSTEP: "PAE"


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPAEModel(ks.Model):
    def __init__(
        self,
        config: "PAE",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, name=config.name.rsplit(maxsplit=1)[-1], **kwargs)
        # --- Config ---
        global PAEMODELSTEP
        PAEMODELSTEP = config
        self.options: TFPAEConfig = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed
        self.rng: tf.random.Generator = tf.random.Generator.from_seed(self.seed)
        self.set_seed()

        # --- Latent Dimensions ---
        self.physical_latents: bool = self.options.physical_latents
        self.n_physical_latents = 3 if self.options.physical_latents else 0
        self.n_z_latents: int = self.options.n_z_latents
        self.n_pae_latents: int = self.n_physical_latents + self.n_z_latents

        # --- Layers ---
        self.encoder: TFPAEEncoder = TFPAEEncoder(self.options, config.name)
        self.decoder: TFPAEDecoder = TFPAEDecoder(self.options, config.name)
        self.decoder.wl_dim = config.wl_dim
        self.decoder.colourlaw = config.colourlaw
        self.decoder.input_wavelength = config.input_wavelength
        self.decoder.input_sigma = config.input_sigma
        self.decoder.input_throughput = config.input_throughput
        self.decoder.input_effective_wavelength = config.input_effective_wavelength
        self.decoder.input_spectra_mask = config.input_spectra_mask
        self.decoder.input_phot_mask = config.input_phot_mask

        # --- Training ---
        self.built: bool = False
        self._epoch: int = 0

        self.batch_size: int = config.batch_size
        self.save_best: bool = self.options.save_best
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )
        self.log_path: str = f"{'best' if self.save_best else 'latest'}_logs/"

        # Data Offsets
        self.phase_offset_scale: Literal[0, -1] | float = (
            self.options.phase_offset_scale
        )
        self.amplitude_offset_scale: float = self.options.amplitude_offset_scale
        self.mask_fraction: float = self.options.mask_fraction

        # Training functions
        self._scheduler: type[LearningRateSchedule] = self.options.scheduler_cls
        self._optimiser: type[ks.optimizers.Optimizer] = self.options.optimiser_cls

        self.stage: PAEStage
        self.latents_physical_mask: tf.Tensor

        # --- Loss ---
        self._loss: Loss
        self._loss_terms: dict[str, tf.Tensor]
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

    @override
    def get_config(self) -> dict[str, "Any"]:
        return {**super().get_config(), "stage": self.stage.name}

    @override
    @classmethod
    def from_config(cls, config: dict[str, "Any"]):
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
        self.rng.reset_from_seed(seed)

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
        inputs: tf.Tensor,
        *,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        wavelength: tf.Tensor | None = None,
        sigma: tf.Tensor | None = None,
        throughput: tf.Tensor | None = None,
        effective_wavelength: tf.Tensor | None = None,
        spectra_mask: tf.Tensor | None = None,
        phot_mask: tf.Tensor | None = None,
        training: bool | None = None,
        testing: bool | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        training = False if training is None else training
        testing = False if testing is None else testing

        self.set_seed(self._epoch)

        input_phase = inputs[..., :1]
        encoded = self.encoder(
            inputs,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            training=training,
            testing=testing,
        )

        decoder_inputs = tf.concat((input_phase, encoded), axis=-1)

        decoded = self.decoder(
            decoder_inputs,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            wavelength=wavelength,
            sigma=sigma,
            throughput=throughput,
            effective_wavelength=effective_wavelength,
            spectra_mask=spectra_mask,
            phot_mask=phot_mask,
            training=training,
            testing=testing,
        )

        return encoded, decoded

    @override
    def __call__(
        self,
        inputs: "TensorLike",
        *,
        mask: "TensorLike | None" = None,
        sn_mask: "TensorLike | None" = None,
        spec_mask: "TensorLike | None" = None,
        wl_mask: "TensorLike | None" = None,
        wavelength: tf.Tensor | None = None,
        sigma: tf.Tensor | None = None,
        throughput: tf.Tensor | None = None,
        effective_wavelength: tf.Tensor | None = None,
        spectra_mask: tf.Tensor | None = None,
        phot_mask: tf.Tensor | None = None,
        training: bool | None = None,
        testing: bool | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        training = False if training is None else training
        testing = False if testing is None else testing
        if isinstance(inputs, tuple | list):
            inputs = tf.concat(inputs, axis=-1)
        else:
            inputs = tf.convert_to_tensor(inputs)
        if mask is not None:
            mask = tf.convert_to_tensor(mask)
        if sn_mask is not None:
            sn_mask = tf.convert_to_tensor(sn_mask)
        if spec_mask is not None:
            spec_mask = tf.convert_to_tensor(spec_mask)
        if wl_mask is not None:
            wl_mask = tf.convert_to_tensor(wl_mask)
        if wavelength is not None:
            wavelength = tf.convert_to_tensor(wavelength, dtype=inputs.dtype)
        if sigma is not None:
            sigma = tf.convert_to_tensor(sigma, dtype=inputs.dtype)
        if throughput is not None:
            throughput = tf.convert_to_tensor(throughput, dtype=inputs.dtype)
        if effective_wavelength is not None:
            effective_wavelength = tf.convert_to_tensor(
                effective_wavelength, dtype=inputs.dtype
            )
        if spectra_mask is not None:
            spectra_mask = tf.cast(
                tf.convert_to_tensor(spectra_mask), dtype=inputs.dtype
            )
        if phot_mask is not None:
            phot_mask = tf.cast(tf.convert_to_tensor(phot_mask), dtype=inputs.dtype)
        return super().__call__(
            inputs,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            wavelength=wavelength,
            sigma=sigma,
            throughput=throughput,
            effective_wavelength=effective_wavelength,
            spectra_mask=spectra_mask,
            phot_mask=phot_mask,
            training=training,
            testing=testing,
        )

    @override
    def compute_loss(
        self,
        x: tf.Tensor | None = None,
        y: tf.Tensor | None = None,
        y_pred: tf.Tensor | None = None,
        sample_weight: tf.Tensor | None = None,
        mask: tf.Tensor | None = None,
        sn_mask: tf.Tensor | None = None,
        spec_mask: tf.Tensor | None = None,
        wl_mask: tf.Tensor | None = None,
        training: bool | None = None,
        testing: bool | None = None,
    ) -> tf.Tensor | None:
        if x is None or y is None or y_pred is None or sample_weight is None:
            return None
        training = False if training is None else training
        testing = False if testing is None else testing

        # === Unpack Inputs ===
        # --- Encoder Latents ---
        latents = tf.convert_to_tensor(x, dtype=tf.float32)

        # --- Data Amplitude ---
        input_amp = tf.convert_to_tensor(y, dtype=tf.float32)

        # --- Decoder Amplitude ---
        output_amp = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # --- Data Sigma ---
        input_d_amp = tf.convert_to_tensor(sample_weight, dtype=tf.float32)

        # --- Masks ---
        # Data Mask
        input_mask = tf.ones_like(input_amp, dtype=tf.bool) if mask is None else mask

        # Wavelength Range Mask
        input_wl_mask = tf.ones_like(input_mask) if wl_mask is None else wl_mask

        # Phase Range Mask
        input_spec_mask = (
            tf.math.reduce_any(input_wl_mask, axis=-1, keepdims=True)
            if spec_mask is None
            else spec_mask
        )

        # Redshift Range Mask
        input_sn_mask = (
            tf.math.reduce_any(input_spec_mask, axis=-2, keepdims=True)
            if sn_mask is None
            else sn_mask
        )

        # === Setup Masks ===
        # Apply sn and spec masks
        input_mask &= input_sn_mask & input_spec_mask & input_wl_mask

        # ~(~input_mask & input_wl_mask)
        # Extracts unmasked wavelengths from the valid wavelength range provided by wl_mask
        valid_wl_mask = tf.logical_not(
            tf.logical_and(tf.logical_not(input_mask), input_wl_mask)
        )

        # Determine which spectra to keep
        # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
        mask_spec = tf.logical_and(
            tf.math.reduce_all(valid_wl_mask, axis=-1, keepdims=True), input_spec_mask
        )

        # Determine which SNe to keep
        # Will mask out any SN with *no* unmasked spectra
        mask_sn = tf.logical_and(
            tf.math.reduce_any(mask_spec, axis=-2, keepdims=True), input_sn_mask
        )[..., 0]

        loss = self.options.loss_cls()
        loss.input_mask = input_mask
        loss.input_d_amp = input_d_amp
        loss.model = self
        self._loss = loss

        pred_loss = loss(y_true=input_amp, y_pred=output_amp)
        model_loss = tf.reduce_sum(self.losses)

        loss_terms = {
            self.pred_loss_tracker.name: pred_loss,
            self.model_loss_tracker.name: model_loss,
        }

        # --- Penalties ---
        # Penalise larger residuals between the input amplitude and the output amplitude
        if self.loss_residual_penalty > 0:
            residual_penalty = self.loss_residual_penalty * tf.math.reduce_mean(
                tf.abs(
                    tf.reduce_sum(
                        tf.where(
                            input_mask, input_amp - output_amp, tf.zeros_like(input_amp)
                        ),
                        axis=(-2, -1),
                    )
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
            median_latent_values_per_sn = mask_sn * tfp.stats.percentile(
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
                tf.where(
                    self.latents_physical_mask,
                    latents_penalty_scale * physical_latents_offset,
                    tf.zeros_like(physical_latents_offset),
                )
            )
            loss_terms[self.delta_loss_tracker.name] = physical_latents_penalty

        if self.loss_covariance_penalty > 0:
            eps = ks.backend.epsilon()
            mask_latents = mask_sn
            n_unmasked_latents = tf.math.count_nonzero(
                mask_latents[:, 0], dtype=tf.float32
            )
            masked_latents = tf.where(
                mask_latents, latents[:, 0, :], tf.zeros_like(latents[:, 0, :])
            )
            latents_mean = (
                tf.reduce_sum(masked_latents, axis=0, keepdims=True)
                / n_unmasked_latents
            )
            latents_norm = tf.where(
                mask_latents,
                (masked_latents - latents_mean),
                tf.zeros_like(mask_latents, dtype=tf.float32),
            )
            latents_cov = (
                tf.matmul(latents_norm, latents_norm, transpose_a=True)
                / n_unmasked_latents
            )

            latents_std = tf.sqrt(
                tf.reduce_sum(latents_norm * latents_norm, axis=0) / n_unmasked_latents
            )
            # Avoid nans
            latents_std = tf.where(
                latents_std < eps,
                tf.ones_like(latents_std) * eps,
                latents_std,
            )

            std_outer = tf.matmul(
                tf.expand_dims(latents_std, axis=-1),
                tf.expand_dims(latents_std, axis=0),
            )

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

                decorrelate = tf.concat(decorrelate_latents, axis=0)
                decorrelate *= tf.transpose(decorrelate)
                cov_mask *= decorrelate

            loss_cov = tf.reduce_sum(
                tf.square(
                    cov_mask * latents_cov_norm,
                )
            ) / tf.reduce_sum(cov_mask)

            loss_covariance_penalty = self.loss_covariance_penalty * loss_cov
            loss_terms[self.cov_loss_tracker.name] = loss_covariance_penalty

        total_loss = tf.add_n(loss_terms.values())
        loss_terms[self.loss_tracker.name] = total_loss

        self._loss_terms = loss_terms
        return total_loss

    @override
    def train_step(
        self, data: tuple["TensorLike", ...], *, dummy: bool = False
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:
        training = not dummy
        testing = False

        # === Per Epoch Setup ===
        self._epoch += 1

        # --- Setup Data ---
        if dummy:
            (
                phase,
                amplitude,
                d_amplitude,
                mask,
                sn_mask,
                spec_mask,
                wl_mask,
                wavelength,
                sigma,
                throughput,
                effective_wavelength,
                spectra_mask,
                phot_mask,
            ) = (tf.convert_to_tensor(d) for d in data)
        else:
            (
                phase,
                amplitude,
                d_amplitude,
                mask,
                sn_mask,
                spec_mask,
                wl_mask,
                wavelength,
                sigma,
                throughput,
                effective_wavelength,
                spectra_mask,
                phot_mask,
            ) = self.prep_data_per_epoch(data)

        pae_input = tf.concat((phase, amplitude), axis=-1)

        with tf.GradientTape() as tape:
            latents, pred_amplitude = self(
                pae_input,
                mask=mask,
                sn_mask=sn_mask,
                spec_mask=spec_mask,
                wl_mask=wl_mask,
                wavelength=wavelength,
                sigma=sigma,
                throughput=throughput,
                effective_wavelength=effective_wavelength,
                spectra_mask=spectra_mask,
                phot_mask=phot_mask,
                training=training,
                testing=testing,
            )
            loss = self.compute_loss(
                x=latents,
                y=amplitude,
                y_pred=pred_amplitude,
                sample_weight=d_amplitude,
                mask=mask,
                sn_mask=sn_mask,
                spec_mask=spec_mask,
                wl_mask=wl_mask,
                training=training,
                testing=testing,
            )
        if loss is None:
            return {m.name: m.result() for m in self.metrics}

        gradients = tape.gradient(loss, self.trainable_variables)

        cast("ks.optimizers.Optimizer", self.optimizer).apply_gradients(
            zip(gradients, self.trainable_variables, strict=True)
        )

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            metric_loss = self._loss_terms[metric.name]
            metric.update_state(metric_loss, None)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @override
    def test_step(
        self, data: tuple["TensorLike", ...]
    ) -> dict[str, tf.Tensor | dict[str, tf.Tensor]]:

        self.get_moving_means()

        training = False
        testing = True

        # --- Setup Data ---
        (
            phase,
            _d_phase,
            amplitude,
            d_amplitude,
            mask,
            sn_mask,
            spec_mask,
            wl_mask,
            wavelength,
            sigma,
            throughput,
            effective_wavelength,
            spectra_mask,
            phot_mask,
        ) = data[0]
        pae_input = tf.concat((phase, amplitude), axis=-1)

        latents, pred_amplitude = self(
            pae_input,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            wavelength=wavelength,
            sigma=sigma,
            throughput=throughput,
            effective_wavelength=effective_wavelength,
            spectra_mask=spectra_mask,
            phot_mask=phot_mask,
            training=training,
            testing=testing,
        )

        loss = self.compute_loss(
            x=latents,
            y=amplitude,
            y_pred=pred_amplitude,
            sample_weight=d_amplitude,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            training=training,
            testing=testing,
        )
        if loss is None:
            return {m.name: m.result() for m in self.val_metrics}

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            metric.update_state(self._loss_terms[metric.name], None)
        for i, metric in enumerate(self.val_metrics):
            metric.update_state(self._loss_terms[self.metrics[i].name], None)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def get_moving_means(self) -> None:
        phase = tf.convert_to_tensor(self.stage.train_data.time, dtype=tf.float32)
        amplitude = tf.convert_to_tensor(
            self.stage.train_data.amplitude, dtype=tf.float32
        )

        self.stage.train_data.clear()

        mask = self.stage.train_mask
        sn_mask = self.stage.train_sn_mask
        spec_mask = self.stage.train_spec_mask
        wl_mask = self.stage.train_wl_mask
        pae_input = tf.concat((phase, amplitude), axis=-1)

        self.encoder.moving_means.assign(tf.zeros(self.encoder.n_pae_latents))

        encoded = self.encoder(
            pae_input,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
        )[:, 0, :]

        mask &= sn_mask & spec_mask & wl_mask
        valid_wl_mask = tf.logical_not(tf.logical_and(tf.logical_not(mask), wl_mask))
        mask_spec = tf.logical_and(
            tf.math.reduce_all(valid_wl_mask, axis=-1, keepdims=True), spec_mask
        )
        n_unmasked_spec = tf.math.count_nonzero(
            mask_spec[..., 0], axis=-1, keepdims=True, dtype=tf.float32
        )
        n_unmasked_spec = tf.math.maximum(n_unmasked_spec, 1)
        mask_sn = tf.logical_and(
            tf.math.reduce_any(mask_spec, axis=-2, keepdims=True), sn_mask
        )[..., 0]
        n_unmasked_sn = tf.math.count_nonzero(mask_sn[:, 0], dtype=tf.float32)

        # latents_mean = tf.nn.weighted_moments(
        #     encoded, [0], n_unmasked_spec * tf.cast(mask_sn, tf.float32)
        # )[0]
        # latents_mean = tfp.stats.percentile(encoded, 50.0, axis=0)
        latents_mean = (
            tf.reduce_sum(tf.where(mask_sn, encoded, tf.zeros_like(encoded)), axis=0)
            / n_unmasked_sn
        )

        self.encoder.moving_means.assign(latents_mean)
        self.log.debug(self.encoder.moving_means)

    def train_model(self, stage: "PAEStage") -> None:
        self.stage = stage

        self.stage.mask = tf.convert_to_tensor(self.stage.mask, dtype=tf.bool)
        self.stage.sn_mask = tf.convert_to_tensor(self.stage.sn_mask, dtype=tf.bool)
        self.stage.spec_mask = tf.convert_to_tensor(self.stage.spec_mask, dtype=tf.bool)
        self.stage.wl_mask = tf.convert_to_tensor(self.stage.wl_mask, dtype=tf.bool)

        self.stage.train_mask = tf.convert_to_tensor(
            self.stage.train_mask, dtype=tf.bool
        )
        self.stage.train_sn_mask = tf.convert_to_tensor(
            self.stage.train_sn_mask, dtype=tf.bool
        )
        self.stage.train_spec_mask = tf.convert_to_tensor(
            self.stage.train_spec_mask, dtype=tf.bool
        )
        self.stage.train_wl_mask = tf.convert_to_tensor(
            self.stage.train_wl_mask, dtype=tf.bool
        )

        self.stage.test_mask = tf.convert_to_tensor(self.stage.test_mask, dtype=tf.bool)
        self.stage.test_sn_mask = tf.convert_to_tensor(
            self.stage.test_sn_mask, dtype=tf.bool
        )
        self.stage.test_spec_mask = tf.convert_to_tensor(
            self.stage.test_spec_mask, dtype=tf.bool
        )
        self.stage.test_wl_mask = tf.convert_to_tensor(
            self.stage.test_wl_mask, dtype=tf.bool
        )

        self.stage.val_mask = tf.convert_to_tensor(self.stage.val_mask, dtype=tf.bool)
        self.stage.val_sn_mask = tf.convert_to_tensor(
            self.stage.val_sn_mask, dtype=tf.bool
        )
        self.stage.val_spec_mask = tf.convert_to_tensor(
            self.stage.val_spec_mask, dtype=tf.bool
        )
        self.stage.val_wl_mask = tf.convert_to_tensor(
            self.stage.val_wl_mask, dtype=tf.bool
        )

        n_batches_per_epoch = self.stage.train_mask.shape[0] / self.batch_size

        # === Setup Callbacks ===
        callbacks: list[ks.callbacks.Callback] = []

        # --- Terminate on NaN ---
        # Terminate training when a NaN loss is encountered
        callbacks.append(ks.callbacks.TerminateOnNaN())

        patience = self.stage.patience
        if isinstance(patience, float):
            patience = int(self.stage.epochs * patience)
        callbacks.extend([
            ks.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                start_from_epoch=patience,
            ),
            # ks.callbacks.EarlyStopping(
            #     monitor="val_loss_pred",
            #     patience=2 * patience,
            #     mode="min",
            #     start_from_epoch=patience,
            # ),
            # ks.callbacks.EarlyStopping(
            #     monitor="val_loss_cov",
            #     patience=2 * patience,
            #     mode="min",
            #     start_from_epoch=patience,
            # ),
        ])

        # --- Backup & Restore ---
        # Backup checkpoints each epoch and restore if training got cancelled midway through
        if not self.force and self.stage.savepath is not None:
            backup_dir = self.stage.savepath / "backups"
            backup_callback = ks.callbacks.BackupAndRestore(
                str(backup_dir),
                save_freq=max(1, int(0.25 * self.stage.epochs * n_batches_per_epoch)),
            )
            callbacks.append(backup_callback)

        # --- TQDM Progress Bar ---
        callbacks.append(TqdmCallback(epochs=self.stage.epochs, verbose=0))

        if self.stage.profile and self.stage.savepath is not None:
            callbacks.append(
                ks.callbacks.TensorBoard(
                    log_dir=self.stage.savepath.parent
                    / self.log_path
                    / self.stage.savepath.stem,
                    write_graph=False,
                    write_images=False,
                    write_steps_per_second=False,
                    update_freq="epoch",
                    # profile_batch=(
                    #     int(n_batches_per_epoch * self.stage.epochs * 0.1),
                    #     int(n_batches_per_epoch * self.stage.epochs * 0.1) + 10,
                    # ),
                    embeddings_freq=0,
                ),
            )

        if self.stage.loadpath is not None:
            self.load_checkpoint(self.stage.loadpath)
        self.build_model()

        # === Prep Data ===
        _data = (
            tf.cast(
                tf.where(
                    self.stage.spec_mask,
                    self.stage.data.time,
                    -HUGE * tf.ones_like(self.stage.data.time),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.spec_mask,
                    self.stage.data.dphase,
                    -HUGE * tf.ones_like(self.stage.data.dphase),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.mask,
                    self.stage.data.amplitude,
                    -HUGE * tf.ones_like(self.stage.data.amplitude),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.mask,
                    self.stage.data.sigma,
                    -HUGE * tf.ones_like(self.stage.data.sigma),
                ),
                tf.float32,
            ),
            self.stage.mask,
            self.stage.sn_mask,
            self.stage.spec_mask,
            self.stage.wl_mask,
            self.stage.data.wavelength,
            self.stage.data.sigma,
            self.stage.data.throughput,
            self.stage.data.effective_wavelength,
            self.stage.data.spectra_mask,
            self.stage.data.phot_mask,
        )
        self.stage.data.clear()

        train_data = (
            tf.cast(
                tf.where(
                    self.stage.train_spec_mask,
                    self.stage.train_data.time,
                    -HUGE * tf.ones_like(self.stage.train_data.time),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.train_spec_mask,
                    self.stage.train_data.dphase,
                    -HUGE * tf.ones_like(self.stage.train_data.dphase),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.train_mask,
                    self.stage.train_data.amplitude,
                    -HUGE * tf.ones_like(self.stage.train_data.amplitude),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.train_mask,
                    self.stage.train_data.sigma,
                    -HUGE * tf.ones_like(self.stage.train_data.sigma),
                ),
                tf.float32,
            ),
            self.stage.train_mask,
            self.stage.train_sn_mask,
            self.stage.train_spec_mask,
            self.stage.train_wl_mask,
            self.stage.train_data.wavelength,
            self.stage.train_data.sigma,
            self.stage.train_data.throughput,
            self.stage.train_data.effective_wavelength,
            self.stage.train_data.spectra_mask,
            self.stage.train_data.phot_mask,
        )
        self.stage.train_data.clear()

        _test_data = (
            tf.cast(
                tf.where(
                    self.stage.test_spec_mask,
                    self.stage.test_data.time,
                    -HUGE * tf.ones_like(self.stage.test_data.time),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.test_spec_mask,
                    self.stage.test_data.dphase,
                    -HUGE * tf.ones_like(self.stage.test_data.dphase),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.test_mask,
                    self.stage.test_data.amplitude,
                    -HUGE * tf.ones_like(self.stage.test_data.amplitude),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.test_mask,
                    self.stage.test_data.sigma,
                    -HUGE * tf.ones_like(self.stage.test_data.sigma),
                ),
                tf.float32,
            ),
            self.stage.test_mask,
            self.stage.test_sn_mask,
            self.stage.test_spec_mask,
            self.stage.test_wl_mask,
            self.stage.test_data.wavelength,
            self.stage.test_data.sigma,
            self.stage.test_data.throughput,
            self.stage.test_data.effective_wavelength,
            self.stage.test_data.spectra_mask,
            self.stage.test_data.phot_mask,
        )
        self.stage.test_data.clear()

        val_data = (
            tf.cast(
                tf.where(
                    self.stage.val_spec_mask,
                    self.stage.val_data.time,
                    -HUGE * tf.ones_like(self.stage.val_data.time),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.val_spec_mask,
                    self.stage.val_data.dphase,
                    -HUGE * tf.ones_like(self.stage.val_data.dphase),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.val_mask,
                    self.stage.val_data.amplitude,
                    -HUGE * tf.ones_like(self.stage.val_data.amplitude),
                ),
                tf.float32,
            ),
            tf.cast(
                tf.where(
                    self.stage.val_mask,
                    self.stage.val_data.sigma,
                    -HUGE * tf.ones_like(self.stage.val_data.sigma),
                ),
                tf.float32,
            ),
            self.stage.val_mask,
            self.stage.val_sn_mask,
            self.stage.val_spec_mask,
            self.stage.val_wl_mask,
            self.stage.val_data.wavelength,
            self.stage.val_data.sigma,
            self.stage.val_data.throughput,
            self.stage.val_data.effective_wavelength,
            self.stage.val_data.spectra_mask,
            self.stage.val_data.phot_mask,
        )
        self.stage.val_data.clear()

        # === Train ===
        self._epoch = 0
        self.fit(
            x=train_data,
            initial_epoch=self._epoch,
            epochs=self.stage.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            validation_data=(val_data,),
            validation_freq=1,
        )
        clear_session()

    def build_model(self, *, update: bool = False) -> None:
        self.set_seed()
        if not self.built or update:
            # Mask tensors to select specific latents
            if self.physical_latents:
                latents_z_mask = tf.cast(
                    tf.concat(
                        (
                            tf.zeros(1),
                            tf.ones(self.n_z_latents),
                            tf.zeros(1),
                            tf.zeros(1),
                        ),
                        axis=0,
                    ),
                    tf.bool,
                )
            else:
                latents_z_mask = tf.ones(self.n_z_latents, dtype=tf.bool)
            self.latents_physical_mask = tf.math.logical_not(latents_z_mask)

            # === Setup Encoder ===
            self.encoder.stage_num.assign(self.stage.stage)
            self.encoder.latents_physical_mask.assign(self.latents_physical_mask)

            schedule = self._scheduler(
                initial_learning_rate=self.stage.learning_rate,
                decay_steps=self.stage.learning_rate_decay_steps,
                decay_rate=self.stage.learning_rate_decay_rate,
            )
            if update:
                self.optimizer.learning_rate.assign(self.stage.learning_rate)
                self.optimizer.iterations.assign(0)
                self.optimizer.weight_decay = self.stage.learning_rate_weight_decay_rate
            else:
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
                    jit_compile=JIT_COMPILE and not self.stage.debug,
                )

                phase = tf.convert_to_tensor(self.stage.data.time, dtype=tf.float32)
                amplitude = tf.convert_to_tensor(
                    self.stage.data.amplitude, dtype=tf.float32
                )
                self.stage.data.clear()
                pae_input = tf.concat((phase, amplitude), axis=-1)
                self(
                    pae_input,
                    mask=self.stage.mask,
                    sn_mask=self.stage.sn_mask,
                    spec_mask=self.stage.spec_mask,
                    wl_mask=self.stage.wl_mask,
                    wavelength=self.stage.data.wavelength,
                    sigma=self.stage.data.sigma,
                    throughput=self.stage.data.throughput,
                    effective_wavelength=self.stage.data.effective_wavelength,
                    spectra_mask=self.stage.data.spectra_mask,
                    phot_mask=self.stage.data.phot_mask,
                )
                if self.stage.debug:
                    self.log.debug("Trainable variables:")
                    for var in self.trainable_variables:
                        self.log.debug(f"{var.name}: {var.shape}")
                    self.summary(
                        print_fn=self.log.debug, show_trainable=True
                    )  # Will show number of parameters
            self.built = True

    def get_loss(self, loss: str) -> tf.Tensor:
        return self._loss_terms.get(loss, tf.constant(0, dtype=tf.float32))

    def save_checkpoint(self, savepath: "Path") -> None:
        (savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
        tf.train.Checkpoint(
            self,
        ).save(f"{savepath / self.ckpt_path}/")

        clear_session()

    def load_checkpoint(
        self,
        loadpath: "Path",
        *,
        reset_weights: bool | None = None,
    ) -> None:
        self.set_seed()
        stage_num = self.stage.stage
        if self.stage.prev_stage is not None:
            self.stage.stage = self.stage.prev_stage

        self.build_model()
        init_weights = self.encoder.encode_output_layer.get_weights()[0]

        tf.train.Checkpoint(
            self,
        ).restore(
            tf.train.latest_checkpoint(f"{loadpath / self.ckpt_path}/")
        ).expect_partial()

        self.stage.stage = stage_num

        self.build_model(update=True)

        reset_weights = (
            (self.stage.prev_stage is not None)
            and (self.stage.stage <= self.n_pae_latents)
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

        if self.physical_latents:
            self.get_moving_means()

        clear_session()

    def prep_data_per_epoch(
        self, data: tuple["TensorLike", ...]
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        (
            phase,
            d_phase,
            amplitude,
            d_amplitude,
            mask,
            sn_mask,
            spec_mask,
            wl_mask,
            wavelength,
            sigma,
            throughput,
            effective_wavelength,
            spectra_mask,
            phot_mask,
        ) = data[0]

        # === Randomised Data Offsets ===
        # Every epoch we want to randomly shift some of the data as a countermeasure to overfitting

        # --- Phase Offset ---
        if self.phase_offset_scale != 0:
            d_phase_shape = tf.shape(d_phase)
            if self.phase_offset_scale < 0:
                phase_offset = (
                    abs(self.phase_offset_scale)
                    * d_phase
                    * self.rng.normal(d_phase_shape)
                )
            else:
                phase_offset = (
                    tf.ones_like(d_phase)
                    * self.phase_offset_scale
                    * self.rng.normal(d_phase_shape)
                )

            phase += phase_offset

        # --- Amplitude Offset ---
        if self.amplitude_offset_scale != 0:
            amplitude_offset = (
                self.amplitude_offset_scale
                * d_amplitude
                * self.rng.normal(tf.shape(d_amplitude))
            )
            amplitude += amplitude_offset

        # --- Spectral Masking ---
        if self.mask_fraction != 0:
            # Identify any spectra which aren't completely masked out
            unmasked_spectra = tf.math.reduce_any(mask, axis=-1)

            # The number of unmasked spectra for each SN
            n_unmasked_spectra_per_sn = tf.math.count_nonzero(
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

            n_total = tf.shape(unmasked_spectra)[-1]

            # def shuffle_and_pad(n_unmasked: tf.Tensor) -> tf.Tensor:
            #     shuffled = tf.random.shuffle(tf.range(n_unmasked))
            #     padding = tf.range(n_unmasked, n_total)
            #     return tf.concat([shuffled, padding], axis=0)
            #
            # shuffled_inds = tf.map_fn(
            #     shuffle_and_pad,
            #     n_unmasked_spectra_per_sn[:, 0],
            #     parallel_iterations=NPROC,
            #     fn_output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
            # )
            #
            # shuffled_mask = tf.expand_dims(
            #     tf.gather(unshuffled_mask, shuffled_inds, axis=1, batch_dims=1),
            #     axis=-1,
            # )
            # mask &= shuffled_mask

            batch = tf.shape(unmasked_spectra)[0]
            n_total = tf.shape(unmasked_spectra)[-1]

            # how many spectra to mask (int32), shape [batch]
            n_spectra_to_mask = tf.cast(
                tf.reshape(n_unmasked_spectra_per_sn, [-1]), tf.int32
            )  # shape [batch]

            # 1) Generate batched random priorities and get a batched permutation:
            #    priorities: [batch, n_total]
            priorities = self.rng.uniform([batch, n_total], dtype=tf.float32)

            # shuffled_inds: for each row, gives a permutation of original indices in shuffled order.
            shuffled_inds = tf.argsort(priorities, axis=1, direction="ASCENDING")
            shuffled_inds = tf.cast(shuffled_inds, tf.int32)  # ensure int32

            # 2) Build "keep" mask in shuffled order: positions >= n_spectra_to_mask are kept.
            #    pos = [0,1,2,...,n_total-1] shape [n_total]
            pos = tf.range(n_total, dtype=tf.int32)[tf.newaxis, :]  # shape [1, n_total]
            # broadcast compare -> shape [batch, n_total]
            keep_in_shuffled = pos >= tf.expand_dims(n_spectra_to_mask, axis=1)

            # 3) Convert keep mask from "shuffled-order positions" back to original index order.
            #    Compute inverse permutation: inverse_perm[orig_idx] = position_in_shuffled
            inverse_perm = tf.argsort(
                shuffled_inds, axis=1
            )  # shape [batch, n_total], int32

            # gather keep_in_shuffled at positions given by inverse_perm -> yields keep mask in original order
            keep_in_original = tf.gather(
                keep_in_shuffled, inverse_perm, axis=1, batch_dims=1
            )
            # keep_in_original dtype == bool, shape [batch, n_total]

            # 4) Expand dims to match your mask shape ([batch, n_total, 1]) and apply
            shuffled_mask = tf.expand_dims(
                keep_in_original, axis=-1
            )  # [batch, n_total, 1]
            mask &= shuffled_mask

        return (
            phase,
            amplitude,
            d_amplitude,
            mask,
            sn_mask,
            spec_mask,
            wl_mask,
            wavelength,
            sigma,
            throughput,
            effective_wavelength,
            spectra_mask,
            phot_mask,
        )

    def recon_error(
        self,
        data: tuple[
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
        ],
        min_phase: float,
        max_phase: float,
        *,
        fractional_error: bool = False,
        weighted_error: bool = False,
        measurement_error: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        (
            phase,
            amp_true,
            d_amp,
            mask,
            sn_mask,
            spec_mask,
            wl_mask,
            wavelength,
            sigma,
            throughput,
            effective_wavelength,
            spectra_mask,
            phot_mask,
        ) = data
        phase = tf.convert_to_tensor(phase, dtype=tf.float32)
        amp_true = tf.convert_to_tensor(amp_true, dtype=tf.float32)
        d_amp = tf.convert_to_tensor(d_amp, dtype=tf.float32)
        mask = tf.convert_to_tensor(mask, dtype=tf.bool)
        sn_mask = tf.convert_to_tensor(sn_mask, dtype=tf.bool)
        spec_mask = tf.convert_to_tensor(spec_mask, dtype=tf.bool)
        wl_mask = tf.convert_to_tensor(wl_mask, dtype=tf.bool)
        wavelength = tf.convert_to_tensor(wavelength, dtype=tf.float32)
        sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
        throughput = tf.convert_to_tensor(throughput, dtype=tf.float32)
        effective_wavelength = tf.convert_to_tensor(
            effective_wavelength, dtype=tf.float32
        )
        spectra_mask = tf.convert_to_tensor(spectra_mask, dtype=tf.float32)
        phot_mask = tf.convert_to_tensor(phot_mask, dtype=tf.float32)

        time = (phase - min_phase) / (max_phase - min_phase)

        _, amp_pred = self(
            (time, amp_true),
            training=False,
            mask=mask,
            sn_mask=sn_mask,
            spec_mask=spec_mask,
            wl_mask=wl_mask,
            wavelength=wavelength,
            sigma=sigma,
            throughput=throughput,
            effective_wavelength=effective_wavelength,
            spectra_mask=spectra_mask,
            phot_mask=phot_mask,
        )
        wl_dim = tf.shape(mask)[-1]

        time_bin_width = 0.1
        time_min = 0.0
        time_max = 1.0
        num_time_bins = tf.cast(((time_max - time_min) // time_bin_width) + 2, tf.int32)

        # === Setup Masks ===
        # Apply sn and spec masks
        recon_mask = mask & sn_mask & spec_mask & wl_mask

        has_valid_data = tf.math.reduce_any(recon_mask, axis=-1)

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
        recon_mask = tf.boolean_mask(recon_mask, has_valid_data)
        time = tf.boolean_mask(time, has_valid_data)

        amp_true = tf.reshape(amp_true, [-1, wl_dim])
        amp_pred = tf.reshape(amp_pred, [-1, wl_dim])
        d_amp = tf.reshape(d_amp, [-1, wl_dim])
        recon_mask = tf.reshape(recon_mask, [-1, wl_dim])

        error = tf.abs(amp_true - amp_pred)
        eps = ks.backend.epsilon()

        meas_error = d_amp
        weights = tf.ones_like(error)
        if weighted_error:
            weights = 1 / tf.math.maximum(d_amp, eps)
        if fractional_error:
            scale = tf.math.maximum(tf.sqrt(amp_pred * amp_pred + d_amp * d_amp), eps)
            error /= scale
            meas_error /= scale
            # if weighted_error:
            #     weights *= scale
        if not measurement_error:
            meas_error *= 0

        bin_indices = tf.reshape(
            (
                tf.raw_ops.Bucketize(
                    input=time, boundaries=time_bin_edges.numpy().tolist()
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
            bin_meas_error = tf.gather(meas_error, in_bin_idx)
            bin_weights = tf.square(tf.gather(weights, in_bin_idx))
            bin_mask = tf.gather(recon_mask, in_bin_idx)

            # Compute std of masked values
            numerator = tf.reduce_sum(
                tf.where(
                    bin_mask,
                    bin_weights
                    * tf.math.maximum(
                        bin_error * bin_error - bin_meas_error * bin_meas_error, 0
                    ),
                    tf.zeros_like(bin_error),
                ),
                axis=0,
            )
            # meas_numerator = tf.reduce_sum(
            #     tf.where(
            #         bin_mask,
            #         bin_weights * bin_meas_error * bin_meas_error,
            #         tf.zeros_like(bin_meas_error),
            #     ),
            #     axis=0,
            # )
            denominator = tf.math.maximum(
                tf.reduce_sum(
                    tf.where(bin_mask, bin_weights, tf.zeros_like(bin_error)),
                    axis=0,
                ),
                eps,
            )
            var = numerator / denominator
            # meas_var = (meas_numerator / denominator) if measurement_error else 0
            # total_var = tf.math.maximum(var - meas_var, 0)
            # rms_error = tf.sqrt(total_var)
            rms_error = tf.sqrt(var)
            binned_error = binned_error.write(bin_id, rms_error)

        binned_error = tf.transpose(binned_error.stack())

        return binned_error, time_bin_edges, time_bin_centers
