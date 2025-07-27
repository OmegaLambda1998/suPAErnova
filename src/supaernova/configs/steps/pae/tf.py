import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Any, Concatenate, cast, override
from functools import cached_property
from collections.abc import Callable

from pydantic import PositiveFloat, computed_field
import tensorflow as tf
from tensorflow import keras as ks

from supaernova.utils import ConfigInputObject, validate_object
from supaernova.steps.pae.tf import (
    loss as snpae_losses,
)

from .pae import PAEConfig

ActivationObject = Callable[[tf.Tensor], tf.Tensor]


def validate_activation(activation: ConfigInputObject[ActivationObject]):
    return validate_object(activation, dummy_obj=tf.nn.relu, mod=tf.nn)


RegulariserObject = type[ks.regularizers.Regularizer] | Callable[[tf.Tensor], tf.Tensor]


def validate_kernel_regulariser(
    kernel_regulariser: ConfigInputObject[RegulariserObject],
) -> RegulariserObject:
    return validate_object(
        kernel_regulariser, dummy_obj=ks.regularizers.Regularizer, mod=ks.regularizers
    )


SchedulerObject = (
    type[ks.optimizers.schedules.LearningRateSchedule]
    | Callable[[Concatenate[int | tf.Tensor, ...]], tf.Tensor]
)


def validate_scheduler(
    scheduler: ConfigInputObject[SchedulerObject],
) -> SchedulerObject:
    return validate_object(
        scheduler,
        dummy_obj=ks.optimizers.schedules.LearningRateSchedule,
        mod=ks.optimizers.schedules,
    )


OptimiserObject = type[ks.optimizers.Optimizer]


def validate_optimiser(
    optimiser: ConfigInputObject[OptimiserObject],
):
    return validate_object(
        optimiser, dummy_obj=ks.optimizers.Optimizer, mod=ks.optimizers
    )


LossObject = type[ks.losses.Loss] | Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def validate_loss(
    loss: ConfigInputObject[LossObject],
):
    err = f"Could not validate loss: {loss}:\n"
    for dummy_obj in (ks.losses.Loss, ks.losses.mae):
        for mod in (ks.losses, snpae_losses):
            try:
                return validate_object(loss, dummy_obj=dummy_obj, mod=mod)
            except ValueError as e:
                err += f"{e}\n"
    raise ValueError(err)


def get_loss(
    loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
) -> type[ks.losses.Loss]:
    @ks.utils.register_keras_serializable("SuPAErnova")
    class CustomLoss(ks.losses.Loss):
        @override
        def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return loss_fn(y_true, y_pred, model=self.model)

    return CustomLoss


class TFPAEConfig(PAEConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    activation: ConfigInputObject[ActivationObject]

    @computed_field
    @cached_property
    def activation_fn(self) -> ActivationObject:
        return validate_activation(self.activation)

    scheduler: ConfigInputObject[SchedulerObject]

    @computed_field
    @cached_property
    def scheduler_cls(self) -> type[ks.optimizers.schedules.LearningRateSchedule]:
        scheduler = validate_scheduler(self.scheduler)
        if isinstance(scheduler, type):
            return scheduler

        class CustomScheduler(ks.optimizers.schedules.LearningRateSchedule):
            @override
            def __init__(
                self,
                *,
                initial_learning_rate: float,
                decay_steps: int,
                decay_rate: float,
            ) -> None:
                self.initial_learning_rate: float = initial_learning_rate
                self.decay_steps: int = decay_steps
                self.decay_rate: float = decay_rate

            @override
            def __call__(self, step: int | tf.Tensor) -> tf.Tensor:
                return scheduler(
                    step,
                    initial_learning_rate=self.initial_learning_rate,
                    decay_steps=self.decay_steps,
                    decay_rate=self.decay_rate,
                )

        return CustomScheduler

    optimiser: ConfigInputObject[OptimiserObject]

    @computed_field
    @cached_property
    def optimiser_cls(self) -> type[ks.optimizers.Optimizer]:
        return cast(
            "type[ks.optimizers.Optimizer]",
            cast("object", validate_optimiser(self.optimiser)),
        )

    loss: ConfigInputObject[LossObject]

    @computed_field
    @cached_property
    def loss_cls(self) -> type[ks.losses.Loss]:
        loss = validate_loss(self.loss)

        if isinstance(loss, type):
            loss = loss()

        return get_loss(loss)

    # --- Optional ---
    kernel_regulariser: ConfigInputObject[RegulariserObject] | None = None
    kernel_regulariser_penalty: PositiveFloat | None = None

    @computed_field
    @cached_property
    def kernel_regulariser_cls(self) -> type[ks.regularizers.Regularizer] | None:
        if self.kernel_regulariser is None:
            return None
        regulariser = validate_kernel_regulariser(self.kernel_regulariser)
        if isinstance(regulariser, type):
            return regulariser

        class CustomRegulariser(ks.regularizers.Regularizer):
            @override
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

            @override
            def __call__(self, x: tf.Tensor) -> tf.Tensor:
                return regulariser(x)

        return CustomRegulariser

    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===
    # --- Training ---
