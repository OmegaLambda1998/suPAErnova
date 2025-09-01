from typing import Any, Self, Concatenate, cast, override
from functools import cached_property
from collections.abc import Callable

from pydantic import PositiveFloat, computed_field
import tensorflow as tf
from tensorflow import keras as ks

from supaernova.steps.pae.tf import (
    loss as snpae_losses,
)
from supaernova.utils.validation import ConfigInputObject, validate_object
from supaernova.typing.backends.tf import Loss, LossFunc

from .pae import PAEConfig

ActivationObject = Callable[[tf.Tensor], tf.Tensor]


def validate_activation(
    activation: ConfigInputObject[ActivationObject],
) -> ActivationObject:
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
) -> OptimiserObject:
    return validate_object(
        optimiser, dummy_obj=ks.optimizers.Optimizer, mod=ks.optimizers
    )


LossObject = type[Loss] | Callable[..., tf.Tensor]


def validate_loss(
    loss: ConfigInputObject[LossObject],
) -> LossObject:
    err = f"Could not validate loss: {loss}:\n"
    for dummy_obj in (ks.losses.Loss, LossFunc.__call__):
        for mod in (ks.losses, snpae_losses):
            try:
                return validate_object(loss, dummy_obj=dummy_obj, mod=mod)
            except ValueError as e:
                err += f"{e}\n"
    raise ValueError(err)


def get_loss(
    loss_fn: LossFunc,
) -> type[Loss]:
    @ks.utils.register_keras_serializable("SuPAErnova")
    class CustomLoss(Loss):
        @override
        def call(self: Self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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
    def activation_fn(self: Self) -> ActivationObject:
        return validate_activation(self.activation)

    scheduler: ConfigInputObject[SchedulerObject]

    @computed_field
    @cached_property
    def scheduler_cls(self: Self) -> type[ks.optimizers.schedules.LearningRateSchedule]:
        scheduler = validate_scheduler(self.scheduler)
        if isinstance(scheduler, type):
            return scheduler

        class CustomScheduler(ks.optimizers.schedules.LearningRateSchedule):
            @override
            def __init__(
                self: Self,
                *,
                initial_learning_rate: float,
                decay_steps: int,
                decay_rate: float,
            ) -> None:
                self.initial_learning_rate: float = initial_learning_rate
                self.decay_steps: int = decay_steps
                self.decay_rate: float = decay_rate

            @override
            def __call__(self: Self, step: int | tf.Tensor) -> tf.Tensor:
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
    def optimiser_cls(self: Self) -> type[ks.optimizers.Optimizer]:
        return cast(
            "type[ks.optimizers.Optimizer]",
            cast("object", validate_optimiser(self.optimiser)),
        )

    loss: ConfigInputObject[LossObject]

    @computed_field
    @cached_property
    def loss_cls(self: Self) -> type[Loss]:
        loss = validate_loss(self.loss)

        if isinstance(loss, type):
            loss = loss()

        return get_loss(loss)

    # --- Optional ---
    kernel_regulariser: ConfigInputObject[RegulariserObject] | None = None
    kernel_regulariser_penalty: PositiveFloat | None = None

    @computed_field
    @cached_property
    def kernel_regulariser_cls(self: Self) -> type[ks.regularizers.Regularizer] | None:
        if self.kernel_regulariser is None:
            return None
        regulariser = validate_kernel_regulariser(self.kernel_regulariser)
        if isinstance(regulariser, type):
            return regulariser

        class CustomRegulariser(ks.regularizers.Regularizer):
            @override
            def __init__(self: Self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

            @override
            def __call__(self: Self, x: tf.Tensor) -> tf.Tensor:
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
