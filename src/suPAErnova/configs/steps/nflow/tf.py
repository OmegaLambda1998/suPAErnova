import os
from typing import cast, override
from functools import cached_property
from collections.abc import Callable

from pydantic import computed_field

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf
from tensorflow import keras as ks

from suPAErnova.configs.steps import ConfigInputObject, validate_object
from suPAErnova.steps.nflow.tf import (
    loss as snpae_losses,
)

from .model import NFlowModelConfig

ActivationObject = Callable[[tf.Tensor], tf.Tensor]
OptimiserObject = type[ks.optimizers.Optimizer]
LossObject = type[ks.losses.Loss] | Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def validate_activation(activation: ConfigInputObject[ActivationObject]):
    return validate_object(
        activation, dummy_obj=ks.activations.relu, mod=ks.activations
    )


def validate_optimiser(
    optimiser: ConfigInputObject[OptimiserObject],
):
    return validate_object(
        optimiser, dummy_obj=ks.optimizers.Optimizer, mod=ks.optimizers
    )


def validate_loss(loss: ConfigInputObject[LossObject]):
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
            self.reduction = "none"
            return loss_fn(y_true, y_pred, model=self.model)

    return CustomLoss


class TFNFlowModelConfig(NFlowModelConfig):
    activation: ConfigInputObject[ActivationObject]

    @computed_field
    @cached_property
    def activation_fn(self) -> ActivationObject:
        return validate_activation(self.activation)

    optimiser: ConfigInputObject[OptimiserObject]

    @computed_field
    @cached_property
    def optimiser_cls(self) -> type[ks.optimizers.Optimizer]:
        return cast(
            "type[ks.optimizers.Optimizer]",
            cast("object", validate_optimiser(self.optimiser)),
        )

    loss: ConfigInputObject[LossObject] = "NegLogLikelihood"

    @computed_field
    @cached_property
    def loss_cls(self) -> type[ks.losses.Loss] | None:
        if self.loss is None:
            return self.loss
        loss = validate_loss(self.loss)

        if isinstance(loss, type):
            loss = loss()

        return get_loss(loss)
