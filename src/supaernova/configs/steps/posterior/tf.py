from typing import Self, override
from functools import cached_property
from collections.abc import Callable

from pydantic import computed_field
import tensorflow as tf
from tensorflow import keras as ks

from supaernova.utils.validation import ConfigInputObject, validate_object
from supaernova.steps.posterior.tf import (
    loss as snpae_losses,
)
from supaernova.typing.backends.tf import Loss, LossFunc

from .posterior import PosteriorConfig

LossObject = type[Loss] | Callable[..., tf.Tensor]


def validate_loss(loss: ConfigInputObject[LossObject]) -> LossObject:
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
            self.reduction = "none"
            return loss_fn(y_true, y_pred, model=self.model)

    return CustomLoss


class TFPosteriorConfig(PosteriorConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    loss: ConfigInputObject[LossObject] = "NegLogLikelihood"

    @computed_field
    @cached_property
    def loss_cls(self: Self) -> type[Loss] | None:
        if self.loss is None:
            return self.loss
        loss = validate_loss(self.loss)

        if isinstance(loss, type):
            loss = loss()

        return get_loss(loss)

    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===
    # --- Training ---
