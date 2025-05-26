from typing import override
from functools import cached_property

from pydantic import computed_field
import tensorflow as tf
from tensorflow import keras as ks

from suPAErnova.configs.steps import ConfigInputObject
from suPAErnova.configs.steps.pae.tf import (
    LossObject,
    validate_loss,
)

from .model import PosteriorModelConfig


class TFPosteriorModelConfig(PosteriorModelConfig):
    loss: ConfigInputObject[LossObject] | None = None

    @computed_field
    @cached_property
    def loss_cls(self) -> type[ks.losses.Loss] | None:
        if self.loss is None:
            return None
        loss = validate_loss(self.loss)
        if isinstance(loss, type):
            return loss

        @ks.saving.register_keras_serializable("SuPAErnova")
        class CustomLoss(ks.losses.Loss):
            @override
            def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                return loss(y_true, y_pred)

        return CustomLoss
