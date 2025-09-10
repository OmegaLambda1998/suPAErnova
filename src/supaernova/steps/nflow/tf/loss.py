from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from tensorflow import keras as ks


def NegLogLikelihood(
    y_true: "tf.Tensor", y_pred: "tf.Tensor", *, model: "ks.Model"
) -> "tf.Tensor":
    return -tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
