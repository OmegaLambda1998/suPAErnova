from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf
    from tensorflow import keras as ks


def NegLogLikelihood(
    y_true: "tf.Tensor", y_pred: "tf.Tensor", *, model: "ks.Model"
) -> "tf.Tensor":
    return -y_pred
