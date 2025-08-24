from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from tensorflow import keras as ks


def WHuber(
    y_true: "tf.Tensor", y_pred: "tf.Tensor", *, model: "ks.Model"
) -> "tf.Tensor":
    mask = tf.cast(model._loss.input_mask, tf.float32)
    d_amp = model._loss.input_d_amp

    error = mask * (y_true - y_pred) / d_amp
    cond = tf.abs(error) < model.loss_clip_delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = model.loss_clip_delta * (tf.abs(error) - 0.5 * model.loss_clip_delta)
    huber_loss = tf.where(cond, squared_loss, linear_loss)

    return tf.reduce_mean(tf.reduce_sum(huber_loss, axis=(-2, -1)))
