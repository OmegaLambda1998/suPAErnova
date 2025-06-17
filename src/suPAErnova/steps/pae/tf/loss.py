import os
from typing import TYPE_CHECKING

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf

if TYPE_CHECKING:
    from .tf import TFPAEModel


def WHuber(y_true, y_pred, *, model: "TFPAEModel"):
    error = model._loss.input_mask * (y_true - y_pred) / model._loss.input_d_amp
    cond = tf.abs(error) < model.loss_clip_delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = model.loss_clip_delta * (tf.abs(error) - 0.5 * model.loss_clip_delta)
    return tf.reduce_mean(
        tf.reduce_sum(tf.where(cond, squared_loss, linear_loss), axis=(-2, -1))
    )
