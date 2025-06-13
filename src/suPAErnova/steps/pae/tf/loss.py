import os
from typing import TYPE_CHECKING

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf

if TYPE_CHECKING:
    from .tf import TFPAEModel


def WHuber(y_true, y_pred, *, model: "TFPAEModel"):
    # print("y_true", y_true)
    # print("y_pred", y_pred)
    # print("mask", model._loss.input_mask)
    # print("sigma", model._loss.input_d_amp)
    error = model._loss.input_mask * (y_true - y_pred) / model._loss.input_d_amp
    # print("error", error)

    cond = tf.abs(error) < model.loss_clip_delta
    # print("cond", cond)

    squared_loss = 0.5 * tf.square(error)
    linear_loss = model.loss_clip_delta * (tf.abs(error) - 0.5 * model.loss_clip_delta)
    # print("squared_loss", squared_loss * tf.cast(cond, tf.float32))
    # print("linear_loss", linear_loss * tf.abs(tf.cast(cond, tf.float32) - 1))

    # print(
    #     "sum", tf.reduce_sum(tf.where(cond, squared_loss, linear_loss), axis=(-2, -1))
    # )

    # print(
    #     "mean",
    #     tf.reduce_mean(
    #         tf.reduce_sum(tf.where(cond, squared_loss, linear_loss), axis=(-2, -1))
    #     ),
    # )

    return tf.reduce_mean(
        tf.reduce_sum(tf.where(cond, squared_loss, linear_loss), axis=(-2, -1))
    )
