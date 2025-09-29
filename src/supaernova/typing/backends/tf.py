from typing import TYPE_CHECKING, Self, Protocol
from collections.abc import Iterable

from numpy import typing as npt

from supaernova._tf import ks, tf

type TensorLike = tf.Tensor | npt.NDArray | Iterable


class Loss(ks.losses.Loss):
    model: "ks.Model"
    input_mask: "tf.Tensor"
    input_d_amp: "tf.Tensor"


class LossFunc(Protocol):
    def __call__(
        self: Self, y_true: "tf.Tensor", y_pred: "tf.Tensor", *, model: "ks.Model"
    ) -> "tf.Tensor": ...


class LearningRateSchedule(ks.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self: Self,
        initial_learning_rate: float | tf.Tensor,
        decay_steps: int,
        decay_rate: float,
    ) -> None:
        super().__init__(initial_learning_rate, decay_steps, decay_rate)
