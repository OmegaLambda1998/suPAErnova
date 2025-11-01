from typing import TYPE_CHECKING

from supaernova._tf import tf
from supaernova.utils.tf import pp

if TYPE_CHECKING:
    from supaernova._tf import ks


def NegLogLikelihood(
    y_true: "tf.Tensor", y_pred: "tf.Tensor", *, model: "ks.Model"
) -> "tf.Tensor":
    valid_loss = tf.math.is_finite(y_pred)
    loss_num = tf.reduce_sum(tf.where(valid_loss, y_pred, tf.zeros_like(y_pred)))
    loss_sum = tf.math.maximum(tf.math.count_nonzero(valid_loss, dtype=y_pred.dtype), 1)
    loss = loss_num / loss_sum
    return -loss
