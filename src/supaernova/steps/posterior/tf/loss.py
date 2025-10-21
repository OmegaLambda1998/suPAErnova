from typing import TYPE_CHECKING

from supaernova._tf import HUGE, tf
from supaernova.utils.tf import db, pp

if TYPE_CHECKING:
    from supaernova._tf import ks


def NegLogLikelihood(
    y_true: "tf.Tensor", y_pred: "tf.Tensor", *, model: "ks.Model"
) -> "tf.Tensor":
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, -HUGE * tf.ones_like(y_pred))
    return -y_pred * y_true
