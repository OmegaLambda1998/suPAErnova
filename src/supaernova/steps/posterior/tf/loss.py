from typing import TYPE_CHECKING

from supaernova.utils.tf import db, pp

if TYPE_CHECKING:
    from supaernova._tf import ks, tf


def NegLogLikelihood(
    y_true: "tf.Tensor", y_pred: "tf.Tensor", *, model: "ks.Model"
) -> "tf.Tensor":
    return -y_pred * y_true
