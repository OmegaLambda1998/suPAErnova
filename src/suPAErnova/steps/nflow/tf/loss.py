from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf

    from .tf import TFNFlowModel


def NegLogLikelihood(y_true, y_pred, *, model: "TFNFlowModel"):
    return -y_pred
