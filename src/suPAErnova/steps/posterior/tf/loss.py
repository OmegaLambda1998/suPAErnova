from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tf import TFPosteriorModel


def NegLogLikelihood(y_true, y_pred, *, model: "TFPosteriorModel"):
    return -y_pred / 100
