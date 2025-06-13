import os
from typing import TYPE_CHECKING

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf

if TYPE_CHECKING:
    from .tf import TFNFlowModel


def NegLogLikelihood(y_true, y_pred, *, model: "TFNFlowModel"):
    return -y_pred
