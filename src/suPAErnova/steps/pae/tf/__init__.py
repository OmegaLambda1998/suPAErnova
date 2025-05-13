from typing import TYPE_CHECKING

from .tf import TFPAEModel

if TYPE_CHECKING:
    from .tf import S, FTensor, TensorCompatible  # noqa: TC004

    __all__ = ("FTensor", "S", "TFPAEModel", "TensorCompatible")
else:
    __all__ = ("TFPAEModel",)
