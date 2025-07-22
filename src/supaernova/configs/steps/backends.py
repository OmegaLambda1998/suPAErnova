from typing import Literal, get_args

from pydantic import ConfigDict

from .steps import StepConfig

TFBackend = Literal["tf", "tensorflow"]
TCHBackend = Literal["tch", "torch"]
Backend = TFBackend | TCHBackend

BACKENDS = {"TensorFlow": TFBackend, "PyTorch": TCHBackend}
BACKENDS_STR = ", ".join(
    f"{get_args(B)} for {backend}" for backend, B in BACKENDS.items()
)


class AbstractModelConfig(StepConfig):
    model_config: ConfigDict = ConfigDict(extra="allow")

    # === Required ===
    backend: Backend
