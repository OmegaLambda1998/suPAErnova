from typing import ClassVar

from pydantic import PositiveInt, PositiveFloat

from suPAErnova.configs.steps.pae import PAEStepConfig
from suPAErnova.configs.steps.steps import AbstractStepAnalysis
from suPAErnova.analysis.distribution import DistributionPlot
from suPAErnova.configs.steps.backends import AbstractModelConfig


class NFlowStepAnalysis(AbstractStepAnalysis):
    plot_z_latents: DistributionPlot | list[DistributionPlot] | None = None
    plot_u_latents: DistributionPlot | list[DistributionPlot] | None = None


class NFlowModelConfig(AbstractModelConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "nflow_model"
    required_steps: ClassVar[list[str]] = [PAEStepConfig.id]
    analysis: NFlowStepAnalysis

    # === Required ===
    debug: bool = False

    # === Optional ===
    seed: int
    batch_size: PositiveInt
    patience: PositiveFloat

    save_best: bool

    epochs: PositiveInt
    learning_rate: PositiveFloat
    batch_normalisation: bool

    n_hidden_units: PositiveInt
    n_layers: PositiveInt
    physical_latents: bool
