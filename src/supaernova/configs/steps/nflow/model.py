from typing import ClassVar, Annotated

from pydantic import Field, PositiveInt, PositiveFloat

from supaernova.configs.steps.pae import PAEStepConfig
from supaernova.configs.steps.steps import AbstractStepAnalysis
from supaernova.analysis.distribution import DistributionPlot
from supaernova.configs.steps.backends import AbstractModelConfig


class NFlowStepAnalysis(AbstractStepAnalysis):
    plot_z_latents: DistributionPlot | list[DistributionPlot] | None = None
    plot_u_latents: DistributionPlot | list[DistributionPlot] | None = None
    plot_latents: DistributionPlot | list[DistributionPlot] | None = None
    plot_latent_steps: DistributionPlot | list[DistributionPlot] | None = None


class NFlowModelConfig(AbstractModelConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "nflow_model"
    required_steps: ClassVar[list[str]] = [PAEStepConfig.id]
    analysis: NFlowStepAnalysis = NFlowStepAnalysis.model_validate({})

    # === Required ===
    debug: bool = False
    profile: bool = False

    # === Optional ===
    seed: int = 12345
    batch_size: PositiveInt
    patience: PositiveFloat = 0.02
    validation_frac: Annotated[float, Field(ge=0, le=1)] = 0

    save_best: bool = False

    lr: PositiveFloat = 0.0001
    lr_decay_steps: PositiveFloat = 300
    lr_decay_rate: PositiveFloat = 0.95
    lr_weight_decay_rate: PositiveFloat = 0.0001

    epochs: PositiveInt = 5000
    batch_normalisation: bool = False

    n_hidden_units: PositiveInt = 12
    n_layers: PositiveInt = 18
    physical_latents: bool
