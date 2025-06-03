from typing import ClassVar

from pydantic import PositiveInt, PositiveFloat

from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.backends import AbstractModelConfig


class PosteriorModelConfig(AbstractModelConfig):
    # --- Class Variables ---
    required_steps: ClassVar[list[str]] = [NFlowStepConfig.id]
    id: ClassVar[str] = "posterior_model"

    # === Required ===
    debug: bool = False

    # === Optional ===
    seed: int = 12345
    batch_size: PositiveInt = 32
    save_best: bool = False
    random_initial_positions: bool = True
    tolerance: PositiveFloat = 0.01
    max_iterations: PositiveInt = 2500

    n_chains_early: PositiveInt = 10
    n_chains_mid: PositiveInt = 10
    n_chains_final: PositiveInt = 10

    latents_mean: float = 0.0
    latents_std: float = 1.0

    train_delta_av: bool = True
    delta_av_min: float = -1.5
    delta_av_max: float = 1.5
    delta_av_mean: float = 0.0
    delta_av_std: float = 1.0

    train_delta_m: bool = True
    delta_m_min: float = -1.5
    delta_m_max: float = 1.5
    delta_m_mean: float = 0.0
    delta_m_std: float = 1.0

    train_delta_p: bool = True
    delta_p_mean: float = 0.0
    delta_p_min: float = -1.5
    delta_p_max: float = 1.5
    delta_p_std: float = 1.0

    train_bias: bool = True
    bias_min: float = -1.0
    bias_max: float = 1.0
    bias_mean: float = 0.0
    bias_std: float = 1.0
