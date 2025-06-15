from typing import ClassVar
from pathlib import Path

from pydantic import PositiveInt, PositiveFloat

from suPAErnova.configs.steps.nflow import NFlowStepConfig
from suPAErnova.configs.steps.steps import AbstractStepResult, AbstractStepAnalysis
from suPAErnova.configs.steps.backends import AbstractModelConfig


class PosteriorStepPlotDistribution(AbstractStepAnalysis):
    name: str
    savepath: Path | None = None
    ext: str = "svg"
    labels: list[str] = []


class PosteriorStepAnalysis(AbstractStepAnalysis):
    plot_distribution: (
        PosteriorStepPlotDistribution | list[PosteriorStepPlotDistribution] | None
    ) = None


class PosteriorModelConfig(AbstractModelConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "posterior_model"
    required_steps: ClassVar[list[str]] = [NFlowStepConfig.id]

    # === Required ===
    debug: bool = False

    # === Optional ===
    seed: int
    batch_size: PositiveInt
    train: bool = True
    analysis: PosteriorStepAnalysis

    save_best: bool

    n_chains_early: int
    n_chains_mid: int
    n_chains_final: int

    tolerance: PositiveFloat
    max_iterations: PositiveInt

    n_burnin: PositiveInt
    n_samples: PositiveInt
    n_leapfrog: PositiveInt
    target_acceptance_rate: PositiveFloat

    random_initial_positions: bool

    u_delta_av_min: float = -1.0
    u_delta_av_max: float = 1.0
    u_delta_av_mean: float = 0.0
    u_delta_av_std: float = 1.0

    u_latents_mean: float = 0.0
    u_latents_std: float = 1.0

    delta_av_min: float = -0.5
    delta_av_max: float = 0.5
    delta_av_mean: float = 0.0
    delta_av_std: float = 0.5

    train_delta_m: bool
    delta_m_min: float = -1.5
    delta_m_max: float = 1.5
    delta_m_mean: float = 0.0
    delta_m_std: float = 1.5

    train_delta_p: bool
    delta_p_min: float = -1.0
    delta_p_max: float = 1.0
    delta_p_mean: float = 0.0
    delta_p_std: float = 1.0

    train_bias: bool
    bias_min: float = -1.0
    bias_max: float = 1.0
    bias_mean: float = 0.0
    bias_std: float = 1.0
