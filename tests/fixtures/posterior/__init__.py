from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    from tests.fixtures.pae import PAEParams
    from tests.fixtures.data import DataParams
    from tests.fixtures.nflow import NFlowParams
    from suPAErnova.steps.posterior import PosteriorStep
    from suPAErnova.configs.steps.posterior import PosteriorStepResult

    PosteriorParams = dict[str, Any]
    PosteriorResults = dict[str, Any] | PosteriorStep
    PosteriorStepFactory = Callable[
        [DataParams, PAEParams, NFlowParams, PosteriorParams], PosteriorResults
    ]

    PosteriorStepResults = PosteriorStepResult
    PosteriorResultFactory = Callable[
        [DataParams, PAEParams, NFlowParams, PosteriorParams], PosteriorStepResults
    ]
