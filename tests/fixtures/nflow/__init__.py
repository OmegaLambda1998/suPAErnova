from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    from tests.fixtures.pae import PAEParams
    from tests.fixtures.data import DataParams
    from supaernova.steps.nflow import NFlowStep
    from supaernova.configs.steps.nflow import NFlowStepResult

    NFlowParams = dict[str, Any]
    NFlowResults = dict[str, Any] | NFlowStep
    NFlowStepFactory = Callable[
        [DataParams, PAEParams, NFlowParams], tuple[NFlowResults, NFlowParams]
    ]

    NFlowStepResults = NFlowStepResult
    NFlowResultFactory = Callable[
        [DataParams, PAEParams, NFlowParams], NFlowStepResults
    ]
