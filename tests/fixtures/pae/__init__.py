from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    from tests.fixtures.data import DataParams
    from supaernova.steps.pae import PAEStep
    from supaernova.configs.steps.pae import PAEStepResult
    from supaernova.configs.steps.data import DataStepResult

    PAEParams = dict[str, Any]
    PAEResults = dict[str, dict[str, Any]] | PAEStep
    PAEStepFactory = Callable[
        [DataParams, PAEParams], tuple[PAEResults, tuple[PAEParams, DataStepResult]]
    ]

    PAEStepResults = dict[str, PAEStepResult]
    PAEResultFactory = Callable[[DataParams, PAEParams], PAEStepResults]
