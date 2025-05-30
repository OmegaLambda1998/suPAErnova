from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    from tests.fixtures.data import DataParams
    from suPAErnova.steps.pae import PAEStep
    from suPAErnova.configs.steps.pae import PAEStepResult

    PAEParams = dict[str, Any]
    PAEResults = dict[str, dict[str, Any]] | PAEStep
    PAEStepFactory = Callable[[DataParams, PAEParams], PAEResults]

    PAEStepResults = dict[str, PAEStepResult]
    PAEResultFactory = Callable[[DataParams, PAEParams], PAEStepResults]
