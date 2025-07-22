from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    from supaernova.steps.data import DataStep
    from supaernova.configs.steps.data import DataStepResult

    DataParams = dict[str, Any]
    DataResults = dict[str, Any] | DataStep
    DataStepFactory = Callable[[DataParams], tuple[DataResults, DataParams]]

    DataStepResults = DataStepResult
    DataResultFactory = Callable[[DataParams], DataStepResults]
