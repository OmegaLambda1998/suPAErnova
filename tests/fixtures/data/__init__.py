from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.steps.data import DataStepResult

    DataParams = dict[str, Any]
    DataResults = dict[str, Any] | DataStep
    DataStepFactory = Callable[[DataParams], tuple[DataResults, DataParams]]

    DataStepResults = tuple[list[DataStepResult], list[DataStepResult]]
    DataResultFactory = Callable[[DataParams], DataStepResults]
