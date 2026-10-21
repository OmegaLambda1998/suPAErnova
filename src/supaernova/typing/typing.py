from typing import Any

from typing_extensions import TypeVar

T = TypeVar("T", default=Any)
type Config[T] = dict[str, T]
type Cfg = Config[Any]
