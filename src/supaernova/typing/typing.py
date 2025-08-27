from pathlib import Path
from collections.abc import Mapping, Iterable

from pydantic import BaseModel, JsonValue

type T = JsonValue | BaseModel | Path | Mapping[str, "T"] | Iterable["T"]
type Config[T] = dict[str, T]
