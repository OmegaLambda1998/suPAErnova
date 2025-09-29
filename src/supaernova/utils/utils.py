from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from supaernova.typing import T, Config


def pp(expression: object) -> None:
    __import__("pprint").pprint(expression)


def pf(expression: object) -> str:
    return __import__("pprint").pformat(expression)


def deepmerge(d1: "Config[T]", d2: "Config[T]") -> "Config[T]":
    out = d1.copy()
    for k, v in d2.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deepmerge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(
    input_path: str | Path | None = None,
    *,
    default_path: str | Path | None = None,
    relative_path: str | Path,
    mkdir: bool = False,
) -> Path:
    if input_path is None:
        if default_path is None:
            msg = "Cannot resolve `input_path=None` with `default_path=None"
            raise ValueError(msg)
        input_path = default_path
    input_path = Path(input_path)
    relative_path = Path(relative_path)
    if not input_path.is_absolute():
        input_path = relative_path / input_path
    final_path = input_path.resolve()
    if mkdir:
        final_path.mkdir(parents=True, exist_ok=True)
    return final_path
