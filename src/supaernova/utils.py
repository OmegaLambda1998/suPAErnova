from types import ModuleType
from typing import TYPE_CHECKING, Any
from inspect import signature
from pathlib import Path
from collections.abc import Callable

Fn = Callable[..., Any]
type ConfigInputObject[F: Fn] = str | Path | F


if TYPE_CHECKING:
    from supaernova.typing import T, Config


def pp(expression: object) -> None:
    __import__("pprint").pprint(expression)


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


def validate_signature[F: Fn](obj: F, dummy_obj: F, attr: str | None = None) -> F:
    if attr is not None:
        fn_signature = signature(getattr(obj, attr))
        dummy_signature = signature(getattr(dummy_obj, attr))
    else:
        fn_signature = signature(obj)
        dummy_signature = signature(dummy_obj)
    for dummy_param, dummy_sig in dummy_signature.parameters.items():
        if dummy_param in {"args", "kwargs", "self"}:
            continue
        if dummy_param not in fn_signature.parameters:
            err = f"Function `{obj.__name__}` is missing argument `{dummy_param}` of type `{dummy_sig.annotation}`"
            raise ValueError(err)
        fn_sig = fn_signature.parameters[dummy_param]
        if fn_sig.annotation != dummy_sig.annotation:
            err = f"Argument `{dummy_param}` of function `{obj.__name__}` should be of type `{dummy_sig}`, but is instead of type `{fn_sig}`"
            raise ValueError(err)
    dummy_rtn = dummy_signature.return_annotation
    fn_rtn = fn_signature.return_annotation
    if fn_rtn != dummy_rtn:
        err = f"Function `{obj.__name__}` should have a return type of `{dummy_rtn}`, but instead has a return type of `{fn_rtn}`"
        raise ValueError(err)
    return obj


def extract_from_module[F: Fn](name: str, mod: ModuleType, _type_hint: type[F]) -> F:
    if not hasattr(mod, name):
        err = f"Module `{mod}` has no attribute `{name}`"
        raise ValueError(err)
    return getattr(mod, name)


def extract_from_file[F: Fn](name: str, file: Path, _type_hint: type[F]) -> F:
    if not file.exists():
        err = f"File `{file}` does not exist"
        raise ValueError(err)
    if file.suffix != ".py":
        err = f"File `{file}` is not a `.py` file"
        raise ValueError(err)

    with file.open("r", encoding="utf-8") as io:
        code = io.read()
    mod = ModuleType(str(file))
    exec(code, mod.__dict__)
    return extract_from_module(name, mod, _type_hint)


def validate_object[F: Fn](
    obj: ConfigInputObject[F],
    *,
    dummy_obj: F,
    mod: ModuleType | None = None,
    attr: str | None = None,
) -> F:
    type_hint = type(dummy_obj)
    if isinstance(obj, str):
        path = Path(obj)
        if path.exists() and path.suffix == ".py":
            obj = path
        elif mod is None:
            err = "When specifying a function by name, you must also include a module to extract it from via `validate_function(fn, dummy_fn=dummy_fn, mod=module)"
            raise ValueError(err)
        else:
            obj = extract_from_module(obj, mod, type_hint)
    if isinstance(obj, Path):
        obj = extract_from_file(dummy_obj.__name__, obj, type_hint)
    return validate_signature(obj, dummy_obj, attr=attr)
