# Copyright 2025 Patrick Armstrong

from copy import deepcopy
from types import ModuleType
from typing import Any, ClassVar, override
from inspect import signature
from pathlib import Path
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, DirectoryPath

from supaernova.configs import SNPAEConfig
from supaernova.configs.paths import PathConfig

Fn = Callable[..., Any]
type ConfigInputObject[T: Fn] = str | Path | T


def validate_signature[T: Fn](obj: T, dummy_obj: T, attr: str | None = None) -> T:
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


def extract_from_module[T: Fn](name: str, mod: ModuleType, _type_hint: type[T]) -> T:
    if not hasattr(mod, name):
        err = f"Module `{mod}` has no attribute `{name}`"
        raise ValueError(err)
    return getattr(mod, name)


def extract_from_file[T: Fn](name: str, file: Path, _type_hint: type[T]) -> T:
    if not file.exists():
        err = f"File `{file}` does not exist"
        raise ValueError(err)
    if file.suffix != ".py":
        err = f"File `{file}` is not a `.py` file"
        raise ValueError(err)

    with file.open("r") as io:
        code = io.read()
    mod = ModuleType(str(file))
    exec(code, mod.__dict__)
    return extract_from_module(name, mod, _type_hint)


def validate_object[T: Fn](
    obj: ConfigInputObject[T],
    *,
    dummy_obj: T,
    mod: ModuleType | None = None,
    attr: str | None = None,
) -> T:
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


class AbstractStepResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    metadata: dict[str, Any] | None = None


class AbstractStepAnalysis(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, allow="extra"
    )


class StepConfig(SNPAEConfig):
    # Class Variables
    steps: ClassVar[dict[str, type["StepConfig"]]] = {}
    required_steps: ClassVar[list[str]] = []
    id: ClassVar[str]

    external: DirectoryPath | None = None

    @classmethod
    def register_step(cls) -> None:
        cls.steps[cls.id] = cls

    @override
    @classmethod
    def from_config(
        cls,
        input_config: dict[str, Any],
    ) -> "StepConfig":
        step_config = deepcopy(input_config)
        if step_config["paths"].out.parent.name != "outputs":
            step_config["paths"].out = PathConfig.resolve_path(
                step_config["paths"].out
                / "outputs"
                / step_config.get("name", cls.__name__),
                relative_path=step_config["paths"].base,
                mkdir=True,
            )
        step_config["paths"].plots = PathConfig.resolve_path(
            step_config["paths"].plots / step_config.get("name", cls.__name__),
            relative_path=step_config["paths"].base,
            mkdir=True,
        )
        step_config["paths"].log = PathConfig.resolve_path(
            step_config["paths"].log / step_config.get("name", cls.__name__),
            relative_path=step_config["paths"].base,
            mkdir=True,
        )
        if step_config.get("external") is not None:
            step_config["paths"].out = PathConfig.resolve_path(
                Path(step_config["external"]), relative_path=step_config["paths"].base
            )
            step_config["external"] = step_config["paths"].out
        return super().from_config(step_config)
