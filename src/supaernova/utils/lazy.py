from collections.abc import (
    Callable,
    Collection,
    Iterable,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

import numpy as np
from numpy import typing as npt
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


class LazyObj[O, I]:
    meta_keys: ClassVar[set[str]] = {
        # `isinstance` falls back to `__class__`, which would otherwise trigger a load
        "__class__",
        "obj",
        "load",
        "clear",
        "_obj",
        "_lazy_load",
    }

    def __init__(self) -> None:
        self.obj: O | None = None
        self._obj: I | None = None

    def __getattribute__(self, key: str) -> "Any":
        if key in type(self).meta_keys:
            return object.__getattribute__(self, key)
        self.load()
        return object.__getattribute__(self.obj, key)

    def __setattr__(self, key: str, val: "Any") -> None:
        if key in type(self).meta_keys:
            super().__setattr__(key, val)
            return
        self.load()
        setattr(self.obj, key, val)

    def _lazy_load(self) -> None:
        pass

    def load(self) -> None:
        if self.obj is None:
            self._lazy_load()

    def clear(self) -> None:
        self.obj = None


class LazyLambda[O, I](LazyObj[O, I]):
    meta_keys: ClassVar[set[str]] = LazyObj.meta_keys.union(
        {
            "fn",
        }
    )

    def __init__(self, fn: "Callable[[], O]") -> None:
        super().__init__()
        self.fn: Callable = fn

    def _lazy_load(self) -> None:
        super()._lazy_load()
        self.obj = self.fn()


class LazyFile[O, I](LazyObj[O, I]):
    meta_keys: ClassVar[set[str]] = LazyObj.meta_keys.union(
        {
            "path",
        }
    )

    def __init__(self, path: "Path") -> None:
        super().__init__()
        self.path: Path = path


class LazySequence[C: Sequence, L: LazyFile](LazyObj[C, C]):
    meta_keys: ClassVar[set[str]] = LazyObj.meta_keys.union(
        {
            "iter",
            "elem",
            "paths",
        }
    )
    iter: Callable[[Iterable[L]], C]
    elem: type[L]

    def __init__(self, paths: "Collection[Path]") -> None:
        super().__init__()
        self.paths: Collection[Path] = paths
        self._obj = self.iter(self.elem(p) for p in self.paths)

    def __getitem__(self, key: int):
        assert self._obj is not None
        return self._obj[key]

    def __len__(self) -> int:
        assert self._obj is not None
        return self._obj.__len__()

    def __contains__(self, x) -> bool:
        assert self._obj is not None
        return self._obj.__contains__(x)

    def __iter__(self):
        assert self._obj is not None
        return self._obj.__iter__()

    def _lazy_load(self) -> None:
        super()._lazy_load()
        assert self._obj is not None
        for elem in self._obj:
            elem.load()

    def clear(self) -> None:
        assert self._obj is not None
        for elem in self._obj:
            elem.clear()


class LazyMutableSequence[C: MutableSequence, L: LazyFile](LazySequence[C, L]):
    def __setitem__(self, key: int, val: L) -> None:
        assert self._obj is not None
        self._obj[key] = val


class LazyList[L: LazyFile](LazyMutableSequence[list, L]):
    iter = list


class LazyTuple[L: LazyFile](LazySequence[tuple, L]):
    iter = tuple


class LazyDict[K: Any, V: Any](LazyFile[MutableMapping[K, V], MutableMapping[K, V]]):
    def __getitem__(self, key: K) -> V:
        self.load()
        assert self.obj is not None
        return self.obj[key]

    def __setitem__(self, key: K, val: V) -> None:
        self.load()
        assert self.obj is not None
        self.obj[key] = val


class LazyNPZ(LazyDict[str, npt.NDArray]):
    @override
    def _lazy_load(self) -> None:
        super()._lazy_load()
        with np.load(self.path, allow_pickle=True) as io:
            self.obj = cast("dict[str, npt.NDArray]", dict(io.items()))


class LazyPydantic[O: BaseModel, L: LazyFile[Mapping[str, Any], Any]](LazyFile[O, L]):
    meta_keys: "ClassVar[set[str]]" = LazyFile.meta_keys.union(
        {
            "inner",
            "model",
            "model_validate",
            "model_dump",
        }
    )
    inner: type[L]
    model: type[O]

    def __init__(self, path: "Path") -> None:
        super().__init__(path)
        self._obj = self.inner(self.path)

    def _lazy_load(self) -> None:
        super()._lazy_load()
        assert self._obj is not None
        self._obj.load()
        self.obj = self.model.model_validate(self._obj.obj)

    def model_validate(self, *args: Any, **kwargs: Any) -> O:
        self.obj = self.model.model_validate(*args, **kwargs)
        return self.obj

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.load()
        assert self.obj is not None
        return self.obj.model_dump(*args, **kwargs)

    def clear(self) -> None:
        super().clear()
        assert self._obj is not None
        self._obj.clear()
