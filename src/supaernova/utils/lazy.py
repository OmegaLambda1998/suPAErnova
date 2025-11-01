from typing import TYPE_CHECKING, Any, cast, override
from collections.abc import Mapping, Collection

import numpy as np
from numpy import typing as npt
from pydantic import BaseModel

if TYPE_CHECKING:
    from typing import ClassVar
    from pathlib import Path
    from collections.abc import Callable


class LazyObj[O, I]:
    meta_keys: "ClassVar[set[str]]" = {
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
    meta_keys: "ClassVar[set[str]]" = LazyObj.meta_keys.union({
        "fn",
    })

    def __init__(self, fn: "Callable[[], O]") -> None:
        super().__init__()
        self.fn: Callable = fn

    def _lazy_load(self) -> None:
        super()._lazy_load()
        self.obj = self.fn()


class LazyFile[O, I](LazyObj[O, I]):
    meta_keys: "ClassVar[set[str]]" = LazyObj.meta_keys.union({
        "path",
    })

    def __init__(self, path: "Path") -> None:
        super().__init__()
        self.path: Path = path


class LazyCollection[C: Collection, L: LazyFile](
    LazyObj[type[C][L], type[C][L]], Collection
):
    meta_keys: "ClassVar[set[str]]" = LazyObj.meta_keys.union({
        "iter",
        "elem",
        "paths",
    })
    iter: type[C]
    elem: type[L]

    def __init__(self, paths: "Collection[Path]") -> None:
        super().__init__()
        self.paths: Collection[Path] = paths
        self._obj: C[L] = self.iter(self.elem(p) for p in self.paths)

    def __getitem__(self, key: int):
        return self._obj[key]

    def __setitem__(self, key: int, val: L) -> None:
        self._obj[key] = val

    def __len__(self) -> int:
        return self._obj.__len__()

    def __contains__(self) -> bool:
        return self._obj.__contains__()

    def __iter__(self):
        return self._obj.__iter__()

    def __next__(self):
        return self._obj.__next__()

    def _lazy_load(self) -> None:
        super()._lazy_load()
        for elem in self._obj:
            elem.load()

    def clear(self) -> None:
        for elem in self._obj:
            elem.clear()


class LazyList[L: LazyFile](LazyCollection[list, L]):
    iter = list


class LazySet[L: LazyFile](LazyCollection[set, L]):
    iter = set


class LazyTuple[L: LazyFile](LazyCollection[tuple, L]):
    iter = tuple


class LazyDict[K: Any, V: Any](LazyFile[Mapping[K, V], Mapping[K, V]]):
    def __getitem__(self, key: K) -> V:
        self.load()
        return self.obj[key]

    def __setitem__(self, key: K, val: V) -> None:
        self.load()
        self.obj[key] = val


class LazyNPZ(LazyDict[str, npt.NDArray]):
    @override
    def _lazy_load(self) -> None:
        super()._lazy_load()
        with np.load(self.path, allow_pickle=True) as io:
            self.obj = cast("dict[str, npt.NDArray]", dict(io.items()))


class LazyPydantic[O: BaseModel, L: LazyFile[Mapping[str, Any], Any]](LazyFile[O, L]):
    meta_keys: "ClassVar[set[str]]" = LazyFile.meta_keys.union({
        "inner",
        "model",
        "model_validate",
        "model_dump",
    })
    inner: type[L]
    model: type[O]

    def __init__(self, path: "Path") -> None:
        super().__init__(path)
        self._obj: L = self.inner(self.path)

    def _lazy_load(self) -> None:
        super()._lazy_load()
        self._obj.load()
        self.obj = self.model.model_validate(self._obj)

    def model_validate(self, *args: Any, **kwargs: Any) -> O:
        self.obj = self.model.model_validate(*args, **kwargs)
        return self.obj

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.load()
        return self.obj.model_dump(*args, **kwargs)

    def clear(self) -> None:
        super().clear()
        self._obj.clear()
