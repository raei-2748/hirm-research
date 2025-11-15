"""Minimal NumPy stub for offline environments."""
from __future__ import annotations

import builtins
import math
import random as _py_random
from typing import Iterable, List, Sequence

nan = float("nan")
bool_ = bool


class ndarray:
    """Very small 1D ndarray stand-in."""

    def __init__(self, data: Iterable[float] | "ndarray", dtype: type | str | None = float) -> None:
        if isinstance(data, ndarray):
            values = list(data._data)
        else:
            values = list(data)
        caster = self._resolve_dtype(dtype)
        self._data = [caster(v) for v in values]

    @staticmethod
    def _resolve_dtype(dtype: type | str | None):
        if dtype in (float, None, "float", "float64"):
            return float
        if dtype in (int, "int", "int64"):
            return int
        return lambda x: x

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def shape(self) -> tuple[int]:
        return (len(self._data),)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self._data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ndarray(self._data[item])
        return self._data[item]

    def __setitem__(self, key, value):  # pragma: no cover - seldom used
        self._data[key] = value

    def tolist(self) -> list[float]:
        return list(self._data)

    def astype(self, dtype):  # pragma: no cover - compatibility
        return ndarray(self._data, dtype=dtype)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ndarray({self._data!r})"


def _to_list(values: Sequence[float] | ndarray) -> List[float]:
    if isinstance(values, ndarray):
        return list(values._data)
    try:
        return list(values)
    except TypeError:
        return [values]


def array(values: Iterable[float], dtype: type | str | None = float) -> ndarray:
    return ndarray(values, dtype=dtype)


def asarray(values, dtype: type | str | None = float) -> ndarray:
    if isinstance(values, ndarray) and dtype in (None, float, "float", "float64"):
        return values
    return ndarray(values, dtype=dtype)


def empty(length: int, dtype: type | str | None = float) -> ndarray:
    return ndarray([0] * int(length), dtype=dtype)


def empty_like(values: Sequence[float] | ndarray, dtype: type | str | None = None) -> ndarray:
    return ndarray([0] * len(values), dtype=dtype)


def zeros(length: int, dtype: type | str | None = float) -> ndarray:
    return ndarray([0] * int(length), dtype=dtype)


def diff(values: Sequence[float] | ndarray) -> ndarray:
    arr = _to_list(values)
    return ndarray([arr[i + 1] - arr[i] for i in range(len(arr) - 1)])


def log(values):
    if isinstance(values, ndarray):
        return ndarray([math.log(v) for v in values])
    return math.log(values)


def exp(values):
    if isinstance(values, ndarray):
        return ndarray([math.exp(v) for v in values])
    return math.exp(values)


def sqrt(values):
    if isinstance(values, ndarray):
        return ndarray([math.sqrt(v) for v in values])
    return math.sqrt(values)


def std(values, ddof: int = 0):
    arr = _to_list(values)
    n = len(arr)
    if n == 0:
        return 0.0
    mean = sum(arr) / n
    variance = sum((x - mean) ** 2 for x in arr)
    denom = max(n - ddof, 1)
    return math.sqrt(variance / denom)


def isfinite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:  # pragma: no cover
        return False


def isscalar(value) -> bool:
    if isinstance(value, ndarray):
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return False
    return True


def linspace(start: float, stop: float, num: int) -> ndarray:
    if num <= 1:
        return ndarray([start])
    step = (stop - start) / (num - 1)
    return ndarray([start + i * step for i in range(num)])


def all(values) -> bool:  # pragma: no cover
    return builtins.all(values)


class _TestingModule:
    def assert_allclose(self, a, b, rtol: float = 1e-7, atol: float = 0.0) -> None:
        arr_a = _to_list(a)
        arr_b = _to_list(b)
        if len(arr_a) != len(arr_b):
            raise AssertionError("Length mismatch")
        for x, y in zip(arr_a, arr_b):
            if abs(x - y) > atol + rtol * abs(y):
                raise AssertionError(f"Values {x} and {y} differ beyond tolerance")


testing = _TestingModule()


class _Generator:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = _py_random.Random(seed)

    def standard_normal(self, size: int | None = None):
        if size is None:
            return self._rng.gauss(0.0, 1.0)
        return [self._rng.gauss(0.0, 1.0) for _ in range(size)]

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: int | None = None):
        if size is None:
            return loc + scale * self.standard_normal()
        return [loc + scale * self.standard_normal() for _ in range(size)]

    def integers(self, low: int, high: int | None = None, size: int | None = None):
        if high is None:
            high = low
            low = 0
        if size is None:
            return self._rng.randrange(low, high)
        return [self._rng.randrange(low, high) for _ in range(size)]

    def poisson(self, lam: float = 1.0, size: int | None = None):
        def single(lam_val: float) -> int:
            L = math.exp(-lam_val)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= self._rng.random()
            return k - 1

        if size is None:
            return single(lam)
        return [single(lam) for _ in range(size)]


class _RandomModule:
    Generator = _Generator

    def default_rng(self, seed: int | None = None) -> _Generator:
        return _Generator(seed)

    def seed(self, seed: int) -> None:  # pragma: no cover
        _py_random.seed(seed)

    def rand(self, *size: int):
        """Return samples from U[0, 1) similar to ``numpy.random.rand``."""

        if not size:
            return _py_random.random()
        count = 1
        for dim in size:
            count *= int(dim)
        values = [_py_random.random() for _ in range(max(count, 0))]
        if len(size) == 1:
            return values
        # Multi-dimensional support is approximated by returning a flat list.
        return values


random = _RandomModule()


__all__ = [
    "array",
    "asarray",
    "empty",
    "empty_like",
    "zeros",
    "diff",
    "log",
    "exp",
    "sqrt",
    "std",
    "isfinite",
    "isscalar",
    "nan",
    "bool_",
    "ndarray",
    "testing",
    "linspace",
    "random",
]
