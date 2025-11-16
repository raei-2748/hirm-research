"""Objective registry exports."""
from __future__ import annotations

from .base import BaseObjective, build_objective, register_objective
from . import erm as _erm  # noqa: F401 - registers objective
from . import group_dro as _group_dro  # noqa: F401
from . import vrex as _vrex  # noqa: F401
from . import irm as _irm  # noqa: F401
from . import hirm as _hirm  # noqa: F401


__all__ = [
    "BaseObjective",
    "build_objective",
    "register_objective",
]
