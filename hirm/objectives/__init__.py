"""Objective factory that exposes a registry-based API."""
from __future__ import annotations

from hirm.objectives.base import BaseObjective, build_objective, register_objective

# Import objective implementations so that they register themselves.
from . import erm as _erm  # noqa: F401
from . import group_dro as _group_dro  # noqa: F401
from . import hirm as _hirm  # noqa: F401
from . import irm as _irm  # noqa: F401
from . import vrex as _vrex  # noqa: F401


__all__ = ["BaseObjective", "build_objective", "register_objective"]
