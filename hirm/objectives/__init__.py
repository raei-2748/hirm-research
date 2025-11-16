"""Objective factory and registry."""
from __future__ import annotations

from typing import Any

from hirm.objectives.erm import ERMObjective
from hirm.objectives.groupdro import GroupDROObjective
from hirm.objectives.hirm import HIRMObjective
from hirm.objectives.irmv1 import IRMv1Objective
from hirm.objectives.vrex import VRExObjective


def build_objective(cfg_objective: Any):
    """Instantiate one of the Phase-4 objectives from configuration."""

    name = getattr(cfg_objective, "name", None)
    if not name:
        raise ValueError("Objective config must include a 'name'")
    key = str(name).lower()
    if key == "erm":
        return ERMObjective(cfg_objective)
    if key == "groupdro":
        return GroupDROObjective(cfg_objective)
    if key == "vrex":
        return VRExObjective(cfg_objective)
    if key == "irmv1":
        return IRMv1Objective(cfg_objective)
    if key == "hirm":
        return HIRMObjective(cfg_objective)
    raise ValueError(f"Unknown objective '{name}'")


__all__ = [
    "build_objective",
    "ERMObjective",
    "GroupDROObjective",
    "HIRMObjective",
    "IRMv1Objective",
    "VRExObjective",
]
