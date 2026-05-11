"""apeGmsh.mesh.records — Resolved FEM records and kind enums.

Canonical home for the **resolved** dataclasses produced by the
resolvers after meshing: constraint records, load records, mass
records, plus the :class:`ConstraintKind` / :class:`LoadKind`
classifiers that label them.

User-facing **definition** dataclasses (``*Def``) live in
:mod:`apeGmsh.core.constraints.defs`, :mod:`apeGmsh.core.loads.defs`,
and :mod:`apeGmsh.core.masses.defs` — they describe pre-mesh intent.
The **resolvers** that translate defs into records live in
:mod:`apeGmsh.mesh._constraint_resolver`, :mod:`apeGmsh.mesh._load_resolver`,
and :mod:`apeGmsh.mesh._mass_resolver`.
"""

from __future__ import annotations

from ._constraints import (
    ConstraintRecord,
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from ._kinds import ConstraintKind, LoadKind
from ._loads import (
    ElementLoadRecord,
    LoadRecord,
    NodalLoadRecord,
    SPRecord,
)


__all__ = [
    # Kind enums
    "ConstraintKind",
    "LoadKind",
    # Constraint records
    "ConstraintRecord",
    "NodePairRecord",
    "NodeGroupRecord",
    "InterpolationRecord",
    "SurfaceCouplingRecord",
    "NodeToSurfaceRecord",
    # Load records
    "LoadRecord",
    "NodalLoadRecord",
    "ElementLoadRecord",
    "SPRecord",
]
