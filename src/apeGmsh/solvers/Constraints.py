"""
Constraints — Solver-agnostic multi-point constraint engine.
=============================================================

Two-stage pipeline:

    Stage 1  ─  **Definition** (pre-mesh, geometry level)
    ─────────────────────────────────────────────────────────
    The user declares *intent*:  "these two instances are tied
    at this interface" or "this node is rigidly linked to that
    surface."  Definitions are lightweight dataclasses that
    carry the geometry-level information.

    Stage 2  ─  **Resolution** (post-mesh)
    ─────────────────────────────────────────────────────────
    After meshing, :class:`ConstraintResolver` converts each
    definition into concrete :class:`ConstraintRecord` objects
    containing actual node tags, DOFs, weights, and offset
    vectors.  These records are **solver-agnostic** — any
    solver adapter (OpenSees, Abaqus, Code_Aster, …) can
    consume them.

Constraint taxonomy
~~~~~~~~~~~~~~~~~~~

**Level 1 — Node-to-Node** (1 master, 1 slave)

=================  ================================================
``equal_dof``      u_slave = u_master  (selected DOFs)
``rigid_beam``     slave follows master as rigid bar (6-DOF coupling
                   with rotational offset)
``rigid_rod``      only translations coupled through rigid bar
``penalty``        soft spring K_p between two nodes
=================  ================================================

**Level 2 — Node-to-Group** (1 master, N slaves)

=================  ================================================
``rigid_diaphragm``  in-plane DOFs of all slaves follow master
``rigid_body``       all 6 DOFs follow master
``kinematic``        general: user picks which DOFs
=================  ================================================

**Level 2b — Mixed-DOF coupling** (1 6-DOF master, N 3-DOF slaves)

=================  ================================================
``node_to_surface`` 6-DOF master -> phantom nodes -> 3-DOF solid slaves
                   via rigid link + equalDOF (translations only)
=================  ================================================

**Level 3 — Node-to-Surface** (1 node, 1 element face)

=================  ================================================
``tie``            u_slave = Σ N_i · u_master_i  (shape function
                   interpolation on closest master face)
``distributing``   force at master distributed to slave surface
``embedded``       embedded element nodes follow host field
=================  ================================================

**Level 4 — Surface-to-Surface**

=================  ================================================
``tied_contact``   all nodes on surface A tied to surface B
``mortar``         Lagrange multiplier space on interface
=================  ================================================

All constraints ultimately express the linear MPC equation:

    u_slave = C · u_master

where C is the constraint transformation matrix.

Module organisation
~~~~~~~~~~~~~~~~~~~

This module is a thin re-export shim. Implementations live in:

* :mod:`apeGmsh.solvers._constraint_defs` — Stage-1 Def dataclasses
* :mod:`apeGmsh.solvers._constraint_records` — Stage-2 Record dataclasses
* :mod:`apeGmsh.solvers._constraint_geom` — shape functions, spatial index,
  Newton projection
* :mod:`apeGmsh.solvers._constraint_resolver` — :class:`ConstraintResolver`

All public names that were previously imported from
``apeGmsh.solvers.Constraints`` continue to be available here.
"""

from __future__ import annotations

from ._constraint_defs import (
    ConstraintDef,
    DistributingCouplingDef,
    EmbeddedDef,
    EqualDOFDef,
    KinematicCouplingDef,
    MortarDef,
    NodeToSurfaceDef,
    NodeToSurfaceSpringDef,
    PenaltyDef,
    RigidBodyDef,
    RigidDiaphragmDef,
    RigidLinkDef,
    TieDef,
    TiedContactDef,
)
from ._constraint_geom import (  # noqa: F401  (intentional re-exports of private helpers that callers used to import from this module pre-split)
    SHAPE_FUNCTIONS,
    _SpatialIndex,
    _is_inside_parametric,
    _project_point_to_face,
    _shape_quad4,
    _shape_quad8,
    _shape_tri3,
    _shape_tri6,
)
from ._constraint_records import (
    ConstraintRecord,
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from ._constraint_resolver import ConstraintResolver


__all__ = [
    # Defs
    "ConstraintDef",
    "EqualDOFDef",
    "RigidLinkDef",
    "PenaltyDef",
    "RigidDiaphragmDef",
    "RigidBodyDef",
    "KinematicCouplingDef",
    "TieDef",
    "DistributingCouplingDef",
    "EmbeddedDef",
    "NodeToSurfaceDef",
    "NodeToSurfaceSpringDef",
    "TiedContactDef",
    "MortarDef",
    # Records
    "ConstraintRecord",
    "NodePairRecord",
    "NodeGroupRecord",
    "InterpolationRecord",
    "SurfaceCouplingRecord",
    "NodeToSurfaceRecord",
    # Resolver
    "ConstraintResolver",
    # Geom helpers (historically importable)
    "SHAPE_FUNCTIONS",
]
