"""
FEMData — Solver-ready mesh data container.
============================================

Returned by ``Mesh.get_fem_data()``, this module holds everything
needed to build a solver model: node IDs, coordinates, element IDs,
connectivity, mesh statistics, and physical group snapshots.

The data is fully self-contained — once extracted, no live Gmsh
session is required.

Classes
-------
FEMData
    Top-level container with ``.info`` and ``.physical`` sub-objects.
MeshInfo
    Read-only mesh statistics (n_nodes, n_elems, bandwidth).
PhysicalGroupSet
    Snapshot of physical groups mirroring the ``g.physical`` API.

Usage
-----
::

    g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=2)

    # Mesh statistics
    print(fem.info)

    # Physical groups
    fem.physical.get_all()
    fem.physical.get_name(0, 1)
    base = fem.physical.get_nodes(0, 1)

    # Build solver model
    for i in range(fem.info.n_nodes):
        ops.node(int(fem.node_ids[i]),
                 *fem.node_coords[i])

    for i in range(fem.info.n_elems):
        ops.element('ShellMITC4', int(fem.element_ids[i]),
                    *fem.connectivity[i], sec_tag)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from .MeshSelectionSet import MeshSelectionStore


# =====================================================================
# Bandwidth computation
# =====================================================================

def _compute_bandwidth(connectivity: ndarray) -> int:
    """
    Compute the semi-bandwidth of the mesh.

    bandwidth = max over all elements of (max_node_id - min_node_id)

    This is the semi-bandwidth of the assembled stiffness matrix.
    Lower is better for direct solvers.
    """
    if connectivity.size == 0:
        return 0
    row_max = connectivity.max(axis=1)
    row_min = connectivity.min(axis=1)
    return int((row_max - row_min).max())


# =====================================================================
# Mesh info
# =====================================================================

class MeshInfo:
    """
    Read-only summary of mesh statistics.

    Accessed via ``fem.info``.

    Attributes
    ----------
    n_nodes : int
    n_elems : int
    bandwidth : int
    nodes_per_elem : int
        Number of nodes per element (e.g. 4 for tet4, 3 for tri3).
    elem_type_name : str
        Gmsh element type name (e.g. ``"Tetrahedron 4"``).
    """

    # Type declarations for __slots__ (consumed by mypy / pyright)
    n_nodes: int
    n_elems: int
    bandwidth: int
    nodes_per_elem: int
    elem_type_name: str

    __slots__ = ('n_nodes', 'n_elems', 'bandwidth',
                 'nodes_per_elem', 'elem_type_name')

    def __init__(
        self,
        n_nodes: int,
        n_elems: int,
        bandwidth: int,
        nodes_per_elem: int = 0,
        elem_type_name: str = "",
    ) -> None:
        object.__setattr__(self, 'n_nodes', n_nodes)
        object.__setattr__(self, 'n_elems', n_elems)
        object.__setattr__(self, 'bandwidth', bandwidth)
        object.__setattr__(self, 'nodes_per_elem', nodes_per_elem)
        object.__setattr__(self, 'elem_type_name', elem_type_name)

    def __repr__(self) -> str:
        parts = (
            f"MeshInfo(n_nodes={self.n_nodes}, n_elems={self.n_elems}, "
            f"bandwidth={self.bandwidth}"
        )
        if self.elem_type_name:
            parts += f", type={self.elem_type_name!r}"
        return parts + ")"

    def summary(self) -> str:
        """One-line summary string."""
        s = f"{self.n_nodes} nodes, {self.n_elems} elements"
        if self.elem_type_name:
            s += f" ({self.elem_type_name})"
        s += f", bandwidth={self.bandwidth}"
        return s


# =====================================================================
# Physical group snapshot
# =====================================================================

class _GroupSetBase:
    """Shared logic for PhysicalGroupSet and LabelSet.

    Both store a ``{(dim, tag): info_dict}`` structure and expose
    name-first direct-array access.
    """

    def __init__(self, groups: dict[tuple[int, int], dict]) -> None:
        self._groups = groups
        self._name_index: dict[str, tuple[int, int]] | None = None

    # ── Internal resolution ──────────────────────────────────

    def _build_name_index(self) -> dict[str, tuple[int, int]]:
        if self._name_index is None:
            idx: dict[str, tuple[int, int]] = {}
            # Lowest dim first so dim=0 wins over dim=2 for same name
            for (d, t) in sorted(self._groups.keys()):
                name = self._groups[(d, t)].get('name', '')
                if name and name not in idx:
                    idx[name] = (d, t)
            self._name_index = idx
        return self._name_index

    def _resolve(self, target) -> dict:
        """Resolve *target* (str name, int tag, or (dim, tag) tuple)
        to the internal info dict.  Raises KeyError on miss."""
        if isinstance(target, str):
            idx = self._build_name_index()
            key = idx.get(target)
            if key is None:
                raise KeyError(
                    f"No group named {target!r}. "
                    f"Available: {self.names()}"
                )
            return self._groups[key]
        if isinstance(target, tuple):
            info = self._groups.get(target)
            if info is None:
                raise KeyError(
                    f"No group {target}. Available: {self.get_all()}"
                )
            return info
        # int tag — search across dims (lowest first)
        for (d, t) in sorted(self._groups.keys()):
            if t == int(target):
                return self._groups[(d, t)]
        raise KeyError(
            f"No group with tag {target}. Available: {self.get_all()}"
        )

    # ── Name-first, direct-array access ──────────────────────

    def node_ids(self, target) -> ndarray:
        """Node IDs for a group.

        Parameters
        ----------
        target : str, int, or (dim, tag)
            PG name, tag, or ``(dim, tag)`` tuple.

        Returns
        -------
        ndarray(N,) — object dtype (yields Python ``int``).

        Example
        -------
        ::

            base = fem.physical.node_ids("base_supports")
            for nid in base:
                ops.fix(nid, 1, 1, 1)
        """
        info = self._resolve(target)
        return np.asarray(info['node_ids']).astype(object)

    def node_coords(self, target) -> ndarray:
        """Node coordinates for a group.

        Returns
        -------
        ndarray(N, 3) — float64.
        """
        info = self._resolve(target)
        return np.asarray(info['node_coords'], dtype=np.float64)

    def element_ids(self, target) -> ndarray:
        """Element IDs for a group (dim >= 1 only).

        Returns
        -------
        ndarray(E,) — object dtype.

        Raises
        ------
        ValueError
            If the group has no element data (dim=0 groups).
        """
        info = self._resolve(target)
        eids = info.get('element_ids')
        if eids is None:
            name = info.get('name', str(target))
            raise ValueError(
                f"Group '{name}' has no element data "
                f"(element data is only available for dim >= 1)."
            )
        return np.asarray(eids).astype(object)

    def connectivity(self, target) -> ndarray:
        """Element connectivity for a group (dim >= 1 only).

        Returns
        -------
        ndarray(E, npe) — object dtype.
        """
        info = self._resolve(target)
        conn = info.get('connectivity')
        if conn is None:
            name = info.get('name', str(target))
            raise ValueError(
                f"Group '{name}' has no element data "
                f"(element data is only available for dim >= 1)."
            )
        return np.asarray(conn).astype(object)

    # ── Queries ──────────────────────────────────────────────

    def names(self, dim: int = -1) -> list[str]:
        """Return all group names, optionally filtered by dimension.

        Example
        -------
        ::

            fem.physical.names()       # all
            fem.physical.names(dim=0)  # point groups only
        """
        result = []
        for (d, _), info in sorted(self._groups.items()):
            name = info.get('name', '')
            if name and (dim == -1 or d == dim):
                result.append(name)
        return result

    def get_all(self, dim: int = -1) -> list[tuple[int, int]]:
        """Return all groups as ``(dim, tag)`` pairs.

        Parameters
        ----------
        dim : filter to a single dimension (``-1`` = all)
        """
        if dim == -1:
            return sorted(self._groups.keys())
        return sorted(k for k in self._groups if k[0] == dim)

    def get_name(self, dim: int, tag: int) -> str:
        """Return the name of a group, or ``""`` if unnamed."""
        info = self._groups.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No group (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        return info.get('name', '')

    def get_tag(self, dim: int, name: str) -> int | None:
        """Look up the tag of a named group.  Returns None if not found."""
        for (d, pg_tag), info in self._groups.items():
            if d == dim and info.get('name', '') == name:
                return pg_tag
        return None

    # ── Legacy dict-return methods (backward compat) ─────────

    def get_nodes(self, dim: int, tag: int) -> dict:
        """Return ``{'tags': ndarray, 'coords': ndarray}`` for a group.

        Prefer :meth:`node_ids` and :meth:`node_coords` for direct access.
        """
        info = self._groups.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No group (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        return {
            'tags':   np.asarray(info['node_ids']).astype(object),
            'coords': np.asarray(info['node_coords'], dtype=np.float64),
        }

    def get_elements(self, dim: int, tag: int) -> dict:
        """Return ``{'element_ids': ndarray, 'connectivity': ndarray}``.

        Prefer :meth:`element_ids` and :meth:`connectivity` for direct access.
        """
        info = self._groups.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No group (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        elem_ids = info.get('element_ids')
        conn     = info.get('connectivity')
        if elem_ids is None or conn is None:
            name = info.get('name', f'(dim={dim}, tag={tag})')
            raise ValueError(
                f"Group '{name}' has no element data. "
                f"Element data is only available for dim >= 1 groups."
            )
        return {
            'element_ids':  np.asarray(elem_ids).astype(object),
            'connectivity': np.asarray(conn).astype(object),
        }

    # ── Display ──────────────────────────────────────────────

    def summary(self):
        """DataFrame describing every group.

        Returns
        -------
        pd.DataFrame indexed by ``(dim, pg_tag)`` with columns:
        ``name``, ``n_nodes``, ``n_elems``.
        """
        import pandas as pd

        rows: list[dict] = []
        for (dim, pg_tag), info in sorted(self._groups.items()):
            elem_ids = info.get('element_ids')
            rows.append({
                'dim':     dim,
                'pg_tag':  pg_tag,
                'name':    info.get('name', ''),
                'n_nodes': len(info['node_ids']),
                'n_elems': len(elem_ids) if elem_ids is not None else 0,
            })

        if not rows:
            return pd.DataFrame(
                columns=['dim', 'pg_tag', 'name', 'n_nodes', 'n_elems']
            )

        return (
            pd.DataFrame(rows)
            .set_index(['dim', 'pg_tag'])
            .sort_index()
        )

    # ── Dunder ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._groups)

    def __bool__(self) -> bool:
        return bool(self._groups)

    def __iter__(self):
        return iter(sorted(self._groups.keys()))


class PhysicalGroupSet(_GroupSetBase):
    """Snapshot of physical groups captured at ``get_fem_data()`` time.

    Accessed via ``fem.physical``.  Fully self-contained — no live
    Gmsh session required after construction.

    Name-first access
    -----------------
    ::

        fem.physical.node_ids("base")       # ndarray(N,)
        fem.physical.node_coords("base")    # ndarray(N, 3)
        fem.physical.element_ids("slab")    # ndarray(E,)
        fem.physical.connectivity("slab")   # ndarray(E, npe)
        fem.physical.names()                # ['base', 'slab', ...]

    Accepts str name, int tag, or ``(dim, tag)`` tuple — consistent
    with the rest of the library.
    """

    def __repr__(self) -> str:
        return f"PhysicalGroupSet({len(self._groups)} groups)"


class LabelSet(_GroupSetBase):
    """Snapshot of labels (Tier 1) captured at ``get_fem_data()`` time.

    Accessed via ``fem.labels``.  Contains the ``_label:``-prefixed
    physical groups with the prefix stripped.  Same API as
    :class:`PhysicalGroupSet`.

    Example
    -------
    ::

        fem.labels.names()                   # ['col.web', 'col.top_flange', ...]
        web_ids = fem.labels.node_ids("col.web")
        web_xyz = fem.labels.node_coords("col.web")
    """

    def __repr__(self) -> str:
        if not self._groups:
            return "LabelSet(empty)"
        return f"LabelSet({len(self._groups)} labels)"


# =====================================================================
# Constraint snapshot
# =====================================================================

class ConstraintSet:
    """
    Solver-ready snapshot of resolved multi-point constraints.

    Accessed via ``fem.constraints``.  Holds the full
    :class:`ConstraintRecord` list **and** any extra nodes
    (e.g. phantom nodes from :class:`NodeToSurfaceRecord`) that
    solvers must create before emitting constraint commands.

    The class is **solver-agnostic** — it exposes flat iterators
    that any exporter (OpenSees, Abaqus, Code_Aster, …) can consume.

    Construction
    ------------
    ::

        records = g.constraints.resolve(...)
        cs = ConstraintSet(records)
        fem = g.mesh.queries.get_fem_data(dim=3)
        fem.constraints = cs           # or pass at construction

    Iteration
    ---------
    ::

        # Extra nodes that solvers must create first
        for nid, xyz in fem.constraints.extra_nodes():
            ops.node(nid, *xyz)

        # Flat constraint pairs (covers ALL constraint types)
        for c in fem.constraints.node_pairs():
            # c is a NodePairRecord: master_node, slave_node, dofs,
            # offset, penalty_stiffness, kind
            ...

        # All records, ungrouped
        for rec in fem.constraints:
            ...
    """

    def __init__(self, records: list | None = None) -> None:
        self._records: list = list(records) if records else []

    # ── Extra nodes (phantom nodes from compound constraints) ──

    def extra_nodes(self):
        """Iterate ``(node_id, coords)`` for nodes that solvers must
        create **before** emitting constraint commands.

        Currently produced by :class:`NodeToSurfaceRecord` (phantom nodes).
        Returns Python-native types safe for OpenSees.

        Yields
        ------
        (int, list[float])
            Node ID and ``[x, y, z]`` coordinates.

        Example
        -------
        ::

            for nid, xyz in fem.constraints.extra_nodes():
                ops.node(nid, *xyz)
        """
        from apeGmsh.solvers.Constraints import NodeToSurfaceRecord

        for rec in self._records:
            if isinstance(rec, NodeToSurfaceRecord):
                coords = rec.phantom_coords
                if coords is None:
                    continue
                for tag, xyz in zip(rec.phantom_nodes, coords):
                    yield int(tag), xyz.tolist()

    # ── Flat node-pair iteration ─────────────────────────────────

    def node_pairs(self):
        """Iterate over every constraint as individual
        :class:`NodePairRecord` objects.

        Compound records (:class:`NodeGroupRecord`,
        :class:`NodeToSurfaceRecord`) are expanded automatically.
        :class:`InterpolationRecord` and :class:`SurfaceCouplingRecord`
        are skipped here — use :meth:`interpolations` for those.

        Yields
        ------
        NodePairRecord

        Example
        -------
        ::

            for c in fem.constraints.node_pairs():
                if c.kind == "equal_dof":
                    ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
                elif c.kind == "rigid_beam":
                    ops.rigidLink("beam", c.master_node, c.slave_node)
        """
        from apeGmsh.solvers.Constraints import (
            NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        )

        for rec in self._records:
            if isinstance(rec, NodePairRecord):
                yield rec
            elif isinstance(rec, NodeGroupRecord):
                yield from rec.expand_to_pairs()
            elif isinstance(rec, NodeToSurfaceRecord):
                yield from rec.expand()

    # ── Interpolation iteration ──────────────────────────────────

    def interpolations(self):
        """Iterate over :class:`InterpolationRecord` objects.

        Covers ``tie``, ``distributing``, and the slave records
        inside ``tied_contact`` / ``mortar``.

        Yields
        ------
        InterpolationRecord
        """
        from apeGmsh.solvers.Constraints import (
            InterpolationRecord, SurfaceCouplingRecord,
        )

        for rec in self._records:
            if isinstance(rec, InterpolationRecord):
                yield rec
            elif isinstance(rec, SurfaceCouplingRecord):
                yield from rec.slave_records

    # ── Filter by kind ───────────────────────────────────────────

    def by_kind(self, kind: str) -> list:
        """Return all records matching a constraint kind.

        Parameters
        ----------
        kind : str
            e.g. ``"equal_dof"``, ``"rigid_beam"``, ``"node_to_surface"``,
            ``"tie"``, ``"mortar"``.
        """
        return [r for r in self._records if r.kind == kind]

    # ── Summary ──────────────────────────────────────────────────

    def summary(self):
        """Build a DataFrame summarising the constraint set.

        Returns
        -------
        pd.DataFrame
            Columns: ``kind``, ``count``, ``n_node_pairs``.
        """
        import pandas as pd
        from apeGmsh.solvers.Constraints import (
            NodePairRecord, NodeGroupRecord,
            InterpolationRecord, SurfaceCouplingRecord,
            NodeToSurfaceRecord,
        )

        counts: dict[str, dict] = {}
        for rec in self._records:
            kind = rec.kind
            if kind not in counts:
                counts[kind] = {"count": 0, "n_node_pairs": 0}
            counts[kind]["count"] += 1

            if isinstance(rec, NodePairRecord):
                counts[kind]["n_node_pairs"] += 1
            elif isinstance(rec, NodeGroupRecord):
                counts[kind]["n_node_pairs"] += len(rec.slave_nodes)
            elif isinstance(rec, NodeToSurfaceRecord):
                counts[kind]["n_node_pairs"] += len(rec.rigid_link_records) + len(rec.equal_dof_records)
            elif isinstance(rec, InterpolationRecord):
                counts[kind]["n_node_pairs"] += 1
            elif isinstance(rec, SurfaceCouplingRecord):
                counts[kind]["n_node_pairs"] += len(rec.slave_records)

        if not counts:
            return pd.DataFrame(columns=["kind", "count", "n_node_pairs"])

        rows = [
            {"kind": k, "count": v["count"], "n_node_pairs": v["n_node_pairs"]}
            for k, v in counts.items()
        ]
        return pd.DataFrame(rows).set_index("kind").sort_index()

    # ── Dunder ───────────────────────────────────────────────────

    def __iter__(self):
        """Iterate over the raw record list."""
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0

    def __repr__(self) -> str:
        if not self._records:
            return "ConstraintSet(empty)"
        kinds: dict[str, int] = {}
        for r in self._records:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        return f"ConstraintSet({len(self._records)} records: {parts})"


# =====================================================================
# Load snapshot
# =====================================================================

class LoadSet:
    """
    Solver-ready snapshot of resolved loads.

    Accessed via ``fem.loads``.  Holds the full :class:`LoadRecord`
    list (nodal forces and/or element load commands) grouped by
    pattern name.

    Iteration helpers:

    * :meth:`patterns`     — list of unique pattern names
    * :meth:`by_pattern`   — records belonging to a named pattern
    * :meth:`nodal`        — yield NodalLoadRecord objects
    * :meth:`element`      — yield ElementLoadRecord objects
    * :meth:`by_kind`      — filter by record ``kind`` field
    * :meth:`summary`      — DataFrame of pattern × kind counts

    Construction
    ------------
    ::

        records = g.loads.resolve(...)
        ls = LoadSet(records)
        fem = g.mesh.queries.get_fem_data(dim=3)
        fem.loads = ls            # or pass at construction
    """

    def __init__(self, records: list | None = None) -> None:
        self._records: list = list(records) if records else []

    # ── Pattern grouping ─────────────────────────────────────

    def patterns(self) -> list[str]:
        """Return all unique pattern names in insertion order."""
        seen: list[str] = []
        for r in self._records:
            if r.pattern not in seen:
                seen.append(r.pattern)
        return seen

    def by_pattern(self, name: str) -> list:
        """Return all records belonging to the named pattern."""
        return [r for r in self._records if r.pattern == name]

    # ── Type iterators ───────────────────────────────────────

    def nodal(self):
        """Yield :class:`NodalLoadRecord` objects."""
        from apeGmsh.solvers.Loads import NodalLoadRecord
        for r in self._records:
            if isinstance(r, NodalLoadRecord):
                yield r

    def element(self):
        """Yield :class:`ElementLoadRecord` objects."""
        from apeGmsh.solvers.Loads import ElementLoadRecord
        for r in self._records:
            if isinstance(r, ElementLoadRecord):
                yield r

    def by_kind(self, kind: str) -> list:
        """Return records matching a load kind."""
        return [r for r in self._records if r.kind == kind]

    # ── Summary ──────────────────────────────────────────────

    def summary(self):
        """DataFrame of (pattern, kind) -> count."""
        import pandas as pd
        if not self._records:
            return pd.DataFrame(columns=["pattern", "kind", "count"])
        rows: dict[tuple, int] = {}
        for r in self._records:
            key = (r.pattern, r.kind)
            rows[key] = rows.get(key, 0) + 1
        data = [
            {"pattern": p, "kind": k, "count": c}
            for (p, k), c in rows.items()
        ]
        return pd.DataFrame(data).sort_values(["pattern", "kind"]).reset_index(drop=True)

    # ── Dunder ───────────────────────────────────────────────

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)

    def __repr__(self) -> str:
        if not self._records:
            return "LoadSet(empty)"
        n_pats = len(self.patterns())
        n_nodal = sum(1 for _ in self.nodal())
        n_elem = sum(1 for _ in self.element())
        return (
            f"LoadSet({len(self._records)} records, "
            f"{n_pats} pattern(s), {n_nodal} nodal, {n_elem} element)"
        )


# =====================================================================
# Mass snapshot
# =====================================================================

class MassSet:
    """
    Solver-ready snapshot of resolved nodal masses.

    Accessed via ``fem.masses``.  One :class:`MassRecord` per node
    (the composite already accumulates contributions from multiple
    mass definitions).

    There is no pattern grouping — mass is intrinsic to the model.

    Iteration helpers:

    * :meth:`records`     — yield raw MassRecord objects
    * :meth:`total_mass`  — scalar sum of all translational mass
    * :meth:`summary`     — DataFrame of (node_id, mx, my, mz, ...)
    """

    def __init__(self, records: list | None = None) -> None:
        self._records: list = list(records) if records else []

    def records(self):
        """Yield :class:`MassRecord` objects."""
        return iter(self._records)

    def by_node(self, node_id: int):
        """Return the MassRecord for a node, or None if not present."""
        for r in self._records:
            if r.node_id == int(node_id):
                return r
        return None

    def total_mass(self) -> float:
        """Sum of translational mass (mx) over all records.

        Assumes isotropic mass (mx == my == mz).  Useful as a sanity
        check that the resolved total matches the expected
        ``Σ density × volume``.
        """
        return sum(float(r.mass[0]) for r in self._records)

    def summary(self):
        """DataFrame with one row per node: ``node_id, mx, my, mz, Ixx, Iyy, Izz``."""
        import pandas as pd
        if not self._records:
            return pd.DataFrame(
                columns=["node_id", "mx", "my", "mz", "Ixx", "Iyy", "Izz"]
            )
        rows = []
        for r in self._records:
            m = r.mass
            rows.append({
                "node_id": int(r.node_id),
                "mx": float(m[0]), "my": float(m[1]), "mz": float(m[2]),
                "Ixx": float(m[3]), "Iyy": float(m[4]), "Izz": float(m[5]),
            })
        return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)

    def __repr__(self) -> str:
        if not self._records:
            return "MassSet(empty)"
        return (
            f"MassSet({len(self._records)} nodes, "
            f"total={self.total_mass():.6g})"
        )


# =====================================================================
# FEM data container
# =====================================================================

@dataclass
class FEMData:
    """
    Solver-ready FEM mesh data.

    Returned by ``Mesh.get_fem_data()``.  Contains everything needed
    to build a solver model: node IDs, coordinates, element IDs,
    connectivity, mesh statistics, and physical group data.

    Attributes
    ----------
    node_ids : ndarray(N,)
        Node IDs (contiguous if ``renumber_mesh()`` was called first).
    node_coords : ndarray(N, 3)
        Nodal coordinates, same order as ``node_ids``.
    element_ids : ndarray(E,)
        Element IDs (contiguous if ``renumber_mesh()`` was called first).
    connectivity : ndarray(E, npe)
        Element-to-node connectivity in terms of ``node_ids``.
    info : MeshInfo
        Mesh statistics: ``n_nodes``, ``n_elems``, ``bandwidth``.
    physical : PhysicalGroupSet
        Physical group introspection and node retrieval.

    Example
    -------
    ::

        g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
        fem = g.mesh.queries.get_fem_data(dim=2)

        # Mesh stats
        print(fem.info)

        # Physical groups (mirrors g.physical API)
        fem.physical.get_all()              # [(0, 1), (1, 2), (2, 3)]
        fem.physical.get_name(0, 1)         # "base_supports"
        base = fem.physical.get_nodes(0, 1) # {'tags': ..., 'coords': ...}
        fem.physical.summary()              # DataFrame

        # Build solver model
        for i in range(fem.info.n_nodes):
            ops.node(int(fem.node_ids[i]), *fem.node_coords[i])

        # Quick coordinate lookup by node ID
        x, y, z = fem.get_node_coords(42)

        # Array index lookup (node or element)
        idx = fem.node_index(42)
        fem.node_coords[idx]       # same as above

        # Elements by physical group
        cols = fem.physical.get_elements(1, 2)   # columns (dim=1)
        slab = fem.physical.get_elements(2, 3)   # slab (dim=2)
        cols['element_ids'], cols['connectivity']
        slab['element_ids'], slab['connectivity']
    """
    node_ids:      ndarray
    node_coords:   ndarray
    element_ids:   ndarray
    connectivity:  ndarray
    info:          MeshInfo          = field(repr=False)
    physical:      PhysicalGroupSet  = field(repr=False)
    labels:        LabelSet          = field(repr=False, default=None)
    mesh_selection: "MeshSelectionStore" = field(repr=False, default=None)
    constraints: ConstraintSet = field(repr=False, default=None)
    loads: LoadSet = field(repr=False, default=None)
    masses: MassSet = field(repr=False, default=None)

    # -- Lazy lookup caches (not part of __init__) --

    _node_id_to_idx: dict = field(default=None, init=False, repr=False)
    _elem_id_to_idx: dict = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Cast ID/connectivity arrays to ``object`` dtype so iteration
        yields Python ``int``, not ``numpy.int64``.

        OpenSees and other C-extension solvers reject numpy integer
        types.  ``object`` dtype keeps full numpy functionality
        (``np.isin``, ``np.unique``, arithmetic) while making
        ``for x in arr`` yield plain ``int``.
        """
        object.__setattr__(
            self, 'node_ids', np.asarray(self.node_ids).astype(object))
        object.__setattr__(
            self, 'element_ids', np.asarray(self.element_ids).astype(object))
        object.__setattr__(
            self, 'connectivity', np.asarray(self.connectivity).astype(object))
        object.__setattr__(
            self, 'node_coords',
            np.asarray(self.node_coords, dtype=np.float64))
        # Default labels to empty LabelSet if not provided
        if self.labels is None:
            object.__setattr__(self, 'labels', LabelSet({}))
        # Default mesh_selection to empty store if not provided
        if self.mesh_selection is None:
            from .MeshSelectionSet import MeshSelectionStore
            object.__setattr__(self, 'mesh_selection', MeshSelectionStore({}))
        # Default constraints to empty ConstraintSet if not provided
        if self.constraints is None:
            object.__setattr__(self, 'constraints', ConstraintSet())
        # Default loads to empty LoadSet if not provided
        if self.loads is None:
            object.__setattr__(self, 'loads', LoadSet())
        # Default masses to empty MassSet if not provided
        if self.masses is None:
            object.__setattr__(self, 'masses', MassSet())

    # -- Solver-friendly iterators --

    def nodes(self):
        """Iterate ``(node_id, coords)`` as Python-native types.

        Yields ``(int, list[float])`` -- safe for OpenSees.

        Example
        -------
        ::

            for nid, xyz in fem.nodes():
                ops.node(nid, *xyz)
        """
        for i in range(len(self.node_ids)):
            yield int(self.node_ids[i]), self.node_coords[i].tolist()

    def elements(self):
        """Iterate ``(element_id, connectivity)`` as Python-native types.

        Yields ``(int, list[int])`` -- safe for OpenSees.

        Example
        -------
        ::

            for eid, conn in fem.elements():
                ops.element("tri31", eid, *conn, thk, "PlaneStrain", 1)
        """
        for i in range(len(self.element_ids)):
            yield int(self.element_ids[i]), [int(n) for n in self.connectivity[i]]

    # -- Lookup maps --

    def _build_node_map(self) -> dict[int, int]:
        """Build and cache the node-ID -> array-index map."""
        if self._node_id_to_idx is None:
            self._node_id_to_idx = {
                int(nid): i for i, nid in enumerate(self.node_ids)
            }
        return self._node_id_to_idx

    def _build_elem_map(self) -> dict[int, int]:
        """Build and cache the element-ID -> array-index map."""
        if self._elem_id_to_idx is None:
            self._elem_id_to_idx = {
                int(eid): i for i, eid in enumerate(self.element_ids)
            }
        return self._elem_id_to_idx

    # -- Node lookups --

    def node_index(self, nid: int) -> int:
        """
        Return the array index for a node ID.

        Parameters
        ----------
        nid : int
            Node ID (as stored in ``node_ids``).

        Returns
        -------
        int
            Index into ``node_ids``, ``node_coords``.

        Raises
        ------
        KeyError
            If ``nid`` is not in ``node_ids``.

        Example
        -------
        ::

            idx = fem.node_index(42)
            fem.node_coords[idx]   # -> array([x, y, z])
        """
        m = self._build_node_map()
        try:
            return m[int(nid)]
        except KeyError:
            raise KeyError(
                f"Node ID {nid} not found. "
                f"Valid range: {int(self.node_ids.min())}-"
                f"{int(self.node_ids.max())} "
                f"({len(self.node_ids)} nodes)"
            ) from None

    def get_node_coords(self, nid: int) -> ndarray:
        """
        Return the coordinates of a node by its ID.

        Parameters
        ----------
        nid : int
            Node ID (as stored in ``node_ids``).

        Returns
        -------
        ndarray(3,)
            ``[x, y, z]`` coordinates.

        Example
        -------
        ::

            x, y, z = fem.get_node_coords(42)
        """
        return self.node_coords[self.node_index(nid)]

    # -- Element lookups --

    def elem_index(self, eid: int) -> int:
        """
        Return the array index for an element ID.

        Parameters
        ----------
        eid : int
            Element ID (as stored in ``element_ids``).

        Returns
        -------
        int
            Index into ``element_ids``, ``connectivity``.

        Raises
        ------
        KeyError
            If ``eid`` is not in ``element_ids``.

        Example
        -------
        ::

            idx = fem.elem_index(10)
            fem.connectivity[idx]   # -> array([n1, n2, n3])
        """
        m = self._build_elem_map()
        try:
            return m[int(eid)]
        except KeyError:
            raise KeyError(
                f"Element ID {eid} not found. "
                f"Valid range: {int(self.element_ids.min())}-"
                f"{int(self.element_ids.max())} "
                f"({len(self.element_ids)} elements)"
            ) from None

    def get_elem_connectivity(self, eid: int) -> ndarray:
        """
        Return the connectivity (node IDs) of an element by its ID.

        Parameters
        ----------
        eid : int
            Element ID (as stored in ``element_ids``).

        Returns
        -------
        ndarray(npe,)
            Node IDs forming the element.

        Example
        -------
        ::

            n1, n2, n3 = fem.get_elem_connectivity(10)
        """
        return self.connectivity[self.elem_index(eid)]

    # -- Display --

    def __repr__(self) -> str:
        lines = [self.info.summary()]

        # Physical groups
        if self.physical:
            pg_names = self.physical.names()
            lines.append(f"  Physical groups ({len(self.physical)}):")
            for (d, t), info in sorted(self.physical._groups.items()):
                name = info.get('name', '')
                n_n = len(info['node_ids'])
                eids = info.get('element_ids')
                n_e = len(eids) if eids is not None else 0
                label = f'"{name}"' if name else f"tag={t}"
                parts = f"{n_n} nodes"
                if n_e:
                    parts += f", {n_e} elems"
                lines.append(f"    ({d}) {label:24s} {parts}")

        # Labels
        if self.labels:
            lines.append(f"  Labels ({len(self.labels)}):")
            for (d, t), info in sorted(self.labels._groups.items()):
                name = info.get('name', '')
                n_n = len(info['node_ids'])
                eids = info.get('element_ids')
                n_e = len(eids) if eids is not None else 0
                parts = f"{n_n} nodes"
                if n_e:
                    parts += f", {n_e} elems"
                lines.append(f"    ({d}) {name!r:24s} {parts}")

        # Constraints
        if self.constraints:
            lines.append(f"  {self.constraints!r}")
        # Loads
        if self.loads:
            lines.append(f"  {self.loads!r}")
        # Masses
        if self.masses:
            lines.append(f"  {self.masses!r}")

        return "\n".join(lines)
