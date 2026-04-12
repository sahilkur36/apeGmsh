"""
FEMData — Solver-ready FEM mesh broker.
========================================

The main output of apeGmsh's meshing pipeline.  Organized by what the
engineer needs: **Nodes** and **Elements** — with selections, BCs,
loads, and masses as sub-composites.

Top-level composites::

    fem.nodes       → NodeComposite   (IDs, coords, nodal loads, masses, node constraints)
    fem.elements    → ElementComposite (IDs, connectivity, element loads, surface constraints)
    fem.info        → MeshInfo        (mesh statistics)
    fem.inspect     → InspectComposite (introspection and summaries)

Construction::

    fem = FEMData.from_gmsh(dim=3, session=g, ndf=3)
    fem = FEMData.from_msh("bridge.msh", dim=2)
    fem = FEMData(nodes=..., elements=..., info=...)   # direct

Usage::

    # Domain nodes
    for nid, xyz in zip(*fem.nodes.get()):
        ops.node(nid, *xyz)

    # Supports
    for nid in fem.nodes.get_ids(pg="Base"):
        ops.fix(nid, 1, 1, 1)

    # Elements
    for eid, conn in zip(*fem.elements.get()):
        ops.element("tet4", eid, *conn, mat_tag)

    # Constraints
    K = fem.nodes.constraints.Kind
    for c in fem.nodes.constraints.node_pairs():
        if c.kind == K.RIGID_BEAM:
            ops.rigidLink("beam", c.master_node, c.slave_node)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy import ndarray

from ._group_set import (
    NamedGroupSet, PhysicalGroupSet, LabelSet, _to_object,
)
from ._record_set import (
    ConstraintKind, LoadKind,
    NodeConstraintSet, SurfaceConstraintSet,
    NodalLoadSet, ElementLoadSet, MassSet,
)

if TYPE_CHECKING:
    import pandas as pd
    from .MeshSelectionSet import MeshSelectionStore


# =====================================================================
# Result NamedTuples
# =====================================================================

class NodeResult(NamedTuple):
    """Bundled node IDs and coordinates.

    Destructurable::

        ids, coords = fem.nodes.get(pg="Base")

    Or use as an object::

        result = fem.nodes.get(pg="Base")
        result.ids       # ndarray(N,) object dtype
        result.coords    # ndarray(N, 3) float64
        result.to_dataframe()
    """
    ids:    ndarray
    coords: ndarray

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(
            self.coords,
            index=pd.Index(
                [int(x) for x in self.ids], name='node_id'),
            columns=['x', 'y', 'z'],
        )


class ElementResult(NamedTuple):
    """Bundled element IDs and connectivity.

    Destructurable::

        ids, conn = fem.elements.get(pg="Body")
    """
    ids:          ndarray
    connectivity: ndarray

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        npe = self.connectivity.shape[1] if self.connectivity.ndim == 2 else 0
        cols = [f"n{i}" for i in range(npe)]
        return pd.DataFrame(
            [[int(x) for x in row] for row in self.connectivity],
            index=pd.Index(
                [int(x) for x in self.ids], name='elem_id'),
            columns=cols,
        )


# =====================================================================
# Bandwidth computation
# =====================================================================

def _compute_bandwidth(connectivity: ndarray) -> int:
    """Semi-bandwidth = max over all elements of (max_node - min_node)."""
    if connectivity.size == 0:
        return 0
    # Work with numeric dtype for min/max
    c = np.asarray(connectivity, dtype=np.int64)
    row_max = c.max(axis=1)
    row_min = c.min(axis=1)
    return int((row_max - row_min).max())


# =====================================================================
# MeshInfo (unchanged)
# =====================================================================

class MeshInfo:
    """Read-only summary of mesh statistics.

    Accessed via ``fem.info``.

    Attributes
    ----------
    n_nodes : int
    n_elems : int
    bandwidth : int
    nodes_per_elem : int
    elem_type_name : str
    """

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
# NodeComposite
# =====================================================================

class NodeComposite:
    """Access and query nodes from the FEM mesh.

    Primary interface::

        fem.nodes.get(pg="Base")    → NodeResult(ids, coords)
        fem.nodes.get()             → all domain nodes

    Sub-composites::

        fem.nodes.constraints       → NodeConstraintSet
        fem.nodes.loads             → NodalLoadSet
        fem.nodes.masses            → MassSet

    Public properties for raw array access::

        fem.nodes.ids               → ndarray(N,) object dtype
        fem.nodes.coords            → ndarray(N, 3) float64
    """

    def __init__(
        self,
        node_ids: ndarray,
        node_coords: ndarray,
        physical: PhysicalGroupSet,
        labels: LabelSet,
        constraints=None,
        loads=None,
        masses=None,
    ) -> None:
        self._ids    = _to_object(node_ids)
        self._coords = np.asarray(node_coords, dtype=np.float64)
        self.physical = physical
        self.labels   = labels

        # Sub-composites (always present, empty by default)
        self.constraints = NodeConstraintSet(constraints)
        self.loads       = NodalLoadSet(loads)
        self.masses      = MassSet(masses)

        self._id_to_idx: dict[int, int] | None = None

    # ── Public properties (raw arrays) ───────────────────────

    @property
    def ids(self) -> ndarray:
        """All domain node IDs.  ``ndarray(N,)`` object dtype."""
        return self._ids

    @property
    def coords(self) -> ndarray:
        """All domain node coordinates.  ``ndarray(N, 3)`` float64."""
        return self._coords

    # ── Selection API ────────────────────────────────────────

    def get(
        self,
        target: str | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
    ) -> NodeResult:
        """Bundled ``(ids, coords)`` for a selection.

        Parameters
        ----------
        target : str, optional
            Shorthand — searches PGs first, then labels.
        pg : str, optional
            Physical group name (explicit).
        label : str, optional
            Label name (explicit).

        Returns
        -------
        NodeResult
            NamedTuple — destructure as ``ids, coords = fem.nodes.get(...)``

        Examples
        --------
        ::

            fem.nodes.get()                  # all domain nodes
            fem.nodes.get("Base")            # PG-first fallback
            fem.nodes.get(pg="Base")         # explicit PG
            fem.nodes.get(label="col.web")   # explicit label
        """
        if pg is not None:
            return NodeResult(
                self.physical.node_ids(pg),
                self.physical.node_coords(pg))
        if label is not None:
            return NodeResult(
                self.labels.node_ids(label),
                self.labels.node_coords(label))
        if target is not None:
            # PG first, then label
            try:
                return NodeResult(
                    self.physical.node_ids(target),
                    self.physical.node_coords(target))
            except KeyError:
                return NodeResult(
                    self.labels.node_ids(target),
                    self.labels.node_coords(target))
        # No target → all domain nodes
        return NodeResult(self._ids, self._coords)

    def get_ids(
        self,
        target: str | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
    ) -> ndarray:
        """Node IDs only for a selection."""
        return self.get(target, pg=pg, label=label).ids

    def get_coords(
        self,
        target: str | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
    ) -> ndarray:
        """Coordinates only for a selection."""
        return self.get(target, pg=pg, label=label).coords

    # ── Lookups ──────────────────────────────────────────────

    def index(self, nid: int) -> int:
        """Array index for a node ID.  O(1) after first call.

        Raises
        ------
        KeyError
            If ``nid`` is not in the domain nodes.
        """
        if self._id_to_idx is None:
            self._id_to_idx = {
                int(n): i for i, n in enumerate(self._ids)}
        try:
            return self._id_to_idx[int(nid)]
        except KeyError:
            raise KeyError(
                f"Node ID {nid} not found. "
                f"Valid range: {int(self._ids.min())}-"
                f"{int(self._ids.max())} "
                f"({len(self._ids)} nodes)"
            ) from None

    # ── Dunder ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        parts = [f"NodeComposite({len(self._ids)} nodes)"]
        if self.constraints:
            parts.append(f"  constraints: {self.constraints!r}")
        if self.loads:
            parts.append(f"  loads: {self.loads!r}")
        if self.masses:
            parts.append(f"  masses: {self.masses!r}")
        return "\n".join(parts)


# =====================================================================
# ElementComposite
# =====================================================================

class ElementComposite:
    """Access and query elements from the FEM mesh.

    Primary interface::

        fem.elements.get(pg="Body")  → ElementResult(ids, conn)
        fem.elements.get()           → all elements

    Sub-composites::

        fem.elements.constraints     → SurfaceConstraintSet
        fem.elements.loads           → ElementLoadSet

    Public properties for raw array access::

        fem.elements.ids             → ndarray(E,) object dtype
        fem.elements.connectivity    → ndarray(E, npe) object dtype
    """

    def __init__(
        self,
        element_ids: ndarray,
        connectivity: ndarray,
        physical: PhysicalGroupSet,
        labels: LabelSet,
        constraints=None,
        loads=None,
    ) -> None:
        self._ids  = _to_object(element_ids)
        self._conn = _to_object(connectivity)
        self.physical = physical
        self.labels   = labels

        self.constraints = SurfaceConstraintSet(constraints)
        self.loads       = ElementLoadSet(loads)

        self._id_to_idx: dict[int, int] | None = None

    # ── Public properties (raw arrays) ───────────────────────

    @property
    def ids(self) -> ndarray:
        """All element IDs.  ``ndarray(E,)`` object dtype."""
        return self._ids

    @property
    def connectivity(self) -> ndarray:
        """Full connectivity array.  ``ndarray(E, npe)`` object dtype."""
        return self._conn

    # ── Selection API ────────────────────────────────────────

    def get(
        self,
        target: str | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
    ) -> ElementResult:
        """Bundled ``(ids, connectivity)`` for a selection.

        Parameters
        ----------
        target : str, optional
            Shorthand — searches PGs first, then labels.
        pg : str, optional
            Physical group name (explicit).
        label : str, optional
            Label name (explicit).

        Returns
        -------
        ElementResult
            NamedTuple — destructure as
            ``ids, conn = fem.elements.get(...)``
        """
        if pg is not None:
            return ElementResult(
                self.physical.element_ids(pg),
                self.physical.connectivity(pg))
        if label is not None:
            return ElementResult(
                self.labels.element_ids(label),
                self.labels.connectivity(label))
        if target is not None:
            try:
                return ElementResult(
                    self.physical.element_ids(target),
                    self.physical.connectivity(target))
            except (KeyError, ValueError):
                return ElementResult(
                    self.labels.element_ids(target),
                    self.labels.connectivity(target))
        return ElementResult(self._ids, self._conn)

    def get_ids(
        self,
        target: str | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
    ) -> ndarray:
        """Element IDs only for a selection."""
        return self.get(target, pg=pg, label=label).ids

    def get_connectivity(
        self,
        target: str | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
    ) -> ndarray:
        """Connectivity only for a selection."""
        return self.get(target, pg=pg, label=label).connectivity

    # ── Lookups ──────────────────────────────────────────────

    def index(self, eid: int) -> int:
        """Array index for an element ID.  O(1) after first call."""
        if self._id_to_idx is None:
            self._id_to_idx = {
                int(e): i for i, e in enumerate(self._ids)}
        try:
            return self._id_to_idx[int(eid)]
        except KeyError:
            raise KeyError(
                f"Element ID {eid} not found. "
                f"Valid range: {int(self._ids.min())}-"
                f"{int(self._ids.max())} "
                f"({len(self._ids)} elements)"
            ) from None

    # ── Dunder ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        parts = [f"ElementComposite({len(self._ids)} elements)"]
        if self.constraints:
            parts.append(f"  constraints: {self.constraints!r}")
        if self.loads:
            parts.append(f"  loads: {self.loads!r}")
        return "\n".join(parts)


# =====================================================================
# InspectComposite
# =====================================================================

class InspectComposite:
    """Introspection and summary methods.

    Accessed via ``fem.inspect``.

    Example
    -------
    ::

        print(fem.inspect.summary())
        fem.inspect.node_table()
        fem.inspect.constraint_summary()
    """

    def __init__(self, fem: "FEMData") -> None:
        self._fem = fem

    # ── Tables ───────────────────────────────────────────────

    def summary(self) -> str:
        """One-line mesh summary plus sub-composite counts."""
        f = self._fem
        lines = [f.info.summary()]

        # Physical groups
        pg = f.nodes.physical
        if pg:
            lines.append(f"  Physical groups ({len(pg)}):")
            for (d, t), info in sorted(pg._groups.items()):
                name = info.get('name', '')
                n_n = len(info['node_ids'])
                eids = info.get('element_ids')
                n_e = len(eids) if eids is not None else 0
                lbl = f'"{name}"' if name else f"tag={t}"
                parts = f"{n_n} nodes"
                if n_e:
                    parts += f", {n_e} elems"
                lines.append(f"    ({d}) {lbl:24s} {parts}")

        # Labels
        lb = f.nodes.labels
        if lb:
            lines.append(f"  Labels ({len(lb)}):")
            for (d, t), info in sorted(lb._groups.items()):
                name = info.get('name', '')
                n_n = len(info['node_ids'])
                eids = info.get('element_ids')
                n_e = len(eids) if eids is not None else 0
                parts = f"{n_n} nodes"
                if n_e:
                    parts += f", {n_e} elems"
                lines.append(f"    ({d}) {name!r:24s} {parts}")

        # Constraints
        nc = f.nodes.constraints
        sc = f.elements.constraints
        if nc:
            lines.append(f"  Node constraints: {nc!r}")
        if sc:
            lines.append(f"  Surface constraints: {sc!r}")
        # Loads
        if f.nodes.loads:
            lines.append(f"  Nodal loads: {f.nodes.loads!r}")
        if f.elements.loads:
            lines.append(f"  Element loads: {f.elements.loads!r}")
        # Masses
        if f.nodes.masses:
            lines.append(f"  {f.nodes.masses!r}")

        return "\n".join(lines)

    def node_table(self) -> "pd.DataFrame":
        """DataFrame of all nodes: ``node_id, x, y, z``."""
        import pandas as pd
        f = self._fem
        return pd.DataFrame(
            f.nodes.coords,
            index=pd.Index(
                [int(x) for x in f.nodes.ids], name='node_id'),
            columns=['x', 'y', 'z'],
        )

    def element_table(self) -> "pd.DataFrame":
        """DataFrame of all elements: ``elem_id, n0, n1, …``."""
        import pandas as pd
        f = self._fem
        npe = (f.elements.connectivity.shape[1]
               if f.elements.connectivity.ndim == 2 else 0)
        cols = [f"n{i}" for i in range(npe)]
        return pd.DataFrame(
            [[int(x) for x in row]
             for row in f.elements.connectivity],
            index=pd.Index(
                [int(x) for x in f.elements.ids], name='elem_id'),
            columns=cols,
        )

    def physical_table(self) -> "pd.DataFrame":
        """DataFrame of all physical groups."""
        return self._fem.nodes.physical.summary()

    def label_table(self) -> "pd.DataFrame":
        """DataFrame of all labels."""
        return self._fem.nodes.labels.summary()

    # ── Constraint/Load/Mass introspection ───────────────────

    def constraint_summary(self) -> str:
        """Human-readable breakdown of all constraints with sources."""
        f = self._fem
        lines = []

        nc = f.nodes.constraints
        if nc:
            lines.append(f"Node constraints ({len(nc)} records):")
            kinds: dict[str, int] = {}
            for r in nc:
                kinds[r.kind] = kinds.get(r.kind, 0) + 1
            for k, count in sorted(kinds.items()):
                name_hint = ""
                # Try to get source name from first record of this kind
                for r in nc:
                    if r.kind == k and getattr(r, 'name', None):
                        name_hint = f"  (source: {r.name!r})"
                        break
                lines.append(f"  {k:24s} {count:>4d}{name_hint}")
            # Phantom nodes
            n_phantom = sum(1 for _ in nc.extra_nodes())
            if n_phantom:
                lines.append(
                    f"  {'phantom nodes':24s} {n_phantom:>4d}"
                    f"  (created by node_to_surface)")

        sc = f.elements.constraints
        if sc:
            lines.append(f"Surface constraints ({len(sc)} records):")
            kinds = {}
            for r in sc:
                kinds[r.kind] = kinds.get(r.kind, 0) + 1
            for k, count in sorted(kinds.items()):
                name_hint = ""
                for r in sc:
                    if r.kind == k and getattr(r, 'name', None):
                        name_hint = f"  (source: {r.name!r})"
                        break
                lines.append(f"  {k:24s} {count:>4d}{name_hint}")

        if not lines:
            return "No constraints."
        return "\n".join(lines)

    def load_summary(self) -> str:
        """Human-readable breakdown of all loads with sources."""
        f = self._fem
        lines = []

        nl = f.nodes.loads
        if nl:
            lines.append(f"Nodal loads ({len(nl)} records):")
            for pat in nl.patterns():
                recs = nl.by_pattern(pat)
                name_hint = ""
                for r in recs:
                    if getattr(r, 'name', None):
                        name_hint = f"  (source: {r.name!r})"
                        break
                lines.append(
                    f"  Pattern {pat!r:16s} {len(recs):>4d} "
                    f"nodal{name_hint}")

        el = f.elements.loads
        if el:
            lines.append(f"Element loads ({len(el)} records):")
            for pat in el.patterns():
                recs = el.by_pattern(pat)
                name_hint = ""
                for r in recs:
                    if getattr(r, 'name', None):
                        name_hint = f"  (source: {r.name!r})"
                        break
                ltype = getattr(recs[0], 'load_type', 'element') if recs else 'element'
                lines.append(
                    f"  Pattern {pat!r:16s} {len(recs):>4d} "
                    f"{ltype}{name_hint}")

        if not lines:
            return "No loads."
        return "\n".join(lines)

    def mass_summary(self) -> str:
        """Human-readable breakdown of masses."""
        f = self._fem
        ms = f.nodes.masses
        if not ms:
            return "No masses."
        lines = [f"Nodal masses ({len(ms)} nodes):"]
        lines.append(f"  Total mass: {ms.total_mass():.6g}")
        # Source hint from first record
        for r in ms:
            if getattr(r, 'name', None):
                lines.append(f"  Source: {r.name!r}")
                break
        return "\n".join(lines)


# =====================================================================
# FEMData — top-level broker
# =====================================================================

class FEMData:
    """Solver-ready FEM mesh broker.

    Organized by what the user needs::

        fem.nodes       → NodeComposite
        fem.elements    → ElementComposite
        fem.info        → MeshInfo
        fem.inspect     → InspectComposite

    Parameters
    ----------
    nodes : NodeComposite
        Node data with sub-composites for constraints, loads, masses.
    elements : ElementComposite
        Element data with sub-composites for constraints, loads.
    info : MeshInfo
        Mesh statistics.
    mesh_selection : MeshSelectionStore, optional
        Snapshot of mesh selections (if any).
    """

    def __init__(
        self,
        nodes: NodeComposite,
        elements: ElementComposite,
        info: MeshInfo,
        mesh_selection: "MeshSelectionStore | None" = None,
    ) -> None:
        self.nodes    = nodes
        self.elements = elements
        self.info     = info
        self.mesh_selection = mesh_selection
        self.inspect  = InspectComposite(self)

    @classmethod
    def from_gmsh(cls, dim: int, *, session=None, ndf: int = 6):
        """Extract FEMData from a live Gmsh session.

        Parameters
        ----------
        dim : int
            Element dimension to extract (1, 2, or 3).
        session : apeGmsh session, optional
            When provided, auto-resolves constraints, loads, and masses.
        ndf : int
            DOFs per node for load/mass vector padding.

        Returns
        -------
        FEMData
        """
        from ._fem_factory import _from_gmsh
        return _from_gmsh(cls, dim=dim, session=session, ndf=ndf)

    @classmethod
    def from_msh(cls, path: str, dim: int = 2):
        """Load FEMData from an external ``.msh`` file.

        No session, no BCs — pure mesh + physical groups.

        Parameters
        ----------
        path : str
            Path to the ``.msh`` file.
        dim : int
            Element dimension to extract.

        Returns
        -------
        FEMData
        """
        from ._fem_factory import _from_msh
        return _from_msh(cls, path=path, dim=dim)

    def __repr__(self) -> str:
        return self.inspect.summary()
