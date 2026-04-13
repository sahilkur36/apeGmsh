"""
_fem_factory — Factory functions for FEMData construction.
===========================================================

Implements ``FEMData.from_gmsh()`` and ``FEMData.from_msh()`` by
orchestrating the raw Gmsh extraction helpers from ``_fem_extract.py``
and splitting resolved records into node-side vs element-side
sub-composites.
"""

from __future__ import annotations

import logging

import numpy as np

from ._fem_extract import (
    extract_raw, extract_physical_groups, extract_labels,
    extract_partitions,
)

_log = logging.getLogger(__name__)
from ._group_set import PhysicalGroupSet, LabelSet


# =====================================================================
# Constraint splitting
# =====================================================================

def _split_constraints(records: list) -> tuple[list, list]:
    """Split resolved constraint records into node-level and surface-level.

    Returns
    -------
    (node_records, surface_records)
        node_records:    NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord
        surface_records: InterpolationRecord, SurfaceCouplingRecord
    """
    from apeGmsh.solvers.Constraints import (
        NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        InterpolationRecord, SurfaceCouplingRecord,
    )

    node_recs = []
    surface_recs = []

    for rec in records:
        if isinstance(rec, (NodePairRecord, NodeGroupRecord,
                            NodeToSurfaceRecord)):
            node_recs.append(rec)
        elif isinstance(rec, (InterpolationRecord,
                              SurfaceCouplingRecord)):
            surface_recs.append(rec)
        else:
            _log.warning(
                "Unknown constraint record type %s (kind=%r) — "
                "placed in node-level set as fallback.",
                type(rec).__name__, getattr(rec, 'kind', '?'))
            node_recs.append(rec)

    return node_recs, surface_recs


# =====================================================================
# Load splitting
# =====================================================================

def _split_loads(records: list) -> tuple[list, list]:
    """Split resolved load records into nodal and element.

    Returns
    -------
    (nodal_records, element_records)
    """
    from apeGmsh.solvers.Loads import NodalLoadRecord, ElementLoadRecord

    nodal = []
    element = []

    for rec in records:
        if isinstance(rec, NodalLoadRecord):
            nodal.append(rec)
        elif isinstance(rec, ElementLoadRecord):
            element.append(rec)
        else:
            _log.warning(
                "Unknown load record type %s (kind=%r) — "
                "placed in nodal set as fallback.",
                type(rec).__name__, getattr(rec, 'kind', '?'))
            nodal.append(rec)

    return nodal, element


# =====================================================================
# Constraint-connected node collection
# =====================================================================

def _collect_constraint_nodes(
    node_constraints: list,
    surface_constraints: list,
    nodal_loads: list,
    mass_records: list,
) -> set[int]:
    """Collect every mesh node ID referenced by resolved BCs.

    Used by ``_from_gmsh`` to protect constraint-connected nodes
    from orphan removal.

    Walks all record types exhaustively so that no referenced node
    can slip through.
    """
    from apeGmsh.solvers.Constraints import (
        NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        InterpolationRecord, SurfaceCouplingRecord,
    )

    ids: set[int] = set()

    # ── Node-level constraints ──────────────────────────────
    for rec in node_constraints:
        if isinstance(rec, NodePairRecord):
            ids.add(rec.master_node)
            ids.add(rec.slave_node)
        elif isinstance(rec, NodeGroupRecord):
            ids.add(rec.master_node)
            ids.update(rec.slave_nodes)
        elif isinstance(rec, NodeToSurfaceRecord):
            ids.add(rec.master_node)
            ids.update(rec.slave_nodes)
            # phantom_nodes are synthetic — not in Gmsh mesh,
            # no need to protect them (they're added later).

    # ── Surface-level constraints ───────────────────────────
    for rec in surface_constraints:
        if isinstance(rec, InterpolationRecord):
            ids.add(rec.slave_node)
            ids.update(rec.master_nodes)
        elif isinstance(rec, SurfaceCouplingRecord):
            ids.update(rec.master_nodes)
            ids.update(rec.slave_nodes)

    # ── Nodal loads & masses ────────────────────────────────
    for rec in nodal_loads:
        ids.add(rec.node_id)
    for rec in mass_records:
        ids.add(rec.node_id)

    return ids


# =====================================================================
# from_gmsh
# =====================================================================

def _from_gmsh(
    cls,
    *,
    dim: int,
    session=None,
    ndf: int = 6,
    remove_orphans: bool = False,
):
    """Build a FEMData from the live Gmsh session.

    Called by ``FEMData.from_gmsh()``.

    Parameters
    ----------
    cls : type
        The FEMData class (for ``cls(...)`` construction).
    dim : int
        Element dimension to extract.
    session : apeGmsh session, optional
        Provides constraints, loads, masses composites for resolution.
    ndf : int
        DOFs per node for load/mass padding.
    remove_orphans : bool
        If True, remove mesh nodes that are not connected to any
        element (dim >= 1).  Nodes referenced by constraints, loads,
        or masses are always kept.  Default False.

    Returns
    -------
    FEMData
    """
    from .FEMData import (
        NodeComposite, ElementComposite, MeshInfo, _compute_bandwidth,
    )

    # ── 1. Extract raw mesh (no orphan filtering yet) ──────
    (node_tags, node_coords, elem_tags, connectivity,
     used_tags, physical, labels, partitions,
     elem_type_name, nodes_per_elem) = _extract_mesh_core(dim)

    # Start with ALL nodes — resolvers need to see orphans too
    node_ids = np.asarray(node_tags, dtype=int)
    node_coords_all = node_coords

    # ── 2. Resolve BCs (sees all nodes, including orphans) ──
    node_constraints: list = []
    surface_constraints: list = []
    nodal_loads: list = []
    element_loads: list = []
    mass_records: list = []

    if session is not None:
        # Build node/face maps from parts
        parts_comp = getattr(session, "parts", None)
        node_map = None
        face_map = None
        if (parts_comp is not None
                and getattr(parts_comp, "_instances", None)):
            try:
                node_map = parts_comp.build_node_map(
                    node_ids, node_coords_all)
                face_map = parts_comp.build_face_map(node_map)
            except Exception:
                node_map = None
                face_map = None

        resolve_kw = dict(
            elem_tags=elem_tags,
            connectivity=connectivity,
            node_map=node_map,
            face_map=face_map,
        )

        # Constraints
        constraints_comp = getattr(session, "constraints", None)
        if (constraints_comp is not None
                and getattr(constraints_comp, "constraint_defs", None)):
            try:
                all_constraints = constraints_comp.resolve(
                    node_ids, node_coords_all, **resolve_kw)
                node_constraints, surface_constraints = \
                    _split_constraints(all_constraints)
            except Exception as exc:
                _log.warning("Constraint resolve failed: %s", exc)

        # Loads
        loads_comp = getattr(session, "loads", None)
        if (loads_comp is not None
                and getattr(loads_comp, "load_defs", None)):
            try:
                all_loads = loads_comp.resolve(
                    node_ids, node_coords_all,
                    ndf=ndf, **resolve_kw)
                nodal_loads, element_loads = _split_loads(all_loads)
            except Exception as exc:
                _log.warning("Load resolve failed: %s", exc)

        # Masses
        masses_comp = getattr(session, "masses", None)
        if (masses_comp is not None
                and getattr(masses_comp, "mass_defs", None)):
            try:
                mass_records = masses_comp.resolve(
                    node_ids, node_coords_all,
                    ndf=ndf, **resolve_kw)
            except Exception as exc:
                _log.warning("Mass resolve failed: %s", exc)

    # ── 3. Orphan filtering (opt-in, constraint-aware) ──────
    if remove_orphans:
        protected = _collect_constraint_nodes(
            node_constraints, surface_constraints,
            nodal_loads, mass_records,
        )
        node_ids, node_coords_all = _filter_orphans(
            node_tags, node_coords, used_tags, protected)

    # ── 4. Build MeshInfo ──────────────────────────────────
    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=len(elem_tags),
        bandwidth=_compute_bandwidth(connectivity),
        nodes_per_elem=nodes_per_elem,
        elem_type_name=elem_type_name,
    )

    # ── 5. Build composites ────────────────────────────────
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords_all,
        physical=physical,
        labels=labels,
        constraints=node_constraints or None,
        loads=nodal_loads or None,
        masses=mass_records or None,
        partitions=partitions or None,
    )
    elements = ElementComposite(
        element_ids=elem_tags,
        connectivity=connectivity,
        physical=physical,
        labels=labels,
        constraints=surface_constraints or None,
        loads=element_loads or None,
        partitions=partitions or None,
    )

    # ── 6. Snapshot mesh selections if available ────────────
    ms_store = None
    if session is not None:
        ms_comp = getattr(session, "mesh_selection", None)
        if ms_comp is not None and len(ms_comp) > 0:
            try:
                ms_store = ms_comp._snapshot()
            except Exception:
                pass

    return cls(
        nodes=nodes,
        elements=elements,
        info=info,
        mesh_selection=ms_store,
    )


# =====================================================================
# from_msh
# =====================================================================

def _extract_mesh_core(dim: int):
    """Shared extraction: raw arrays + orphan mask (no filtering applied).

    Returns
    -------
    tuple
        (node_tags, node_coords, elem_tags, connectivity,
         used_tags, physical, labels, partitions, elem_type_name,
         nodes_per_elem)

    Orphan filtering is NOT done here — the caller decides via
    :func:`_filter_orphans`.
    """
    raw = extract_raw(dim=dim)

    node_tags    = raw['node_tags']
    node_coords  = raw['node_coords']
    connectivity = raw['connectivity']
    elem_tags    = np.asarray(raw['elem_tags'], dtype=int)
    used_tags    = raw['used_tags']

    physical = PhysicalGroupSet(extract_physical_groups())
    labels   = LabelSet(extract_labels())
    partitions = extract_partitions(dim)

    type_info = raw.get('elem_type_info', {})
    if type_info:
        first = next(iter(type_info.values()))
        elem_type_name = first[0]
        nodes_per_elem = first[2]
    else:
        elem_type_name = ""
        nodes_per_elem = (connectivity.shape[1]
                          if connectivity.size else 0)

    return (node_tags, node_coords, elem_tags, connectivity,
            used_tags, physical, labels, partitions,
            elem_type_name, nodes_per_elem)


# =====================================================================
# Orphan filtering
# =====================================================================

def _filter_orphans(
    node_tags: np.ndarray,
    node_coords: np.ndarray,
    used_tags: set[int],
    protected: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove orphan nodes, optionally protecting some.

    A node is "orphan" if it does not appear in any element's
    connectivity (``used_tags``).  Nodes in *protected* are kept
    even if orphan (e.g. constraint reference nodes).

    Parameters
    ----------
    node_tags : ndarray
        All mesh node tags.
    node_coords : ndarray
        Matching coordinate array.
    used_tags : set[int]
        Node tags that appear in at least one element (dim >= 1).
    protected : set[int], optional
        Node tags to keep regardless of element connectivity.

    Returns
    -------
    (node_ids, node_coords_filtered)
    """
    keep = np.isin(node_tags, list(used_tags))

    if protected:
        also_keep = np.isin(node_tags, list(protected))
        keep = keep | also_keep

    orphan_mask = ~keep
    n_orphans = int(orphan_mask.sum())
    if n_orphans > 0:
        orphan_tags = node_tags[orphan_mask]
        orphan_coords = node_coords[orphan_mask]
        detail = ", ".join(
            f"{int(t)} ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})"
            for t, c in zip(orphan_tags[:20], orphan_coords[:20])
        )
        _log.warning(
            "%d orphan node(s) removed (not connected to any element). "
            "First: [%s]%s",
            n_orphans,
            detail,
            f" ... (+{n_orphans - 20} more)" if n_orphans > 20 else "")

    node_ids = np.asarray(node_tags[keep], dtype=int)
    node_coords_filtered = node_coords[keep]
    return node_ids, node_coords_filtered


def _from_msh(
    cls,
    *,
    path: str,
    dim: int = 2,
    remove_orphans: bool = False,
):
    """Build a FEMData from an external ``.msh`` file.

    Called by ``FEMData.from_msh()``.
    Opens a temporary Gmsh session, merges the file, extracts, closes.
    """
    import gmsh
    from .FEMData import (
        NodeComposite, ElementComposite, MeshInfo, _compute_bandwidth,
    )

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(path))

        (node_tags, node_coords, elem_tags, connectivity,
         used_tags, physical, labels, partitions,
         elem_type_name, nodes_per_elem) = _extract_mesh_core(dim)

        if remove_orphans:
            node_ids, node_coords = _filter_orphans(
                node_tags, node_coords, used_tags)
        else:
            node_ids = np.asarray(node_tags, dtype=int)

        info = MeshInfo(
            n_nodes=len(node_ids),
            n_elems=len(elem_tags),
            bandwidth=_compute_bandwidth(connectivity),
            nodes_per_elem=nodes_per_elem,
            elem_type_name=elem_type_name,
        )

        nodes = NodeComposite(
            node_ids=node_ids, node_coords=node_coords,
            physical=physical, labels=labels,
            partitions=partitions or None,
        )
        elements = ElementComposite(
            element_ids=elem_tags, connectivity=connectivity,
            physical=physical, labels=labels,
            partitions=partitions or None,
        )
    finally:
        gmsh.finalize()

    return cls(nodes=nodes, elements=elements, info=info)
