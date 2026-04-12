"""
_fem_extract — Shared FEM data extraction from a live Gmsh session.
====================================================================

Standalone helper functions that pull node/element/physical-group data
straight from the ``gmsh`` API and package it into a :class:`FEMData`.

Used by:

* ``Mesh.get_fem_data()``   — the normal apeGmsh pipeline
* ``MshLoader.from_msh()``  — loading an external ``.msh`` file

Having the logic here avoids duplicating ~90 lines of Gmsh API calls
across two modules.
"""

from __future__ import annotations

import gmsh
import numpy as np
from numpy import ndarray


# =====================================================================
# Raw extraction (dict of arrays — no FEMData yet)
# =====================================================================

def extract_raw(dim: int = 2) -> dict:
    """
    Pull raw FEM arrays from the current Gmsh session.

    Parameters
    ----------
    dim : int
        Element dimension to extract (1 = lines, 2 = tri/quad,
        3 = tet/hex).

    Returns
    -------
    dict
        Keys: ``node_tags``, ``node_coords``, ``connectivity``,
        ``elem_tags``, ``elem_type_codes``, ``elem_type_info``,
        ``used_tags``.
    """
    # --- nodes (full mesh) ---
    raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
    node_tags   = np.array(raw_tags, dtype=np.int64)
    node_coords = np.array(raw_coords).reshape(-1, 3)

    # --- elements of requested dimension ---
    elem_types, elem_tags_list, node_tags_list = \
        gmsh.model.mesh.getElements(dim=dim, tag=-1)

    conn_blocks:     list[ndarray] = []
    elem_tags:       list[int]     = []
    elem_type_codes: list[int]     = []
    elem_type_info:  dict[int, tuple] = {}

    for etype, etags, enodes in zip(
        elem_types, elem_tags_list, node_tags_list
    ):
        props = gmsh.model.mesh.getElementProperties(int(etype))
        # props: (name, dim, order, n_nodes, local_coords, n_primary)
        npe = props[3]
        conn_blocks.append(
            np.array(enodes, dtype=np.int64).reshape(-1, npe)
        )
        n_this = len(etags)
        elem_tags.extend(int(t) for t in etags)
        elem_type_codes.extend([int(etype)] * n_this)
        elem_type_info[int(etype)] = (props[0], props[1], props[3])

    connectivity = (
        np.vstack(conn_blocks) if conn_blocks
        else np.empty((0, 0), dtype=int)
    )

    # --- used_tags from connectivity (all dims >= 1) ---
    # A node is "used" if it appears in any element's connectivity
    # (lines, surfaces, volumes).  Nodes only referenced by dim=0
    # point elements (arc centres, construction points) are orphans.
    used_tags: set[int] = set()
    # Start with the target-dim connectivity we already extracted
    if connectivity.size > 0:
        used_tags.update(connectivity.ravel().tolist())
    # Also include nodes from other structural dims (1D, 2D, 3D)
    # in case they don't appear in the target-dim connectivity.
    for d in range(1, 4):
        if d == dim:
            continue  # already covered above
        _, _, d_enodes = gmsh.model.mesh.getElements(dim=d, tag=-1)
        for enodes_block in d_enodes:
            used_tags.update(int(n) for n in enodes_block)

    return {
        'node_tags':       node_tags,
        'node_coords':     node_coords,
        'connectivity':    connectivity,
        'elem_tags':       elem_tags,
        'elem_type_codes': elem_type_codes,
        'elem_type_info':  elem_type_info,
        'used_tags':       used_tags,
    }


# =====================================================================
# Physical-group snapshot
# =====================================================================

def extract_physical_groups() -> dict[tuple[int, int], dict]:
    """
    Snapshot every physical group in the current Gmsh session.

    Returns
    -------
    dict
        ``{(dim, pg_tag): {'name', 'node_ids', 'node_coords',
        'element_ids'?, 'connectivity'?}}``
    """
    pg_data: dict[tuple[int, int], dict] = {}

    from apeGmsh.core.Labels import is_label_pg

    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        # Skip internal label PGs (Tier 1 naming) — they are
        # geometry bookkeeping and should not appear in fem.physical.
        if is_label_pg(name):
            continue
        pg_node_tags, pg_coords = \
            gmsh.model.mesh.getNodesForPhysicalGroup(pg_dim, pg_tag)

        entry: dict = {
            'name':        name,
            'node_ids':    np.array(pg_node_tags, dtype=np.int64),
            'node_coords': np.array(pg_coords).reshape(-1, 3),
        }

        # Capture element data for dim >= 1 physical groups
        if pg_dim >= 1:
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(
                pg_dim, pg_tag
            )
            pg_elem_tags:   list[int]     = []
            pg_conn_blocks: list[ndarray] = []

            for ent_tag in entity_tags:
                etypes, etags_list, enodes_list = \
                    gmsh.model.mesh.getElements(
                        dim=pg_dim, tag=int(ent_tag)
                    )
                for etype, etags_arr, enodes in zip(
                    etypes, etags_list, enodes_list
                ):
                    props = gmsh.model.mesh.getElementProperties(
                        int(etype)
                    )
                    npe = props[3]
                    pg_elem_tags.extend(int(t) for t in etags_arr)
                    pg_conn_blocks.append(
                        np.array(enodes, dtype=np.int64).reshape(-1, npe)
                    )

            if pg_conn_blocks:
                entry['element_ids']  = np.array(
                    pg_elem_tags, dtype=np.int64
                )
                entry['connectivity'] = np.vstack(pg_conn_blocks)

        pg_data[(pg_dim, pg_tag)] = entry

    return pg_data


# =====================================================================
# Label snapshot (Tier 1 — _label: prefixed PGs)
# =====================================================================

def extract_labels() -> dict[tuple[int, int], dict]:
    """Snapshot every label (``_label:``-prefixed PG) in the session.

    Same structure as :func:`extract_physical_groups` but captures
    only label PGs and strips the ``_label:`` prefix from names.

    Returns
    -------
    dict
        ``{(dim, pg_tag): {'name', 'node_ids', 'node_coords',
        'element_ids'?, 'connectivity'?}}``
    """
    from apeGmsh.core.Labels import LABEL_PREFIX, is_label_pg

    lbl_data: dict[tuple[int, int], dict] = {}

    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        if not is_label_pg(name):
            continue

        # Strip the _label: prefix for the public name
        clean_name = name[len(LABEL_PREFIX):]

        pg_node_tags, pg_coords = \
            gmsh.model.mesh.getNodesForPhysicalGroup(pg_dim, pg_tag)

        entry: dict = {
            'name':        clean_name,
            'node_ids':    np.array(pg_node_tags, dtype=np.int64),
            'node_coords': np.array(pg_coords).reshape(-1, 3),
        }

        # Capture element data for dim >= 1
        if pg_dim >= 1:
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(
                pg_dim, pg_tag
            )
            lbl_elem_tags:   list[int]     = []
            lbl_conn_blocks: list[ndarray] = []

            for ent_tag in entity_tags:
                etypes, etags_list, enodes_list = \
                    gmsh.model.mesh.getElements(
                        dim=pg_dim, tag=int(ent_tag)
                    )
                for etype, etags_arr, enodes in zip(
                    etypes, etags_list, enodes_list
                ):
                    props = gmsh.model.mesh.getElementProperties(
                        int(etype)
                    )
                    npe = props[3]
                    lbl_elem_tags.extend(int(t) for t in etags_arr)
                    lbl_conn_blocks.append(
                        np.array(enodes, dtype=np.int64).reshape(-1, npe)
                    )

            if lbl_conn_blocks:
                entry['element_ids']  = np.array(
                    lbl_elem_tags, dtype=np.int64
                )
                entry['connectivity'] = np.vstack(lbl_conn_blocks)

        lbl_data[(pg_dim, pg_tag)] = entry

    return lbl_data


# =====================================================================
# Full FEMData assembly
# =====================================================================

def build_fem_data(
    dim: int = 2,
    mesh_selection_composite=None,
    *,
    parent=None,
    auto_resolve: bool = True,
    ndf: int = 6,
):
    """
    Extract a complete :class:`FEMData` from the live Gmsh session.

    Combines :func:`extract_raw` and :func:`extract_physical_groups`
    into a single self-contained ``FEMData`` object.

    Parameters
    ----------
    dim : int
        Element dimension to extract.
    mesh_selection_composite : MeshSelectionSet, optional
        If provided, a snapshot is taken and attached to FEMData.
    parent : _SessionBase, optional
        Owning session.  When provided, ``g.constraints`` and
        ``g.loads`` are automatically resolved against the extracted
        mesh and attached to ``fem.constraints`` / ``fem.loads``.
    auto_resolve : bool
        If True (default) and *parent* is given, auto-resolve any
        registered constraints and loads.  Pass ``False`` to skip
        auto-resolution and assign records manually later.
    ndf : int
        Number of DOFs per node (used for load vector padding).
        Default 6 (full 3D with rotations).

    Returns
    -------
    FEMData
    """
    from .FEMData import (
        FEMData, MeshInfo, PhysicalGroupSet, LabelSet,
        _compute_bandwidth,
    )

    raw = extract_raw(dim=dim)

    node_tags    = raw['node_tags']
    node_coords  = raw['node_coords']
    connectivity = raw['connectivity']
    elem_tags    = np.asarray(raw['elem_tags'], dtype=int)
    used_tags    = raw['used_tags']

    # Filter to only nodes referenced by elements
    mask      = np.isin(node_tags, list(used_tags))
    n_total   = len(node_tags)
    node_ids  = np.asarray(node_tags[mask], dtype=int)
    node_coords = node_coords[mask]
    n_orphans = n_total - len(node_ids)
    if n_orphans > 0:
        orphan_tags = node_tags[~mask]
        print(
            f"[FEMData] WARNING: {n_orphans} orphan node(s) removed "
            f"(not connected to any element). "
            f"Tags: {orphan_tags.tolist()[:20]}"
            + (f" ... (+{n_orphans - 20} more)" if n_orphans > 20 else "")
        )

    # Element type info from extraction
    type_info = raw.get('elem_type_info', {})
    if type_info:
        first_type = next(iter(type_info.values()))
        elem_type_name = first_type[0]   # e.g. "Tetrahedron 4"
        nodes_per_elem = first_type[2]   # e.g. 4
    else:
        elem_type_name = ""
        nodes_per_elem = connectivity.shape[1] if connectivity.size else 0

    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=len(elem_tags),
        bandwidth=_compute_bandwidth(connectivity),
        nodes_per_elem=nodes_per_elem,
        elem_type_name=elem_type_name,
    )

    physical = PhysicalGroupSet(extract_physical_groups())
    labels = LabelSet(extract_labels())

    # Snapshot mesh selections if available
    ms_store = None
    if mesh_selection_composite is not None and len(mesh_selection_composite) > 0:
        ms_store = mesh_selection_composite._snapshot()

    result = FEMData(
        node_ids=node_ids,
        node_coords=node_coords,
        element_ids=elem_tags,
        connectivity=connectivity,
        info=info,
        physical=physical,
        labels=labels,
        mesh_selection=ms_store,
    )

    # ── Auto-resolve constraints and loads ──────────────────────
    if auto_resolve and parent is not None:
        parts_comp = getattr(parent, "parts", None)
        node_map = None
        face_map = None
        if parts_comp is not None and getattr(parts_comp, "_instances", None):
            try:
                node_map = parts_comp.build_node_map(result.node_ids, result.node_coords)
                face_map = parts_comp.build_face_map(node_map)
            except Exception:
                node_map = None
                face_map = None

        # Constraints
        constraints_comp = getattr(parent, "constraints", None)
        if constraints_comp is not None and getattr(
            constraints_comp, "constraint_defs", None
        ):
            try:
                cs = constraints_comp.resolve(
                    result.node_ids, result.node_coords,
                    elem_tags=result.element_ids,
                    connectivity=result.connectivity,
                    node_map=node_map, face_map=face_map,
                )
                object.__setattr__(result, 'constraints', cs)
            except Exception as exc:
                print(f"[FEMData] WARNING: constraint auto-resolve failed: {exc}")

        # Loads
        loads_comp = getattr(parent, "loads", None)
        if loads_comp is not None and getattr(loads_comp, "load_defs", None):
            try:
                ls = loads_comp.resolve(
                    result.node_ids, result.node_coords,
                    elem_tags=result.element_ids,
                    connectivity=result.connectivity,
                    node_map=node_map, face_map=face_map,
                    ndf=ndf,
                )
                object.__setattr__(result, 'loads', ls)
            except Exception as exc:
                print(f"[FEMData] WARNING: load auto-resolve failed: {exc}")

        # Masses
        masses_comp = getattr(parent, "masses", None)
        if masses_comp is not None and getattr(masses_comp, "mass_defs", None):
            try:
                ms = masses_comp.resolve(
                    result.node_ids, result.node_coords,
                    elem_tags=result.element_ids,
                    connectivity=result.connectivity,
                    node_map=node_map, face_map=face_map,
                    ndf=ndf,
                )
                object.__setattr__(result, 'masses', ms)
            except Exception as exc:
                print(f"[FEMData] WARNING: mass auto-resolve failed: {exc}")

    return result
