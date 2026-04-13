"""
MassesComposite -- Define and resolve nodal masses.

Two-stage pipeline mirroring :class:`LoadsComposite` but simpler:

1. **Define** (pre-mesh): factory methods (``point``, ``line``,
   ``surface``, ``volume``) store :class:`MassDef` objects.
2. **Resolve** (post-mesh): :meth:`resolve` delegates to
   :class:`MassResolver` and accumulates per-node mass into a
   :class:`MassSet`.  Auto-called by ``Mesh.get_fem_data()``.

There is **no pattern grouping** — mass is intrinsic to the model.
Multiple mass definitions targeting the same nodes accumulate.

Targets accept any of:
    * a list of ``(dim, tag)`` tuples
    * a part label (``g.parts.instances[label]``)
    * a physical group name (``g.physical``)
    * a mesh selection name (``g.mesh_selection``)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh.mesh._record_set import MassSet
from apeGmsh.solvers.Masses import (
    MassDef,
    MassRecord,
    MassResolver,
    PointMassDef,
    LineMassDef,
    SurfaceMassDef,
    VolumeMassDef,
)


# (MassDefType, reduction) -> method name on MassesComposite
_DISPATCH: dict[type, dict[str, str]] = {
    PointMassDef: {
        "lumped":     "_resolve_point",
        "consistent": "_resolve_point",
    },
    LineMassDef: {
        "lumped":     "_resolve_line_lumped",
        "consistent": "_resolve_line_consistent",
    },
    SurfaceMassDef: {
        "lumped":     "_resolve_surface_lumped",
        "consistent": "_resolve_surface_consistent",
    },
    VolumeMassDef: {
        "lumped":     "_resolve_volume_lumped",
        "consistent": "_resolve_volume_consistent",
    },
}

_MassT = TypeVar("_MassT", bound=MassDef)


class MassesComposite:
    """Mass composite — define + resolve nodal masses."""

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.mass_defs: list[MassDef] = []
        self.mass_records: list[MassRecord] = []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def point(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        mass: float,
        rotational: tuple | None = None,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> PointMassDef:
        """Concentrated mass at the node(s) of *target*.

        Target resolution follows the FEM broker pattern:
        ``pg=`` explicit PG, ``label=`` explicit label,
        ``tag=`` direct Gmsh tag,
        positional *target* tries label first then PG.
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(PointMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            mass=mass, rotational=rotational,
        ))

    def line(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        linear_density: float,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> LineMassDef:
        """Distributed line mass on curve(s) of *target*."""
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(LineMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            linear_density=linear_density,
        ))

    def surface(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        areal_density: float,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> SurfaceMassDef:
        """Distributed surface mass on face(s) of *target*."""
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(SurfaceMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            areal_density=areal_density,
        ))

    def volume(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        density: float,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> VolumeMassDef:
        """Distributed volume mass on volume(s) of *target*."""
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(VolumeMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            density=density,
        ))

    @staticmethod
    def _coalesce_target(target, *, pg=None, label=None, tag=None):
        """Resolve explicit pg=/label=/tag= into (target, source) pair."""
        if tag is not None:
            return tag, "tag"
        if pg is not None:
            return pg, "pg"
        if label is not None:
            return label, "label"
        if target is not None:
            return target, "auto"
        raise ValueError(
            "One of target, pg=, label=, or tag= is required.")

    # ------------------------------------------------------------------
    # Internal: store + validate
    # ------------------------------------------------------------------

    def _add_def(self, defn: _MassT) -> _MassT:
        cfg = _DISPATCH.get(type(defn), {})
        if defn.reduction not in cfg:
            raise ValueError(
                f"{type(defn).__name__} does not support "
                f"reduction={defn.reduction!r}.  Supported: {list(cfg.keys())}"
            )
        self.mass_defs.append(defn)
        return defn

    # ------------------------------------------------------------------
    # Target resolution (same lookup order as LoadsComposite)
    # ------------------------------------------------------------------

    def _resolve_target(self, target, source: str = "auto") -> list[tuple]:
        """Resolve target -> list of ``(dim, tag)`` or mesh-selection sentinel.

        Lookup order (for ``source="auto"``):
            1. Raw DimTag list -> as-is
            2. Mesh selection name
            3. Label name (Tier 1, ``_label:`` prefixed PG)
            4. Physical group name (Tier 2)
            5. Part label
        """
        import gmsh

        if isinstance(target, (list, tuple)) and len(target) > 0 \
                and isinstance(target[0], (list, tuple)):
            return [(int(d), int(t)) for d, t in target]

        if not isinstance(target, str):
            raise TypeError(
                f"target must be a string label or list of (dim, tag), "
                f"got {type(target).__name__}"
            )

        # Mesh selection name (auto only)
        if source == "auto":
            ms = getattr(self._parent, "mesh_selection", None)
            if ms is not None and hasattr(ms, "_sets"):
                for (dim, tag), info in ms._sets.items():
                    if info.get("name") == target:
                        return [("__ms__", dim, tag)]

        # Label name (Tier 1)
        if source in ("auto", "label"):
            try:
                from apeGmsh.core.Labels import add_prefix
                prefixed = add_prefix(target)
                for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                    try:
                        if gmsh.model.getPhysicalName(pg_dim, pg_tag) == prefixed:
                            ents = gmsh.model.getEntitiesForPhysicalGroup(
                                pg_dim, pg_tag)
                            return [(pg_dim, int(t)) for t in ents]
                    except Exception:
                        pass
            except Exception:
                pass

        # Physical group name (Tier 2)
        if source in ("auto", "pg"):
            try:
                for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                    try:
                        if gmsh.model.getPhysicalName(pg_dim, pg_tag) == target:
                            ents = gmsh.model.getEntitiesForPhysicalGroup(
                                pg_dim, pg_tag)
                            return [(pg_dim, int(t)) for t in ents]
                    except Exception:
                        pass
            except Exception:
                pass

        # Part label
        if source == "auto":
            parts = getattr(self._parent, "parts", None)
            if parts is not None and hasattr(parts, "_instances"):
                inst = parts._instances.get(target)
                if inst is not None:
                    out: list = []
                    for d, ts in inst.entities.items():
                        out.extend((int(d), int(t)) for t in ts)
                    return out

        raise KeyError(
            f"Mass target {target!r} not found as label, physical group, "
            f"part label, or mesh selection."
        )

    def _target_nodes(self, target, node_map, all_nodes,
                      source: str = "auto") -> set[int]:
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            _, dim, tag = dts[0]
            ms = self._parent.mesh_selection
            info = ms._sets.get((dim, tag))
            if info is None:
                return set()
            return set(int(n) for n in info.get("node_ids", []))

        parts = getattr(self._parent, "parts", None)
        if isinstance(target, str) and parts is not None:
            if target in getattr(parts, "_instances", {}):
                if node_map is not None and target in node_map:
                    return set(node_map[target])

        import gmsh
        nodes: set[int] = set()
        for d, t in dts:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=int(d), tag=int(t),
                    includeBoundary=True, returnParametricCoord=False,
                )
                nodes.update(int(n) for n in nt)
            except Exception:
                pass
        return nodes

    def _target_edges(self, target, source: str = "auto") -> list[tuple[int, int]]:
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            return []
        import gmsh
        edges: list[tuple[int, int]] = []
        for d, t in dts:
            if d != 1:
                continue
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                npe = 2 if int(etype) == 1 else 3
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    edges.append((int(row[0]), int(row[-1])))
        return edges

    def _target_faces(self, target, source: str = "auto") -> list[list[int]]:
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            return []
        import gmsh
        faces: list[list[int]] = []
        for d, t in dts:
            if d != 2:
                continue
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                etype = int(etype)
                npe = {2: 3, 3: 4, 9: 6, 16: 8}.get(etype, None)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                corners_per = {3: 3, 4: 4, 6: 3, 8: 4}[npe]
                for row in arr:
                    faces.append([int(n) for n in row[:corners_per]])
        return faces

    def _target_elements(self, target, source: str = "auto"):
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            return []
        import gmsh
        conns: list[np.ndarray] = []
        for d, t in dts:
            if d != 3:
                continue
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                etype = int(etype)
                npe_map = {4: 4, 5: 8, 6: 6, 11: 10, 17: 20}
                npe = npe_map.get(etype, None)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    conns.append(row)
        return conns

    # ------------------------------------------------------------------
    # resolve()
    # ------------------------------------------------------------------

    def resolve(
        self,
        node_tags,
        node_coords,
        elem_tags=None,
        connectivity=None,
        *,
        node_map=None,
        face_map=None,
        ndf: int = 6,
    ) -> MassSet:
        """Resolve all stored MassDefs into a :class:`MassSet`.

        Multiple definitions targeting overlapping nodes are
        accumulated — each node ends up with at most one
        :class:`MassRecord` whose vector is the sum of contributions.
        """
        resolver = MassResolver(
            node_tags, node_coords, elem_tags, connectivity, ndf=ndf,
        )
        all_nodes = set(int(t) for t in node_tags)

        # Per-node accumulator across all defs
        accum: dict[int, np.ndarray] = {}

        for defn in self.mass_defs:
            cfg = _DISPATCH[type(defn)]
            method_name = cfg.get(defn.reduction)
            if method_name is None:
                raise ValueError(
                    f"{type(defn).__name__} does not support "
                    f"reduction={defn.reduction!r}"
                )
            method = getattr(self, method_name)
            raw_records = method(resolver, defn, node_map, all_nodes)
            for r in raw_records:
                vec = np.asarray(r.mass, dtype=float)
                if r.node_id in accum:
                    accum[r.node_id] += vec
                else:
                    accum[r.node_id] = vec.copy()

        # Build final flattened MassRecord list (one per node)
        records: list[MassRecord] = [
            MassRecord(
                node_id=int(nid),
                mass=tuple(float(v) for v in vec),
            )
            for nid, vec in sorted(accum.items())
        ]
        self.mass_records = records
        return MassSet(records)

    # ------------------------------------------------------------------
    # Private dispatch methods
    # ------------------------------------------------------------------

    def _resolve_point(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        nodes = self._target_nodes(defn.target, node_map, all_nodes, source=src)
        return resolver.resolve_point_lumped(defn, nodes)

    def _resolve_line_lumped(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        edges = self._target_edges(defn.target, source=src)
        return resolver.resolve_line_lumped(defn, edges)

    def _resolve_line_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        edges = self._target_edges(defn.target, source=src)
        return resolver.resolve_line_consistent(defn, edges)

    def _resolve_surface_lumped(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        faces = self._target_faces(defn.target, source=src)
        return resolver.resolve_surface_lumped(defn, faces)

    def _resolve_surface_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        faces = self._target_faces(defn.target, source=src)
        return resolver.resolve_surface_consistent(defn, faces)

    def _resolve_volume_lumped(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        elements = self._target_elements(defn.target, source=src)
        return resolver.resolve_volume_lumped(defn, elements)

    def _resolve_volume_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        elements = self._target_elements(defn.target, source=src)
        return resolver.resolve_volume_consistent(defn, elements)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.mass_defs)

    def __repr__(self) -> str:
        if not self.mass_defs:
            return "MassesComposite(empty)"
        return f"MassesComposite({len(self.mass_defs)} defs)"
