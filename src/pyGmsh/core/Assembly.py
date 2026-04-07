"""
Assembly — Import, position, mesh, and connect Parts.
======================================================

Mirrors the Abaqus Assembly concept with a clean separation:

* **Part**  =  geometry only (STEP files)
* **Assembly**  =  everything else: import, position, fragment,
  physical groups, mesh control, mesh generation, FEM data
  extraction, and constraint resolution.

Each :class:`Part` is imported as an *instance* (with optional
translation / rotation).  The Assembly owns the mesh.

Workflow
--------
::

    from pyGmsh import Part, Assembly

    # Build parts (each in its own Gmsh session) ──────────────────
    web   = Part("web");    web.begin();  ...  web.save("web.step");    web.end()
    flange= Part("flange"); flange.begin(); ...; flange.save("fl.step"); flange.end()

    # Assemble ─────────────────────────────────────────────────────
    asm = Assembly("I_beam")
    asm.begin()

    asm.add_part(web)                                        # identity placement
    asm.add_part(flange, translate=(0, 0, 100), label="bot") # shift bottom flange
    asm.add_part(flange, translate=(0, 0, 300), label="top") # reuse same Part

    asm.fragment_all()              # conformal mesh at shared interfaces
    asm.mesh.generate(dim=3)        # mesh the whole assembly
    asm.mesh.renumber_mesh(method="rcm", base=1)
    fem = asm.mesh.get_fem_data(dim=3)
    ...
    asm.end()

Design notes
~~~~~~~~~~~~
*  ``add_part`` imports the CAD file into the *assembly's* Gmsh session
   and returns an ``Instance`` dataclass with the resulting entity tags.
*  The same Part can be added multiple times (like Abaqus instances of
   the same part).
*  ``fragment_all()`` fragments *every* volume/surface in the model so
   all interfaces become conformal — this is the simplest and most
   robust strategy for FEA assemblies.
*  ``fragment_pair()`` allows selective fragmentation between two
   specific instances.
*  Constraints (``tie``, ``rigid_link``, ``coupling``) are stored as
   lightweight records; they are consumed later when building the
   solver model (e.g. OpenSees multi-point constraints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import gmsh
import numpy as np

from .._session import _SessionBase

if TYPE_CHECKING:
    from .Part import Part
    from ..viz.Inspect import Inspect
    from .Model import Model
    from ..mesh.Mesh import Mesh
    from ..mesh.PhysicalGroups import PhysicalGroups
    from ..mesh.Partition import Partition
    from ..mesh.View import View
    from ..solvers.Gmsh2OpenSees import Gmsh2OpenSees
    from ..viz.Plot import Plot


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class Instance:
    """
    Bookkeeping record for one imported Part inside the Assembly.

    Attributes
    ----------
    label       : unique name inside the assembly
    part_name   : name of the source Part
    file_path   : CAD file that was imported
    entities    : ``{dim: [tag, ...]}`` of imported entities
    translate   : applied translation (dx, dy, dz)
    rotate      : applied rotation   (angle_rad, ax, ay, az, cx, cy, cz)
    properties  : metadata copied from the Part
    """
    label: str
    part_name: str
    file_path: Path
    entities: dict[int, list[int]] = field(default_factory=dict)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate: tuple[float, ...] | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    bbox: tuple[float, float, float, float, float, float] | None = None


from ..solvers.Constraints import (
    ConstraintDef, ConstraintRecord, ConstraintResolver,
    # Level 1
    EqualDOFDef, RigidLinkDef, PenaltyDef,
    # Level 2
    RigidDiaphragmDef, RigidBodyDef, KinematicCouplingDef,
    # Level 3
    TieDef, DistributingCouplingDef, EmbeddedDef,
    # Level 4
    TiedContactDef, MortarDef,
)


# ======================================================================
# Assembly class
# ======================================================================

class Assembly(_SessionBase):
    """
    Multi-part assembly manager.

    Parameters
    ----------
    name : str
        Assembly name (used as the Gmsh model name).
    """

    _COMPOSITES = (
        ("inspect",   ".viz.Inspect",           "Inspect",        False),
        ("model",     ".core.Model",            "Model",          False),
        ("mesh",      ".mesh.Mesh",             "Mesh",           False),
        ("physical",  ".mesh.PhysicalGroups",   "PhysicalGroups", False),
        ("partition", ".mesh.Partition",        "Partition",      False),
        ("view",      ".mesh.View",             "View",           False),
        ("g2o",       ".solvers.Gmsh2OpenSees", "Gmsh2OpenSees",  False),
        ("plot",      ".viz.Plot",              "Plot",           True),
    )

    # -- Static type declarations for composites --
    inspect: Inspect
    model: Model
    mesh: Mesh
    physical: PhysicalGroups
    partition: Partition
    view: View
    g2o: Gmsh2OpenSees
    plot: Plot

    def __init__(self, name: str = "Assembly") -> None:
        super().__init__(name=name, verbose=False)

        # Instance storage (insertion order preserved)
        self.instances: dict[str, Instance] = {}
        self._instance_counter: int = 0

        # Constraint storage (definitions, pre-mesh)
        self.constraint_defs: list[ConstraintDef] = []
        # Resolved records (post-mesh, populated by resolve_constraints)
        self.constraint_records: list[ConstraintRecord] = []

    @staticmethod
    def _compute_bbox(
        dimtags: list[tuple[int, int]],
    ) -> tuple[float, float, float, float, float, float] | None:
        """Return the union bounding box of the provided entities."""
        if not dimtags:
            return None

        mins = np.array([np.inf, np.inf, np.inf], dtype=float)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        for dim, tag in dimtags:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            mins = np.minimum(mins, [xmin, ymin, zmin])
            maxs = np.maximum(maxs, [xmax, ymax, zmax])
        return (
            float(mins[0]),
            float(mins[1]),
            float(mins[2]),
            float(maxs[0]),
            float(maxs[1]),
            float(maxs[2]),
        )

    @staticmethod
    def _nodes_in_bbox(
        node_tags,
        node_coords,
        bbox: tuple[float, float, float, float, float, float] | None,
    ) -> set[int]:
        """Return node tags whose coordinates lie inside a bounding box."""
        tags = np.asarray(node_tags, dtype=int)
        coords = np.asarray(node_coords, dtype=float)
        if bbox is None or len(tags) == 0:
            return set(int(tag) for tag in tags)

        mins = np.array(bbox[:3], dtype=float)
        maxs = np.array(bbox[3:], dtype=float)
        span = max(float((maxs - mins).max()), 1.0)
        tol = 1e-6 * span
        mask = np.all((coords >= (mins - tol)) & (coords <= (maxs + tol)), axis=1)
        return set(int(tag) for tag in tags[mask])

    def _build_instance_node_map(self, node_tags, node_coords) -> dict[str, set[int]]:
        """Infer instance node ownership from stored instance bounding boxes."""
        return {
            label: self._nodes_in_bbox(node_tags, node_coords, inst.bbox)
            for label, inst in self.instances.items()
        }

    @staticmethod
    def _coerce_entities(
        entities: list[tuple[int, int]] | None,
    ) -> list[tuple[int, int]]:
        if not entities:
            return []
        return [(int(dim), int(tag)) for dim, tag in entities]

    def _get_nodes_for_entities(
        self,
        entities: list[tuple[int, int]] | None,
    ) -> set[int]:
        """Collect mesh node tags attached to the selected geometric entities."""
        if not self._active or not entities:
            return set()

        tags: set[int] = set()
        for dim, tag in self._coerce_entities(entities):
            try:
                node_tags, _, _ = gmsh.model.mesh.getNodes(
                    dim=dim,
                    tag=tag,
                    includeBoundary=True,
                    returnParametricCoord=False,
                )
            except Exception:
                continue
            tags.update(int(node_tag) for node_tag in node_tags)
        return tags

    def _surface_entities_from_selection(
        self,
        entities: list[tuple[int, int]] | None,
    ) -> list[tuple[int, int]]:
        """Expand a selection to the surface entities needed for face queries."""
        if not entities:
            return []

        selected: set[tuple[int, int]] = set()
        for dim, tag in self._coerce_entities(entities):
            if dim == 2:
                selected.add((2, tag))
            elif dim == 3:
                for bdim, btag in gmsh.model.getBoundary([(dim, tag)], oriented=False):
                    if bdim == 2:
                        selected.add((bdim, btag))
        return sorted(selected)

    def _collect_surface_faces(
        self,
        entities: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Collect a rectangular array of surface element connectivity."""
        if not self._active:
            return np.empty((0, 0), dtype=int)

        if entities is None:
            surface_entities = list(gmsh.model.getEntities(2))
        else:
            surface_entities = self._surface_entities_from_selection(entities)

        blocks: list[np.ndarray] = []
        n_nodes_per_face: int | None = None

        for _, tag in surface_entities:
            elem_types, _, node_tags = gmsh.model.mesh.getElements(dim=2, tag=tag)
            for etype, enodes in zip(elem_types, node_tags):
                if len(enodes) == 0:
                    continue

                _, _, _, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
                if n_nodes_per_face is None:
                    n_nodes_per_face = int(n_nodes)
                elif n_nodes_per_face != int(n_nodes):
                    raise ValueError(
                        "Mixed surface element types are not yet supported in "
                        "automatic constraint face extraction."
                    )

                blocks.append(np.array(enodes, dtype=int).reshape(-1, int(n_nodes)))

        if not blocks:
            return np.empty((0, 0), dtype=int)
        return np.vstack(blocks)

    def _build_instance_face_map(
        self,
        instance_node_map: dict[str, set[int]],
    ) -> dict[str, np.ndarray]:
        """Infer per-instance surface faces by filtering global surface elements."""
        faces = self._collect_surface_faces()
        if faces.size == 0:
            return {label: np.empty((0, 0), dtype=int) for label in self.instances}

        out: dict[str, np.ndarray] = {}
        for label, nodes in instance_node_map.items():
            if not nodes:
                out[label] = np.empty((0, faces.shape[1]), dtype=int)
                continue
            mask = np.all(np.isin(faces, list(nodes)), axis=1)
            out[label] = faces[mask]
        return out

    def _resolve_constraint_nodes(
        self,
        label: str,
        role: str,
        defn: ConstraintDef,
        instance_node_map: dict[str, set[int]],
        all_nodes: set[int],
    ) -> set[int]:
        """Resolve the node set for one side of a constraint definition."""
        selected_entities = getattr(defn, f"{role}_entities", None)
        if selected_entities is not None:
            selected_nodes = self._get_nodes_for_entities(selected_entities)
            if selected_nodes:
                return selected_nodes
        return instance_node_map.get(label, all_nodes)

    def _resolve_constraint_faces(
        self,
        label: str,
        role: str,
        defn: ConstraintDef,
        instance_face_map: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Resolve the face connectivity array for one side of a constraint."""
        selected_entities = getattr(defn, f"{role}_entities", None)
        if selected_entities is not None:
            return self._collect_surface_faces(selected_entities)
        return instance_face_map.get(label, np.empty((0, 0), dtype=int))

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------

    def add_part(
        self,
        part: "Part",
        *,
        label: str | None = None,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
        highest_dim_only: bool = True,
    ) -> Instance:
        """
        Import a Part into the Assembly and apply placement transforms.

        Parameters
        ----------
        part : Part
            A Part that has been ``save()``-d to disk.
        label : str, optional
            Unique instance name.  Auto-generated as
            ``"{part.name}_1"``, ``"{part.name}_2"``, … if omitted.
        translate : (dx, dy, dz)
            Translation applied *after* import.
        rotate : tuple, optional
            ``(angle_rad, ax, ay, az)`` or
            ``(angle_rad, ax, ay, az, cx, cy, cz)``.
            Rotation applied *before* translation.
        highest_dim_only : bool
            Only keep the highest-dimension entities from the CAD file.

        Returns
        -------
        Instance
            The bookkeeping record (also stored in ``self.instances``).
        """
        if not self._active:
            raise RuntimeError("Assembly session is not active — call begin() first.")

        if not part.has_file:
            raise FileNotFoundError(
                f"Part '{part.name}' has not been saved to disk.  "
                f"Call part.save('file.step') first."
            )

        # Auto-label
        if label is None:
            self._instance_counter += 1
            label = f"{part.name}_{self._instance_counter}"
        if label in self.instances:
            raise ValueError(f"Instance label '{label}' already exists.")

        # ── Import (always CAD — Parts are geometry-only) ─────────
        file_path = part.file_path

        raw: list[tuple[int, int]] = gmsh.model.occ.importShapes(
            str(file_path),
            highestDimOnly=highest_dim_only,
        )
        gmsh.model.occ.synchronize()
        entities: dict[int, list[int]] = {}
        for dim, tag in raw:
            entities.setdefault(dim, []).append(tag)

        # Build dimtag list for transforms
        dimtags = [
            (d, t) for d, tags in entities.items() for t in tags
        ]

        # ── Rotate (before translation) ───────────────────────────
        if rotate is not None and dimtags:
            if len(rotate) == 4:
                angle, ax, ay, az = rotate
                cx = cy = cz = 0.0
            elif len(rotate) == 7:
                angle, ax, ay, az, cx, cy, cz = rotate
            else:
                raise ValueError(
                    "rotate must be (angle, ax, ay, az) or "
                    "(angle, ax, ay, az, cx, cy, cz)"
                )
            gmsh.model.occ.rotate(dimtags, cx, cy, cz, ax, ay, az, angle)
            gmsh.model.occ.synchronize()

        # ── Translate ─────────────────────────────────────────────
        dx, dy, dz = translate
        if (dx != 0.0 or dy != 0.0 or dz != 0.0) and dimtags:
            gmsh.model.occ.translate(dimtags, dx, dy, dz)
            gmsh.model.occ.synchronize()

        # ── Store ─────────────────────────────────────────────────
        inst = Instance(
            label=label,
            part_name=part.name,
            file_path=file_path,
            entities=entities,
            translate=(dx, dy, dz),
            rotate=rotate,
            properties=dict(part.properties),
            bbox=self._compute_bbox(dimtags),
        )
        self.instances[label] = inst
        return inst

    def add_file(
        self,
        file_path: str | Path,
        *,
        label: str | None = None,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
        highest_dim_only: bool = True,
        properties: dict[str, Any] | None = None,
    ) -> Instance:
        """
        Import a CAD file directly (without a Part object).

        Useful when you already have STEP/IGES files from external CAD
        software.

        Parameters
        ----------
        file_path : path
            STEP or IGES file.
        label : str, optional
            Instance name (auto-generated from file stem if omitted).
        translate, rotate, highest_dim_only
            Same as :meth:`add_part`.
        properties : dict, optional
            Arbitrary metadata to attach.

        Returns
        -------
        Instance
        """
        if not self._active:
            raise RuntimeError("Assembly session is not active — call begin() first.")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CAD file not found: {file_path}")

        if label is None:
            self._instance_counter += 1
            label = f"{file_path.stem}_{self._instance_counter}"
        if label in self.instances:
            raise ValueError(f"Instance label '{label}' already exists.")

        # ── Import ────────────────────────────────────────────────
        ext = file_path.suffix.lower()
        if ext == '.msh':
            entities = self._import_msh(file_path)
        else:
            raw = gmsh.model.occ.importShapes(
                str(file_path),
                highestDimOnly=highest_dim_only,
            )
            gmsh.model.occ.synchronize()
            entities = {}
            for dim, tag in raw:
                entities.setdefault(dim, []).append(tag)

        # Build dimtag list for transforms
        dimtags = [
            (d, t) for d, tags in entities.items() for t in tags
        ]

        # ── Rotate ────────────────────────────────────────────────
        if rotate is not None and dimtags:
            if len(rotate) == 4:
                angle, ax, ay, az = rotate
                cx = cy = cz = 0.0
            elif len(rotate) == 7:
                angle, ax, ay, az, cx, cy, cz = rotate
            else:
                raise ValueError(
                    "rotate must be (angle, ax, ay, az) or "
                    "(angle, ax, ay, az, cx, cy, cz)"
                )
            gmsh.model.occ.rotate(dimtags, cx, cy, cz, ax, ay, az, angle)
            gmsh.model.occ.synchronize()

        # ── Translate ─────────────────────────────────────────────
        dx, dy, dz = translate
        if (dx != 0.0 or dy != 0.0 or dz != 0.0) and dimtags:
            gmsh.model.occ.translate(dimtags, dx, dy, dz)
            gmsh.model.occ.synchronize()

        inst = Instance(
            label=label,
            part_name=file_path.stem,
            file_path=file_path.resolve(),
            entities=entities,
            translate=(dx, dy, dz),
            rotate=rotate,
            properties=properties or {},
            bbox=self._compute_bbox(dimtags),
        )
        self.instances[label] = inst
        return inst

    # ------------------------------------------------------------------
    # MSH import helper
    # ------------------------------------------------------------------

    @staticmethod
    def _import_msh(file_path: Path) -> dict[int, list[int]]:
        """
        Import a ``.msh`` file using ``gmsh.merge``.

        Unlike ``occ.importShapes``, ``merge`` preserves:
        * Physical groups and their names
        * Existing mesh (nodes + elements)
        * Partition data

        Returns
        -------
        dict[int, list[int]]
            ``{dim: [tag, ...]}`` of all entities present after merge.
        """
        # Record entities before merge so we can identify the new ones
        pre_entities: set[tuple[int, int]] = set()
        for d in range(4):
            for dim, tag in gmsh.model.getEntities(d):
                pre_entities.add((dim, tag))

        gmsh.merge(str(file_path))

        # Collect new entities
        entities: dict[int, list[int]] = {}
        for d in range(4):
            for dim, tag in gmsh.model.getEntities(d):
                if (dim, tag) not in pre_entities:
                    entities.setdefault(dim, []).append(tag)

        # If nothing was "new" (first import into empty model),
        # collect everything
        if not entities:
            for d in range(4):
                for dim, tag in gmsh.model.getEntities(d):
                    entities.setdefault(dim, []).append(tag)

        return entities

    # ------------------------------------------------------------------
    # Fragment / Boolean
    # ------------------------------------------------------------------

    def fragment_all(self, *, dim: int | None = None) -> list[int]:
        """
        Fragment **every** entity in the assembly so all interfaces
        become conformal.

        This is the most robust approach: after fragmentation, shared
        surfaces between adjacent parts have matching nodes, producing
        a single continuous FE mesh.

        Parameters
        ----------
        dim : int, optional
            Entity dimension to fragment.  If ``None``, uses the
            highest dimension present (3 for volumes, 2 for surfaces).

        Returns
        -------
        list[int]
            Tags of all surviving entities at the target dimension.
        """
        if not self._active:
            raise RuntimeError("Assembly session is not active.")

        # Determine working dimension
        if dim is None:
            for d in (3, 2, 1):
                ents = gmsh.model.getEntities(d)
                if ents:
                    dim = d
                    break
            else:
                raise RuntimeError("No entities found in the assembly.")

        all_ents = gmsh.model.getEntities(dim)
        if len(all_ents) < 2:
            return [t for _, t in all_ents]

        obj  = [all_ents[0]]
        tool = list(all_ents[1:])

        # input_ents: obj + tool, in the order fragment() expects
        input_ents = obj + tool

        result, result_map = gmsh.model.occ.fragment(
            obj, tool,
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        # Build old-tag → set-of-new-tags mapping for this dimension
        old_to_new: dict[int, list[int]] = {}
        for old_dt, new_dts in zip(input_ents, result_map):
            old_dim, old_tag = old_dt
            if old_dim == dim:
                old_to_new[old_tag] = [t for d, t in new_dts if d == dim]

        # Update each Instance.entities so they track post-fragment tags
        for inst in self.instances.values():
            old_tags = inst.entities.get(dim, [])
            new_tags: list[int] = []
            for ot in old_tags:
                if ot in old_to_new:
                    new_tags.extend(old_to_new[ot])
                else:
                    # Entity wasn't part of this fragment (different dim)
                    new_tags.append(ot)
            inst.entities[dim] = new_tags

        surviving = [t for _, t in result]
        return surviving

    def fragment_pair(
        self,
        label_a: str,
        label_b: str,
        *,
        dim: int | None = None,
    ) -> list[int]:
        """
        Fragment only two specific instances against each other.

        Useful when you want conformal mesh at one interface but not
        everywhere (e.g. two plates sharing an edge but a third plate
        kept separate).

        Parameters
        ----------
        label_a, label_b : str
            Instance labels to fragment.
        dim : int, optional
            Entity dimension (auto-detected if ``None``).

        Returns
        -------
        list[int]
            Tags of all surviving entities at the target dimension.
        """
        if not self._active:
            raise RuntimeError("Assembly session is not active.")

        inst_a = self.instances[label_a]
        inst_b = self.instances[label_b]

        if dim is None:
            for d in (3, 2, 1):
                if d in inst_a.entities and d in inst_b.entities:
                    dim = d
                    break
            else:
                raise RuntimeError(
                    f"No common entity dimension between "
                    f"'{label_a}' and '{label_b}'."
                )

        obj  = [(dim, t) for t in inst_a.entities.get(dim, [])]
        tool = [(dim, t) for t in inst_b.entities.get(dim, [])]

        result, _ = gmsh.model.occ.fragment(
            obj, tool,
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        surviving = [t for _, t in result]
        return surviving

    # ------------------------------------------------------------------
    # Constraint definitions (Stage 1 — pre-mesh)
    # ------------------------------------------------------------------

    def _check_labels(self, *labels: str) -> None:
        for lbl in labels:
            if lbl not in self.instances:
                raise KeyError(f"Instance '{lbl}' not found.")

    def _add_def(self, defn: ConstraintDef) -> ConstraintDef:
        self._check_labels(defn.master_label, defn.slave_label)
        self.constraint_defs.append(defn)
        return defn

    # ── Level 1: Node-to-Node ────────────────────────────────────

    def equal_dof(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_entities: list[tuple[int, int]] | None = None,
        slave_entities: list[tuple[int, int]] | None = None,
        dofs: list[int] | None = None,
        tolerance: float = 1e-6,
        name: str | None = None,
    ) -> EqualDOFDef:
        """
        Co-located nodes share selected DOFs.

        After meshing + ``resolve_constraints()``, produces one
        :class:`NodePairRecord` per matched node pair.
        """
        return self._add_def(EqualDOFDef(
            master_label=master_label,
            slave_label=slave_label,
            master_entities=master_entities,
            slave_entities=slave_entities,
            dofs=dofs,
            tolerance=tolerance,
            name=name,
        ))

    def rigid_link(
        self,
        master_label: str,
        slave_label: str,
        *,
        link_type: str = "beam",
        master_point: tuple[float, float, float] | None = None,
        slave_entities: list[tuple[int, int]] | None = None,
        tolerance: float = 1e-6,
        name: str | None = None,
    ) -> RigidLinkDef:
        """
        Rigid bar between master and slave nodes.

        ``link_type="beam"``  →  full 6-DOF coupling
        ``link_type="rod"``   →  translations only
        """
        return self._add_def(RigidLinkDef(
            master_label=master_label,
            slave_label=slave_label,
            link_type=link_type,
            master_point=master_point,
            slave_entities=slave_entities,
            tolerance=tolerance,
            name=name,
        ))

    def penalty(
        self,
        master_label: str,
        slave_label: str,
        *,
        stiffness: float = 1e10,
        dofs: list[int] | None = None,
        tolerance: float = 1e-6,
        name: str | None = None,
    ) -> PenaltyDef:
        """Soft spring between co-located node pairs."""
        return self._add_def(PenaltyDef(
            master_label=master_label,
            slave_label=slave_label,
            stiffness=stiffness,
            dofs=dofs,
            tolerance=tolerance,
            name=name,
        ))

    # ── Level 2: Node-to-Group ───────────────────────────────────

    def rigid_diaphragm(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        constrained_dofs: list[int] | None = None,
        plane_tolerance: float = 1.0,
        name: str | None = None,
    ) -> RigidDiaphragmDef:
        """
        In-plane rigid body constraint (e.g. floor diaphragm).
        """
        return self._add_def(RigidDiaphragmDef(
            master_label=master_label,
            slave_label=slave_label,
            master_point=master_point,
            plane_normal=plane_normal,
            constrained_dofs=constrained_dofs or [1, 2, 6],
            plane_tolerance=plane_tolerance,
            name=name,
        ))

    def rigid_body(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str | None = None,
    ) -> RigidBodyDef:
        """
        All 6 DOFs of every slave node follow the master rigidly.
        """
        return self._add_def(RigidBodyDef(
            master_label=master_label,
            slave_label=slave_label,
            master_point=master_point,
            name=name,
        ))

    def kinematic_coupling(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dofs: list[int] | None = None,
        name: str | None = None,
    ) -> KinematicCouplingDef:
        """
        Generalised coupling: user picks which DOFs are constrained.
        """
        return self._add_def(KinematicCouplingDef(
            master_label=master_label,
            slave_label=slave_label,
            master_point=master_point,
            dofs=dofs or [1, 2, 3, 4, 5, 6],
            name=name,
        ))

    # ── Level 3: Node-to-Surface ─────────────────────────────────

    def tie(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_entities: list[tuple[int, int]] | None = None,
        slave_entities: list[tuple[int, int]] | None = None,
        dofs: list[int] | None = None,
        tolerance: float = 1.0,
        name: str | None = None,
    ) -> TieDef:
        """
        Surface tie via shape function interpolation.

        Each slave node is projected onto the closest master element
        face.  Produces :class:`InterpolationRecord` objects.
        """
        return self._add_def(TieDef(
            master_label=master_label,
            slave_label=slave_label,
            master_entities=master_entities,
            slave_entities=slave_entities,
            dofs=dofs,
            tolerance=tolerance,
            name=name,
        ))

    def distributing_coupling(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dofs: list[int] | None = None,
        weighting: str = "uniform",
        name: str | None = None,
    ) -> DistributingCouplingDef:
        """
        Load at master distributed to slave surface.
        """
        return self._add_def(DistributingCouplingDef(
            master_label=master_label,
            slave_label=slave_label,
            master_point=master_point,
            dofs=dofs,
            weighting=weighting,
            name=name,
        ))

    def embedded(
        self,
        host_label: str,
        embedded_label: str,
        *,
        tolerance: float = 1.0,
        name: str | None = None,
    ) -> EmbeddedDef:
        """
        Embedded element: nodes of embedded instance follow host field.
        """
        return self._add_def(EmbeddedDef(
            master_label=host_label,
            slave_label=embedded_label,
            tolerance=tolerance,
            name=name,
        ))

    # ── Level 4: Surface-to-Surface ──────────────────────────────

    def tied_contact(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_entities: list[tuple[int, int]] | None = None,
        slave_entities: list[tuple[int, int]] | None = None,
        dofs: list[int] | None = None,
        tolerance: float = 1.0,
        name: str | None = None,
    ) -> TiedContactDef:
        """
        Full surface-to-surface tie (bidirectional projection).
        """
        return self._add_def(TiedContactDef(
            master_label=master_label,
            slave_label=slave_label,
            master_entities=master_entities,
            slave_entities=slave_entities,
            dofs=dofs,
            tolerance=tolerance,
            name=name,
        ))

    def mortar(
        self,
        master_label: str,
        slave_label: str,
        *,
        master_entities: list[tuple[int, int]] | None = None,
        slave_entities: list[tuple[int, int]] | None = None,
        dofs: list[int] | None = None,
        integration_order: int = 2,
        name: str | None = None,
    ) -> MortarDef:
        """
        Mortar coupling with Lagrange multiplier space.

        .. note::
           Currently uses a simplified node-to-surface projection
           as placeholder.  Full segment-based mortar integration
           is architecturally supported but not yet implemented.
        """
        return self._add_def(MortarDef(
            master_label=master_label,
            slave_label=slave_label,
            master_entities=master_entities,
            slave_entities=slave_entities,
            dofs=dofs,
            integration_order=integration_order,
            name=name,
        ))

    # ------------------------------------------------------------------
    # Constraint resolution (Stage 2 — post-mesh)
    # ------------------------------------------------------------------

    def resolve_constraints(
        self,
        node_tags,
        node_coords,
        elem_tags=None,
        connectivity=None,
        *,
        instance_node_map: dict[str, set[int]] | None = None,
        instance_face_map: dict[str, "np.ndarray"] | None = None,
    ) -> list[ConstraintRecord]:
        """
        Resolve all stored constraint definitions into concrete records.

        Call this **after** meshing.  Pass the mesh data extracted from
        ``get_fem_data()`` or equivalent.

        Parameters
        ----------
        node_tags : array-like
            Node tags from the mesh.
        node_coords : array-like, shape (n, 3)
            Nodal coordinates.
        elem_tags : array-like, optional
            Element tags (needed for Level 3+).
        connectivity : array-like, optional
            Element connectivity (needed for Level 3+).
        instance_node_map : dict, optional
            ``{instance_label: {node_tags...}}``.  If not provided,
            inferred automatically from instance bounding boxes.
        instance_face_map : dict, optional
            ``{instance_label: face_connectivity_array}``.
            Needed for Level 3+ (tie, mortar).

        Returns
        -------
        list[ConstraintRecord]
            Resolved records ready for solver consumption.
        """
        resolver = ConstraintResolver(
            node_tags=node_tags,
            node_coords=node_coords,
            elem_tags=elem_tags,
            connectivity=connectivity,
        )

        all_nodes = set(int(t) for t in node_tags)
        records: list[ConstraintRecord] = []

        if instance_node_map is None:
            instance_node_map = self._build_instance_node_map(node_tags, node_coords)

        needs_faces = any(
            isinstance(defn, (TieDef, TiedContactDef, MortarDef))
            for defn in self.constraint_defs
        )
        if needs_faces and instance_face_map is None:
            instance_face_map = self._build_instance_face_map(instance_node_map)

        for defn in self.constraint_defs:
            ml = defn.master_label
            sl = defn.slave_label

            m_nodes = self._resolve_constraint_nodes(
                ml, "master", defn, instance_node_map, all_nodes
            )
            s_nodes = self._resolve_constraint_nodes(
                sl, "slave", defn, instance_node_map, all_nodes
            )

            if isinstance(defn, EqualDOFDef):
                records.extend(
                    resolver.resolve_equal_dof(defn, m_nodes, s_nodes)
                )

            elif isinstance(defn, RigidLinkDef):
                records.extend(
                    resolver.resolve_rigid_link(defn, m_nodes, s_nodes)
                )

            elif isinstance(defn, PenaltyDef):
                records.extend(
                    resolver.resolve_penalty(defn, m_nodes, s_nodes)
                )

            elif isinstance(defn, RigidDiaphragmDef):
                combined = m_nodes | s_nodes
                records.append(
                    resolver.resolve_rigid_diaphragm(defn, combined)
                )

            elif isinstance(defn, (RigidBodyDef, KinematicCouplingDef)):
                records.append(
                    resolver.resolve_kinematic_coupling(defn, m_nodes, s_nodes)
                )

            elif isinstance(defn, TieDef):
                m_faces = self._resolve_constraint_faces(
                    ml, "master", defn, instance_face_map or {}
                )
                if m_faces.size:
                    records.extend(
                        resolver.resolve_tie(defn, m_faces, s_nodes)
                    )

            elif isinstance(defn, DistributingCouplingDef):
                records.append(
                    resolver.resolve_distributing(defn, m_nodes, s_nodes)
                )

            elif isinstance(defn, TiedContactDef):
                m_faces = self._resolve_constraint_faces(
                    ml, "master", defn, instance_face_map or {}
                )
                s_faces = self._resolve_constraint_faces(
                    sl, "slave", defn, instance_face_map or {}
                )
                if m_faces.size and s_faces.size:
                    records.append(
                        resolver.resolve_tied_contact(
                            defn, m_faces, s_faces, m_nodes, s_nodes,
                        )
                    )

            elif isinstance(defn, MortarDef):
                m_faces = self._resolve_constraint_faces(
                    ml, "master", defn, instance_face_map or {}
                )
                s_faces = self._resolve_constraint_faces(
                    sl, "slave", defn, instance_face_map or {}
                )
                if m_faces.size and s_faces.size:
                    records.append(
                        resolver.resolve_mortar(
                            defn, m_faces, s_faces, m_nodes, s_nodes,
                        )
                    )

            elif isinstance(defn, EmbeddedDef):
                raise NotImplementedError(
                    "Embedded constraint resolution is not implemented yet."
                )

        self.constraint_records = records
        return records

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_instance(self, label: str) -> Instance:
        """Return an Instance by label."""
        return self.instances[label]

    def list_instances(self) -> list[str]:
        """Return all instance labels in insertion order."""
        return list(self.instances.keys())

    def list_constraint_defs(self) -> list[dict]:
        """Return constraint definitions as a list of dicts."""
        return [
            {"kind": d.kind, "master": d.master_label,
             "slave": d.slave_label, "name": d.name}
            for d in self.constraint_defs
        ]

    def list_constraint_records(self) -> list[dict]:
        """Return resolved records as a list of dicts (after resolve)."""
        out = []
        for r in self.constraint_records:
            d = {"kind": r.kind, "name": r.name}
            if hasattr(r, 'master_node'):
                d["master_node"] = r.master_node
            if hasattr(r, 'slave_node'):
                d["slave_node"] = r.slave_node
            if hasattr(r, 'slave_nodes'):
                d["n_slaves"] = len(r.slave_nodes)
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def add_physical_groups_from_instances(self, dim: int | None = None) -> dict[str, int]:
        """
        Create one physical group per instance using the instance label
        as the group name.

        Must be called **after** ``fragment_all()`` so that instance
        entity tags are up to date.

        Parameters
        ----------
        dim : int, optional
            Entity dimension for the groups.  If ``None``, uses the
            highest dimension present in any instance.

        Returns
        -------
        dict[str, int]
            ``{instance_label: physical_group_tag}``.
        """
        if dim is None:
            for d in (3, 2, 1, 0):
                if any(d in inst.entities for inst in self.instances.values()):
                    dim = d
                    break
            else:
                raise RuntimeError("No entities found in any instance.")

        groups: dict[str, int] = {}
        for label, inst in self.instances.items():
            tags = inst.entities.get(dim, [])
            if tags:
                pg = gmsh.model.addPhysicalGroup(dim, tags, name=label)
                groups[label] = pg
        return groups

    @property
    def is_active(self) -> bool:
        return self._active

    def __repr__(self) -> str:
        n = len(self.instances)
        c = len(self.constraint_defs)
        r = len(self.constraint_records)
        status = "active" if self._active else "closed"
        return (
            f"Assembly('{self.name}', {status}, "
            f"{n} instance{'s' if n != 1 else ''}, "
            f"{c} constraint def{'s' if c != 1 else ''}, "
            f"{r} resolved record{'s' if r != 1 else ''})"
        )
