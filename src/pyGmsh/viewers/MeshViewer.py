"""
MeshViewer
==========

Interactive mesh viewer with dual-level picking (mesh nodes vs BRep patches),
multiple coloring modes, label overlays, and selection sets.

Renders a Gmsh mesh (nodes + elements) using one VTK actor per BRep entity,
enabling BRep-level hide/isolate operations.  At the mesh level, individual
node picking is supported via nearest-node search.

Usage
-----
::

    with pyGmsh(model_name="Example") as g:
        g.model.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.generate(3)
        g.mesh.viewer()

Keyboard
--------
    [M]  switch to mesh-level picking (nodes)
    [B]  switch to BRep-level picking (patches)
    [H]  hide selected
    [I]  isolate selected
    [R]  reveal all
    [Q]  close window
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np
import pyvista as pv

from .BaseViewer import BaseViewer, DimTag, _hex_to_rgb

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase
    from pyGmsh.mesh.Mesh import Mesh


# ======================================================================
# Gmsh element type -> VTK cell type mapping
# ======================================================================

_GMSH_TO_VTK = {
    1:  3,    # 2-node line          -> VTK_LINE
    2:  5,    # 3-node triangle      -> VTK_TRIANGLE
    3:  9,    # 4-node quad          -> VTK_QUAD
    4:  10,   # 4-node tet           -> VTK_TETRA
    5:  12,   # 8-node hex           -> VTK_HEXAHEDRON
    6:  13,   # 6-node prism         -> VTK_WEDGE
    7:  14,   # 5-node pyramid       -> VTK_PYRAMID
    8:  21,   # 6-node tri (2nd)     -> VTK_QUADRATIC_TRIANGLE
    9:  23,   # 8-node quad (2nd)    -> VTK_QUADRATIC_QUAD
    11: 24,   # 10-node tet (2nd)    -> VTK_QUADRATIC_TETRA
    15: 1,    # 1-node point         -> VTK_VERTEX
}

# Maximum number of labels rendered before camera-distance culling kicks in.
MAX_LABELS = 2000

# ======================================================================
# Colour palettes
# ======================================================================

_PARTITION_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

_ELEM_TYPE_COLORS = {
    "Triangle":      "#4363d8",
    "Quadrilateral": "#3cb44b",
    "Tetrahedron":   "#e6194b",
    "Hexahedron":    "#f58231",
    "Prism":         "#911eb4",
    "Pyramid":       "#42d4f4",
    "Line":          "#aaaaaa",
    "Point":         "#ffffff",
}

_DEFAULT_MESH_COLOR = "#5B8DB8"
_PICK_COLOR         = "#E74C3C"
_HOVER_COLOR        = "#FFD700"
_NODE_COLOR         = "#E8D5B7"
_EDGE_COLOR         = "#2C4A6E"


# ======================================================================
# Element-type name normalisation
# ======================================================================

def _elem_type_category(name: str) -> str:
    """Map a Gmsh element type name to a colour-palette key."""
    low = name.lower()
    if "triangle" in low:
        return "Triangle"
    if "quadrilateral" in low or "quad" in low:
        return "Quadrilateral"
    if "tetrahedron" in low:
        return "Tetrahedron"
    if "hexahedron" in low:
        return "Hexahedron"
    if "prism" in low:
        return "Prism"
    if "pyramid" in low:
        return "Pyramid"
    if "line" in low:
        return "Line"
    if "point" in low:
        return "Point"
    return "Line"


# ======================================================================
# MeshViewer
# ======================================================================

class MeshViewer(BaseViewer):
    """
    Interactive mesh viewer with dual-level picking.

    Builds one VTK actor per BRep entity (grouping all elements on that
    entity into a single ``pyvista.UnstructuredGrid``), plus a single
    point-cloud actor for all mesh nodes.

    Two pick levels are supported:

    * **mesh** -- clicking picks the nearest mesh node.
    * **brep** -- clicking picks the BRep entity (surface/volume/curve)
      that owns the element under the cursor.

    Parameters
    ----------
    parent : _SessionBase
        Owning session instance.
    mesh_composite : Mesh
        The ``Mesh`` composite -- used for renumbering access.
    dims : list[int] or None
        Dimensions to render.  Default ``[1, 2, 3]``.
    point_size : float
        Visual size for the node point cloud.
    line_width : float
        VTK line width for 1-D element actors.
    surface_opacity : float
        Opacity for 2-D / 3-D element actors.
    show_surface_edges : bool
        Whether to draw mesh edges on surface actors.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        mesh_composite: "Mesh",
        *,
        dims: list[int] | None = None,
        point_size: float = 6.0,
        line_width: float = 3.0,
        surface_opacity: float = 1.0,
        show_surface_edges: bool = True,
    ) -> None:
        super().__init__(
            parent, parent.model,
            dims=dims if dims is not None else [1, 2, 3],
            point_size=point_size,
            line_width=line_width,
            surface_opacity=surface_opacity,
            show_surface_edges=show_surface_edges,
        )
        self._mesh = mesh_composite
        self._pick_level: str = "mesh"  # "mesh" or "brep"

        # ---- Mesh data (populated in _build_scene) ----
        self._node_tags: np.ndarray | None = None       # (N,) node tags
        self._node_coords: np.ndarray | None = None      # (N, 3) xyz
        self._node_tag_to_idx: dict[int, int] = {}       # tag -> row index

        # Element bookkeeping
        self._elem_data: dict[int, dict[str, Any]] = {}  # elem_tag -> info dict
        # {type_name, nodes, dim, brep_dt}

        # ---- BRep-level actors: one VTK actor per BRep entity ----
        self._brep_actors: dict[DimTag, Any] = {}
        self._brep_grids: dict[DimTag, Any] = {}
        self._brep_to_elems: dict[DimTag, list[int]] = {}  # BRep -> elem tags
        self._elem_to_brep: dict[int, DimTag] = {}          # elem -> parent BRep

        # Per-BRep dominant element type (for type-coloring)
        self._brep_dominant_type: dict[DimTag, str] = {}

        # ---- Node-level actor ----
        self._node_actor: Any = None  # single point cloud

        # ---- Visual properties for mesh ----
        self._node_marker_size: float = point_size
        self._edge_color: str = "#2C4A6E"

        # ---- Element-level picking ----
        # Maps (brep_dt, cell_idx_in_grid) -> gmsh elem tag
        self._grid_cell_to_elem: dict[tuple[DimTag, int], int] = {}

        # ---- Physical-group level picking ----
        self._group_to_breps: dict[str, list[DimTag]] = {}  # group name -> BRep entities
        self._brep_to_group: dict[DimTag, str] = {}          # BRep -> first group name
        self._selected_groups: list[str] = []

        # ---- Selection sets ----
        self._selection_sets: dict[str, list[int]] = {}  # name -> tags
        self._selected_nodes: list[int] = []
        self._selected_elems: list[int] = []  # gmsh element tags

        # ---- Color mode ----
        self._color_mode: str = "default"
        # "default", "partition", "quality", "type", "group"

        # ---- Label actors ----
        self._node_label_actor: object | None = None
        self._elem_label_actor: object | None = None

        # ---- Own cell picker ----
        import vtk
        self._click_picker = vtk.vtkCellPicker()
        self._click_picker.SetTolerance(0.005)

        # ---- Opt-1: vectorized tag-to-idx lookup array ----
        self._tag_to_idx: np.ndarray | None = None  # dense lookup, O(max_tag)

        # ---- Opt-2: KD-tree for nearest-node picking ----
        self._node_tree: Any = None  # scipy cKDTree (or None)

        # ---- Opt-3: persistent node cloud PolyData ----
        self._node_cloud: pv.PolyData | None = None

        # ---- Opt-7: cached gmsh API calls ----
        self._cached_elem_props: dict[int, tuple] = {}
        self._cached_qualities: dict[DimTag, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Gmsh element-property cache (Opt-7)
    # ------------------------------------------------------------------

    def _get_elem_props(self, etype: int) -> tuple:
        """Return cached ``gmsh.model.mesh.getElementProperties(etype)``."""
        cached = self._cached_elem_props.get(etype)
        if cached is not None:
            return cached
        props = gmsh.model.mesh.getElementProperties(int(etype))
        self._cached_elem_props[etype] = props
        return props

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:  # noqa: C901
        """Populate the plotter with one actor per BRep entity plus a
        node point cloud."""
        plotter = self._plotter

        # 1. Get all mesh nodes
        node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        node_coords = np.asarray(node_coords_flat, dtype=np.float64).reshape(-1, 3)

        self._node_tags = node_tags
        self._node_coords = node_coords
        self._node_tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # --- Opt-1: build dense tag_to_idx lookup array ---
        if len(node_tags) > 0:
            max_tag = int(node_tags.max())
            tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
            tag_to_idx[node_tags] = np.arange(len(node_tags), dtype=np.int64)
            self._tag_to_idx = tag_to_idx
        else:
            self._tag_to_idx = np.array([], dtype=np.int64)

        # Pre-compute the default BRep color as uint8 for grid cell_data
        _default_rgb_u8 = np.array(
            [int(c * 255) for c in _hex_to_rgb(_DEFAULT_MESH_COLOR)],
            dtype=np.uint8,
        )

        # 2. For each BRep entity with dim in self._dims, build actors
        for dim in sorted(self._dims):
            for _, tag in gmsh.model.getEntities(dim=dim):
                dt: DimTag = (dim, tag)
                try:
                    elem_types, elem_tags_list, elem_node_tags_list = (
                        gmsh.model.mesh.getElements(dim, tag)
                    )
                except Exception:
                    continue

                if not elem_types:
                    continue

                # Collect VTK cells, cell types, and elem tags across all
                # element-type blocks on this BRep entity.
                all_cells_parts: list[np.ndarray] = []
                all_cell_types_parts: list[np.ndarray] = []
                brep_elem_tags: list[int] = []
                dominant_type_name = ""
                dominant_count = 0

                for etype, etags, enodes in zip(
                    elem_types, elem_tags_list, elem_node_tags_list,
                ):
                    vtk_type = _GMSH_TO_VTK.get(int(etype))
                    if vtk_type is None:
                        continue

                    # --- Opt-7: cached element properties ---
                    props = self._get_elem_props(int(etype))
                    type_name: str = props[0]
                    n_nodes: int = props[3]

                    etags_arr = np.asarray(etags, dtype=np.int64)
                    enodes_arr = np.asarray(enodes, dtype=np.int64)
                    n_elems = len(etags_arr)

                    if n_elems == 0:
                        continue

                    # Track dominant element type on this BRep entity
                    if n_elems > dominant_count:
                        dominant_count = n_elems
                        dominant_type_name = type_name

                    # --- Opt-1: vectorized cell construction ---
                    node_rows = enodes_arr.reshape(n_elems, n_nodes)

                    # Check that all node tags are within the lookup table
                    # and map to valid indices.
                    if self._tag_to_idx is not None and len(self._tag_to_idx) > 0:
                        max_allowed = len(self._tag_to_idx) - 1
                        in_range = np.all(node_rows <= max_allowed) and np.all(
                            node_rows >= 0,
                        )
                        if in_range:
                            idx_arr = self._tag_to_idx[node_rows]
                            valid_mask = np.all(idx_arr >= 0, axis=1)
                        else:
                            # Fallback: clip and check
                            clipped = np.clip(node_rows, 0, max_allowed)
                            idx_arr = self._tag_to_idx[clipped]
                            valid_mask = (
                                np.all(idx_arr >= 0, axis=1)
                                & np.all(node_rows >= 0, axis=1)
                                & np.all(node_rows <= max_allowed, axis=1)
                            )
                    else:
                        idx_arr = np.zeros_like(node_rows)
                        valid_mask = np.zeros(n_elems, dtype=bool)

                    valid_idx_arr = idx_arr[valid_mask]
                    valid_etags = etags_arr[valid_mask]
                    n_valid = int(valid_mask.sum())

                    if n_valid == 0:
                        continue

                    # Build VTK cells array: [n_nodes, idx0, idx1, ...] per row
                    prefix_col = np.full(
                        (n_valid, 1), n_nodes, dtype=np.int64,
                    )
                    cell_block = np.hstack([prefix_col, valid_idx_arr])
                    all_cells_parts.append(cell_block.ravel())
                    all_cell_types_parts.append(
                        np.full(n_valid, vtk_type, dtype=np.uint8),
                    )

                    # Extend elem tag list and build _elem_data / _elem_to_brep
                    valid_node_rows = node_rows[valid_mask]
                    for ri in range(n_valid):
                        elem_tag = int(valid_etags[ri])
                        brep_elem_tags.append(elem_tag)
                        self._elem_to_brep[elem_tag] = dt
                        self._elem_data[elem_tag] = {
                            "type_name": type_name,
                            "nodes": valid_node_rows[ri].tolist(),
                            "dim": dim,
                            "brep_dt": dt,
                        }

                if not all_cells_parts:
                    continue

                # Map grid cell index -> gmsh element tag for element picking
                for cell_idx, etag in enumerate(brep_elem_tags):
                    self._grid_cell_to_elem[(dt, cell_idx)] = etag

                self._brep_to_elems[dt] = brep_elem_tags
                self._brep_dominant_type[dt] = _elem_type_category(
                    dominant_type_name,
                )

                cells_flat = np.concatenate(all_cells_parts)
                cell_types_flat = np.concatenate(all_cell_types_parts)

                grid = pv.UnstructuredGrid(
                    cells_flat,
                    cell_types_flat,
                    node_coords,
                )
                self._brep_grids[dt] = grid

                # --- Opt-4: pre-allocate per-cell RGB so mapper is configured ---
                default_colors = np.tile(
                    _default_rgb_u8, (grid.n_cells, 1),
                )
                grid["colors"] = default_colors

                # Visual properties depend on dimension
                show_edges = self._show_surface_edges and dim >= 2
                opacity = self._surface_opacity if dim >= 2 else 1.0

                actor = plotter.add_mesh(
                    grid,
                    scalars="colors",
                    rgb=True,
                    opacity=opacity,
                    show_edges=show_edges,
                    edge_color=_EDGE_COLOR,
                    line_width=self._line_width if dim == 1 else 0.5,
                    render_lines_as_tubes=(dim == 1),
                    smooth_shading=False,
                    pickable=True,
                )
                self._brep_actors[dt] = actor
                self._register_actor(actor, ("brep", dim, tag))

        # 3. Node point cloud -- Opt-3: persistent PolyData with pre-set scalars
        if self._node_coords is not None and len(self._node_coords) > 0:
            node_cloud = pv.PolyData(self._node_coords)
            n_nodes = len(self._node_coords)
            default_node_rgb = np.array(
                [int(c * 255) for c in _hex_to_rgb(_NODE_COLOR)],
                dtype=np.uint8,
            )
            node_cloud["pick_colors"] = np.tile(
                default_node_rgb, (n_nodes, 1),
            )
            self._node_cloud = node_cloud
            self._node_actor = plotter.add_mesh(
                node_cloud,
                scalars="pick_colors",
                rgb=True,
                point_size=self._node_marker_size,
                render_points_as_spheres=True,
                pickable=True,
                opacity=0.8,
            )
            self._register_actor(self._node_actor, ("nodes",))

        # --- Opt-2: build KD-tree for nearest-node picking ---
        if self._node_coords is not None and len(self._node_coords) > 0:
            try:
                from scipy.spatial import cKDTree
                self._node_tree = cKDTree(self._node_coords)
            except ImportError:
                self._node_tree = None

        # 4. Build physical-group <-> BRep mappings
        self._group_to_breps.clear()
        self._brep_to_group.clear()
        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                pg_name = f"Group_{pg_dim}_{pg_tag}"
            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
            breps = [(pg_dim, int(t)) for t in ents]
            self._group_to_breps[pg_name] = breps
            for dt in breps:
                if dt not in self._brep_to_group:
                    self._brep_to_group[dt] = pg_name

        # 5. Apply initial coloring
        if self._color_mode != "default":
            self._apply_coloring()

    # ------------------------------------------------------------------
    # Keybindings
    # ------------------------------------------------------------------

    def _install_keybindings(self) -> None:
        plotter = self._plotter
        plotter.add_key_event("q", lambda: plotter.close())
        plotter.add_key_event("Q", lambda: plotter.close())
        plotter.add_key_event("h", self._hide_selected)
        plotter.add_key_event("H", self._hide_selected)
        plotter.add_key_event("i", self._isolate_selected)
        plotter.add_key_event("I", self._isolate_selected)
        plotter.add_key_event("r", self._show_all)
        plotter.add_key_event("R", self._show_all)
        plotter.add_key_event("m", lambda: self._set_pick_level("mesh"))
        plotter.add_key_event("M", lambda: self._set_pick_level("mesh"))
        plotter.add_key_event("g", lambda: self._set_pick_level("group"))
        plotter.add_key_event("G", lambda: self._set_pick_level("group"))

    # ------------------------------------------------------------------
    # Pick level
    # ------------------------------------------------------------------

    def _set_pick_level(self, level: str) -> None:
        """Switch between ``"mesh"`` (node/element) and ``"group"`` picking."""
        self._pick_level = level
        self._update_status()
        self._fire_pick_changed()

    # ------------------------------------------------------------------
    # Picker hooks (called by BaseViewer._install_picker)
    # ------------------------------------------------------------------

    def _on_lmb_click(self, x: int, y: int, ctrl: bool) -> None:
        """Handle a left-mouse-button click at display coords *(x, y)*."""

        renderer = self._plotter.renderer
        self._click_picker.Pick(x, y, 0, renderer)
        prop = self._click_picker.GetViewProp()
        if prop is None:
            return

        entity_id = self._actor_to_id.get(id(prop))
        if entity_id is None:
            return

        if self._pick_level == "group":
            # Group-level: find which physical group the clicked BRep belongs to
            if isinstance(entity_id, tuple) and entity_id[0] == "brep":
                dt: DimTag = (entity_id[1], entity_id[2])
                group_name = self._brep_to_group.get(dt)
                if group_name is not None:
                    if ctrl:
                        if group_name in self._selected_groups:
                            self._selected_groups.remove(group_name)
                    else:
                        if group_name in self._selected_groups:
                            self._selected_groups.remove(group_name)
                        else:
                            self._selected_groups.append(group_name)
                    self._recolor_group(group_name)
                    # --- Opt-6: single render at end of handler ---
                    self._plotter.render()
                    self._update_status()
                    self._fire_pick_changed()

        elif self._pick_level == "mesh":
            # Mesh-level: identify which element cell was clicked
            cell_id = self._click_picker.GetCellId()
            if cell_id >= 0 and isinstance(entity_id, tuple) and entity_id[0] == "brep":
                brep_dt: DimTag = (entity_id[1], entity_id[2])
                elem_tag = self._grid_cell_to_elem.get((brep_dt, cell_id))
                if elem_tag is not None:
                    if ctrl:
                        if elem_tag in self._selected_elems:
                            self._selected_elems.remove(elem_tag)
                    else:
                        if elem_tag in self._selected_elems:
                            self._selected_elems.remove(elem_tag)
                        else:
                            self._selected_elems.append(elem_tag)
                    self._update_elem_highlight()
                    self._update_status()
                    self._fire_pick_changed()
                    return

            # Fallback: nearest node picking (for node cloud clicks)
            pos = self._click_picker.GetPickPosition()
            nearest = self._find_nearest_node(pos)
            if nearest is not None:
                if ctrl:
                    if nearest in self._selected_nodes:
                        self._selected_nodes.remove(nearest)
                else:
                    if nearest in self._selected_nodes:
                        self._selected_nodes.remove(nearest)
                    else:
                        self._selected_nodes.append(nearest)
                self._update_node_highlight()
                self._update_status()
                self._fire_pick_changed()

    def _on_box_select(
        self,
        x0: int, y0: int, x1: int, y1: int,
        crossing: bool,
        ctrl: bool,
    ) -> None:
        """Box-select at the current pick level."""
        # DPI scaling
        try:
            rw = self._plotter.render_window
            vw, vh = rw.GetSize()
            aw, ah = rw.GetActualSize()
            sx_ratio = aw / vw if vw else 1.0
            sy_ratio = ah / vh if vh else 1.0
        except Exception:
            sx_ratio = sy_ratio = 1.0
        bx0, bx1 = x0 * sx_ratio, x1 * sx_ratio
        by0, by1 = y0 * sy_ratio, y1 * sy_ratio

        if self._pick_level == "mesh":
            self._box_select_nodes(bx0, by0, bx1, by1, ctrl)
        elif self._pick_level == "group":
            self._box_select_groups(bx0, by0, bx1, by1, ctrl)

    def _box_select_nodes(
        self,
        x0: float, y0: float, x1: float, y1: float,
        ctrl: bool,
    ) -> None:
        """Select/unselect nodes whose screen projection falls in the box."""
        if self._node_coords is None or self._node_tags is None:
            return

        # --- Opt-5: batch projection via camera composite matrix ---
        renderer = self._plotter.renderer
        rw = self._plotter.render_window
        cam = renderer.GetActiveCamera()
        proj_matrix = cam.GetCompositeProjectionTransformMatrix(
            renderer.GetTiledAspectRatio(), 0, 1,
        )
        M = np.array(
            [[proj_matrix.GetElement(i, j) for j in range(4)] for i in range(4)],
            dtype=np.float64,
        )

        n = len(self._node_coords)
        ones = np.ones((n, 1), dtype=np.float64)
        world4 = np.hstack([self._node_coords, ones])
        clip = world4 @ M.T
        w_clip = clip[:, 3].copy()
        w_clip[w_clip == 0] = 1e-12  # avoid div by zero
        ndc = clip[:, :2] / w_clip[:, None]

        w, h = rw.GetSize()
        sx = (ndc[:, 0] + 1.0) * 0.5 * w
        sy = (ndc[:, 1] + 1.0) * 0.5 * h

        mask = (x0 <= sx) & (sx <= x1) & (y0 <= sy) & (sy <= y1)
        hit_indices = np.nonzero(mask)[0]

        changed = False
        selected_set = set(self._selected_nodes)

        for i in hit_indices:
            tag = int(self._node_tags[i])
            if ctrl:
                if tag in selected_set:
                    self._selected_nodes.remove(tag)
                    selected_set.discard(tag)
                    changed = True
            else:
                if tag not in selected_set:
                    self._selected_nodes.append(tag)
                    selected_set.add(tag)
                    changed = True

        if changed:
            self._update_node_highlight()
            self._update_status()
            self._fire_pick_changed()

    def _box_select_groups(
        self,
        x0: float, y0: float, x1: float, y1: float,
        ctrl: bool,
    ) -> None:
        """Select/unselect physical groups -- any BRep actor whose centroid
        falls in the box adds its entire group to the selection."""
        hit_groups: set[str] = set()
        for dt, actor in self._brep_actors.items():
            if dt in self._hidden:
                continue
            pt = self._project_centroid(actor)
            if pt is None:
                continue
            sx, sy = pt
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                gname = self._brep_to_group.get(dt)
                if gname is not None:
                    hit_groups.add(gname)

        changed = False
        for gname in hit_groups:
            if ctrl:
                if gname in self._selected_groups:
                    self._selected_groups.remove(gname)
                    changed = True
            else:
                if gname not in self._selected_groups:
                    self._selected_groups.append(gname)
                    changed = True

        if changed:
            for gname in hit_groups:
                self._recolor_group(gname)
            # --- Opt-6: single render at end ---
            self._plotter.render()
            self._update_status()
            self._fire_pick_changed()

    # ------------------------------------------------------------------
    # Hover
    # ------------------------------------------------------------------

    def _on_hover_changed_internal(self, old_id: Any, new_id: Any) -> None:
        """Recolor actors on hover transition."""
        if old_id is not None and isinstance(old_id, tuple) and old_id[0] == "brep":
            dt = (old_id[1], old_id[2])
            self._recolor_brep(dt)
        if new_id is not None and isinstance(new_id, tuple) and new_id[0] == "brep":
            dt = (new_id[1], new_id[2])
            actor = self._brep_actors.get(dt)
            if actor is not None and dt not in self._picks:
                actor.GetProperty().SetColor(*_hex_to_rgb(_HOVER_COLOR))
                self._plotter.render()

    # ------------------------------------------------------------------
    # Node utilities
    # ------------------------------------------------------------------

    def _find_nearest_node(
        self, world_pos: tuple[float, float, float],
    ) -> int | None:
        """Return the node tag closest to *world_pos*, or ``None``."""
        if self._node_coords is None or len(self._node_coords) == 0:
            return None

        pos = np.asarray(world_pos, dtype=np.float64)

        # --- Opt-2: KD-tree query ---
        if self._node_tree is not None:
            _, idx = self._node_tree.query(pos)
            return int(self._node_tags[idx])

        # Fallback: brute-force numpy
        dists = np.linalg.norm(self._node_coords - pos, axis=1)
        idx = int(np.argmin(dists))
        return int(self._node_tags[idx])

    def _update_node_highlight(self) -> None:
        """Refresh the node point cloud to reflect the current selection.

        Opt-3: in-place scalar update -- no actor rebuild.
        """
        if self._node_cloud is None or self._node_tags is None:
            return

        selected_set = set(self._selected_nodes)
        n = len(self._node_tags)
        default_rgb = np.array(
            [int(c * 255) for c in _hex_to_rgb(_NODE_COLOR)], dtype=np.uint8,
        )
        pick_rgb = np.array(
            [int(c * 255) for c in _hex_to_rgb(_PICK_COLOR)], dtype=np.uint8,
        )
        colors = np.tile(default_rgb, (n, 1))

        if selected_set:
            for i, tag in enumerate(self._node_tags):
                if int(tag) in selected_set:
                    colors[i] = pick_rgb

        # In-place update on persistent PolyData
        self._node_cloud["pick_colors"] = colors
        self._node_cloud.Modified()
        # --- Opt-6: single render at call site ---
        self._plotter.render()

    def _update_elem_highlight(self) -> None:
        """Refresh BRep grid colors to highlight selected elements.

        Opt-4: in-place scalar update -- no actor rebuild.
        """
        selected_set = set(self._selected_elems)
        pick_rgb = np.array(
            [int(c * 255) for c in _hex_to_rgb(_PICK_COLOR)], dtype=np.uint8,
        )
        default_rgb = np.array(
            [int(c * 255) for c in _hex_to_rgb(_DEFAULT_MESH_COLOR)],
            dtype=np.uint8,
        )

        for dt, grid in self._brep_grids.items():
            elem_tags = self._brep_to_elems.get(dt, [])
            if not elem_tags:
                continue
            n_cells = grid.n_cells
            colors = np.tile(default_rgb, (n_cells, 1))
            any_selected = False
            for ci, etag in enumerate(elem_tags):
                if ci < n_cells and etag in selected_set:
                    colors[ci] = pick_rgb
                    any_selected = True

            if any_selected or dt in self._picks:
                grid["colors"] = colors
                grid.Modified()
            else:
                # Revert to uniform default color
                grid["colors"] = np.tile(default_rgb, (n_cells, 1))
                grid.Modified()
                self._recolor_brep(dt, _skip_render=True)

        # --- Opt-6: single render at end ---
        self._plotter.render()

    # ------------------------------------------------------------------
    # BRep-level recoloring
    # ------------------------------------------------------------------

    def _recolor_brep(self, dt: DimTag, *, _skip_render: bool = False) -> None:
        """Set the colour of a single BRep actor based on group selection /
        hover / color-mode state.

        Opt-6: ``_skip_render`` suppresses the per-call ``render()`` so the
        caller can batch many updates and issue a single render.
        """
        actor = self._brep_actors.get(dt)
        if actor is None:
            return
        # Check if this BRep's group is selected
        gname = self._brep_to_group.get(dt)
        if gname and gname in self._selected_groups:
            actor.GetProperty().SetColor(*_hex_to_rgb(_PICK_COLOR))
        else:
            self._apply_coloring_single(dt, actor)
        if not _skip_render:
            self._plotter.render()

    def _recolor_group(self, group_name: str) -> None:
        """Recolor all BRep actors belonging to a physical group.

        Opt-6: individual ``_recolor_brep`` calls skip their own render;
        the caller is responsible for issuing a single ``render()`` after.
        """
        breps = self._group_to_breps.get(group_name, [])
        for dt in breps:
            self._recolor_brep(dt, _skip_render=True)

    def _recolor_all_brep(self) -> None:
        """Recolor every BRep actor."""
        for dt in self._brep_actors:
            self._recolor_brep(dt, _skip_render=True)
        # --- Opt-6: single render ---
        self._plotter.render()

    # ------------------------------------------------------------------
    # Color mode
    # ------------------------------------------------------------------

    def _set_color_mode(self, mode: str) -> None:
        """Switch the color mode and reapply."""
        self._color_mode = mode
        self._apply_coloring()

    def _apply_coloring(self) -> None:
        """Recolor all BRep actors according to the current color mode."""
        if self._color_mode == "partition":
            self._color_by_partition()
        elif self._color_mode == "quality":
            self._color_by_quality()
        elif self._color_mode == "type":
            self._color_by_type()
        elif self._color_mode == "group":
            self._color_by_group()
        else:
            self._color_default()
        # --- Opt-6: single render at end ---
        self._plotter.render()

    def _apply_coloring_single(self, dt: DimTag, actor: object) -> None:
        """Apply the current color mode to a single actor (used after
        un-picking to restore the correct base color)."""
        if self._color_mode == "partition":
            self._color_single_partition(dt, actor)
        elif self._color_mode == "type":
            self._color_single_type(dt, actor)
        elif self._color_mode == "group":
            self._color_single_group(dt, actor)
        else:
            actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))
            actor.GetProperty().SetOpacity(self._surface_opacity)

    # ---- default ----

    def _color_default(self) -> None:
        for dt, actor in self._brep_actors.items():
            if dt not in self._picks:
                actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))
                actor.GetProperty().SetOpacity(self._surface_opacity)

    # ---- partition ----

    def _color_by_partition(self) -> None:
        for dt, actor in self._brep_actors.items():
            if dt not in self._picks:
                self._color_single_partition(dt, actor)

    def _color_single_partition(self, dt: DimTag, actor: object) -> None:
        try:
            parts = gmsh.model.mesh.getPartitions(dt[0], dt[1])
            if parts:
                idx = int(parts[0]) % len(_PARTITION_COLORS)
                actor.GetProperty().SetColor(
                    *_hex_to_rgb(_PARTITION_COLORS[idx]),
                )
            else:
                actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))
        except Exception:
            actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))

    # ---- quality ----

    def _color_by_quality(self) -> None:
        """Scalar-field coloring using element quality."""
        for dt, actor in self._brep_actors.items():
            if dt in self._picks:
                continue
            grid = self._brep_grids.get(dt)
            if grid is None:
                continue
            etags = self._brep_to_elems.get(dt, [])
            if not etags:
                continue
            try:
                # --- Opt-7: lazy quality cache ---
                q_arr = self._cached_qualities.get(dt)
                if q_arr is None:
                    qualities = gmsh.model.mesh.getElementQualities(
                        np.array(etags, dtype=np.int64),
                    )
                    q_arr = np.asarray(qualities, dtype=np.float64)
                    self._cached_qualities[dt] = q_arr

                if len(q_arr) == grid.n_cells:
                    grid.cell_data["quality"] = q_arr
                    # Remove old actor and re-add with scalar coloring
                    try:
                        self._plotter.remove_actor(actor)
                    except Exception:
                        pass
                    new_actor = self._plotter.add_mesh(
                        grid,
                        scalars="quality",
                        cmap="RdYlGn",
                        clim=[0.0, 1.0],
                        show_edges=self._show_surface_edges,
                        edge_color=_EDGE_COLOR,
                        line_width=0.5,
                        pickable=True,
                    )
                    self._brep_actors[dt] = new_actor
                    self._register_actor(new_actor, ("brep", dt[0], dt[1]))
            except Exception:
                pass

    # ---- type ----

    def _color_by_type(self) -> None:
        for dt, actor in self._brep_actors.items():
            if dt not in self._picks:
                self._color_single_type(dt, actor)

    def _color_single_type(self, dt: DimTag, actor: object) -> None:
        cat = self._brep_dominant_type.get(dt, "Line")
        hex_color = _ELEM_TYPE_COLORS.get(cat, _DEFAULT_MESH_COLOR)
        actor.GetProperty().SetColor(*_hex_to_rgb(hex_color))

    # ---- group (physical group membership) ----

    def _color_by_group(self) -> None:
        for dt, actor in self._brep_actors.items():
            if dt not in self._picks:
                self._color_single_group(dt, actor)

    def _color_single_group(self, dt: DimTag, actor: object) -> None:
        """Color a BRep actor by physical-group membership.  Entities
        belonging to a physical group get a palette colour; others stay
        at the default."""
        try:
            groups = gmsh.model.getPhysicalGroups()
        except Exception:
            actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))
            return
        for pg_dim, pg_tag in groups:
            if pg_dim != dt[0]:
                continue
            try:
                ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
                if dt[1] in ents:
                    idx = pg_tag % len(_PARTITION_COLORS)
                    actor.GetProperty().SetColor(
                        *_hex_to_rgb(_PARTITION_COLORS[idx]),
                    )
                    return
            except Exception:
                continue
        actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _toggle_node_labels(self, show: bool) -> None:
        """Show or hide node-tag labels at every mesh node."""
        if self._node_label_actor is not None:
            try:
                self._plotter.remove_actor(self._node_label_actor)
            except Exception:
                pass
            self._node_label_actor = None

        if show and self._node_tags is not None and len(self._node_tags) > 0:
            labels = [str(int(t)) for t in self._node_tags]
            pts = self._node_coords

            # --- Opt-8: label cap ---
            if len(labels) > MAX_LABELS:
                cam_pos = np.array(
                    self._plotter.camera_position[0], dtype=np.float64,
                )
                dists = np.linalg.norm(self._node_coords - cam_pos, axis=1)
                closest = np.argsort(dists)[:MAX_LABELS]
                labels = [labels[i] for i in closest]
                pts = self._node_coords[closest]

            self._node_label_actor = self._plotter.add_point_labels(
                pts,
                labels,
                font_size=8,
                font_family="arial",
                text_color="white",
                show_points=False,
                always_visible=True,
                name="_mesh_node_labels",
            )
        self._plotter.render()

    def _toggle_elem_labels(self, show: bool) -> None:
        """Show or hide element-tag labels at element centroids."""
        if self._elem_label_actor is not None:
            try:
                self._plotter.remove_actor(self._elem_label_actor)
            except Exception:
                pass
            self._elem_label_actor = None

        if show and self._elem_data:
            centroids = []
            labels = []
            for etag, info in self._elem_data.items():
                node_tags = info["nodes"]
                pts = []
                for nt in node_tags:
                    idx = self._node_tag_to_idx.get(nt)
                    if idx is not None:
                        pts.append(self._node_coords[idx])
                if pts:
                    centroid = np.mean(pts, axis=0)
                    centroids.append(centroid)
                    labels.append(str(etag))

            if centroids:
                centroids_arr = np.array(centroids)

                # --- Opt-8: label cap ---
                if len(labels) > MAX_LABELS:
                    cam_pos = np.array(
                        self._plotter.camera_position[0], dtype=np.float64,
                    )
                    dists = np.linalg.norm(
                        centroids_arr - cam_pos, axis=1,
                    )
                    closest = np.argsort(dists)[:MAX_LABELS]
                    labels = [labels[i] for i in closest]
                    centroids_arr = centroids_arr[closest]

                self._elem_label_actor = self._plotter.add_point_labels(
                    centroids_arr,
                    labels,
                    font_size=7,
                    font_family="arial",
                    text_color="yellow",
                    show_points=False,
                    always_visible=True,
                    name="_mesh_elem_labels",
                )
        self._plotter.render()

    # ------------------------------------------------------------------
    # Renumbering
    # ------------------------------------------------------------------

    def _apply_renumbering(self, method: str = "rcmk") -> None:
        """Renumber the mesh and rebuild the scene.

        Delegates to the Mesh composite's renumber methods, then fully
        rebuilds the viewer scene with the new numbering.
        """
        # Use Numberer if available for advanced methods
        try:
            from pyGmsh.solvers.Numberer import Numberer
            fem = self._mesh.get_fem_data()
            numb = Numberer(fem)
            numb.renumber(method=method)
        except Exception:
            pass

        # Rebuild scene
        self._actor_to_id.clear()
        self._id_to_actor.clear()
        self._brep_actors.clear()
        self._brep_grids.clear()
        self._brep_to_elems.clear()
        self._elem_to_brep.clear()
        self._elem_data.clear()
        self._brep_dominant_type.clear()
        self._node_actor = None
        self._node_cloud = None
        self._node_tree = None
        self._cached_elem_props.clear()
        self._cached_qualities.clear()
        self._tag_to_idx = None
        self._node_label_actor = None
        self._elem_label_actor = None
        self._selected_nodes.clear()
        self._picks.clear()
        self._plotter.clear()
        self._build_scene()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def _apply_partitioning(self, n_partitions: int) -> None:
        """Partition the mesh and switch to partition coloring."""
        if hasattr(self._parent, "partition") and self._parent.partition:
            self._parent.partition.auto(n_partitions)
            self._set_color_mode("partition")

    # ------------------------------------------------------------------
    # Selection sets
    # ------------------------------------------------------------------

    def _save_selection_set(self, name: str) -> None:
        """Save the current selection as a named set."""
        if self._pick_level == "mesh":
            self._selection_sets[name] = list(self._selected_nodes)
        else:
            self._selection_sets[name] = [
                (dt[0] * 10000 + dt[1]) for dt in self._picks
            ]

    def _load_selection_set(self, name: str) -> None:
        """Restore a previously saved selection set."""
        if name not in self._selection_sets:
            return
        tags = self._selection_sets[name]
        if self._pick_level == "mesh":
            self._selected_nodes = list(tags)
            self._update_node_highlight()
        else:
            self._picks = list(tags)
            self._recolor_all_brep()
        self._update_status()
        self._fire_pick_changed()

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        """Refresh the HUD text on the plotter."""
        try:
            self._plotter.remove_actor("_picker_hud")
        except Exception:
            pass

        if self._pick_level == "mesh":
            n = len(self._selected_nodes)
            text = f"Mode: Mesh | Nodes selected: {n}"
        else:
            n = len(self._picks)
            text = f"Mode: BRep | Patches selected: {n}"

        try:
            self._plotter.add_text(
                text,
                position="lower_left",
                font_size=9,
                color="white",
                name="_picker_hud",
            )
        except Exception:
            pass

    def _deselect_all(self) -> None:
        """Clear all picks, selected nodes, elements, and groups."""
        self._picks.clear()
        self._selected_nodes.clear()
        self._selected_elems.clear()
        self._selected_groups.clear()
        self._apply_coloring()
        self._update_node_highlight()
        self._update_elem_highlight()
        self._update_status()
        self._fire_pick_changed()

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def show(
        self, *, title: str | None = None, maximized: bool = True,
    ) -> "MeshViewer":
        """Open the mesh viewer window.  Blocks until closed."""
        gmsh.model.occ.synchronize()
        default_title = (
            f"MeshViewer — {self._parent.model_name} - Ladruño"
        )
        window = self._create_window(
            title=title or default_title, maximized=maximized,
        )
        window.exec()
        self._on_window_closed()
        return self

    def _create_window(self, *, title: str, maximized: bool):
        from .MeshViewerUI import MeshViewerWindow
        return MeshViewerWindow(self, title=title, maximized=maximized)

    def _on_window_closed(self) -> None:
        """Cleanup after the viewer window closes."""
        if self._parent._verbose:
            print(
                f"[MeshViewer] closed — "
                f"{len(self._selected_nodes)} nodes selected, "
                f"{len(self._picks)} BRep patches selected"
            )
