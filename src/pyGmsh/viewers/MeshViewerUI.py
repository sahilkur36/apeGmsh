"""
MeshViewerUI
============

Qt (pyvistaqt) front-end for :class:`MeshViewer`.

Inherits from :class:`BaseViewerWindow` (shared window shell, toolbar,
console, prefs, camera controls) and adds mesh-specific UI:

* **Mesh Browser tab**: physical groups, BRep patches, partitions tree.
* **Display tab**: color mode, node/element labels, wireframe, edges.
* **Tools tab**: renumbering (RCMK/Hilbert/Simple) and partitioning.
* **Preferences tab**: base prefs extended with node marker size and
  edge color picker.
* **Node/Element Info dock** (bottom): details for the picked node or
  BRep patch -- coordinates, connectivity, element types.
* **Selection Sets dock** (right, below tabs): save / load / delete
  named selection sets.
* **Toolbar extras**: mesh/BRep pick level toggle, hide/isolate/show.
* **Status bar**: pick level, pick count, color mode.

The window is blocking -- ``MeshViewer.show()`` calls
``MeshViewerWindow(...).exec()`` and waits.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from .BaseViewerUI import BaseViewerWindow

if TYPE_CHECKING:
    from pyGmsh.viewers.MeshViewer import MeshViewer


# Mapping from Gmsh element type integer to human-readable name.
# Only the most common types are listed; unknown types fall back to
# "Type <N>".
_ELEM_TYPE_NAMES: dict[int, str] = {
    1: "Line2",
    2: "Tri3",
    3: "Quad4",
    4: "Tet4",
    5: "Hex8",
    6: "Prism6",
    7: "Pyramid5",
    8: "Line3",
    9: "Tri6",
    10: "Quad9",
    11: "Tet10",
    15: "Point1",
    16: "Quad8",
    17: "Hex20",
    18: "Prism15",
    21: "Tri10",
    26: "Line4",
    29: "Tet20",
    36: "Quad16",
    92: "Hex27",
    93: "Hex64",
}

_DIM_NAMES = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}


# ======================================================================
# MeshViewerWindow
# ======================================================================

class MeshViewerWindow(BaseViewerWindow):
    """QMainWindow hosting the 3D viewport + mesh tree + display controls.

    Inherits from :class:`BaseViewerWindow` which provides the Qt shell,
    console, toolbar (camera buttons), preferences tab, and exec loop.
    This subclass adds all mesh-specific UI.
    """

    def __init__(
        self,
        viewer: "MeshViewer",
        *,
        title: str = "MeshViewer",
        maximized: bool = True,
    ) -> None:
        # Typed reference (mypy sees self._viewer as BaseViewer)
        self._viewer: "MeshViewer" = viewer  # type: ignore[assignment]

        # Selection-set storage: name -> list of picked items
        self._selection_sets: dict[str, list] = {}

        # Base class sets self._viewer = viewer, builds window, wires
        # base observer callbacks.
        super().__init__(viewer, title=title, maximized=maximized)

        # Wire mesh-specific callbacks
        viewer._on_pick_changed.append(self._refresh_info_dock)
        viewer._on_hover_changed.append(self._refresh_info_dock)

        # Populate the mesh browser tree after the window is built.
        self._populate_mesh_browser()

        # Keyboard shortcuts
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui
        window = self._window

        sc_q = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), window)
        sc_q.activated.connect(window.close)

        sc_esc = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), window)
        sc_esc.activated.connect(self._action_deselect_all)

        sc_m = QtWidgets.QShortcut(QtGui.QKeySequence("M"), window)
        sc_m.activated.connect(lambda: self._set_pick_level_ui("mesh"))

        sc_b = QtWidgets.QShortcut(QtGui.QKeySequence("B"), window)
        sc_b.activated.connect(lambda: self._set_pick_level_ui("brep"))

        sc_h = QtWidgets.QShortcut(QtGui.QKeySequence("H"), window)
        sc_h.activated.connect(self._viewer._hide_selected)

        sc_i = QtWidgets.QShortcut(QtGui.QKeySequence("I"), window)
        sc_i.activated.connect(self._viewer._isolate_selected)

        sc_r = QtWidgets.QShortcut(QtGui.QKeySequence("R"), window)
        sc_r.activated.connect(self._viewer._show_all)

    # ==================================================================
    # Deselect-all helper
    # ==================================================================

    def _action_deselect_all(self) -> None:
        """Clear all picks and refresh the UI."""
        try:
            self._viewer._deselect_all()
        except Exception:
            pass
        self._refresh_statusbar()

    # ==================================================================
    # BaseViewerWindow hook overrides
    # ==================================================================

    def _build_tabs(self):
        """Return tabs: Mesh Browser, Display, Tools, Preferences."""
        return [
            ("Browser", self._build_browser_tab()),
            ("Display", self._build_display_tab()),
            ("Tools", self._build_tools_tab()),
            ("Preferences", self._build_prefs_tab()),  # from base
        ]

    def _build_docks(self):
        """Return extra docks: Selection Sets (right) and
        Node/Element Info (right, below sets).
        The base class adds all returned docks to the right area."""
        QtWidgets = self._QtWidgets

        # ---- Selection Sets dock ----
        sets_dock = QtWidgets.QDockWidget("Selection Sets")
        sets_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        sets_dock.setWidget(self._build_selection_sets_widget())
        self._sets_dock = sets_dock

        # ---- Node/Element Info dock ----
        info_dock = QtWidgets.QDockWidget("Node / Element Info")
        info_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        info_dock.setWidget(self._build_info_dock_widget())
        self._info_dock = info_dock

        return [sets_dock, info_dock]

    def _build_toolbar_extra(self, bar) -> None:
        """Add mesh-specific buttons to the toolbar.

        Adds pick-level toggles (Mesh / BRep) and visibility actions
        (hide, isolate, show all) before returning control to the base.
        """
        _IC = "#2d2d2d"

        # ---- Pick level toggle ----
        self._act_mesh_level = bar.addAction(self._make_icon("M", _IC), "")
        self._act_mesh_level.setToolTip("Mesh level picking  [M]")
        self._act_mesh_level.setCheckable(True)
        self._act_mesh_level.setChecked(True)
        self._act_mesh_level.toggled.connect(
            lambda c: self._set_pick_level_ui("mesh" if c else "group"),
        )

        self._act_group_level = bar.addAction(self._make_icon("G", _IC), "")
        self._act_group_level.setToolTip("Physical group picking  [G]")
        self._act_group_level.setCheckable(True)
        self._act_group_level.toggled.connect(
            lambda c: self._set_pick_level_ui("group" if c else "mesh"),
        )

        bar.addSeparator()

        # ---- Visibility ----
        act_hide = bar.addAction(self._make_icon("\u25CB", _IC), "")
        act_hide.setToolTip("Hide selected  [H]")
        act_hide.triggered.connect(self._viewer._hide_selected)

        act_isolate = bar.addAction(self._make_icon("\u25CE", _IC), "")
        act_isolate.setToolTip("Isolate selected  [I]")
        act_isolate.triggered.connect(self._viewer._isolate_selected)

        act_show = bar.addAction(self._make_icon("\u25C9", _IC), "")
        act_show.setToolTip("Show all  [R]")
        act_show.triggered.connect(self._viewer._show_all)

        bar.addSeparator()

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _apply_visual_changes(self) -> None:
        """Rebuild node cloud + recolor after pref changes."""
        v = self._viewer
        v._update_node_highlight()
        v._recolor_all_brep()

    def _refresh_statusbar(self) -> None:
        """Show pick level, pick count, and color mode."""
        v = self._viewer
        level = getattr(v, "_pick_level", "mesh").upper()
        color_mode = getattr(v, "_color_mode", "Default")
        if getattr(v, "_pick_level", "mesh") == "mesh":
            nn = len(getattr(v, "_selected_nodes", []))
            ne = len(getattr(v, "_selected_elems", []))
            self._statusbar.showMessage(
                f"level={level}  nodes={nn}  elems={ne}  color={color_mode}",
            )
        else:
            n = len(getattr(v, "_selected_groups", []))
            self._statusbar.showMessage(
                f"level={level}  groups={n}  color={color_mode}",
            )

    # ------------------------------------------------------------------
    # Hover / pick info update
    # ------------------------------------------------------------------

    def _on_hover_changed_ui(self) -> None:
        """Called when the hover entity changes -- refresh the info dock."""
        self._refresh_info_dock()

    # ------------------------------------------------------------------
    # Help rows
    # ------------------------------------------------------------------

    def _help_extra_rows(self):
        """Additional shortcut rows for the mesh viewer."""
        return [
            ("M", "Switch to mesh (node/element) picking"),
            ("G", "Switch to physical group picking"),
            ("H", "Hide selected"),
            ("I", "Isolate selected"),
            ("R", "Show all"),
        ]

    # ------------------------------------------------------------------
    # Pick level UI synchronisation
    # ------------------------------------------------------------------

    def _set_pick_level_ui(self, level: str) -> None:
        """Switch pick level on the viewer and sync toolbar toggle state.

        Parameters
        ----------
        level : str
            Either ``"mesh"`` or ``"group"``.
        """
        self._viewer._set_pick_level(level)

        # Block signals to avoid re-entrant toggling
        self._act_mesh_level.blockSignals(True)
        self._act_group_level.blockSignals(True)
        self._act_mesh_level.setChecked(level == "mesh")
        self._act_group_level.setChecked(level == "group")
        self._act_mesh_level.blockSignals(False)
        self._act_group_level.blockSignals(False)

        self._refresh_statusbar()

    # ==================================================================
    # Mesh Browser tab
    # ==================================================================

    def _build_browser_tab(self):
        """Build the Mesh Browser tree widget.

        The tree is populated later by :meth:`_populate_mesh_browser`
        once the window is fully constructed and the gmsh model is
        available.
        """
        QtWidgets = self._QtWidgets

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        self._browser_tree = QtWidgets.QTreeWidget()
        self._browser_tree.setHeaderLabels(["Entity", "Details"])
        self._browser_tree.setColumnCount(2)
        self._browser_tree.setAlternatingRowColors(True)
        self._browser_tree.setRootIsDecorated(True)
        layout.addWidget(self._browser_tree)

        return panel

    def _populate_mesh_browser(self) -> None:
        """Fill the browser tree from the current gmsh model data.

        Creates three top-level sections:

        * **Physical Groups** -- for each group, list member entities
          and total element count.
        * **BRep Patches** -- for each (dim, tag) entity, show element
          type breakdown and node count.
        * **Partitions** -- if partitioned, for each partition list its
          entities.
        """
        tree = self._browser_tree
        tree.clear()

        # ----------------------------------------------------------
        # 1. Physical Groups
        # ----------------------------------------------------------
        pg_root = self._QtWidgets.QTreeWidgetItem(tree, ["Physical Groups", ""])
        pg_root.setExpanded(True)
        try:
            phys_groups = gmsh.model.getPhysicalGroups()
        except Exception:
            phys_groups = []

        for dim, tag in phys_groups:
            name = ""
            try:
                name = gmsh.model.getPhysicalName(dim, tag)
            except Exception:
                pass
            label = name if name else f"PhysGrp({dim},{tag})"

            # Count elements in all entities belonging to this group
            entities = []
            try:
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            except Exception:
                pass

            total_elems = 0
            for ent_tag in entities:
                try:
                    elem_types, _, _ = gmsh.model.mesh.getElements(dim, ent_tag)
                    for et in elem_types:
                        _, node_tags = gmsh.model.mesh.getElementsByType(et)
                        # node_tags is flat; divide by nodes-per-element
                        props = gmsh.model.mesh.getElementProperties(et)
                        npe = props[3]  # nodes per element
                        total_elems += len(node_tags) // npe if npe else 0
                except Exception:
                    pass

            pg_item = self._QtWidgets.QTreeWidgetItem(
                pg_root, [label, f"{total_elems} elems"],
            )
            # List member entities under the group
            for ent_tag in entities:
                dim_name = _DIM_NAMES.get(dim, f"Dim{dim}")
                self._QtWidgets.QTreeWidgetItem(
                    pg_item, [f"{dim_name} {ent_tag}", ""],
                )

        # ----------------------------------------------------------
        # 2. BRep Patches
        # ----------------------------------------------------------
        brep_root = self._QtWidgets.QTreeWidgetItem(tree, ["BRep Patches", ""])
        brep_root.setExpanded(False)
        try:
            all_entities = gmsh.model.getEntities()
        except Exception:
            all_entities = []

        for dim, tag in all_entities:
            dim_name = _DIM_NAMES.get(dim, f"Dim{dim}")
            entity_label = f"{dim_name} {tag}"

            # Element type breakdown + node count
            elem_info_parts: list[str] = []
            total_nodes = 0
            try:
                elem_types, elem_tags_list, node_tags_list = (
                    gmsh.model.mesh.getElements(dim, tag)
                )
                for i, et in enumerate(elem_types):
                    type_name = _ELEM_TYPE_NAMES.get(et, f"Type{et}")
                    n_elems = len(elem_tags_list[i])
                    elem_info_parts.append(f"{type_name}:{n_elems}")

                # Node count for this entity
                node_tags_ent, _ = gmsh.model.mesh.getNodes(dim, tag)
                total_nodes = len(node_tags_ent)
            except Exception:
                pass

            details = ", ".join(elem_info_parts) if elem_info_parts else "no elems"
            details += f"  ({total_nodes} nodes)"

            ent_item = self._QtWidgets.QTreeWidgetItem(
                brep_root, [entity_label, details],
            )

            # Children: one row per element type
            for part in elem_info_parts:
                self._QtWidgets.QTreeWidgetItem(ent_item, [part, ""])

        # ----------------------------------------------------------
        # 3. Partitions
        # ----------------------------------------------------------
        partitions_root = self._QtWidgets.QTreeWidgetItem(
            tree, ["Partitions", ""],
        )
        partitions_root.setExpanded(False)
        try:
            partition_entities = gmsh.model.getPartitions()
        except Exception:
            partition_entities = None

        if partition_entities:
            # partition_entities is a list of partition tags
            for part_id in partition_entities:
                part_item = self._QtWidgets.QTreeWidgetItem(
                    partitions_root,
                    [f"Partition {part_id}", ""],
                )
                # List entities in this partition
                try:
                    part_ents = gmsh.model.getEntitiesForPartition(part_id)
                    for dim, tag in part_ents:
                        dim_name = _DIM_NAMES.get(dim, f"Dim{dim}")
                        self._QtWidgets.QTreeWidgetItem(
                            part_item, [f"{dim_name} {tag}", ""],
                        )
                except Exception:
                    pass
        else:
            self._QtWidgets.QTreeWidgetItem(
                partitions_root, ["(not partitioned)", ""],
            )

        # Resize columns to content
        tree.resizeColumnToContents(0)
        tree.resizeColumnToContents(1)

    # ==================================================================
    # Display tab
    # ==================================================================

    def _build_display_tab(self):
        """Build the Display tab with visualization controls.

        Controls:
        - Color mode combo (Default/Partition/Quality/Element Type/Physical Group)
        - Show node labels checkbox
        - Show element labels checkbox
        - Wireframe checkbox
        - Show edges checkbox
        """
        QtWidgets = self._QtWidgets

        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        # ---- Color mode ----
        self._combo_color_mode = QtWidgets.QComboBox()
        self._combo_color_mode.addItems([
            "Default", "Partition", "Quality", "Element Type", "Physical Group",
        ])
        self._combo_color_mode.currentTextChanged.connect(
            self._on_color_mode_changed,
        )
        form.addRow("Color mode", self._combo_color_mode)

        # ---- Show node labels ----
        self._cb_node_labels = QtWidgets.QCheckBox("Show node labels")
        self._cb_node_labels.setChecked(False)
        self._cb_node_labels.toggled.connect(self._on_node_labels_toggled)
        form.addRow(self._cb_node_labels)

        # ---- Show element labels ----
        self._cb_elem_labels = QtWidgets.QCheckBox("Show element labels")
        self._cb_elem_labels.setChecked(False)
        self._cb_elem_labels.toggled.connect(self._on_elem_labels_toggled)
        form.addRow(self._cb_elem_labels)

        # ---- Wireframe ----
        self._cb_wireframe = QtWidgets.QCheckBox("Wireframe")
        self._cb_wireframe.setChecked(False)
        self._cb_wireframe.toggled.connect(self._on_wireframe_toggled)
        form.addRow(self._cb_wireframe)

        # ---- Show edges ----
        self._cb_show_edges = QtWidgets.QCheckBox("Show edges")
        self._cb_show_edges.setChecked(True)
        self._cb_show_edges.toggled.connect(self._on_show_edges_toggled)
        form.addRow(self._cb_show_edges)

        return panel

    # ---- Display callbacks ----

    def _on_color_mode_changed(self, mode: str) -> None:
        """Forward color mode change to the viewer."""
        try:
            self._viewer._set_color_mode(mode)
        except Exception:
            pass
        self._refresh_statusbar()
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_node_labels_toggled(self, checked: bool) -> None:
        """Toggle node label overlays."""
        try:
            self._viewer._toggle_node_labels(checked)
        except Exception:
            pass
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_elem_labels_toggled(self, checked: bool) -> None:
        """Toggle element label overlays."""
        try:
            self._viewer._toggle_elem_labels(checked)
        except Exception:
            pass
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_wireframe_toggled(self, checked: bool) -> None:
        """Toggle wireframe / solid rendering on all mesh actors."""
        try:
            plotter = self._qt_interactor
            for actor in plotter.renderer.actors.values():
                try:
                    prop = actor.GetProperty()
                    if prop is not None:
                        if checked:
                            prop.SetRepresentationToWireframe()
                        else:
                            prop.SetRepresentationToSurface()
                except Exception:
                    pass
            plotter.render()
        except Exception:
            pass

    def _on_show_edges_toggled(self, checked: bool) -> None:
        """Toggle mesh edge visibility on all actors."""
        try:
            plotter = self._qt_interactor
            for actor in plotter.renderer.actors.values():
                try:
                    prop = actor.GetProperty()
                    if prop is not None:
                        prop.SetEdgeVisibility(checked)
                except Exception:
                    pass
            plotter.render()
        except Exception:
            pass

    # ==================================================================
    # Tools tab
    # ==================================================================

    def _build_tools_tab(self):
        """Build the Tools tab with renumbering and partitioning controls.

        Sections:
        - **Renumber**: method combo (RCMK/Hilbert/Simple) + Apply button.
        - **Partition**: N partitions spin box + Apply button.
        """
        QtWidgets = self._QtWidgets

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ---- Renumber section ----
        renumber_group = QtWidgets.QGroupBox("Renumber")
        renumber_form = QtWidgets.QFormLayout(renumber_group)
        renumber_form.setSpacing(6)

        self._combo_renumber = QtWidgets.QComboBox()
        self._combo_renumber.addItems(["RCMK", "Hilbert", "Simple"])
        renumber_form.addRow("Method", self._combo_renumber)

        btn_renumber = QtWidgets.QPushButton("Apply")
        btn_renumber.clicked.connect(self._on_renumber_apply)
        renumber_form.addRow(btn_renumber)

        layout.addWidget(renumber_group)

        # ---- Partition section ----
        partition_group = QtWidgets.QGroupBox("Partition")
        partition_form = QtWidgets.QFormLayout(partition_group)
        partition_form.setSpacing(6)

        self._spin_partitions = QtWidgets.QSpinBox()
        self._spin_partitions.setRange(1, 128)
        self._spin_partitions.setValue(4)
        partition_form.addRow("N partitions", self._spin_partitions)

        btn_partition = QtWidgets.QPushButton("Apply")
        btn_partition.clicked.connect(self._on_partition_apply)
        partition_form.addRow(btn_partition)

        layout.addWidget(partition_group)

        # Spacer at the bottom
        layout.addStretch(1)

        return panel

    # ---- Tools callbacks ----

    def _on_renumber_apply(self) -> None:
        """Apply the selected renumbering method."""
        method = self._combo_renumber.currentText()
        try:
            self._viewer._apply_renumbering(method)
            self.log(f"Renumbering applied: {method}")
        except Exception as exc:
            self.log(f"Renumbering failed: {exc}")
        self._populate_mesh_browser()
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_partition_apply(self) -> None:
        """Apply mesh partitioning with the specified N."""
        n = self._spin_partitions.value()
        try:
            self._viewer._apply_partitioning(n)
            self.log(f"Partitioning applied: {n} partitions")
        except Exception as exc:
            self.log(f"Partitioning failed: {exc}")
        self._populate_mesh_browser()
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    # ==================================================================
    # Preferences extras
    # ==================================================================

    def _build_prefs_extra(self, form) -> None:
        """Extend the Preferences tab with mesh-specific controls.

        Adds:
        - Node marker size slider
        - Edge color picker button
        """
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        form.addRow(QtWidgets.QLabel(""))  # visual spacer
        form.addRow(QtWidgets.QLabel("--- Mesh-specific ---"))

        # ---- Node marker size ----
        self._s_node_marker = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_node_marker.setRange(1, 30)
        self._s_node_marker.setValue(
            int(getattr(self._viewer, "_node_marker_size", 6)),
        )
        self._s_node_marker.valueChanged.connect(self._on_node_marker_changed)
        form.addRow("Node marker size", self._s_node_marker)

        # ---- Edge color picker ----
        self._btn_edge_color = QtWidgets.QPushButton()
        self._btn_edge_color.setFixedSize(60, 24)
        edge_color = getattr(self._viewer, "_edge_color", "#000000")
        self._btn_edge_color.setStyleSheet(
            f"background-color: {edge_color}; border: 1px solid #999;",
        )
        self._btn_edge_color.clicked.connect(self._on_edge_color_pick)
        form.addRow("Edge color", self._btn_edge_color)

    def _on_node_marker_changed(self, value: int) -> None:
        """Update node marker size on the viewer and re-render."""
        try:
            self._viewer._node_marker_size = float(value)
        except Exception:
            pass
        self._apply_visual_changes()
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_edge_color_pick(self) -> None:
        """Open a color dialog and apply the chosen edge color."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui

        current = getattr(self._viewer, "_edge_color", "#000000")
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(current), self._window, "Edge Color",
        )
        if color.isValid():
            hex_color = color.name()
            self._viewer._edge_color = hex_color
            self._btn_edge_color.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #999;",
            )
            self._apply_visual_changes()
            try:
                self._qt_interactor.render()
            except Exception:
                pass

    # ==================================================================
    # Node / Element Info dock (bottom)
    # ==================================================================

    def _build_info_dock_widget(self):
        """Build the QTreeWidget used by the Node/Element Info dock.

        The tree is updated dynamically when the user hovers or picks
        a node / BRep patch via :meth:`_refresh_info_dock`.
        """
        QtWidgets = self._QtWidgets

        self._info_tree = QtWidgets.QTreeWidget()
        self._info_tree.setHeaderLabels(["Property", "Value"])
        self._info_tree.setColumnCount(2)
        self._info_tree.setAlternatingRowColors(True)
        self._info_tree.setRootIsDecorated(True)
        return self._info_tree

    def _refresh_info_dock(self) -> None:
        """Rebuild the info-dock tree from the viewer's current state.

        At **mesh** level, shows the picked node's tag, coordinates,
        connected elements, and physical group membership.

        At **BRep** level, shows the entity (dim, tag), element count,
        element types, and node count.
        """
        tree = self._info_tree
        tree.clear()

        v = self._viewer
        pick_level = getattr(v, "_pick_level", "mesh")

        if pick_level == "mesh":
            self._refresh_info_mesh_level(tree, v)
        else:
            self._refresh_info_brep_level(tree, v)

        tree.expandAll()
        tree.resizeColumnToContents(0)
        tree.resizeColumnToContents(1)

    def _refresh_info_mesh_level(self, tree, v) -> None:
        """Populate info tree for mesh-level (node) picking."""
        QtWidgets = self._QtWidgets

        selected_nodes = getattr(v, "_selected_nodes", [])
        if not selected_nodes:
            QtWidgets.QTreeWidgetItem(tree, ["(no node selected)", ""])
            return

        # Show the last selected node's details
        node_tag = selected_nodes[-1]

        QtWidgets.QTreeWidgetItem(tree, ["Node tag", str(node_tag)])

        # Coordinates
        try:
            coords, _, _, _ = gmsh.model.mesh.getNode(node_tag)
            x, y, z = coords[0], coords[1], coords[2]
            coord_item = QtWidgets.QTreeWidgetItem(
                tree, ["Coordinates", ""],
            )
            QtWidgets.QTreeWidgetItem(coord_item, ["x", f"{x:.6g}"])
            QtWidgets.QTreeWidgetItem(coord_item, ["y", f"{y:.6g}"])
            QtWidgets.QTreeWidgetItem(coord_item, ["z", f"{z:.6g}"])
        except Exception:
            QtWidgets.QTreeWidgetItem(
                tree, ["Coordinates", "(unavailable)"],
            )

        # Connected elements
        try:
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements()
            connected = []
            for i, et in enumerate(elem_types):
                props = gmsh.model.mesh.getElementProperties(et)
                npe = props[3]
                _, all_node_tags = gmsh.model.mesh.getElementsByType(et)
                for e_idx in range(len(all_node_tags) // npe):
                    e_nodes = all_node_tags[e_idx * npe : (e_idx + 1) * npe]
                    if node_tag in e_nodes:
                        connected.append(int(elem_tags[i][e_idx]))
            if connected:
                conn_item = QtWidgets.QTreeWidgetItem(
                    tree,
                    ["Connected elements", f"{len(connected)} element(s)"],
                )
                for eid in connected[:20]:  # cap display at 20
                    QtWidgets.QTreeWidgetItem(
                        conn_item, [f"Elem {eid}", ""],
                    )
                if len(connected) > 20:
                    QtWidgets.QTreeWidgetItem(
                        conn_item, [f"... +{len(connected) - 20} more", ""],
                    )
            else:
                QtWidgets.QTreeWidgetItem(
                    tree, ["Connected elements", "none"],
                )
        except Exception:
            QtWidgets.QTreeWidgetItem(
                tree, ["Connected elements", "(error)"],
            )

        # Physical group membership
        try:
            phys_groups = gmsh.model.getPhysicalGroups()
            memberships: list[str] = []
            # Determine which entity owns this node
            for dim, tag in gmsh.model.getEntities():
                try:
                    node_tags_ent, _, _ = gmsh.model.mesh.getNodes(dim, tag)
                    if node_tag in node_tags_ent:
                        for pdim, ptag in phys_groups:
                            if pdim == dim:
                                ents = gmsh.model.getEntitiesForPhysicalGroup(
                                    pdim, ptag,
                                )
                                if tag in ents:
                                    name = gmsh.model.getPhysicalName(
                                        pdim, ptag,
                                    )
                                    label = name if name else f"({pdim},{ptag})"
                                    memberships.append(label)
                except Exception:
                    pass
            if memberships:
                pg_item = QtWidgets.QTreeWidgetItem(
                    tree,
                    ["Physical groups", f"{len(memberships)} group(s)"],
                )
                for m in memberships:
                    QtWidgets.QTreeWidgetItem(pg_item, [m, ""])
            else:
                QtWidgets.QTreeWidgetItem(
                    tree, ["Physical groups", "none"],
                )
        except Exception:
            pass

    def _refresh_info_brep_level(self, tree, v) -> None:
        """Populate info tree for BRep-level (patch) picking."""
        QtWidgets = self._QtWidgets

        picks = getattr(v, "_picks", [])
        if not picks:
            QtWidgets.QTreeWidgetItem(tree, ["(no patch selected)", ""])
            return

        # Show the last picked entity
        dim, tag = picks[-1]
        dim_name = _DIM_NAMES.get(dim, f"Dim{dim}")

        QtWidgets.QTreeWidgetItem(
            tree, ["BRep entity", f"{dim_name} {tag}"],
        )
        QtWidgets.QTreeWidgetItem(tree, ["Dimension", str(dim)])
        QtWidgets.QTreeWidgetItem(tree, ["Tag", str(tag)])

        # Element count and types
        try:
            elem_types, elem_tags_list, _ = gmsh.model.mesh.getElements(
                dim, tag,
            )
            total_elems = sum(len(et) for et in elem_tags_list)
            QtWidgets.QTreeWidgetItem(
                tree, ["Element count", str(total_elems)],
            )

            types_item = QtWidgets.QTreeWidgetItem(
                tree, ["Element types", ""],
            )
            for i, et in enumerate(elem_types):
                type_name = _ELEM_TYPE_NAMES.get(et, f"Type{et}")
                n_elems = len(elem_tags_list[i])
                QtWidgets.QTreeWidgetItem(
                    types_item, [type_name, str(n_elems)],
                )
        except Exception:
            QtWidgets.QTreeWidgetItem(
                tree, ["Element count", "(unavailable)"],
            )

        # Node count
        try:
            node_tags_ent, _, _ = gmsh.model.mesh.getNodes(dim, tag)
            QtWidgets.QTreeWidgetItem(
                tree, ["Node count", str(len(node_tags_ent))],
            )
        except Exception:
            QtWidgets.QTreeWidgetItem(
                tree, ["Node count", "(unavailable)"],
            )

    # ==================================================================
    # Selection Sets dock (right)
    # ==================================================================

    def _build_selection_sets_widget(self):
        """Build the widget for the Selection Sets dock.

        Contains a list of saved sets, a name input, and
        Save / Load / Delete buttons.  Double-clicking a set loads it.
        """
        QtWidgets = self._QtWidgets

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # List of saved sets (name + count)
        self._sets_list = QtWidgets.QListWidget()
        self._sets_list.setAlternatingRowColors(True)
        self._sets_list.itemDoubleClicked.connect(self._on_set_double_click)
        layout.addWidget(self._sets_list)

        # Name input
        name_row = QtWidgets.QHBoxLayout()
        name_label = QtWidgets.QLabel("Name:")
        self._set_name_input = QtWidgets.QLineEdit()
        self._set_name_input.setPlaceholderText("Selection set name")
        name_row.addWidget(name_label)
        name_row.addWidget(self._set_name_input)
        layout.addLayout(name_row)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()

        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setToolTip("Save current selection as a named set")
        btn_save.clicked.connect(self._on_set_save)
        btn_row.addWidget(btn_save)

        btn_load = QtWidgets.QPushButton("Load")
        btn_load.setToolTip("Load the selected set into the current picks")
        btn_load.clicked.connect(self._on_set_load)
        btn_row.addWidget(btn_load)

        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.setToolTip("Delete the selected set")
        btn_delete.clicked.connect(self._on_set_delete)
        btn_row.addWidget(btn_delete)

        layout.addLayout(btn_row)

        return container

    def _refresh_sets_list(self) -> None:
        """Rebuild the selection sets list widget from internal storage."""
        self._sets_list.clear()
        for name, items in self._selection_sets.items():
            self._sets_list.addItem(f"{name}  ({len(items)})")

    def _on_set_save(self) -> None:
        """Save the current viewer selection as a named set."""
        name = self._set_name_input.text().strip()
        if not name:
            self._statusbar.showMessage("Enter a name for the set", 3000)
            return

        v = self._viewer
        pick_level = getattr(v, "_pick_level", "mesh")

        if pick_level == "mesh":
            items = list(getattr(v, "_selected_nodes", []))
        else:
            items = list(getattr(v, "_picks", []))

        if not items:
            self._statusbar.showMessage("Nothing selected to save", 3000)
            return

        self._selection_sets[name] = items
        self._refresh_sets_list()
        self.log(f"Saved selection set '{name}' ({len(items)} items)")
        self._set_name_input.clear()

    def _on_set_load(self) -> None:
        """Load the currently highlighted set into the viewer picks."""
        current = self._sets_list.currentItem()
        if current is None:
            self._statusbar.showMessage("Select a set to load", 3000)
            return
        self._load_set_by_row(self._sets_list.currentRow())

    def _on_set_delete(self) -> None:
        """Delete the currently highlighted selection set."""
        current = self._sets_list.currentItem()
        if current is None:
            self._statusbar.showMessage("Select a set to delete", 3000)
            return
        name = self._get_set_name_from_row(self._sets_list.currentRow())
        if name and name in self._selection_sets:
            del self._selection_sets[name]
            self._refresh_sets_list()
            self.log(f"Deleted selection set '{name}'")

    def _on_set_double_click(self, item) -> None:
        """Double-click a selection set to load it."""
        row = self._sets_list.row(item)
        self._load_set_by_row(row)

    def _load_set_by_row(self, row: int) -> None:
        """Load a selection set given its row index in the list."""
        name = self._get_set_name_from_row(row)
        if name is None or name not in self._selection_sets:
            return

        items = self._selection_sets[name]
        v = self._viewer
        pick_level = getattr(v, "_pick_level", "mesh")

        try:
            if pick_level == "mesh":
                v._selected_nodes = list(items)
            else:
                v._picks = list(items)
            # Trigger UI refresh through the viewer's callbacks
            for cb in getattr(v, "_on_pick_changed", []):
                try:
                    cb()
                except Exception:
                    pass
        except Exception:
            pass

        self.log(f"Loaded selection set '{name}' ({len(items)} items)")
        self._refresh_statusbar()

    def _get_set_name_from_row(self, row: int) -> str | None:
        """Extract the set name from a list widget row.

        The list items are formatted as ``"name  (count)"``; this
        extracts just the name part.
        """
        names = list(self._selection_sets.keys())
        if 0 <= row < len(names):
            return names[row]
        return None
