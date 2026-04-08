"""
SelectionPickerUI
=================

Qt (pyvistaqt) front-end for :class:`SelectionPicker`.

Inherits from :class:`BaseViewerWindow` (shared window shell, toolbar,
console, prefs, camera controls) and adds BRep-selection-specific UI:

* **Browser tab**: physical-groups + unassigned-entities tree, with
  integrated group management controls at the bottom.
* **View tab**: entity label overlays (dim toggles, font, names/tags).
* **Filter tab**: dimension filter, entity-label filter, text filter.
* **Entity Info dock**: BRep topology tree for hovered entity.
* **Toolbar extras**: new/rename/delete group, hide/isolate/show all.
* **Prefs extras**: selection color, label font family/size/color.

The window is blocking -- ``SelectionPicker.show()`` calls
``SelectionPickerWindow(...).exec()`` and waits.  On close, the picker
core flushes every staged group to Gmsh.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import numpy as np

from .BaseViewerUI import BaseViewerWindow

if TYPE_CHECKING:
    from pyGmsh.viewers.SelectionPicker import SelectionPicker


DimTag = tuple[int, int]

_DIM_NAMES = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}
_DIM_ABBR  = {0: "P", 1: "C", 2: "S", 3: "V"}


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class SelectionPickerWindow(BaseViewerWindow):
    """QMainWindow hosting the 3D viewport + model tree + prefs dock.

    Inherits from :class:`BaseViewerWindow` which provides the Qt shell,
    console, toolbar (camera buttons), preferences tab, and exec loop.
    This subclass adds all BRep-selection-specific UI.
    """

    def __init__(
        self,
        picker: "SelectionPicker",
        *,
        title: str = "SelectionPicker",
        maximized: bool = True,
    ) -> None:
        # Typed reference (mypy sees self._viewer as BaseViewer)
        self._picker: "SelectionPicker" = picker

        # Fast lookups from DimTag / group-name -> tree item
        # (must exist before _build_window, which calls _build_tabs etc.)
        self._tree_items_by_dt: dict = {}
        self._tree_items_by_group: dict = {}

        # Re-entry guard to avoid pick <-> tree feedback loops
        self._syncing = False

        # Alias must exist BEFORE super().__init__ because _build_tabs()
        # is called during window construction and references self._picker.
        self._picker = picker

        super().__init__(picker, title=title, maximized=maximized)

        # Wire picker -> UI callbacks (selection-specific)
        picker._on_pick_changed.append(self._refresh_tree_picks)
        picker._on_pick_changed.append(self._log_pick_changed)
        picker._on_visibility_changed.append(self._refresh_tree_visibility)
        picker._on_visibility_changed.append(self._sync_dim_checkboxes)
        picker._on_hover_changed.append(self._refresh_info)

        # Initial tree populate (must happen after window is built)
        self._populate_tree()

        # Keyboard shortcuts (Qt-level, so they work even when VTK
        # doesn't have focus -- e.g. Tab, which Qt normally eats for
        # widget focus-cycling).
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui
        QtCore = self._QtCore
        window = self._window

        # Q -> close window
        sc_q = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), window)
        sc_q.activated.connect(window.close)
        # Esc / Space -> deselect (two-stage: revert to applied, then clear)
        sc_esc = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), window)
        sc_esc.activated.connect(self._action_deselect_all)
        sc_space = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Space), window,
        )
        sc_space.activated.connect(self._action_deselect_all)
        # Tab -> cycle overlapping entities (Revit-style).
        sc_tab = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Tab), window,
        )
        sc_tab.setContext(QtCore.Qt.ApplicationShortcut)
        sc_tab.activated.connect(self._picker._cycle_pick)

    # ==================================================================
    # BaseViewerWindow hook overrides
    # ==================================================================

    def _build_tabs(self):
        """Return tabs: Browser, View, Filter, Preferences (from base)."""
        return [
            ("Browser", self._build_browser_tab()),
            ("View", self._build_view_tab()),
            ("Filter", self._build_filter_tab()),
            ("Preferences", self._build_prefs_tab()),  # from base
        ]

    def _build_docks(self):
        """Return extra docks: Entity Info only."""
        info_dock = self._QtWidgets.QDockWidget("Entity Info")
        info_dock.setFeatures(
            self._QtWidgets.QDockWidget.DockWidgetMovable
            | self._QtWidgets.QDockWidget.DockWidgetFloatable
            | self._QtWidgets.QDockWidget.DockWidgetClosable
        )
        info_dock.setWidget(self._build_info_tab())
        self._info_dock = info_dock
        info_dock.hide()
        return [info_dock]

    def _build_toolbar_extra(self, bar) -> None:
        """Add BRep-specific buttons BEFORE the generic camera buttons.

        Note: BaseViewerWindow._build_toolbar calls this hook *after*
        the generic camera/screenshot/help buttons.  The extra buttons
        are appended at the end of the toolbar.
        """
        _IC = "#2d2d2d"

        bar.addSeparator()

        # ---- Group operations ----
        act_new = bar.addAction(self._make_icon("+", _IC), "")
        act_new.setToolTip("New physical group")
        act_new.triggered.connect(self._action_new_group)

        act_rename = bar.addAction(self._make_icon("\u270F", _IC), "")
        act_rename.setToolTip("Rename active group")
        act_rename.triggered.connect(self._action_rename_active)

        act_delete = bar.addAction(self._make_icon("\u2212", _IC), "")
        act_delete.setToolTip("Delete active group")
        act_delete.triggered.connect(self._action_delete_active)

        bar.addSeparator()

        # ---- Visibility ----
        act_hide = bar.addAction(self._make_icon("\u25CB", _IC), "")
        act_hide.setToolTip("Hide selected  [H]")
        act_hide.triggered.connect(self._picker._hide_selected)

        act_isolate = bar.addAction(self._make_icon("\u25CE", _IC), "")
        act_isolate.setToolTip("Isolate selected  [I]")
        act_isolate.triggered.connect(self._picker._isolate_selected)

        act_show = bar.addAction(self._make_icon("\u25C9", _IC), "")
        act_show.setToolTip("Show all  [R]")
        act_show.triggered.connect(self._picker._show_all)

    def _build_prefs_extra(self, form) -> None:
        """Append selection-specific prefs to the base prefs form."""
        QtWidgets = self._QtWidgets

        picker = self._picker

        # Selection color
        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.setFixedSize(60, 24)
        self._update_color_btn_style(picker._pick_color)
        self._color_btn.clicked.connect(self._on_pick_color_clicked)
        form.addRow("Selection color", self._color_btn)

        # ---- Font settings ----
        form.addRow(QtWidgets.QLabel(""))  # spacer

        # Font family (VTK only supports these three)
        self._font_combo = QtWidgets.QComboBox()
        self._font_combo.addItems(["Arial", "Courier", "Times"])
        self._font_combo.setCurrentIndex(0)
        self._font_combo.currentIndexChanged.connect(self._on_font_changed)
        form.addRow("Label font", self._font_combo)

        # Font size
        self._font_size_spin = QtWidgets.QSpinBox()
        self._font_size_spin.setRange(6, 30)
        self._font_size_spin.setValue(10)
        self._font_size_spin.valueChanged.connect(self._on_font_changed)
        form.addRow("Label font size", self._font_size_spin)

        # Font color
        self._font_color_btn = QtWidgets.QPushButton()
        self._font_color_btn.setFixedSize(60, 24)
        self._font_color = "#ffffff"
        self._font_color_btn.setStyleSheet(
            f"background-color: {self._font_color}; border: 1px solid #888;"
        )
        self._font_color_btn.clicked.connect(self._on_font_color_clicked)
        form.addRow("Label font color", self._font_color_btn)

    def _refresh_statusbar(self) -> None:
        """Selection-specific status: active group + pick breakdown."""
        picker = self._picker
        active = picker.active_group or "(none)"
        tags = picker.tags
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for d, _ in tags:
            counts[d] = counts.get(d, 0) + 1
        bits = [
            f"{n} {_DIM_NAMES[d].lower()}"
            for d, n in counts.items() if n > 0
        ]
        breakdown = ", ".join(bits) or "no picks"
        self._statusbar.showMessage(
            f"active={active}   picks: {len(tags)} ({breakdown})"
        )

    def _on_hover_changed_ui(self) -> None:
        """Refresh the Entity Info dock on hover change."""
        self._refresh_info()

    def _apply_visual_changes(self) -> None:
        """Reapply colors after a pref change."""
        picker = self._picker

        # For batched mode: rebuild dim=0 glyph if point size changed
        if getattr(picker, '_batched', False) and 0 in picker._batch_actors:
            try:
                self._rebuild_batched_points()
            except Exception:
                pass

        # For non-batched: SetScale updates + opacity/edge refresh
        picker._recolor_all()

    def _rebuild_batched_points(self) -> None:
        """Rebuild the batched dim=0 glyph actor with current point_size."""
        import pyvista as pv
        picker = self._picker
        plotter = picker._plotter

        old_actor = picker._batch_actors.get(0)
        if old_actor is not None:
            plotter.remove_actor(old_actor)

        # Reuse stored centroids to rebuild
        centers = []
        tags = []
        for dt, xyz in picker._batch_centroids.items():
            if dt[0] == 0:
                centers.append(xyz)
                tags.append(dt[1])
        if not centers:
            return

        from .SelectionPicker import _IDLE_POINT, _hex_to_rgb
        idle_rgb = np.array(
            [int(c * 255) for c in _hex_to_rgb(_IDLE_POINT)],
            dtype=np.uint8,
        )

        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm(
                [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
            ))
            if diag <= 0.0:
                diag = 1.0
        except Exception:
            diag = 1.0

        centers_arr = np.array(centers)
        cloud = pv.PolyData(centers_arr)
        sphere_src = pv.Sphere(
            radius=0.005 * diag * picker._point_size,
            theta_resolution=10, phi_resolution=10,
        )
        glyphs = cloud.glyph(geom=sphere_src, orient=False, scale=False)

        n_cells_per_pt = glyphs.n_cells // len(centers) if len(centers) else 1
        entity_tags = np.empty(glyphs.n_cells, dtype=np.int64)
        colors = np.tile(idle_rgb, (glyphs.n_cells, 1))
        cell_to_dt: dict[int, tuple] = {}

        for i, tag in enumerate(tags):
            start = i * n_cells_per_pt
            end = start + n_cells_per_pt
            entity_tags[start:end] = tag
            dt = (0, tag)
            picker._batch_dt_to_cells[dt] = list(range(start, end))
            for ci in range(start, end):
                cell_to_dt[ci] = dt

        glyphs.cell_data["entity_tag"] = entity_tags
        glyphs.cell_data["colors"] = colors

        actor = plotter.add_mesh(
            glyphs, scalars="colors", rgb=True,
            smooth_shading=True, pickable=True,
            reset_camera=False,
        )
        picker._batch_actors[0] = actor
        picker._batch_meshes[0] = glyphs
        picker._batch_cell_to_dt[0] = cell_to_dt
        for tag in tags:
            picker._id_to_actor[(0, tag)] = actor
        picker._actor_to_id[id(actor)] = (0, -1)

    def _help_extra_rows(self):
        """BRep-specific shortcut rows for the help dialog."""
        return [
            ("1", "Points only"),
            ("2", "Curves only"),
            ("3", "Surfaces only"),
            ("4", "Volumes only"),
            ("0", "All dims (reset filter)"),
            ("H", "Hide currently picked entities"),
            ("I", "Isolate picks (hide everything else)"),
            ("R", "Reveal all hidden entities"),
            ("U", "Undo last pick"),
            ("Tab", "Cycle through overlapping entities at the "
                    "last click position (Revit-style)"),
            ("Esc", "Deselect all (clear the working set)"),
            ("Q", "Close window"),
            ("Click group header",
             "Activate that physical group"),
            ("Click entity leaf",
             "Toggle its pick in the active group"),
            ("Right-click group",
             "Rename / delete / set-active menu"),
            ("Click instance",
             "Select all entities of that Assembly instance"),
            ("Right-click instance",
             "Create a physical group from an instance"),
        ]

    # ==================================================================
    # Browser tab (model tree + group management)
    # ==================================================================

    def _build_browser_tab(self):
        """Tab 1 -- Project Browser (model tree) with integrated group
        management controls at the bottom."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Tree (takes most space) ---
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Entity / group", "Info"])
        self._tree.setColumnWidth(0, 200)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._tree.itemClicked.connect(self._on_tree_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_tree_item_dblclicked)
        layout.addWidget(self._tree, stretch=1)  # tree expands

        # --- Group management section (fixed at bottom) ---
        # Name input
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Name:"))
        self._group_name_edit = QtWidgets.QLineEdit()
        self._group_name_edit.setPlaceholderText("group name\u2026")
        name_row.addWidget(self._group_name_edit)
        layout.addLayout(name_row)

        # Buttons row 1
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setToolTip("Create / overwrite group with current picks")
        btn_apply.clicked.connect(self._grp_apply)
        row1.addWidget(btn_apply)
        btn_modify = QtWidgets.QPushButton("Modify")
        btn_modify.setToolTip("Load group members into picks for editing")
        btn_modify.clicked.connect(self._grp_modify)
        row1.addWidget(btn_modify)
        layout.addLayout(row1)

        # Buttons row 2
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(4)
        btn_select = QtWidgets.QPushButton("Select")
        btn_select.setToolTip("Preview group members (load into picks)")
        btn_select.clicked.connect(self._grp_select)
        row2.addWidget(btn_select)
        btn_deselect = QtWidgets.QPushButton("Deselect")
        btn_deselect.setToolTip("Clear all current picks")
        btn_deselect.clicked.connect(self._action_deselect_all)
        row2.addWidget(btn_deselect)
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.setToolTip("Delete the named group")
        btn_delete.clicked.connect(self._grp_delete)
        row2.addWidget(btn_delete)
        layout.addLayout(row2)

        return panel

    # ==================================================================
    # View tab (entity labels)
    # ==================================================================

    def _build_view_tab(self):
        """View tab -- toggle entity labels/tags on screen."""
        QtWidgets = self._QtWidgets

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        group = QtWidgets.QGroupBox("Show entity labels on screen")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._view_label_cbs: dict[int, QtWidgets.QCheckBox] = {}
        for dim, name in _DIM_NAMES.items():
            cb = QtWidgets.QCheckBox(f"{name} tags")
            cb.setChecked(False)
            cb.toggled.connect(self._on_view_labels_changed)
            group_layout.addWidget(cb)
            self._view_label_cbs[dim] = cb

        layout.addWidget(group)

        # Label style options
        style_group = QtWidgets.QGroupBox("Label style")
        style_layout = QtWidgets.QFormLayout(style_group)
        style_layout.setSpacing(4)

        self._label_size_spin = QtWidgets.QSpinBox()
        self._label_size_spin.setRange(6, 24)
        self._label_size_spin.setValue(10)
        self._label_size_spin.valueChanged.connect(self._on_view_labels_changed)
        style_layout.addRow("Font size", self._label_size_spin)

        self._label_use_names = QtWidgets.QCheckBox("Show names instead of tags")
        self._label_use_names.setChecked(False)
        self._label_use_names.toggled.connect(self._on_view_labels_changed)
        style_layout.addRow(self._label_use_names)

        layout.addWidget(style_group)
        layout.addStretch(1)

        return panel

    def _on_view_labels_changed(self) -> None:
        """Add or remove entity label overlays in the 3D viewport."""
        import numpy as np

        plotter = self._picker._plotter
        if plotter is None:
            return

        # Remove existing label actors
        for actor in self._view_label_actors:
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        self._view_label_actors.clear()

        registry = self._picker._model._registry
        try:
            use_names = self._label_use_names.isChecked()
            font_size = self._label_size_spin.value()
        except RuntimeError:
            # Qt widget was deleted (e.g. View tab removed by subclass)
            return

        # Read font prefs (available after prefs tab is built).
        font_family = "arial"
        if hasattr(self, "_font_combo"):
            font_family = self._font_combo.currentText().lower()
        if hasattr(self, "_font_size_spin"):
            font_size = self._font_size_spin.value()
        if hasattr(self, "_font_color"):
            text_color = self._font_color
        else:
            text_color = "white"

        for dim, cb in self._view_label_cbs.items():
            if not cb.isChecked():
                continue
            points = []
            labels = []
            for _, tag in gmsh.model.getEntities(dim=dim):
                try:
                    bb = gmsh.model.getBoundingBox(dim, tag)
                    cx = (bb[0] + bb[3]) * 0.5
                    cy = (bb[1] + bb[4]) * 0.5
                    cz = (bb[2] + bb[5]) * 0.5
                    points.append([cx, cy, cz])
                except Exception:
                    continue
                if use_names:
                    info = registry.get((dim, tag), {})
                    lbl = info.get("label", f"{_DIM_ABBR[dim]}{tag}")
                else:
                    lbl = f"{_DIM_ABBR[dim]}{tag}"
                labels.append(lbl)

            if not points:
                continue

            pts = np.array(points)
            try:
                actor = plotter.add_point_labels(
                    pts, labels,
                    font_size=font_size,
                    font_family=font_family,
                    text_color=text_color,
                    shape_color="#333333",
                    shape_opacity=0.6,
                    show_points=False,
                    always_visible=True,
                    name=f"_view_labels_dim{dim}",
                )
                self._view_label_actors.append(actor)
            except Exception:
                pass

        plotter.render()

    # ==================================================================
    # Filter tab
    # ==================================================================

    def _build_filter_tab(self):
        """Filter tab -- dimension filter, entity label filter, text filter."""
        QtWidgets = self._QtWidgets

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Dimension filter
        dim_group = QtWidgets.QGroupBox("Dimension filter")
        dim_layout = QtWidgets.QVBoxLayout(dim_group)
        self._dim_cbs = {}
        pickable = self._picker._pickable_dims
        for d, name in _DIM_NAMES.items():
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(d in pickable)
            cb.toggled.connect(self._on_dim_filter_changed)
            dim_layout.addWidget(cb)
            self._dim_cbs[d] = cb
        layout.addWidget(dim_group)

        # Entity labels filter (checkboxes for each label found in registry)
        label_group = QtWidgets.QGroupBox("Entity labels")
        label_layout = QtWidgets.QVBoxLayout(label_group)
        self._label_cbs: dict[str, QtWidgets.QCheckBox] = {}
        # Collect unique labels from the model registry
        registry = self._picker._model._registry
        labels: set[str] = set()
        for dt, info in registry.items():
            lbl = info.get("label", "")
            if lbl:
                labels.add(lbl)
        if labels:
            for lbl in sorted(labels):
                cb = QtWidgets.QCheckBox(lbl)
                cb.setChecked(True)
                cb.toggled.connect(self._on_label_filter_changed)
                label_layout.addWidget(cb)
                self._label_cbs[lbl] = cb
        else:
            label_layout.addWidget(
                QtWidgets.QLabel("(no labelled entities)")
            )
        # Scroll area in case there are many labels
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label_group)
        scroll.setMaximumHeight(200)
        layout.addWidget(scroll)

        # Name / text filter
        name_group = QtWidgets.QGroupBox("Name filter")
        name_layout = QtWidgets.QVBoxLayout(name_group)
        self._name_filter = QtWidgets.QLineEdit()
        self._name_filter.setPlaceholderText("Type to filter entities\u2026")
        self._name_filter.textChanged.connect(self._on_name_filter_changed)
        name_layout.addWidget(self._name_filter)
        layout.addWidget(name_group)

        layout.addStretch(1)
        return panel

    def _on_dim_filter_changed(self) -> None:
        """Sync dim-filter checkboxes -> picker._set_pickable_dims."""
        dims = set()
        for d, cb in self._dim_cbs.items():
            if cb.isChecked():
                dims.add(d)
        if not dims:
            dims = set(self._picker._dims)  # at least one
        self._picker._set_pickable_dims(dims)
        self._populate_tree()

    def _on_label_filter_changed(self) -> None:
        """Show/hide entities in the 3D view and tree based on which
        label checkboxes are checked."""
        if not hasattr(self, "_label_cbs"):
            return
        checked_labels = {
            lbl for lbl, cb in self._label_cbs.items() if cb.isChecked()
        }
        registry = self._picker._model._registry
        for dt, actor in self._picker._dimtag_to_actor.items():
            info = registry.get(dt, {})
            lbl = info.get("label", "")
            # If no label, always show; if label exists, check filter
            if lbl and lbl not in checked_labels:
                actor.VisibilityOff()
                actor.SetPickable(False)
            else:
                if dt not in self._picker._hidden:
                    actor.VisibilityOn()
                    if dt[0] in self._picker._pickable_dims:
                        actor.SetPickable(True)
        self._picker._plotter.render()
        # Also filter tree leaves
        for dt, leaf in self._tree_items_by_dt.items():
            info = registry.get(dt, {})
            lbl = info.get("label", "")
            if lbl and lbl not in checked_labels:
                leaf.setHidden(True)
            else:
                leaf.setHidden(False)

    def _on_name_filter_changed(self, text: str) -> None:
        """Hide tree leaves whose label doesn't match the text filter."""
        text = text.strip().lower()
        for dt, leaf in self._tree_items_by_dt.items():
            if not text:
                leaf.setHidden(False)
            else:
                label = leaf.text(0).lower()
                leaf.setHidden(text not in label)

    def _sync_dim_checkboxes(self) -> None:
        """Update the Filter tab checkboxes to match the current
        ``_pickable_dims`` (called after keyboard shortcuts 1-4/0)."""
        if not hasattr(self, "_dim_cbs"):
            return
        pickable = self._picker._pickable_dims
        for d, cb in self._dim_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(d in pickable)
            cb.blockSignals(False)

    # ==================================================================
    # Entity Info dock
    # ==================================================================

    def _build_info_tab(self):
        """Entity Info -- BRep topology tree for the hovered entity."""
        QtWidgets = self._QtWidgets

        self._info_tree = QtWidgets.QTreeWidget()
        self._info_tree.setHeaderLabels(["Entity", "Details"])
        self._info_tree.setColumnWidth(0, 180)
        self._info_tree.setAlternatingRowColors(True)
        self._info_tree.setRootIsDecorated(True)
        return self._info_tree

    def _refresh_info(self) -> None:
        """Rebuild the BRep tree for the hovered entity."""
        QtWidgets = self._QtWidgets

        self._info_tree.clear()
        dt = self._picker._hover_dt
        if dt is None:
            item = QtWidgets.QTreeWidgetItem(["(hover over an entity)"])
            self._info_tree.addTopLevelItem(item)
            return

        registry = self._picker._model._registry

        def _make_node(dim: int, tag: int, parent_item=None):
            """Create a tree item for (dim, tag) and recurse into
            its boundary children."""
            info = registry.get((dim, tag), {})
            label = info.get("label", f"{_DIM_ABBR[dim]}{tag}")
            kind = info.get("kind", "")

            # Details column
            details = f"dim={dim}  tag={tag}"
            try:
                bb = gmsh.model.getBoundingBox(dim, tag)
                details += (
                    f"  bbox=({bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f})"
                    f"\u2192({bb[3]:.1f},{bb[4]:.1f},{bb[5]:.1f})"
                )
            except Exception:
                pass
            if kind:
                details += f"  {kind}"

            # Pick state badge
            if (dim, tag) in self._picker._picks:
                label += "  [PICKED]"
            elif (dim, tag) in self._picker._hidden:
                label += "  [hidden]"

            node = QtWidgets.QTreeWidgetItem([label, details])
            if parent_item is None:
                self._info_tree.addTopLevelItem(node)
            else:
                parent_item.addChild(node)

            # Recurse into boundary (one level down)
            if dim > 0:
                try:
                    boundary = gmsh.model.getBoundary(
                        [(dim, tag)], combined=False,
                        oriented=False, recursive=False,
                    )
                    seen = set()
                    for bd, bt in boundary:
                        if (bd, bt) not in seen:
                            seen.add((bd, bt))
                            _make_node(bd, bt, parent_item=node)
                except Exception:
                    pass
            return node

        dim, tag = dt
        root = _make_node(dim, tag)

        # Physical groups this entity belongs to
        try:
            groups = []
            for d, pg in gmsh.model.getPhysicalGroups():
                if d != dim:
                    continue
                ents = gmsh.model.getEntitiesForPhysicalGroup(d, pg)
                if tag in ents:
                    name = gmsh.model.getPhysicalName(d, pg) or f"pg={pg}"
                    groups.append(name)
            if groups:
                pg_node = QtWidgets.QTreeWidgetItem(
                    ["Groups", ", ".join(groups)]
                )
                root.addChild(pg_node)
        except Exception:
            pass

        # Instance info
        parent = self._picker._parent
        if hasattr(parent, "instances"):
            for lbl, inst in parent.instances.items():
                if tag in inst.entities.get(dim, []):
                    inst_node = QtWidgets.QTreeWidgetItem(
                        ["Instance", f"{lbl} (Part '{inst.part_name}')"]
                    )
                    root.addChild(inst_node)
                    break

        root.setExpanded(True)

    # ==================================================================
    # Group management methods
    # ==================================================================

    def _collect_groups(self) -> dict[str, list[DimTag]]:
        """Union of Gmsh physical groups and session-staged groups.
        Staged entries override Gmsh entries on name collision.  The
        active group is always shown with its current working picks."""
        groups: dict[str, list[DimTag]] = {}

        # From Gmsh
        for d, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(d, pg_tag)
            except Exception:
                name = ""
            if not name:
                name = f"(unnamed:dim={d},tag={pg_tag})"
            bucket = groups.setdefault(name, [])
            try:
                for t in gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag):
                    bucket.append((d, int(t)))
            except Exception:
                pass

        # Overlay staged groups (explicit edits this session)
        for name, members in self._picker.staged_groups.items():
            groups[name] = list(members)

        # Active group always reflects the *current* working picks
        if self._picker.is_editing:
            groups[self._picker.active_group] = list(self._picker.tags)

        return groups

    def _grp_name(self) -> str:
        return self._group_name_edit.text().strip()

    def _grp_apply(self) -> None:
        """Apply: stage current picks as the named group."""
        name = self._grp_name()
        if not name:
            self._statusbar.showMessage("Enter a group name first.", 3000)
            return
        if self._picker.group_exists(name):
            if not self._confirm_overwrite(name):
                return
        self._picker.apply_group(name)
        self._populate_tree()
        self._refresh_statusbar()
        self.log(f"Applied '{name}' ({len(self._picker.tags)} entities)")

    def _grp_modify(self) -> None:
        """Modify: overwrite the named group with the current picks."""
        name = self._grp_name()
        if not name:
            self._statusbar.showMessage("Enter a group name first.", 3000)
            return
        if not self._picker.tags:
            self._statusbar.showMessage("Nothing selected to assign.", 3000)
            return
        self._picker.apply_group(name)
        self._populate_tree()
        self._refresh_statusbar()
        self.log(f"Modified '{name}' ({len(self._picker.tags)} entities)")

    def _grp_select(self) -> None:
        """Select: preview the named group's members (load into picks).

        Does NOT set the group as active for editing.
        """
        from pyGmsh.viewers.SelectionPicker import _load_physical_group_members
        name = self._grp_name()
        if not name:
            return
        members = self._picker.staged_groups.get(name)
        if members is None:
            members = _load_physical_group_members(name)
        if not members:
            self._statusbar.showMessage(
                f"Group '{name}' is empty or does not exist.", 3000,
            )
            return
        self._picker.select_dimtags(members, replace=True)
        self.log(f"Previewing group '{name}' ({len(members)} entities)")

    def _grp_delete(self) -> None:
        """Delete: remove the named group."""
        name = self._grp_name()
        if not name:
            return
        self._delete_group(name)

    def _group_name_taken(self, name: str) -> bool:
        """True if *name* already refers to a group -- either staged in
        this session with non-empty members, or present in Gmsh."""
        return self._picker.group_exists(name)

    def _confirm_overwrite(self, name: str) -> bool:
        """Prompt before overwriting an existing group."""
        QtWidgets = self._QtWidgets
        reply = QtWidgets.QMessageBox.question(
            self._window,
            "Group name already exists",
            f"A physical group called '{name}' already exists.\n"
            f"Overwrite its members?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        return reply == QtWidgets.QMessageBox.Yes

    # ==================================================================
    # Group action toolbar methods
    # ==================================================================

    def _action_new_group(self) -> None:
        QtWidgets = self._QtWidgets
        name, ok = QtWidgets.QInputDialog.getText(
            self._window, "New physical group", "Group name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._group_name_taken(name):
            if not self._confirm_overwrite(name):
                return
        self._picker.set_active_group(name)
        self._populate_tree()
        self._refresh_statusbar()
        self.log(f"Created group '{name}'")

    def _action_rename_active(self) -> None:
        active = self._picker.active_group
        if not active:
            self._statusbar.showMessage("No active group to rename.", 3000)
            return
        self._rename_group(active)

    def _action_delete_active(self) -> None:
        active = self._picker.active_group
        if not active:
            self._statusbar.showMessage("No active group to delete.", 3000)
            return
        self._delete_group(active)

    def _rename_group(self, old_name: str) -> None:
        QtWidgets = self._QtWidgets
        new_name, ok = QtWidgets.QInputDialog.getText(
            self._window, "Rename group", "New name:", text=old_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if self._group_name_taken(new_name):
            if not self._confirm_overwrite(new_name):
                return
        self._picker.rename_group(old_name, new_name)
        self._populate_tree()
        self._refresh_statusbar()
        self.log(f"Renamed '{old_name}' -> '{new_name}'")

    def _delete_group(self, name: str) -> None:
        self._picker.delete_group(name)
        self._populate_tree()
        self.log(f"Deleted group '{name}'")
        self._refresh_statusbar()

    def _action_deselect_all(self) -> None:
        """Esc / Space -- two-stage deselect.

        **Stage 1**: if picks differ from the last Apply, revert to the
        applied group members.

        **Stage 2**: if picks already match the applied state (or no
        group is active), clear all picks and deactivate the group.

        Every applied group takes exactly 2 presses to fully clear.
        Unapplied selections clear in 1 press.  No cycling.
        """
        if not self._picker.revert():
            self._picker.clear()
        self._populate_tree()

    # ==================================================================
    # Tree management
    # ==================================================================

    def _populate_tree(self) -> None:
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        self._tree.clear()
        self._tree_items_by_dt.clear()
        self._tree_items_by_group.clear()

        groups = self._collect_groups()
        assigned: set[DimTag] = {dt for mems in groups.values() for dt in mems}
        active = self._picker.active_group
        registry = self._picker._model._registry

        # ---- Physical groups root ----
        pg_root = QtWidgets.QTreeWidgetItem(
            ["Physical groups", f"{len(groups)}"]
        )
        pg_root.setFont(0, _bold(QtWidgets))
        self._tree.addTopLevelItem(pg_root)
        pg_root.setExpanded(True)

        for gname in sorted(groups.keys()):
            members = groups[gname]
            header = QtWidgets.QTreeWidgetItem(
                [_group_label(gname, active, len(members)),
                 f"{len(members)} entities"]
            )
            header.setData(0, QtCore.Qt.UserRole, ("group", gname))
            if gname == active:
                header.setFont(0, _bold(QtWidgets))
            pg_root.addChild(header)
            self._tree_items_by_group[gname] = header

            for dt in members:
                leaf = self._make_entity_leaf(dt, registry)
                header.addChild(leaf)

        # ---- Unassigned root ----
        unassigned: dict[int, list[DimTag]] = {0: [], 1: [], 2: [], 3: []}
        for dim in (0, 1, 2, 3):
            for _, tag in gmsh.model.getEntities(dim=dim):
                dt = (dim, int(tag))
                if dt not in assigned:
                    unassigned[dim].append(dt)
        total_unassigned = sum(len(v) for v in unassigned.values())

        u_root = QtWidgets.QTreeWidgetItem(
            ["Unassigned", f"{total_unassigned}"]
        )
        u_root.setFont(0, _bold(QtWidgets))
        self._tree.addTopLevelItem(u_root)
        u_root.setExpanded(False)

        for dim in (0, 1, 2, 3):
            members = unassigned[dim]
            if not members:
                continue
            header = QtWidgets.QTreeWidgetItem(
                [_DIM_NAMES[dim], f"{len(members)}"]
            )
            header.setData(0, QtCore.Qt.UserRole, ("dim-bucket", dim))
            u_root.addChild(header)
            for dt in members:
                leaf = self._make_entity_leaf(dt, registry)
                header.addChild(leaf)

        # ---- Instances root (Assembly context only) ----
        self._build_instances_root()

        # Apply pick coloring in the fresh tree
        self._refresh_tree_picks()

    def _build_instances_root(self) -> None:
        """Append an 'Instances' root showing each Assembly instance and
        its entities bucketed by dim.  No-op when the picker's parent is
        not an Assembly."""
        parent = self._picker._parent
        if not hasattr(parent, "instances"):
            return
        instances = parent.instances
        if not instances:
            return

        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        i_root = QtWidgets.QTreeWidgetItem(
            ["Instances", f"{len(instances)}"]
        )
        i_root.setFont(0, _bold(QtWidgets))
        self._tree.addTopLevelItem(i_root)
        i_root.setExpanded(True)

        for label, inst in instances.items():
            total = sum(len(v) for v in inst.entities.values())
            header = QtWidgets.QTreeWidgetItem(
                [f"{label}  \u2190 Part '{inst.part_name}'",
                 f"{total} entities"]
            )
            header.setData(
                0, QtCore.Qt.UserRole, ("instance", label),
            )
            i_root.addChild(header)
            for dim in (0, 1, 2, 3):
                tags = inst.entities.get(dim) or []
                if not tags:
                    continue
                bucket = QtWidgets.QTreeWidgetItem(
                    [_DIM_NAMES[dim], f"{len(tags)}"]
                )
                bucket.setData(
                    0, QtCore.Qt.UserRole,
                    ("instance-dim", label, dim),
                )
                header.addChild(bucket)

    def _make_entity_leaf(self, dt: DimTag, registry: dict):
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore
        dim, tag = dt
        info = registry.get(dt, {})
        label = info.get("label", f"{_DIM_ABBR[dim]}{tag}")
        kind = info.get("kind", "")
        leaf = QtWidgets.QTreeWidgetItem(
            [label, f"dim={dim} tag={tag}  {kind}"]
        )
        leaf.setData(0, QtCore.Qt.UserRole, ("entity", dim, tag))
        self._tree_items_by_dt[dt] = leaf
        return leaf

    # ------------------------------------------------------------------
    # Tree interactions
    # ------------------------------------------------------------------

    def _on_tree_item_clicked(self, item, _column: int) -> None:
        QtCore = self._QtCore
        if self._syncing:
            return
        data = item.data(0, QtCore.Qt.UserRole)
        if data is None:
            return
        kind = data[0]
        if kind == "entity":
            dt = (int(data[1]), int(data[2]))
            self._picker._toggle_pick(dt)
        elif kind == "group":
            gname = data[1]
            self._group_name_edit.setText(gname)  # populate name field
            # Don't call set_active_group -- that's what Modify is for
        elif kind == "instance":
            label = data[1]
            dts = self._instance_dimtags(label)
            self._picker.select_dimtags(dts, replace=True)
        elif kind == "instance-dim":
            label, dim = data[1], int(data[2])
            dts = self._instance_dimtags(label, dim=dim)
            self._picker.select_dimtags(dts, replace=True)

    def _on_tree_item_dblclicked(self, item, _column):
        """Double-clicking a group header previews its members."""
        QtCore = self._QtCore
        data = item.data(0, QtCore.Qt.UserRole)
        if data is None:
            return
        kind = data[0]
        if kind == "group":
            gname = data[1]
            self._group_name_edit.setText(gname)
            self._grp_select()

    def _instance_dimtags(
        self, label: str, *, dim: int | None = None,
    ) -> list[DimTag]:
        """Return the list of DimTags for one Assembly instance."""
        parent = self._picker._parent
        if not hasattr(parent, "instances"):
            return []
        inst = parent.instances.get(label)
        if inst is None:
            return []
        out: list[DimTag] = []
        for d, tags in inst.entities.items():
            if dim is not None and d != dim:
                continue
            for t in tags:
                out.append((int(d), int(t)))
        return out

    def _on_tree_context_menu(self, pos) -> None:
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore
        item = self._tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, QtCore.Qt.UserRole)
        if data is None:
            return
        kind = data[0]

        if kind == "group":
            gname = data[1]
            menu = QtWidgets.QMenu(self._tree)
            act_select = menu.addAction("Select")
            act_modify = menu.addAction("Modify (assign current picks)")
            menu.addSeparator()
            act_rename = menu.addAction("Rename\u2026")
            act_delete = menu.addAction("Delete")
            chosen = menu.exec_(self._tree.mapToGlobal(pos))
            if chosen is None:
                return
            self._group_name_edit.setText(gname)
            if chosen == act_select:
                self._grp_select()
            elif chosen == act_modify:
                self._grp_modify()
            elif chosen == act_rename:
                self._rename_group(gname)
            elif chosen == act_delete:
                self._delete_group(gname)
            return

        if kind in ("instance", "instance-dim"):
            if kind == "instance":
                label = data[1]
                dts = self._instance_dimtags(label)
                suggested = label
            else:
                label, dim = data[1], int(data[2])
                dts = self._instance_dimtags(label, dim=dim)
                suggested = f"{label}_{_DIM_NAMES[dim].lower()}"
            menu = QtWidgets.QMenu(self._tree)
            act_select = menu.addAction("Select entities")
            act_group = menu.addAction(
                "Create physical group from instance\u2026"
            )
            chosen = menu.exec_(self._tree.mapToGlobal(pos))
            if chosen == act_select:
                self._picker.select_dimtags(dts, replace=True)
            elif chosen == act_group:
                self._create_group_from_dimtags(dts, suggested=suggested)
            return

    def _create_group_from_dimtags(
        self, dts: list[DimTag], *, suggested: str,
    ) -> None:
        """Prompt for a group name, stage it, and refresh the tree."""
        QtWidgets = self._QtWidgets
        if not dts:
            self._statusbar.showMessage(
                "Instance has no entities in the current scene.", 3000,
            )
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self._window, "New physical group", "Group name:",
            text=suggested,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._group_name_taken(name):
            if not self._confirm_overwrite(name):
                return
        self._picker._staged_groups[name] = list(dts)
        self._picker.set_active_group(name)
        self._populate_tree()
        self._refresh_statusbar()

    # ------------------------------------------------------------------
    # Tree refresh (bidirectional sync)
    # ------------------------------------------------------------------

    def _refresh_tree_picks(self) -> None:
        """Highlight leaves whose DimTag is in the current pick set."""
        QtGui = self._QtGui
        self._syncing = True
        try:
            picks = set(self._picker.tags)
            pick_brush = QtGui.QBrush(
                QtGui.QColor(self._picker._pick_color)
            )
            idle_brush = QtGui.QBrush()  # default
            active = self._picker.active_group
            if active is not None and active in self._tree_items_by_group:
                header = self._tree_items_by_group[active]
                header.setText(1, f"{len(picks)} entities")
            for dt, leaf in self._tree_items_by_dt.items():
                leaf.setForeground(
                    0, pick_brush if dt in picks else idle_brush,
                )
            self._refresh_statusbar()
        finally:
            self._syncing = False

    def _refresh_tree_visibility(self) -> None:
        """Dim leaves whose DimTag is in the hidden set."""
        QtGui = self._QtGui
        hidden = self._picker._hidden
        dim_brush = QtGui.QBrush(QtGui.QColor("#777777"))
        normal_brush = QtGui.QBrush()
        for dt, leaf in self._tree_items_by_dt.items():
            leaf.setForeground(
                1, dim_brush if dt in hidden else normal_brush,
            )

    # ==================================================================
    # Selection-specific preference callbacks
    # ==================================================================

    def _on_point_size_changed(self, value: int) -> None:
        """Point-spheres use SetScale -- override base to handle dim=0
        actors specially."""
        scale = float(value)
        self._picker._point_size = scale
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d != 0:
                continue
            try:
                actor.SetScale(scale, scale, scale)
            except Exception:
                pass
        self._qt_interactor.render()

    def _on_line_width_changed(self, value: int) -> None:
        self._picker._line_width = float(value)
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d != 1:
                continue
            try:
                actor.GetProperty().SetLineWidth(float(value))
            except Exception:
                pass
        self._qt_interactor.render()

    def _on_opacity_changed(self, value: int) -> None:
        """Apply the slider alpha to dim=2 / dim=3 actors, respecting
        the current pickable_dims filter."""
        alpha = float(value) / 100.0
        self._picker._surface_opacity = alpha
        vol_alpha = max(0.05, alpha * 0.6)
        pickable_dims = self._picker._pickable_dims
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d not in (2, 3):
                continue
            if d not in pickable_dims:
                continue   # dim is filtered out -- keep dimmed
            try:
                actor.GetProperty().SetOpacity(
                    alpha if d == 2 else vol_alpha,
                )
            except Exception:
                pass
        self._qt_interactor.render()

    def _on_edges_toggled(self, checked: bool) -> None:
        self._picker._show_surface_edges = bool(checked)
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d != 2:
                continue
            try:
                actor.GetProperty().SetEdgeVisibility(bool(checked))
            except Exception:
                pass
        self._qt_interactor.render()

    def _update_color_btn_style(self, hex_color: str) -> None:
        """Set the button's background to the current pick colour."""
        self._color_btn.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #888;"
        )
        self._color_btn.setToolTip(hex_color)

    def _on_pick_color_clicked(self) -> None:
        """Open a QColorDialog and apply the chosen colour."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui
        cur = QtGui.QColor(self._picker._pick_color)
        color = QtWidgets.QColorDialog.getColor(
            cur, self._window, "Selection highlight color",
        )
        if not color.isValid():
            return
        hex_color = color.name()   # "#RRGGBB"
        self._picker._pick_color = hex_color
        self._update_color_btn_style(hex_color)
        self._picker._recolor_all()
        self._refresh_tree_picks()

    def _on_font_changed(self) -> None:
        """Re-render entity labels in the viewport when font changes."""
        if hasattr(self, "_view_label_cbs"):
            self._on_view_labels_changed()

    def _on_font_color_clicked(self) -> None:
        """Open a color dialog for the label font color."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui
        cur = QtGui.QColor(self._font_color)
        color = QtWidgets.QColorDialog.getColor(
            cur, self._window, "Label font color",
        )
        if not color.isValid():
            return
        self._font_color = color.name()
        self._font_color_btn.setStyleSheet(
            f"background-color: {self._font_color}; border: 1px solid #888;"
        )
        if hasattr(self, "_view_label_cbs"):
            self._on_view_labels_changed()

    # ==================================================================
    # Console logging (selection-specific)
    # ==================================================================

    def _log_pick_changed(self) -> None:
        """Console-log the pick state after each change."""
        tags = self._picker.tags
        n = len(tags)
        if n == 0:
            self.log("Selection cleared")
        else:
            last = tags[-1] if tags else None
            self.log(f"Picks: {n} total (last: {last})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_label(name: str, active: str | None, count: int) -> str:
    marker = " \u2605" if name == active else ""
    return f"{name}{marker}  ({count})"


def _bold(QtWidgets):
    """Return a bold QFont for tree headers."""
    from qtpy import QtGui
    f = QtGui.QFont()
    f.setBold(True)
    return f
