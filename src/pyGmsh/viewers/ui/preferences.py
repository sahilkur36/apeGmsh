"""
PreferencesTab — Shared visual settings panel.

A standalone QWidget that can be added as a tab in any viewer.
Fires callbacks for each setting change — the viewer wires these
to ColorManager, PickEngine, and actor properties.

Usage::

    from pyGmsh.viewers.ui.preferences import PreferencesTab
    prefs = PreferencesTab(
        point_size=10, line_width=6, surface_opacity=0.35,
        on_point_size=lambda v: ...,
        on_line_width=lambda v: ...,
        on_opacity=lambda v: ...,
    )
"""
from __future__ import annotations

from typing import Callable


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


class PreferencesTab:
    """Preferences panel as a standalone QWidget."""

    def __init__(
        self,
        *,
        point_size: float = 10.0,
        line_width: float = 6.0,
        surface_opacity: float = 0.35,
        show_surface_edges: bool = False,
        drag_threshold: int = 8,
        # Callbacks
        on_point_size: Callable[[float], None] | None = None,
        on_line_width: Callable[[float], None] | None = None,
        on_opacity: Callable[[float], None] | None = None,
        on_edges: Callable[[bool], None] | None = None,
        on_aa: Callable[[bool], None] | None = None,
        on_drag_threshold: Callable[[int], None] | None = None,
        on_theme: Callable[[str], None] | None = None,
        on_pick_color: Callable[[str], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()

        self.widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.widget)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        # ── Point size (pixels) ─────────────────────────────────────
        self._s_point = QtWidgets.QSpinBox()
        self._s_point.setRange(1, 50)
        self._s_point.setValue(int(point_size))
        self._s_point.setSuffix(" px")
        if on_point_size:
            self._s_point.valueChanged.connect(
                lambda v: on_point_size(float(v))
            )
        form.addRow("Point size", self._s_point)

        # ── Line width ──────────────────────────────────────────────
        self._s_line = QtWidgets.QDoubleSpinBox()
        self._s_line.setRange(0.5, 20.0)
        self._s_line.setSingleStep(0.5)
        self._s_line.setDecimals(1)
        self._s_line.setValue(float(line_width))
        if on_line_width:
            self._s_line.valueChanged.connect(on_line_width)
        form.addRow("Line width", self._s_line)

        # ── Surface opacity ─────────────────────────────────────────
        self._s_alpha = QtWidgets.QDoubleSpinBox()
        self._s_alpha.setRange(0.0, 1.0)
        self._s_alpha.setSingleStep(0.05)
        self._s_alpha.setDecimals(2)
        self._s_alpha.setValue(float(surface_opacity))
        if on_opacity:
            self._s_alpha.valueChanged.connect(on_opacity)
        form.addRow("Surface \u03b1", self._s_alpha)

        # ── Show edges ──────────────────────────────────────────────
        self._cb_edges = QtWidgets.QCheckBox("Show surface edges")
        self._cb_edges.setChecked(show_surface_edges)
        if on_edges:
            self._cb_edges.toggled.connect(on_edges)
        form.addRow(self._cb_edges)

        # ── Anti-aliasing ───────────────────────────────────────────
        self._cb_aa = QtWidgets.QCheckBox("Anti-aliasing (SSAA)")
        self._cb_aa.setChecked(True)
        if on_aa:
            self._cb_aa.toggled.connect(on_aa)
        form.addRow(self._cb_aa)

        # ── Drag threshold ──────────────────────────────────────────
        self._s_drag = QtWidgets.QSpinBox()
        self._s_drag.setRange(2, 30)
        self._s_drag.setValue(drag_threshold)
        if on_drag_threshold:
            self._s_drag.valueChanged.connect(on_drag_threshold)
        form.addRow("Drag threshold (px)", self._s_drag)

        # ── Theme ───────────────────────────────────────────────────
        form.addRow(QtWidgets.QLabel(""))  # spacer
        self._theme_combo = QtWidgets.QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        if on_theme:
            self._theme_combo.currentTextChanged.connect(on_theme)
        form.addRow("Theme", self._theme_combo)

        # ── Pick color ──────────────────────────────────────────────
        self._btn_pick_color = QtWidgets.QPushButton()
        self._btn_pick_color.setFixedSize(60, 24)
        self._pick_color_hex = "#E74C3C"
        self._btn_pick_color.setStyleSheet(
            f"background-color: {self._pick_color_hex}; border: 1px solid #999;"
        )
        self._on_pick_color = on_pick_color

        def _pick_color_clicked():
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(self._pick_color_hex),
                self.widget,
                "Pick Selection Color",
            )
            if color.isValid():
                self._pick_color_hex = color.name()
                self._btn_pick_color.setStyleSheet(
                    f"background-color: {self._pick_color_hex}; "
                    f"border: 1px solid #999;"
                )
                if self._on_pick_color:
                    self._on_pick_color(self._pick_color_hex)

        self._btn_pick_color.clicked.connect(_pick_color_clicked)
        form.addRow("Selection color", self._btn_pick_color)

    def add_extra_row(self, label: str, widget) -> None:
        """Add an extra row to the form (for viewer-specific prefs)."""
        form = self.widget.layout()
        form.addRow(label, widget)

    def add_separator(self, text: str = "") -> None:
        """Add a visual separator with optional label."""
        from qtpy import QtWidgets
        form = self.widget.layout()
        form.addRow(QtWidgets.QLabel(""))
        if text:
            form.addRow(QtWidgets.QLabel(f"--- {text} ---"))
