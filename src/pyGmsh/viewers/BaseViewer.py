"""
BaseViewer
==========

Generic base class for 3D viewers backed by PyVista / VTK.

Contains the shared plotter infrastructure (background, axes widget,
anti-aliasing, depth peeling), the custom interactor wiring (LMB pick /
box-select, MMB pan / quaternion orbit, RMB pan, hover highlight), and
hide / isolate / reveal helpers.

Subclasses override the virtual hooks to plug in domain-specific logic:

* **SelectionPicker** (BRep selection) -- entity IDs are ``DimTag`` tuples,
  actors are coloured per BRep-dimension, box-select projects centroids.
* **MeshViewer** (mesh display) -- entity IDs are node / element tags,
  actors are coloured by result fields, etc.

Usage
-----
Subclass ``BaseViewer``, implement at least ``_build_scene`` (populate the
plotter with actors via ``_register_actor``) and ``_create_window``
(return the Qt dialog class), then call ``.show()`` to open the viewer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import gmsh

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase
    from pyGmsh.core.Model import Model


# ======================================================================
# Module-level constants & helpers
# ======================================================================

DimTag = tuple[int, int]

_BG_TOP    = "#1a1a2e"   # viewer dark gradient — top colour
_BG_BOTTOM = "#16213e"   # viewer dark gradient — bottom colour


def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert ``"#RRGGBB"`` to an ``(R, G, B)`` tuple of floats in [0, 1]."""
    s = hex_str.lstrip("#")
    if len(s) != 6:
        return (1.0, 1.0, 1.0)
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


# ======================================================================
# BaseViewer
# ======================================================================

class BaseViewer:
    """
    Generic 3D viewer base backed by PyVista.

    Manages the VTK plotter lifecycle (background, axes widget, AA, depth
    peeling), the custom interactor style (LMB pick / box-select, MMB pan
    / quaternion orbit, RMB pan, hover highlight), hide / isolate / reveal
    helpers, and the observer callback lists that subclass UIs wire into.

    Parameters
    ----------
    parent : _SessionBase
        Owning session instance (pyGmsh, Assembly, or Part).
    model : Model
        The Model composite -- used for sync and naming.
    dims : list[int] or None
        Dimensions to render / make pickable.  Default ``[0, 1, 2, 3]``.
    point_size : float
        Visual size multiplier for dim-0 entities.
    line_width : float
        VTK line width for dim-1 entities.
    surface_opacity : float
        Transparency for dim-2 faces.
    show_surface_edges : bool
        Whether to draw UV-grid edges on surfaces.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        model: "Model",
        *,
        dims: list[int] | None = None,
        point_size: float = 2.0,
        line_width: float = 6.0,
        surface_opacity: float = 0.35,
        show_surface_edges: bool = False,
    ) -> None:
        self._parent = parent
        self._model = model

        self._dims = list(dims) if dims is not None else [0, 1, 2, 3]

        # Visual properties (can be mutated by a preferences panel)
        self._point_size = point_size
        self._line_width = line_width
        self._surface_opacity = surface_opacity
        self._show_surface_edges = show_surface_edges

        # Populated by _setup_on()
        self._plotter: Any = None

        # Actor <-> entity-ID registries.
        # Keys in _actor_to_id are ``id(actor)`` (int);
        # keys in _id_to_actor are whatever the subclass uses as entity
        # identifiers (DimTag for BRep, node/elem tag for mesh, etc.).
        self._actor_to_id: dict[int, Any] = {}
        self._id_to_actor: dict[Any, Any] = {}

        # Hover-highlight state
        self._hover_id: Any | None = None
        self._hover_picker: Any = None   # lazy vtkCellPicker
        self._hover_throttle: int = 0    # skip N-1 of N move events

        # Drag threshold (px) before LMB-move becomes a box-select
        self._drag_threshold: int = 8

        # Working set -- the entities currently selected / picked.
        # Subclass decides the element type (DimTag, int, ...).
        self._picks: list = []

        # Hidden entities (manually hidden via H / isolated via I)
        self._hidden: set = set()

        # Active pick-filter dims (which dims respond to clicks)
        self._pickable_dims: set[int] = set(self._dims)

        # Observer callbacks -- the Qt UI wires these to refresh the
        # tree / statusbar / info panel.
        self._on_pick_changed: list[Callable[[], None]] = []
        self._on_visibility_changed: list[Callable[[], None]] = []
        self._on_hover_changed: list[Callable[[], None]] = []

        # Interactive axes widget (assigned in _setup_on)
        self._orientation_widget = None

    # ------------------------------------------------------------------
    # Callback helpers
    # ------------------------------------------------------------------

    def _fire_pick_changed(self) -> None:
        """Notify all registered pick-changed listeners."""
        for cb in self._on_pick_changed:
            try:
                cb()
            except Exception:
                pass

    def _fire_visibility_changed(self) -> None:
        """Notify all registered visibility-changed listeners."""
        for cb in self._on_visibility_changed:
            try:
                cb()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Actor registration
    # ------------------------------------------------------------------

    def _register_actor(self, actor, entity_id: Any) -> None:
        """Store a bidirectional mapping between *actor* and *entity_id*."""
        self._actor_to_id[id(actor)] = entity_id
        self._id_to_actor[entity_id] = actor

    # ------------------------------------------------------------------
    # Plotter configuration (called by the Qt window)
    # ------------------------------------------------------------------

    def _setup_on(self, plotter) -> None:
        """
        Configure a PyVista ``BasePlotter`` (``pv.Plotter`` or
        ``pyvistaqt.QtInteractor``) with background, axes, anti-aliasing,
        depth peeling, then build the scene, install the custom interactor
        style (LMB pick, LMB drag box-select, MMB pan / orbit, RMB pan,
        hover highlight), install keybindings, and render.
        """
        self._plotter = plotter

        # ------ render-window knobs (MSAA + smoothing) ------
        try:
            rwin = plotter.render_window
            rwin.SetMultiSamples(8)           # 8x MSAA
            rwin.SetLineSmoothing(True)
            rwin.SetPointSmoothing(True)
            rwin.SetPolygonSmoothing(True)
        except Exception:
            pass

        # ------ background gradient ------
        plotter.set_background(_BG_TOP, top=_BG_BOTTOM)

        # ------ interactive axes widget ------
        self._orientation_widget = None
        try:
            import vtk as _vtk

            axes = _vtk.vtkAxesActor()
            axes.SetShaftTypeToCylinder()
            axes.SetCylinderRadius(0.04)
            axes.SetConeRadius(0.3)
            axes.SetTotalLength(1.0, 1.0, 1.0)
            axes.SetNormalizedLabelPosition(1.3, 1.3, 1.3)

            # Axis colours: X=red, Y=green, Z=blue
            axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(
                1.0, 0.2, 0.2,
            )
            axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(
                0.2, 0.8, 0.2,
            )
            axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(
                0.3, 0.5, 1.0,
            )

            for cap in (
                axes.GetXAxisCaptionActor2D(),
                axes.GetYAxisCaptionActor2D(),
                axes.GetZAxisCaptionActor2D(),
            ):
                cap.GetCaptionTextProperty().SetFontSize(14)
                cap.GetCaptionTextProperty().SetBold(True)
                cap.GetCaptionTextProperty().SetShadow(False)
                cap.SetBorder(False)
                cap.GetTextActor().SetTextScaleModeToNone()

            widget = _vtk.vtkOrientationMarkerWidget()
            widget.SetOrientationMarker(axes)
            widget.SetInteractor(plotter.iren.interactor)
            widget.SetViewport(0.0, 0.0, 0.15, 0.15)
            widget.SetOutlineColor(0, 0, 0)
            widget.OutlineOff()
            widget.SetInteractive(True)
            widget.EnabledOn()
            self._orientation_widget = widget
        except Exception:
            plotter.add_axes(
                interactive=False, line_width=2, color="white",
                xlabel="X", ylabel="Y", zlabel="Z",
            )

        # ------ SSAA anti-aliasing ------
        try:
            plotter.enable_anti_aliasing("ssaa")
        except Exception:
            try:
                plotter.enable_anti_aliasing()
            except Exception:
                pass

        # ------ depth peeling ------
        try:
            plotter.enable_depth_peeling(number_of_peels=8)
        except Exception:
            pass

        # ------ clear actor registries ------
        self._actor_to_id.clear()
        self._id_to_actor.clear()

        # ------ build scene (abstract) ------
        self._build_scene()

        # ------ keybindings (virtual, empty default) ------
        self._install_keybindings()

        # ------ custom interactor / picker ------
        self._install_picker()

        # ------ status display (virtual, no-op default) ------
        self._update_status()

        # ------ initial render ------
        try:
            plotter.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Abstract / virtual hooks
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        """Populate the plotter with actors.  **Must be overridden.**"""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_scene()"
        )

    def _install_keybindings(self) -> None:
        """Install keyboard shortcuts.  Empty default -- subclass overrides."""

    def _update_status(self) -> None:
        """Refresh status display.  Default: remove stale HUD actor."""
        try:
            self._plotter.remove_actor("_picker_hud")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ViewCube helpers
    # ------------------------------------------------------------------

    def _is_over_widget(self, x: int, y: int) -> bool:
        """Return ``True`` if screen coord *(x, y)* falls inside the
        orientation-marker widget viewport."""
        w = self._orientation_widget
        if w is None:
            return False
        try:
            rw = self._plotter.render_window
            sz = rw.GetSize()          # (width, height) in pixels
            vp = w.GetViewport()       # (xmin, ymin, xmax, ymax) normalised
            return (vp[0] * sz[0] <= x <= vp[2] * sz[0]
                    and vp[1] * sz[1] <= y <= vp[3] * sz[1])
        except Exception:
            return False

    def _handle_cube_click(self, x: int, y: int) -> None:
        """Pick inside the axes widget's renderer -- if a cone tip was
        clicked, snap to that axis view."""
        import vtk

        w = self._orientation_widget
        if w is None:
            return
        ren = w.GetCurrentRenderer()
        if ren is None:
            return

        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0, ren)
        pos = picker.GetPickPosition()
        if pos is None:
            return

        px, py, pz = pos
        ax = (abs(px), abs(py), abs(pz))
        dominant = ax.index(max(ax))
        if ax[dominant] < 0.35:
            return  # clicked near the centre -- not a tip
        sign = [px, py, pz][dominant] > 0
        directions = {
            (0, True):  "right",  (0, False): "left",
            (1, True):  "back",   (1, False): "front",
            (2, True):  "top",    (2, False): "bottom",
        }
        view = directions.get((dominant, sign))
        if view is not None:
            self._do_snap_view(view)

    def _do_snap_view(self, direction: str) -> None:
        """Snap the main camera to a standard orthogonal view."""
        try:
            p = self._plotter
            views = {
                "top":    lambda: p.view_xy(negative=False),
                "bottom": lambda: p.view_xy(negative=True),
                "front":  lambda: p.view_xz(negative=False),
                "back":   lambda: p.view_xz(negative=True),
                "right":  lambda: p.view_yz(negative=False),
                "left":   lambda: p.view_yz(negative=True),
                "iso":    lambda: p.view_isometric(),
            }
            views[direction]()
            p.reset_camera()
            p.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Centroid projection
    # ------------------------------------------------------------------

    def _project_centroid(self, actor) -> tuple[float, float] | None:
        """Project an actor's 3D bounding-box centre to display coords.

        Returns ``(sx, sy)`` or ``None`` if bounds are degenerate.
        """
        bx0, bx1, by0, by1, bz0, bz1 = actor.GetBounds()
        if bx0 > bx1:
            return None
        cx = (bx0 + bx1) * 0.5
        cy = (by0 + by1) * 0.5
        cz = (bz0 + bz1) * 0.5
        renderer = self._plotter.renderer
        renderer.SetWorldPoint(cx, cy, cz, 1.0)
        renderer.WorldToDisplay()
        dx, dy, _dz = renderer.GetDisplayPoint()
        return (dx, dy)

    # ------------------------------------------------------------------
    # Picker / interactor installation
    # ------------------------------------------------------------------

    def _install_picker(self) -> None:
        """Install camera-control style + high-priority VTK observers for
        click picking, rubber-band box-select, and hover highlight.

        Gesture mapping
        ~~~~~~~~~~~~~~~
        LEFT click           : pick / toggle entity (pixel-accurate)
        LEFT drag            : rubber-band box-select (additive)
                                   L->R = window (enclosed)
                                   R->L = crossing (overlap)
        Ctrl+LEFT click      : unpick entity under cursor
        Ctrl+LEFT drag       : rubber-band box-UNselect
        MIDDLE drag          : pan camera
        Shift+MIDDLE drag    : rotate camera (quaternion orbit)
        RIGHT drag           : pan camera
        WHEEL                : zoom
        hover (no button)    : highlight entity under cursor

        The observer pattern (with AbortFlagOn) is used instead of a
        subclassed interactor style -- it composes reliably with
        pyvistaqt's QtInteractor, which otherwise can swallow or
        re-install styles.
        """
        import vtk

        iren_wrap = self._plotter.iren
        iren = iren_wrap.interactor
        renderer = self._plotter.renderer

        # Base trackball style -- we override ALL button handling via
        # observers.  The style is only used for its OnMouseMove
        # camera-update logic (reads state set by StartRotate / StartPan
        # and moves the camera).
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
        self._interactor_style = style

        # Per-gesture state
        self._press_pos: tuple[int, int] | None = None
        self._dragging: bool = False
        self._ctrl_held: bool = False
        self._widget_press: tuple[int, int] | None = None
        self._rubberband_actor = None
        self._rubberband_pts = None
        self._rubberband_poly = None

        picker_self = self

        # ---- rubber-band helpers ----

        def _ensure_rubberband() -> None:
            if picker_self._rubberband_actor is not None:
                return
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(4)
            for i in range(4):
                pts.SetPoint(i, 0.0, 0.0, 0.0)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(5)
            for i in (0, 1, 2, 3, 0):
                lines.InsertCellPoint(i)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            mapper.SetTransformCoordinate(coord)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            actor.GetProperty().SetLineWidth(1.5)
            actor.VisibilityOff()
            renderer.AddActor2D(actor)
            picker_self._rubberband_actor = actor
            picker_self._rubberband_pts = pts
            picker_self._rubberband_poly = poly

        def _update_rubberband(x0, y0, x1, y1) -> None:
            _ensure_rubberband()
            pts = picker_self._rubberband_pts
            pts.SetPoint(0, x0, y0, 0.0)
            pts.SetPoint(1, x1, y0, 0.0)
            pts.SetPoint(2, x1, y1, 0.0)
            pts.SetPoint(3, x0, y1, 0.0)
            pts.Modified()
            picker_self._rubberband_poly.Modified()
            crossing = x1 < x0
            prop = picker_self._rubberband_actor.GetProperty()
            try:
                prop.SetLineStipplePattern(0x3333 if crossing else 0xFFFF)
                prop.SetLineStippleRepeatFactor(1)
            except Exception:
                pass
            picker_self._rubberband_actor.VisibilityOn()
            picker_self._plotter.render()

        def _hide_rubberband() -> None:
            if picker_self._rubberband_actor is not None:
                picker_self._rubberband_actor.VisibilityOff()
                picker_self._plotter.render()

        # ---- quaternion orbit ----

        _orbit_pivot: list = [None]   # (cx, cy, cz) or None
        _orbit_last:  list = [None]   # last mouse (x, y)

        def _selection_centroid():
            """Return centroid of picked entities, or ``None``."""
            picks = picker_self._picks
            if not picks:
                return None
            xs, ys, zs = [], [], []
            for entity_id in picks:
                actor = picker_self._id_to_actor.get(entity_id)
                if actor is None:
                    continue
                b = actor.GetBounds()
                if b[0] <= b[1]:
                    xs.append((b[0] + b[1]) * 0.5)
                    ys.append((b[2] + b[3]) * 0.5)
                    zs.append((b[4] + b[5]) * 0.5)
            if not xs:
                return None
            return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))

        def _scene_center():
            """Return the centre of the scene bounding box."""
            try:
                b = renderer.ComputeVisiblePropBounds()
                if b[0] <= b[1]:
                    return (
                        (b[0] + b[1]) * 0.5,
                        (b[2] + b[3]) * 0.5,
                        (b[4] + b[5]) * 0.5,
                    )
            except Exception:
                pass
            return (0.0, 0.0, 0.0)

        def _orbit_around(pivot, dx_px, dy_px):
            """Orbit camera around *pivot* using unit-quaternion rotation.

            Composes an azimuth quaternion (around world-up) with an
            elevation quaternion (around the camera right vector) into a
            single rotation, then applies it to both the camera position
            and focal point relative to the pivot.  Quaternion product is
            norm-preserving -- no drift, no gimbal lock, numerically
            stable over long drag sequences.
            """
            import math

            cam = renderer.GetActiveCamera()
            pos = cam.GetPosition()
            fp  = cam.GetFocalPoint()
            up  = cam.GetViewUp()

            az_rad = -dx_px * 0.005
            el_rad =  dy_px * 0.005

            # ---- quaternion helpers (w, x, y, z) ----
            def quat_from_axis_angle(ax, ang):
                ha = ang * 0.5
                s = math.sin(ha)
                return (math.cos(ha), ax[0] * s, ax[1] * s, ax[2] * s)

            def quat_mul(a, b):
                return (
                    a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                    a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                    a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                    a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
                )

            def quat_conj(q):
                return (q[0], -q[1], -q[2], -q[3])

            def quat_rotate(q, v):
                """Rotate vector *v* by quaternion *q*: q * v * q*."""
                vq = (0.0, v[0], v[1], v[2])
                r = quat_mul(quat_mul(q, vq), quat_conj(q))
                return (r[1], r[2], r[3])

            # ---- build rotation ----
            vd = [fp[i] - pos[i] for i in range(3)]
            d = math.sqrt(sum(v * v for v in vd))
            if d < 1e-12:
                return
            vd = [v / d for v in vd]

            # Right = viewdir x up (normalised)
            right = [
                vd[1] * up[2] - vd[2] * up[1],
                vd[2] * up[0] - vd[0] * up[2],
                vd[0] * up[1] - vd[1] * up[0],
            ]
            rn = math.sqrt(sum(v * v for v in right))
            if rn < 1e-12:
                return
            right = [v / rn for v in right]

            # Normalise up for the azimuth axis
            un = math.sqrt(sum(v * v for v in up))
            up_n = [v / un for v in up] if un > 1e-12 else [0, 0, 1]

            q_az = quat_from_axis_angle(up_n, az_rad)
            q_el = quat_from_axis_angle(right, el_rad)
            q = quat_mul(q_el, q_az)       # combined rotation

            # ---- apply to position & focal point around pivot ----
            pv = pivot
            p_rel  = (pos[0] - pv[0], pos[1] - pv[1], pos[2] - pv[2])
            fp_rel = (fp[0] - pv[0],  fp[1] - pv[1],  fp[2] - pv[2])

            rp  = quat_rotate(q, p_rel)
            rfp = quat_rotate(q, fp_rel)

            cam.SetPosition(pv[0] + rp[0],  pv[1] + rp[1],  pv[2] + rp[2])
            cam.SetFocalPoint(pv[0] + rfp[0], pv[1] + rfp[1], pv[2] + rfp[2])
            cam.OrthogonalizeViewUp()
            renderer.ResetCameraClippingRange()

        # ---- abort helper ----

        _tag_press:       list[int] = [0]
        _tag_move:        list[int] = [0]
        _tag_release:     list[int] = [0]
        _tag_mmb_press:   list[int] = [0]
        _tag_mmb_release: list[int] = [0]
        _tag_rmb_press:   list[int] = [0]
        _tag_rmb_release: list[int] = [0]

        def _abort(caller, tag: int) -> None:
            cmd = caller.GetCommand(tag)
            if cmd is not None:
                cmd.SetAbortFlag(1)

        # ---- observer callbacks ----

        def on_lmb_press(caller, _event):
            x, y = caller.GetEventPosition()
            # Let the ViewCube widget handle its own events
            if picker_self._is_over_widget(x, y):
                picker_self._widget_press = (x, y)
                return
            picker_self._widget_press = None
            # All LMB gestures are ours -- pick, box-select, or
            # Ctrl variants for unpick / box-unselect.
            picker_self._press_pos = (x, y)
            picker_self._dragging = False
            picker_self._ctrl_held = bool(caller.GetControlKey())
            _abort(caller, _tag_press[0])

        def on_mouse_move(caller, _event):
            # Custom orbit around selection pivot
            if _orbit_pivot[0] is not None and _orbit_last[0] is not None:
                px, py = caller.GetEventPosition()
                lx, ly = _orbit_last[0]
                _orbit_last[0] = (px, py)
                _orbit_around(_orbit_pivot[0], px - lx, py - ly)
                picker_self._plotter.render()
                _abort(caller, _tag_move[0])
                return
            # ViewCube is handling a drag -- don't interfere
            if getattr(picker_self, '_widget_press', None) is not None:
                return
            # LMB-drag in our custom mode -> rubber-band
            if picker_self._press_pos is not None:
                px, py = caller.GetEventPosition()
                sx, sy = picker_self._press_pos
                if not picker_self._dragging:
                    if (px - sx) ** 2 + (py - sy) ** 2 > picker_self._drag_threshold ** 2:
                        picker_self._dragging = True
                if picker_self._dragging:
                    _update_rubberband(sx, sy, px, py)
                _abort(caller, _tag_move[0])
                return
            # Idle hover
            px, py = caller.GetEventPosition()
            picker_self._update_hover(px, py)
            # Do NOT abort -- allow trackball to see move events for MMB

        def on_lmb_release(caller, _event):
            # If the press started on the ViewCube, detect face click
            wp = getattr(picker_self, '_widget_press', None)
            if wp is not None:
                rx, ry = caller.GetEventPosition()
                dx, dy = rx - wp[0], ry - wp[1]
                picker_self._widget_press = None
                # Only snap if it was a click (not a drag-to-orbit)
                if dx * dx + dy * dy <= picker_self._drag_threshold ** 2:
                    picker_self._handle_cube_click(rx, ry)
                return
            if picker_self._press_pos is None:
                return
            x0, y0 = picker_self._press_pos
            x1, y1 = caller.GetEventPosition()
            ctrl = picker_self._ctrl_held
            _hide_rubberband()
            if picker_self._dragging:
                rx0, rx1 = min(x0, x1), max(x0, x1)
                ry0, ry1 = min(y0, y1), max(y0, y1)
                crossing = x1 < x0
                picker_self._on_box_select(
                    rx0, ry0, rx1, ry1, crossing, ctrl,
                )
                picker_self._update_status()
            else:
                picker_self._on_lmb_click(x1, y1, ctrl)
            picker_self._press_pos = None
            picker_self._dragging = False
            picker_self._ctrl_held = False
            _abort(caller, _tag_release[0])

        # ---- MMB / RMB observers (camera pan + rotate) ----

        def on_mmb_press(caller, _event):
            if caller.GetShiftKey():
                # Quaternion orbit: selection centroid or scene centre
                pivot = _selection_centroid() or _scene_center()
                _orbit_pivot[0] = pivot
                _orbit_last[0] = caller.GetEventPosition()
            else:
                _orbit_pivot[0] = None
                picker_self._interactor_style.StartPan()
            _abort(caller, _tag_mmb_press[0])

        def on_mmb_release(caller, _event):
            if _orbit_pivot[0] is not None:
                _orbit_pivot[0] = None
                _orbit_last[0] = None
                _abort(caller, _tag_mmb_release[0])
                return
            state = picker_self._interactor_style.GetState()
            if state == 2:    # VTKIS_PAN
                picker_self._interactor_style.EndPan()
            _abort(caller, _tag_mmb_release[0])

        def on_rmb_press(caller, _event):
            picker_self._interactor_style.StartPan()
            _abort(caller, _tag_rmb_press[0])

        def on_rmb_release(caller, _event):
            state = picker_self._interactor_style.GetState()
            if state == 2:
                picker_self._interactor_style.EndPan()
            _abort(caller, _tag_rmb_release[0])

        # ---- register observers at priority 10.0 ----
        _tag_press[0] = iren.AddObserver(
            "LeftButtonPressEvent", on_lmb_press, 10.0,
        )
        _tag_move[0] = iren.AddObserver(
            "MouseMoveEvent", on_mouse_move, 10.0,
        )
        _tag_release[0] = iren.AddObserver(
            "LeftButtonReleaseEvent", on_lmb_release, 10.0,
        )
        _tag_mmb_press[0] = iren.AddObserver(
            "MiddleButtonPressEvent", on_mmb_press, 10.0,
        )
        _tag_mmb_release[0] = iren.AddObserver(
            "MiddleButtonReleaseEvent", on_mmb_release, 10.0,
        )
        _tag_rmb_press[0] = iren.AddObserver(
            "RightButtonPressEvent", on_rmb_press, 10.0,
        )
        _tag_rmb_release[0] = iren.AddObserver(
            "RightButtonReleaseEvent", on_rmb_release, 10.0,
        )

    # ------------------------------------------------------------------
    # Virtual hooks called by _install_picker()
    # ------------------------------------------------------------------

    def _on_lmb_click(self, x: int, y: int, ctrl: bool) -> None:
        """Handle a left-mouse-button click at display coords *(x, y)*.

        *ctrl* is ``True`` when the Ctrl key was held during the click.
        Subclass overrides to implement pixel-picking logic.
        """

    def _on_box_select(
        self,
        x0: int, y0: int, x1: int, y1: int,
        crossing: bool,
        ctrl: bool,
    ) -> None:
        """Handle a rubber-band box-select from *(x0, y0)* to *(x1, y1)*.

        *crossing* is ``True`` when the drag went right-to-left (crossing
        selection mode).  *ctrl* is ``True`` for box-unselect.
        Subclass overrides to implement box-selection logic.
        """

    # ------------------------------------------------------------------
    # Hover highlight
    # ------------------------------------------------------------------

    def _update_hover(self, x: int, y: int) -> None:
        """Pixel-precise hover highlight using a software cell picker
        (``vtkCellPicker``).  Throttled via ``_hover_throttle`` -- only
        every 3rd call runs the picker to keep idle-move dispatch cheap.
        """
        import vtk

        if self._plotter is None:
            return

        # Throttle: only pick every 3rd move event
        self._hover_throttle -= 1
        if self._hover_throttle > 0:
            return
        self._hover_throttle = 3

        # Lazy picker
        if self._hover_picker is None:
            p = vtk.vtkCellPicker()
            p.SetTolerance(0.005)
            self._hover_picker = p

        self._hover_picker.Pick(x, y, 0, self._plotter.renderer)
        prop = self._hover_picker.GetViewProp()

        new_id = None
        if prop is not None:
            candidate = self._actor_to_id.get(id(prop))
            if candidate is not None:
                new_id = candidate

        if new_id == self._hover_id:
            return

        old_id = self._hover_id
        self._hover_id = new_id
        self._on_hover_changed_internal(old_id, new_id)

        for cb in self._on_hover_changed:
            try:
                cb()
            except Exception:
                pass

    def _on_hover_changed_internal(self, old_id: Any, new_id: Any) -> None:
        """Hook for subclass to recolor actors on hover change.

        Called before the ``_on_hover_changed`` callback list is fired.
        Empty default -- subclass overrides to recolor the old / new
        hovered entity.
        """

    # ------------------------------------------------------------------
    # Visibility helpers (hide / isolate / reveal)
    # ------------------------------------------------------------------

    def _hide_selected(self) -> None:
        """Hide every currently-picked entity."""
        if not self._picks:
            return
        for entity_id in list(self._picks):
            actor = self._id_to_actor.get(entity_id)
            if actor is not None:
                actor.VisibilityOff()
                actor.SetPickable(False)
            self._hidden.add(entity_id)
        self._picks.clear()
        # Clear pick_history if the subclass has one
        if hasattr(self, '_pick_history'):
            self._pick_history.clear()
        self._plotter.render()
        self._update_status()
        self._fire_pick_changed()
        self._fire_visibility_changed()

    def _isolate_selected(self) -> None:
        """Hide every entity that is NOT currently picked."""
        if not self._picks:
            return
        visible_ids = set(self._picks)
        for entity_id, actor in self._id_to_actor.items():
            if entity_id not in visible_ids:
                actor.VisibilityOff()
                actor.SetPickable(False)
                self._hidden.add(entity_id)
        self._plotter.render()
        self._update_status()
        self._fire_visibility_changed()

    def _show_all(self) -> None:
        """Reveal every hidden entity and restore pickability."""
        if not self._hidden:
            return
        for entity_id in list(self._hidden):
            actor = self._id_to_actor.get(entity_id)
            if actor is not None:
                actor.VisibilityOn()
                actor.SetPickable(True)
        self._hidden.clear()
        self._plotter.render()
        self._update_status()
        self._fire_visibility_changed()

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def show(self, *, title: str | None = None, maximized: bool = True):
        """Open the viewer window.  Blocks until closed."""
        gmsh.model.occ.synchronize()
        default_title = (
            f"{type(self).__name__} — {self._parent.model_name} - Ladruño"
        )
        window = self._create_window(
            title=title or default_title, maximized=maximized,
        )
        window.exec()
        self._on_window_closed()
        return self

    def _create_window(self, *, title: str, maximized: bool):
        """Return the Qt dialog / window instance.  **Must be overridden.**"""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _create_window()"
        )

    def _on_window_closed(self) -> None:
        """Hook for cleanup after the window closes.  Empty default."""
